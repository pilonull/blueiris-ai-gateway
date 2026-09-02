import asyncio
from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import logging
import os
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple
import urllib.request
from facenet_pytorch import MTCNN, InceptionResnetV1
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, status
import numpy as np
from PIL import Image
import tensorrt
import torch
import torch.nn.functional as F
from ultralytics import YOLO

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("bi-gateway")

# --- Configuration ---
MODELS_DIR = Path("/app/models")
FACES_DB_PATH = MODELS_DIR / "faces_db.pt"
DEFAULT_MODEL_NAME = os.getenv("DEFAULT_MODEL", "yolo11m")
PRELOAD_MODELS_ENV = os.getenv("PRELOAD_MODELS", "").strip()
ALLOW_LAZY_LOAD = os.getenv("ALLOW_LAZY_LOAD", "false").lower() in ("true", "1")
HALF_PRECISION = os.getenv("HALF_PRECISION", "true").lower() in ("true", "1")
INFERENCE_TIMEOUT = float(os.getenv("INFERENCE_TIMEOUT", "8.0"))
RECOVERY_GRACE_PERIOD = float(os.getenv("RECOVERY_GRACE_PERIOD", "4.0"))
ALERT_WEBHOOK_URL = os.getenv("ALERT_WEBHOOK_URL", "").strip()

MODEL_EXT_PRIORITY = [".engine", ".onnx", ".pt"]


@dataclass
class ModelEntry:
    model: YOLO
    is_pt: bool
    format_name: str


# Global YOLO State
loaded_models: Dict[str, ModelEntry] = {}
failed_models: Dict[str, str] = {}
cache_lock = asyncio.Lock()

# Global Face Recognition State
face_detector: Optional[MTCNN] = None
face_recognizer: Optional[InceptionResnetV1] = None
registered_faces: Dict[str, List[torch.Tensor]] = {}

# Global GPU Execution Gate & Watchdog State
gpu_lock = asyncio.Lock()
gpu_tainted: bool = False
gpu_taint_generation: int = 0
recovery_timer_handle: Optional[asyncio.TimerHandle] = None

# Telemetry State for Concurrency/Queue Tracking
queue_depth: int = 0
active_inferences: int = 0

is_healthy: bool = True
health_failure_reason: Optional[str] = None


def discover_models() -> Dict[str, Path]:
    """Map model stem -> best available file on disk (Engine > ONNX > PyTorch)."""
    all_files = list(MODELS_DIR.glob("*.*"))
    stems = {f.stem for f in all_files if f.suffix.lower() in MODEL_EXT_PRIORITY}

    found: Dict[str, Path] = {}
    for stem in stems:
        for ext in MODEL_EXT_PRIORITY:
            candidate = MODELS_DIR / f"{stem}{ext}"
            if candidate.exists():
                found[stem] = candidate
                break
    return found


def ensure_tensorrt_engine(stem: str):
    """Auto-compiles a TensorRT .engine from a .pt file if missing or if runtime environment changes."""
    engine_path = MODELS_DIR / f"{stem}.engine"
    pt_path = MODELS_DIR / f"{stem}.pt"
    stamp_path = MODELS_DIR / f"{stem}.trt_version"

    current_version = f"TRT_{tensorrt.__version__}_CUDA_{torch.version.cuda}"
    needs_rebuild = False

    if not engine_path.exists() and pt_path.exists():
        needs_rebuild = True
    elif stamp_path.exists() and stamp_path.read_text().strip() != current_version:
        logger.warning(f"TensorRT/CUDA environment changed for '{stem}'. Rebuilding engine...")
        needs_rebuild = True

    if needs_rebuild and pt_path.exists():
        logger.info(f"Compiling optimized TensorRT engine for '{stem}' on RTX 5060 Ti (Takes ~60-90s)...")
        try:
            model = YOLO(str(pt_path))
            model.export(format="engine", device=0, quantize=16, imgsz=640, dynamic=False)
            stamp_path.write_text(current_version)
            logger.info(f"Successfully compiled and cached: {engine_path.name}")
        except Exception as e:
            logger.error(f"Failed to auto-compile engine for '{stem}': {e}")


def sync_predict(model: YOLO, img: Image.Image, min_conf: float, is_pt: bool):
    """Synchronous inference worker executed inside threadpool."""
    kwargs = {
        "conf": min_conf,
        "device": 0,
        "verbose": False,
    }
    if is_pt and HALF_PRECISION:
        kwargs["quantize"] = 16

    return model.predict(img, **kwargs)


def load_and_warmup_sync(path: Path) -> Tuple[YOLO, bool, str]:
    """Loads weights and forces CUDA/TensorRT buffer allocation."""
    model = YOLO(str(path))
    dummy_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    model.predict(dummy_frame, device=0, verbose=False)

    ext = path.suffix.lower()
    is_pt = (ext == ".pt")
    format_name = "TensorRT" if ext == ".engine" else ("ONNX" if ext == ".onnx" else "PyTorch")
    return model, is_pt, format_name


def save_faces_db():
    """Persists registered face embeddings atomically to prevent file corruption."""
    try:
        temp_path = FACES_DB_PATH.with_suffix(".tmp")
        torch.save(registered_faces, temp_path)
        os.replace(temp_path, FACES_DB_PATH)  # Atomic on POSIX/Linux
        logger.info(f"Faces database saved atomically ({len(registered_faces)} people enrolled).")
    except Exception as e:
        logger.error(f"Failed to persist face database: {e}")


def load_faces_db():
    """Loads enrolled face embeddings from disk on boot."""
    global registered_faces
    if FACES_DB_PATH.exists():
        try:
            registered_faces = torch.load(FACES_DB_PATH, map_location="cpu", weights_only=False)
            logger.info(f"Loaded {len(registered_faces)} enrolled people from {FACES_DB_PATH}")
        except Exception as e:
            logger.error(f"Error loading {FACES_DB_PATH}: {e}")
            registered_faces = {}


def sync_face_register(img: Image.Image, user_id: str) -> bool:
    """Extracts face embedding via MTCNN + InceptionResnetV1 and registers it."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return False

    faces = face_detector(img)
    if faces is None or len(faces) == 0:
        return False

    face_tensor = faces[0].unsqueeze(0).to("cuda:0")
    with torch.no_grad():
        emb = face_recognizer(face_tensor)
        emb = F.normalize(emb, p=2, dim=1).cpu()

    if user_id not in registered_faces:
        registered_faces[user_id] = []
    registered_faces[user_id].append(emb)
    save_faces_db()
    return True


def sync_face_recognize(img: Image.Image, min_conf: float) -> List[dict]:
    """Detects faces and compares embeddings against enrolled database."""
    assert face_detector is not None and face_recognizer is not None
    boxes, _ = face_detector.detect(img)
    if boxes is None or len(boxes) == 0:
        return []

    faces = face_detector(img)
    if faces is None or len(faces) == 0:
        return []

    faces = faces.to("cuda:0")
    with torch.no_grad():
        embeddings = face_recognizer(faces)
        embeddings = F.normalize(embeddings, p=2, dim=1)

    predictions = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box.tolist()
        emb = embeddings[i : i + 1]

        best_user = "unknown"
        best_sim = 0.0

        for user, user_embs in registered_faces.items():
            if not user_embs:
                continue
            db_tensor = torch.cat(user_embs, dim=0).to("cuda:0")
            similarities = F.cosine_similarity(emb, db_tensor)
            max_sim = float(similarities.max().item())

            if max_sim > best_sim:
                best_sim = max_sim
                best_user = user

        matched_id = best_user if best_sim >= min_conf else "unknown"

        predictions.append({
            "confidence": round(best_sim, 2),
            "userid": matched_id,
            "label": matched_id,
            "x_min": int(max(0, x1)),
            "y_min": int(max(0, y1)),
            "x_max": int(x2),
            "y_max": int(y2),
        })

    return predictions


def send_crash_alert(reason: str):
    if not ALERT_WEBHOOK_URL:
        return
    try:
        req = urllib.request.Request(
            ALERT_WEBHOOK_URL,
            data=f"Blue Iris Gateway Fatal: {reason}".encode("utf-8"),
            headers={"Title": "AI Gateway GPU Deadlock", "Priority": "urgent", "Tags": "warning,gpu"},
            method="POST",
        )
        urllib.request.urlopen(req, timeout=1.5)
        logger.info("External crash alert sent.")
    except Exception as e:
        logger.error(f"Failed to dispatch alert webhook: {e}")


def trigger_self_termination(reason: str):
    global is_healthy, health_failure_reason
    if not is_healthy:
        return
    is_healthy = False
    health_failure_reason = reason
    logger.critical(f"FATAL: {reason} — terminating process for container restart in 200ms.")

    if ALERT_WEBHOOK_URL:
        import threading
        threading.Thread(target=send_crash_alert, args=(reason,), daemon=True).start()

    loop = asyncio.get_running_loop()
    loop.call_later(0.2, os._exit, 1)


def on_orphaned_worker_done(fut: asyncio.Future, generation: int):
    global gpu_tainted, gpu_taint_generation, recovery_timer_handle
    if generation != gpu_taint_generation or fut.cancelled():
        return

    exc = fut.exception()
    if exc:
        logger.critical(f"Orphaned worker (gen {generation}) failed: {exc}")
        trigger_self_termination(f"CUDA exception in worker: {exc}")
    else:
        logger.warning(f"Orphaned worker (gen {generation}) recovered within grace window.")
        gpu_tainted = False
        if recovery_timer_handle and not recovery_timer_handle.cancelled():
            recovery_timer_handle.cancel()
            recovery_timer_handle = None


def on_grace_period_expired(generation: int):
    if generation == gpu_taint_generation:
        trigger_self_termination(f"CUDA operation failed to exit within {RECOVERY_GRACE_PERIOD}s grace window.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global is_healthy, health_failure_reason, face_detector, face_recognizer
    load_faces_db()

    # Initialize and warm up Face Recognition models
    logger.info("Initializing FaceNet (MTCNN + InceptionResnetV1) on CUDA...")
    try:
        face_detector = MTCNN(keep_all=True, device="cuda:0", post_process=True)
        face_recognizer = InceptionResnetV1(pretrained="vggface2").eval().to("cuda:0")
        dummy = Image.fromarray(np.zeros((160, 160, 3), dtype=np.uint8))
        face_detector.detect(dummy)
        logger.info("FaceNet pipeline initialized and warmed on GPU.")
    except Exception as e:
        logger.error(f"Failed to initialize FaceNet: {e}")

    # Determine target stems to preload/compile
    target_stems = set()
    if PRELOAD_MODELS_ENV.lower() == "all":
        discovered = discover_models()
        target_stems = set(discovered.keys())
    elif PRELOAD_MODELS_ENV:
        target_stems = {s.strip() for s in PRELOAD_MODELS_ENV.split(",") if s.strip()}
    else:
        target_stems = {Path(DEFAULT_MODEL_NAME).stem}

    logger.info("Checking TensorRT engine compilation status...")
    for stem in target_stems:
        await asyncio.to_thread(ensure_tensorrt_engine, stem)

    # Discover and preload models
    discovered = discover_models()
    default_stem = Path(DEFAULT_MODEL_NAME).stem

    logger.info("Starting model preloading sequence...")
    for stem in target_stems:
        model_path = discovered[stem] if stem in discovered else MODELS_DIR / f"{stem}.pt"
        try:
            model, is_pt, fmt = await asyncio.to_thread(load_and_warmup_sync, model_path)
            loaded_models[stem] = ModelEntry(model=model, is_pt=is_pt, format_name=fmt)
        except Exception as e:
            failed_models[stem] = str(e)
            if stem == default_stem:
                logger.critical(f"Default model '{stem}' failed to load: {e}")
                send_crash_alert(f"Startup crash: default model '{stem}' failed to load.")
                os._exit(1)

    logger.info("==================================================")
    logger.info("            AI GATEWAY STARTUP SUMMARY            ")
    logger.info("==================================================")
    for name, m in loaded_models.items():
        fp_mode = str(HALF_PRECISION).lower() if m.is_pt else "Native"
        logger.info(f"  [+] {name:<16} Format: {m.format_name:<8} FP16: {fp_mode}")
    logger.info(f"  [+] Face Recognition Enrolled: {len(registered_faces)} person(s)")
    for name, err in failed_models.items():
        logger.warning(f"  [!] {name:<16} Error: {err}")
    logger.info("==================================================")

    yield
    loaded_models.clear()
    failed_models.clear()


app = FastAPI(title="Blue Iris YOLO & Face Gateway", lifespan=lifespan)


async def get_model_entry(requested_name: str) -> Optional[ModelEntry]:
    if gpu_tainted:
        return None

    stem = Path(requested_name).stem
    if stem in loaded_models:
        return loaded_models[stem]

    if not ALLOW_LAZY_LOAD:
        return None

    async with cache_lock:
        if gpu_tainted:
            return None
        if stem in loaded_models:
            return loaded_models[stem]

        discovered = discover_models()
        if stem not in discovered:
            return None

        target_path = discovered[stem]
        logger.info(f"Lazy loading model: {target_path.name}")
        try:
            model, is_pt, fmt = await asyncio.to_thread(load_and_warmup_sync, target_path)
            entry = ModelEntry(model=model, is_pt=is_pt, format_name=fmt)
            loaded_models[stem] = entry
            return entry
        except Exception as e:
            logger.error(f"Failed to lazy load {target_path.name}: {e}")
            failed_models[stem] = str(e)
            return None


async def run_inference(image_bytes: bytes, min_conf: float, model_name: str, t_start: float):
    global gpu_tainted, gpu_taint_generation, recovery_timer_handle, queue_depth, active_inferences

    if not is_healthy:
        return {"success": False, "error": f"Gateway shutting down: {health_failure_reason}", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}
    if gpu_tainted:
        return {"success": False, "error": "GPU execution frozen during recovery grace window.", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    try:
        entry = await get_model_entry(model_name)
        if entry is None:
            return {"success": False, "error": f"Model '{model_name}' is not loaded or available on server.", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return {"success": False, "error": f"Bad request/corrupt image: {e}", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    queue_depth += 1
    try:
        async with gpu_lock:
            queue_depth -= 1
            active_inferences += 1
            try:
                if gpu_tainted:
                    return {"success": False, "error": "GPU frozen.", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

                loop = asyncio.get_running_loop()
                t_infer_start = time.perf_counter()
                worker_future = loop.run_in_executor(None, sync_predict, entry.model, img, min_conf, entry.is_pt)

                try:
                    results = await asyncio.wait_for(asyncio.shield(worker_future), timeout=INFERENCE_TIMEOUT)
                    t_infer_end = time.perf_counter()
                except asyncio.TimeoutError:
                    gpu_tainted = True
                    gpu_taint_generation += 1
                    current_gen = gpu_taint_generation
                    msg = f"Inference timed out after {INFERENCE_TIMEOUT}s on '{model_name}'."
                    logger.error(msg)
                    worker_future.add_done_callback(lambda fut, gen=current_gen: on_orphaned_worker_done(fut, gen))
                    recovery_timer_handle = loop.call_later(RECOVERY_GRACE_PERIOD, lambda gen=current_gen: on_grace_period_expired(gen))
                    return {"success": False, "error": msg, "inferenceMs": int((time.perf_counter() - t_infer_start) * 1000), "processMs": int((time.perf_counter() - t_start) * 1000)}
                except Exception as e:
                    logger.error(f"Inference error on '{model_name}': {e}", exc_info=True)
                    return {"success": False, "error": str(e), "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}
            finally:
                active_inferences -= 1
    except asyncio.CancelledError:
        queue_depth -= 1
        raise

    infer_ms = int((t_infer_end - t_infer_start) * 1000)
    process_ms = int((time.perf_counter() - t_start) * 1000)

    predictions = []
    for r in results:
        for box in r.boxes:
            coords = box.xyxy[0].tolist()
            predictions.append({
                "confidence": round(float(box.conf[0]), 2),
                "label": entry.model.names[int(box.cls[0])],
                "x_min": int(coords[0]), "y_min": int(coords[1]),
                "x_max": int(coords[2]), "y_max": int(coords[3]),
            })

    return {
        "success": True,
        "message": f"{len(predictions)} object(s) detected",
        "predictions": predictions,
        "count": len(predictions),
        "command": "detect",
        "moduleName": f"YOLO Gateway ({entry.format_name})",
        "executionProvider": "CUDA",
        "canUseGPU": True,
        "inferenceMs": infer_ms,
        "processMs": process_ms,
    }


# ============================================================================
# FACE RECOGNITION ENDPOINTS
# ============================================================================

@app.api_route("/v1/vision/face/list", methods=["GET", "POST"])
async def face_list():
    return {"success": True, "faces": sorted(list(registered_faces.keys()))}


@app.post("/v1/vision/face/register")
async def face_register(
    image: UploadFile = File(...),
    userid: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
):
    target_id = (userid or name or "").strip()
    if not target_id:
        return {"success": False, "error": "Missing user ID or name."}

    contents = await image.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {"success": False, "error": f"Invalid image: {e}"}

    async with gpu_lock:
        loop = asyncio.get_running_loop()
        success = await loop.run_in_executor(None, sync_face_register, img, target_id)

    if success:
        return {"success": True, "message": f"Face registered for {target_id}"}
    return {"success": False, "error": "No face detected in provided image."}


@app.post("/v1/vision/face/delete")
async def face_delete(
    userid: Optional[str] = Form(None),
    name: Optional[str] = Form(None),
):
    target_id = (userid or name or "").strip()
    if target_id in registered_faces:
        del registered_faces[target_id]
        save_faces_db()
        return {"success": True, "message": f"Face deleted for {target_id}"}
    return {"success": False, "error": f"User '{target_id}' not found."}


@app.post("/v1/vision/face/recognize")
async def face_recognize(
    image: UploadFile = File(...),
    min_confidence: float = Form(0.60),
):
    t_start = time.perf_counter()
    contents = await image.read()

    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        return {"success": False, "error": f"Invalid image: {e}", "inferenceMs": 0, "processMs": int((time.perf_counter() - t_start) * 1000)}

    async with gpu_lock:
        loop = asyncio.get_running_loop()
        t_infer_start = time.perf_counter()
        predictions = await loop.run_in_executor(None, sync_face_recognize, img, min_confidence)
        t_infer_end = time.perf_counter()

    return {
        "success": True,
        "message": f"{len(predictions)} face(s) detected",
        "predictions": predictions,
        "count": len(predictions),
        "command": "recognize",
        "moduleName": "Face Recognition (FaceNet)",
        "executionProvider": "CUDA",
        "canUseGPU": True,
        "inferenceMs": int((t_infer_end - t_infer_start) * 1000),
        "processMs": int((time.perf_counter() - t_start) * 1000),
    }


# ============================================================================
# DIAGNOSTIC & DETECTION ENDPOINTS
# ============================================================================

@app.get("/")
async def root_health(response: Response):
    if not is_healthy or gpu_tainted:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "gpu_tainted": gpu_tainted, "reason": health_failure_reason}
    return {
        "status": "ok",
        "loaded_models": list(loaded_models.keys()),
        "registered_faces": sorted(list(registered_faces.keys())),
    }


@app.api_route("/status", methods=["GET"])
@app.api_route("/v1/status", methods=["GET"])
async def detailed_status(response: Response):
    if not is_healthy or gpu_tainted:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return {
        "success": is_healthy and not gpu_tainted,
        "status": "ok" if (is_healthy and not gpu_tainted) else "degraded",
        "canUseGPU": True,
        "executionProvider": "CUDA",
        "queue_depth": queue_depth,
        "active_inferences": active_inferences,
        "models": {k: {"status": "ready", "format": v.format_name} for k, v in loaded_models.items()},
        "registered_faces": sorted(list(registered_faces.keys())),
    }


@app.api_route("/v1/vision/custom/list", methods=["GET", "POST"])
async def list_custom_models():
    return {"success": True, "models": sorted(discover_models().keys())}


@app.post("/v1/vision/detection")
async def detection(
    image: UploadFile = File(...),
    min_confidence: float = Form(0.4),
    model: Optional[str] = Form(None),
):
    t_start = time.perf_counter()
    image_bytes = await image.read()
    return await run_inference(image_bytes, min_confidence, model or DEFAULT_MODEL_NAME, t_start=t_start)


@app.post("/v1/vision/custom/{model_name}")
async def custom_detection(
    model_name: str,
    image: UploadFile = File(...),
    min_confidence: float = Form(0.4),
):
    t_start = time.perf_counter()
    image_bytes = await image.read()
    return await run_inference(image_bytes, min_confidence, model_name, t_start=t_start)
