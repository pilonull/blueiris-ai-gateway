# !!!WARNING!!! THIS IS AI CODED SLOP !!!WARNING!!!
# Blue Iris AI Gateway

A high-performance, containerized unified AI gateway designed specifically for **Blue Iris**, combining **YOLO Object Detection** (via TensorRT / CUDA) and **Face Recognition** (via `facenet-pytorch`) into a single, highly reliable FastAPI application. 

Engineered for reliability in homelab environments, it features a built-in **GPU watchdog**, **atomic face database persistence**, and **automatic TensorRT engine compilation**.

---

## Key Features

* **Unified API Pipeline:** Routes both object detection and facial recognition through a single port/service, keeping all GPU operations serialized and safe under a shared lock.
* **Maximized Hardware Acceleration:** Leverages NVIDIA TensorRT `.engine` compilation for lightning-fast YOLO inference on modern NVIDIA hardware, with seamless fallbacks to ONNX and PyTorch (`.pt`).
* **Built-In Face Recognition:** Uses `facenet-pytorch` (`MTCNN` + `InceptionResnetV1`) natively inside PyTorch/CUDA without requiring heavy external runtimes or separate containers.
* **Resilient GPU Watchdog:** Monitors inference threads with a strict timeout. If a CUDA driver deadlock or hang occurs, the watchdog taints the GPU gate, fires an optional webhook alert, and terminates the container so Docker can cleanly spin up a fresh instance.
* **Atomic Face Database Storage:** Persists enrolled facial embeddings safely to disk using atomic file replacements (`os.replace`) to prevent file corruption during concurrent registration requests.
* **Automated CI/CD:** Automatically builds and pushes container images to **GitHub Container Registry (GHCR)** via GitHub Actions on every push.

---

## API Endpoints

### Object Detection
* **`POST /v1/vision/detection`** — Runs default model object detection on an uploaded image file.
* **`POST /v1/vision/custom/{model_name}`** — Runs detection using a specific custom model on disk.
* **`GET /v1/vision/custom/list`** — Lists all available models discovered in the models directory.

### Face Recognition (Blue Iris Compatible)
* **`GET /v1/vision/face/list`** / **`POST /v1/vision/face/list`** — Returns a list of all enrolled user IDs.
* **`POST /v1/vision/face/register`** — Enrolls a face image under a specified user ID or name (`userid` / `name`).
* **`POST /v1/vision/face/delete`** — Deletes an enrolled user profile from the database.
* **`POST /v1/vision/face/recognize`** — Detects faces in an image frame and compares them against the enrolled database.

### System & Health
* **`GET /`** or **`GET /status`** — Returns system health, active models, and enrolled face counts. Returns `503 Service Unavailable` if the GPU pipeline is degraded or tainted.

---

## Docker Compose Deployment

Add the following stack definition to your **Portainer** or Docker Compose setup:

```yaml
services:
  blueiris-ai:
    image: ghcr.io/pilonull/blueiris-ai-gateway:latest
    container_name: "blueiris-ai"
    ports:
      - "32169:32168"
    shm_size: "1gb"
    environment:
      - TZ=America/New_York
      - PYTHONUNBUFFERED=1
      - DEFAULT_MODEL=yolo11m
      - PRELOAD_MODELS=yolo11m
      - ALLOW_LAZY_LOAD=false
      - HALF_PRECISION=true
      - INFERENCE_TIMEOUT=8.0
      - RECOVERY_GRACE_PERIOD=4.0
      - ALERT_WEBHOOK_URL=
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - /mnt/Docker_Data/bi-ai/models:/app/models
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python3", "-c", "import urllib.request as u, sys; sys.exit(0 if u.urlopen('[http://127.0.0.1:32168/').status](http://127.0.0.1:32168/').status) == 200 else 1)"]
      interval: 30s
      timeout: 5s
      retries: 3
      start_period: 90s
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Environment Variables

| Variable | Default | Description |
| :--- | :--- | :--- |
| `DEFAULT_MODEL` | `yolo11m` | The primary model stem loaded at startup. |
| `PRELOAD_MODELS` | `yolo11m` | Comma-separated list of models to preload on boot (or `all`). |
| `ALLOW_LAZY_LOAD` | `false` | Whether to load models on-demand if requested via custom endpoints. |
| `HALF_PRECISION` | `true` | Enables FP16 quantization for PyTorch (`.pt`) model inference. |
| `INFERENCE_TIMEOUT` | `8.0` | Seconds before an active inference call triggers the GPU watchdog timeout. |
| `RECOVERY_GRACE_PERIOD` | `4.0` | Grace window allowed for a lagging worker thread before forcing a container exit. |
| `ALERT_WEBHOOK_URL` | *blank* | Optional webhook URL (e.g., ntfy.sh or Discord) to dispatch crash notifications. |

---

## Auto-Compilation Hook

On container startup, the gateway scans your mounted `/app/models` directory. If it finds a PyTorch model (`.pt`) without a corresponding TensorRT engine (`.engine`), or if the container environment/CUDA version changes, it automatically compiles an optimized TensorRT plan file tailored specifically to your GPU's architecture on the fly.
