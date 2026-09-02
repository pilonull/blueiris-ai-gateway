FROM ultralytics/ultralytics:latest

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn python-multipart tqdm
RUN pip install --no-cache-dir --no-deps facenet-pytorch

# Pre-cache the VGGFace2 model weights into the container image
RUN python3 -c "from facenet_pytorch import InceptionResnetV1; InceptionResnetV1(pretrained='vggface2')"

COPY app.py /app/app.py

EXPOSE 32168

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "32168"]
