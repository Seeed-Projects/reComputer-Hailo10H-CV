FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hailort-packages/hailort-5.1.1-cp313-cp313-linux_aarch64.whl /tmp/
RUN pip install --no-cache-dir /tmp/hailort-5.1.1-cp313-cp313-linux_aarch64.whl && rm -f /tmp/*.whl

COPY . .

EXPOSE 8000

CMD ["python", "web_detection.py", "--model_path", "model/stdc1.hef", "--video_path", "video/test.mp4"]
