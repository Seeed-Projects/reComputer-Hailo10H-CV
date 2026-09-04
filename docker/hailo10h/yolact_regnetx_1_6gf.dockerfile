# rebuild for hailort 5.1.1
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends     libgl1     libglib2.0-0     libgomp1     libsm6     libxext6     libxrender1     ffmpeg     && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hailort-packages/ /tmp/hailort-packages/
RUN if ls /tmp/hailort-packages/hailort-*.whl 1>/dev/null 2>&1; then       pip install --no-cache-dir /tmp/hailort-packages/hailort-*.whl;     fi;     rm -rf /tmp/hailort-packages

COPY . .

EXPOSE 8000

CMD ["python", "web_detection.py", "--model_path", "model/yolact_regnetx_1_6gf.hef", "--video_path", "video/test.mp4"]
