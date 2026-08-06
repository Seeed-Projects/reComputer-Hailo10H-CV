# rebuild for hailort 5.1.1
FROM python:3.13-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*


# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Install HailoRT Python API
COPY hailort-packages/ /tmp/hailort-packages/

RUN if ls /tmp/hailort-packages/hailort-*.whl 1>/dev/null 2>&1; then \
      pip install --no-cache-dir /tmp/hailort-packages/hailort-*.whl; \
    fi; \
    rm -rf /tmp/hailort-packages


# Copy application files
COPY . .


EXPOSE 8000


CMD ["python", "web_detection.py", \
     "--model_path", "model/mspn_regnetx_800mf.hef", \
     "--video_path", "video/test.mp4"]
