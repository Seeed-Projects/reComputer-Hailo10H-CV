# STDC1 Semantic Segmentation - Hailo-10H
# Base image with HailoRT support for Raspberry Pi CM5

FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
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

# Copy application code
COPY . .

# Note: HailoRT library (libhailort.so) is mounted from host at runtime
# The HEF model is bundled in the image under model/

EXPOSE 8000

# Start inference service
CMD ["python", "web_service.py"]