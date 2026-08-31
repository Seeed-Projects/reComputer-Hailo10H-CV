# PaddleOCR v5 Mobile Recognition · CM5 + Hailo-10H

This independent recognition module accepts one cropped text line and returns CTC-decoded text with a confidence value. Its `video/test.png` is a single-line crop from the self-authored English OCR sample.

| Component | Value |
| --- | --- |
| Input | 48 x 320 x 3 |
| Output | Decoded text and CTC confidence |
| Runtime | HailoRT 5.1.1, Python 3.13, aarch64 |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/paddle_ocr_v5_mobile_recognition:latest \
  python web_detection.py --model_path model/paddle_ocr_v5_mobile_recognition.hef --video_path video/test.mp4
```

Upload a single text-line image:

```bash
curl -X POST "http://<PI_IP>:8000/api/models/paddle_ocr_v5_mobile_recognition/predict" -F "file=@video/test.png"
```
