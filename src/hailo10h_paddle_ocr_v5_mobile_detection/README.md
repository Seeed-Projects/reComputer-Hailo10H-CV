# PaddleOCR v5 Mobile Detection · CM5 + Hailo-10H

This independent text-detection module receives a full document image and returns DB text-region polygons. Pair it with the recognition module only at the application level; the two Hailo models and images are built separately.

| Component | Value |
| --- | --- |
| Input | 544 x 960 x 3 |
| Output | Text-region polygons and axis-aligned boxes |
| Runtime | HailoRT 5.1.1, Python 3.13, aarch64 |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/paddle_ocr_v5_mobile_detection:latest \
  python web_detection.py --model_path model/paddle_ocr_v5_mobile_detection.hef --video_path video/test.mp4
```

Upload an image:

```bash
curl -X POST "http://<PI_IP>:8000/api/models/paddle_ocr_v5_mobile_detection/predict" -F "file=@video/test.png"
```
