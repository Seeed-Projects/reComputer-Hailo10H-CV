# YOLOv5s on CM5 + Hailo-10H

YOLOv5s performs COCO 80-class object detection with **on-chip Hailo HPP NMS**
(zoo `base/yolo.yaml`, `meta_arch=yolo_v5`): the HEF emits already-decoded
detections and the application only parses the post-NMS tensor. The FastAPI
service provides image prediction, video and camera input, an MJPEG preview and
offline video analysis.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 640x640x3 RGB (normalize_in_net mean=0/std=255) |
| Output | on-chip NMS tensor, zoo post-NMS shape 80x5x80 |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 7.46M |
| Operations | 17.44G |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/yolov5s.dockerfile \
  -t yolov5s:latest \
  src/hailo10h_yolov5s
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov5-s \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov5s:latest \
  python web_detection.py --model_path model/yolov5s.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov5s/predict" \
  -F "file=@bus.jpg" -F "conf=0.25" -F "iou=0.45"
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/yolov5s/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- On-chip NMS (HPP, zoo `base/yolo.yaml`): the app only parses the post-NMS
  tensor, so `nms_thresh` is accepted for API parity but ignored.
- Pre-processing letterboxes to 640x640x3 with the YOLOv5 gray padding (114),
  converts BGR to RGB and feeds raw uint8; the `/255` normalization
  (`normalize_in_net`, mean 0 / std 255) is compiled into the HEF.
- The parser handles the HailoRT NMS layouts: the compact per-class buffer
  (count followed by 5-value rows), the dense `80x5x80` / `80x80x5` forms and
  the ragged (NMS-by-score) list.
- `cls_id` (0..79) indexes the standard COCO class list directly.
- The first inference logs the raw output type and shape once
  (`[YOLOv5] raw output type=..., shape=...`) so the layout can be confirmed on
  hardware.

## Hardware acceptance checklist

1. `hailortcli --version` reports 5.1.1 and `/dev/hailo0` exists.
2. The startup log reports the HEF input size (`Model input size: ...`).
3. The demo video shows COCO boxes with class labels.
4. `POST /api/models/yolov5s/predict` returns detections with class, confidence
   and box.
5. USB camera mode keeps the preview live without stale frames.

## Tests

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mocks `hailo_platform` and checks the HailoRT
  5.1.1 wrapper (persistent binding set, `run([bindings], timeout=10_000)`,
  FLOAT32 output buffers).
- `tests/test_nms_parse.py` checks the post-NMS parsing (compact buffer, dense
  layouts, ragged list, score threshold, un-letterboxing).

Neither test replaces the hardware acceptance above.
