# YOLOv10x on CM5 + Hailo-10H

YOLOv10x performs COCO 80-class object detection on **Hailo-10H** through
HailoRT. The HEF runs NMS on-chip (HPP, `meta_arch=yolov8`, one2one head) and
exposes a single output vstream; the application parses the compact
NMS-by-class buffer HailoRT 5.1.1 returns and keeps handlers for the other
HailoRT NMS layouts. The FastAPI service provides image prediction, video and
camera input, an MJPEG preview and offline video analysis.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 640x640x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| Output | on-chip NMS, 80x5x100 (max 100 boxes per class) |
| Classes | 80 (COCO, 0-indexed) |
| Parameters | 31.72M |
| Operations | 160.56G |
| COCO mAP (full precision) | 53.7 |
| HEF | Hailo Model Zoo v5.4.0 (Hailo-10H) |
| Model licence | AGPL-3.0 (upstream: THU-MIG/yolov10) |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/yolov10x.dockerfile \
  -t yolov10x:latest \
  src/hailo10h_yolov10x
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov10x \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov10x:latest \
  python web_detection.py --model_path model/yolov10x.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov10x/predict" \
  -F "file=@bus.jpg" -F "conf=0.25"
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/yolov10x/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- The HEF performs NMS on-chip: the parser nodes are the one2one head
  (`one2one_cv2.*` box, `one2one_cv3.*` class), so the host decodes nothing and
  runs no NMS. `nms_thresh` is accepted for API parity but ignored.
- HPP output on HailoRT 5.1.1 is the flat compact NMS-by-class buffer (per
  class: detection count followed by that many `[ymin, xmin, ymax, xmax,
  score]` rows, all normalized to [0, 1]). Fixed-stride readers misalign every
  class after the first; `_per_class_iterable` also keeps handlers for the
  ragged NMS-by-score and dense `(1,C,5,D)` / `(1,C,D,5)` layouts.
- Pre-processing letterboxes to 640x640x3 with gray (114) padding, converts BGR
  to RGB and feeds raw uint8; the `/255` normalization is compiled into the HEF.
- `cls_id` (0..79) indexes the standard COCO class list directly.
- The first-inference layout log is disabled in production
  (`_OUTPUT_LOGGED = True`); set it to `False` to dump the raw output
  type/shape/range and per-class rows when validating a new HEF (SOP §10/§14).

## Hardware acceptance checklist

1. `hailortcli --version` reports 5.1.1 and `/dev/hailo0` exists.
2. The startup log reports the HEF input size (`Model input size: ...`).
3. The demo video shows COCO boxes with class labels.
4. `POST /api/models/yolov10x/predict` returns detections with class,
   confidence and box.
5. USB camera mode keeps the preview live without stale frames.
