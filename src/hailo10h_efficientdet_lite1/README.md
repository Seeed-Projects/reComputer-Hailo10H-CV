# EfficientDet-Lite1 on CM5 + Hailo-10H

EfficientDet-Lite1 performs COCO object detection with **on-chip Hailo HPP NMS and
sigmoid**: the HEF emits already-decoded detections and the application only
parses the post-NMS tensor. The FastAPI service provides image prediction,
video and camera input, an MJPEG preview and offline video analysis.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 384x384x3 RGB (normalization compiled into the HEF: mean=127, std=128) |
| Output | on-chip NMS tensor, post-NMS shape 89x5x100 |
| Classes | 89 slots (COCO category IDs 1..89 via `labels_offset=1`; 10 unused IDs) |
| Parameters | 4.73M |
| Operations | 4G |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/efficientdet_lite1.dockerfile \
  -t efficientdet_lite1:latest \
  src/hailo10h_efficientdet_lite1
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-effdet-lite1 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/efficientdet_lite1:latest \
  python web_detection.py --model_path model/efficientdet_lite1.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`. For a USB camera, mount `/dev/video0` and
replace `--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/efficientdet_lite1/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/health` | GET | Health check |
| `/api/models/efficientdet_lite1/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- The HEF runs NMS and sigmoid on-chip (`device_pre_post_layers: nms=true,
  sigmoid=true`, `hpp=true`); the app only parses the post-NMS tensor, so
  `nms_thresh` is accepted for API parity but ignored.
- Post-NMS rows are `[ymin, xmin, ymax, xmax, score]`, normalized to [0,1] of
  the letterboxed input; the app scales to pixels and un-letterboxes.
- `normalize_in_net` with mean=127/std=128 and `padding_color=127`: the app
  letterboxes with gray (127) padding and feeds raw uint8 RGB pixels — no
  manual normalization.
- The parser handles the HailoRT NMS layouts: the compact per-class buffer, the
  dense `89x5x100` / `89x100x5` forms and the ragged (NMS-by-score) list.
- Class mapping: `cls_id` (0..88) maps to COCO category ID `cls_id + 1`
  (`labels_offset=1`); the 10 unused IDs are listed as "N/A" and not drawn.
- The first inference logs the raw output type and shape once
  (`[EfficientDet] raw output type=..., shape=...`) so the layout can be
  confirmed on hardware.

## Hardware acceptance checklist

1. `hailortcli --version` reports 5.1.1 and `/dev/hailo0` exists.
2. The startup log reports the HEF input size (`Model input size: 38484`).
3. The demo video shows COCO boxes with class labels.
4. `POST /api/models/efficientdet_lite1/predict` returns detections with class, confidence
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
  layouts, ragged list, score threshold, un-letterboxing, class mapping).

Neither test replaces the hardware acceptance above.
