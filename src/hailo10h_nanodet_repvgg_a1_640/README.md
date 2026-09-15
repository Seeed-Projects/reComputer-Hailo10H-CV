# NanoDet-RepVGG on reComputer CM5 + Hailo-10H

This module runs NanoDet-RepVGG object detection (COCO 80 classes) with
**on-chip NMS** (Hailo HPP). The HEF performs NMS on-device and emits
already-decoded detections; the app only parses the post-NMS tensor. The
FastAPI service supports images, video files, USB cameras, an MJPEG preview,
and REST prediction endpoints.

Ported from the validated Hailo-8 module `rpi5_hailo8_nanodet_repvgg_a1_640`; the
executor was replaced with the HailoRT 5.1.1 `create_infer_model` multi-output
wrapper (outputs requested as FLOAT32).

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 640x640x3 (shape read from the HEF at startup) |
| Output | on-chip NMS tensor, post-NMS shape 80x5x100 |
| Classes | 80 (COCO, 0-indexed — no labels_offset) |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/nanodet_repvgg_a1_640.dockerfile \
  -t nanodet_repvgg_a1_640:latest \
  src/hailo10h_nanodet_repvgg_a1_640
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-nanodet-repvgg-a1-640 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/nanodet_repvgg_a1_640:latest \
  python web_detection.py --model_path model/nanodet_repvgg_a1_640.hef --video_path video/test.mp4
```

Open `http://<CM5_IP>:8000`. For a USB camera, mount `/dev/video0` and replace
`--video_path video/test.mp4` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<CM5_IP>:8000/api/models/nanodet_repvgg_a1_640/predict" \
  -F "file=@test.jpg"
```

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Health check |
| `/api/models/nanodet_repvgg_a1_640/predict` | POST | Detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |

## Implementation notes

- The HEF runs NMS on-chip; the app only parses the post-NMS tensor, so
  `nms_thresh` is ignored (kept for API parity).
- Post-NMS rows are `[ymin, xmin, ymax, xmax, score]`, normalized to [0,1] of
  the letterboxed input; the app scales to pixels and un-letterboxes.
- Input preprocessing (normalization / color conversion) is compiled into the
  HEF; the app letterboxes and feeds raw uint8 pixels.
- HailoRT may return the NMS vstream as a ragged per-class list
  (NMS-by-score); the parser handles that plus object/dense layouts. First
  inference logs the raw type/shape for on-device verification.
- Executor outputs are requested as `FormatType.FLOAT32` (dequantized by the
  SDK); reading the raw quantized buffers yields values in the tens of
  thousands and zero valid detections.
- Class mapping: `cls_id` (0..79) → standard COCO 80-class list directly.

## Model source

`nanodet_repvgg_a1_640.hef` is the Hailo-10H build from Hailo Model Zoo v5.4.0 — see
`model/README.md` for the URL, SHA256 and size.
