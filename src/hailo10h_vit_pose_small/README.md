# ViTPose-Small on CM5 + Hailo-10H

ViTPose-Small performs single-person 2D pose estimation with 17 COCO keypoints. The
HEF outputs one heatmap (64x48x17); the application takes the argmax per
channel, scales the coordinates to the 256x192 network input and un-letterboxes
them back to the frame, then draws the COCO skeleton. The FastAPI service
supports images, video files, USB cameras, an MJPEG preview and a REST
prediction endpoint.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 256x192x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | Heatmap 64x48x17 (17 COCO keypoints, argmax decode, no NMS) |
| Parameters | 24.29M |
| Operations | 17.17G |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/vit_pose_small.dockerfile \
  -t vit_pose_small:latest \
  src/hailo10h_vit_pose_small
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-vit-pose-small \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/vit_pose_small:latest \
  python web_detection.py --model_path model/vit_pose_small.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/vit_pose_small/predict" \
  -F "file=@test.jpg"
```

The response contains the 17 keypoints (`x`, `y`, `score`) of the single person
in the frame. Keypoints below the confidence threshold (0.30) are dropped from
the skeleton overlay.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/vit_pose_small/predict` | POST | 17 COCO keypoints (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- **Single-person model**: the network assumes the person is centred in the
  crop; for multi-person scenes a detector has to crop each person first.
- **Pre-processing**: letterbox to 256x192 with black padding, BGR to RGB, raw
  uint8. The HEF carries the ImageNet normalization (mean `[123.675, 116.28,
  103.53]`, std `[58.395, 57.12, 57.375]` on RGB), so nothing is normalized on
  the host.
- **Post-processing**: argmax per heatmap channel gives `(x, y)` in the 64x48
  heatmap, scaled to the 256x192 input and un-letterboxed to the frame. There
  is no on-chip NMS and no DARK sub-pixel refinement (`nms_thresh` is kept for
  API parity only).
- The first inference logs the raw output shape once (`[ViTPose] raw output
  shape=...`) so the layout can be confirmed on hardware.

## Hardware acceptance checklist

1. `hailortcli --version` reports 5.1.1 and `/dev/hailo0` exists.
2. The startup log reports the HEF input size (`Model input: 192x256`).
3. The demo video shows a COCO skeleton aligned with the person.
4. `POST /api/models/vit_pose_small/predict` returns 17 keypoints with scores.
5. USB camera mode keeps the preview live without stale frames.

## Tests

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mocks `hailo_platform` and checks the HailoRT
  5.1.1 wrapper (persistent binding set, `run([bindings], timeout=10_000)`,
  FLOAT32 output buffers).
- `tests/test_pose_heatmap.py` checks the heatmap decode (argmax, layout
  normalisation, coordinate scaling and un-letterboxing).

Neither test replaces the hardware acceptance above.
