# YOLOv8m Pose on CM5 + Hailo-10H

YOLOv8 Pose runs single-class person keypoint estimation on a Hailo-10H
accelerator. The module reuses the YOLO service template of this repository
(FastAPI, MJPEG preview, offline video analysis, USB camera) and implements the
YOLOv8 Pose post-processing: the on-chip NMS rows carry the person box, its
score, and 17 COCO keypoints.

| Model | Path | Size | Notes |
|---|---|---:|---|
| YOLOv8m Pose | `model/yolov8m_pose.hef` | 29,335,552 bytes | Larger, usually more accurate and slower |

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 640x640x3 RGB (letterbox, gray padding, normalization in the HEF) |
| Output | Person box, score, 17 COCO keypoints (on-chip NMS) |
| Classes | 1 (`person`) |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/yolov8m_pose.dockerfile \
  -t yolov8m_pose:latest \
  src/hailo10h_yolov8m_pose
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov8m-pose \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov8m_pose:latest \
  python web_detection.py \
    --model_path model/yolov8m_pose.hef \
    --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov8_pose/predict" \
  -F "file=@test.jpg"
```

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "person",
      "confidence": 0.91,
      "box": {"x1": 120, "y1": 80, "x2": 360, "y2": 520},
      "keypoints": [{"x": 180, "y": 120, "score": 0.87}]
    }
  ],
  "image": {"width": 1280, "height": 720}
}
```

Keypoints below `KEYPOINT_SCORE_THRESH` (0.30) are omitted from the response.
The service also exposes MJPEG preview, uploaded-video analysis, and camera
mode through the same web UI.

## Post-processing

- The HEF performs NMS on-chip, so `nms_thresh` is accepted but ignored.
- Each NMS row is `[ymin, xmin, ymax, xmax, score, kpt0_y, kpt0_x, kpt0_s, ...]`
  with 17 joints, i.e. 56 values per row, all normalized to the network input.
  Boxes and keypoints are scaled to `640x640` first and then un-letterboxed
  back to the original frame.
- The buffer is parsed in the compact per-class form (count followed by rows),
  and the dense `(1, C, N, 56)` / `(C, N, 56)` and ragged layouts are handled
  too. A flat buffer that does not validate as 56-wide rows is retried with the
  detection-only width (5) so a different post-process revision still decodes
  boxes.
- The first inference logs the raw output type, shape and a sample row:
  `[YOLOv8 Pose] raw output type=..., shape=..., dtype=..., row_width=...`.
  Use it to confirm the joint ordering on hardware: the decoder reads joints as
  `(y, x, score)` and returns `(x, y, score)`.

## Hardware acceptance checklist

1. `hailortcli --version` reports 5.1.1 and `/dev/hailo0` exists.
2. The first inference log shows the expected output name/shape and a plausible
   sample row (box and joints inside `[0, 1]`).
3. The demo video shows person boxes with COCO skeletons aligned to the body.
4. `POST /api/models/yolov8_pose/predict` returns boxes plus 17 keypoints.
5. `model/yolov8m_pose.hef` runs with the same code path.
6. USB camera mode keeps the preview live without stale frames.

## Tests

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mocks `hailo_platform` and checks the HailoRT
  5.1.1 wrapper (single persistent binding set, `run([bindings], timeout=10_000)`,
  FLOAT32 output buffers).
- `tests/test_pose_postprocess.py` checks the NMS row decoding, keypoint
  ordering, score filtering and un-letterboxing.

Both run without a device; they do not replace the hardware acceptance above.