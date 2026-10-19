# YOLOv11n_obb on CM5 + Hailo-10H

YOLOv11n_obb performs oriented object detection (rotated boxes, DOTA v1
15 classes) on **Hailo-10H** through HailoRT. The compiled HEF has **no on-chip
NMS**: it exposes nine raw heads (per stride one 64-channel DFL box tensor, one
15-channel class tensor and one 1-channel angle tensor) and the host decodes
them and runs the rotated probabilistic-IoU NMS — the same decode the Hailo
Model Zoo uses for its published DOTA mAP.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 1024x1024x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| Output | rotated boxes `[cx, cy, w, h, theta]`, best class per box, 15 DOTA classes |
| Parameters | 3.80M |
| Operations | 17.25G |
| DOTA mAP (hardware) | 48.4 (float 49.7) |
| FPS on Hailo-10H (batch 1) | 166 |
| HEF | Hailo Model Zoo v5.4.0 (Hailo-10H), 6,893,568 bytes |
| Model licence | AGPL-3.0 (upstream: ultralytics/ultralytics) |

HEF SHA256: `6188275241433279adc067c9a419c216aa1bc7212e0baf1297731015112f1008`

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

From the repository root:

```bash
sudo docker build -f docker/hailo10h/yolov11n_obb.dockerfile \
  -t yolov11n_obb:latest \
  src/hailo10h_yolov11n_obb
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov11n_obb \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov11n_obb:latest \
  python web_detection.py --model_path model/yolov11n_obb.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

The bundled clip is a ~6 s **aerial (drone) sample**, so the DOTA-trained model
detects vehicles and other aerial objects directly and the rotated boxes show up
in the preview. Drone-dataset classes that are not part of DOTA (e.g.
pedestrians) are not reported. For a USB camera, mount `/dev/video0` and replace
`--video_path ...` with `--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov11n_obb/predict" \
  -F "file=@aerial.jpg" -F "conf=0.25" -F "iou=0.7"
```

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/yolov11n_obb/predict` | POST | Rotated detections (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

Each prediction carries `class`, `confidence` and
`obb: {x_center, y_center, width, height, angle}` with the angle in **degrees**
(counter-clockwise from +x, range [-45, 135)).

## Implementation notes

- Head layout: nine raw tensors, classified by channel count (64 / 15 / 1) and
  paired per stride (128x128, 64x64, 32x32). No on-chip NMS and no device
  sigmoid (`base/yolo_obb.yaml`), so everything below runs on the host.
- Angle: `theta = (sigmoid(a) - 0.25) * pi` ([-pi/4, 3pi/4)), exactly as in the
  Model Zoo's `YoloOBBPostProc` and ultralytics' OBB head.
- Boxes: DFL softmax expectation over 16 bins gives the l/t/r/b distances in
  stride units; `w = l + r`, `h = t + b`, the centre is the anchor plus the
  rotated half-axis offset, and the distances scale by the stride.
- Selection: best class per anchor above the confidence slider, then
  probabilistic-IoU Fast-NMS with a per-class offset (max_wh 7680, at most
  30000 candidates in and 300 boxes out) — the confidence and IOU sliders are
  both live.
- Pre-processing letterboxes to 1024x1024x3 with gray (114) padding, converts
  BGR to RGB and feeds raw uint8; the `/255` normalization is compiled into the
  HEF.
- Class ids 0..14 index the DOTA v1 list (plane, ship, storage-tank, ...) as in
  the Model Zoo's `dotav1_names.json`.
- First-inference diagnostics are disabled in production
  (`_DET_OUTPUT_LOGGED = True`); set it to `False` to dump the head layout,
  value ranges and decoded angle span when validating a new HEF (SOP §10/§14).

## Hardware acceptance checklist

1. `hailortcli --version` reports 5.1.1 and `/dev/hailo0` exists.
2. The startup log reports the HEF input size (`Model input size: 1024x1024`).
3. An aerial test image through `predict` returns rotated boxes whose angles look
   consistent with the objects (if they are off by 90 degrees, regularize w/h).
4. The demo video keeps the preview live without stale frames.
5. USB camera mode works.
