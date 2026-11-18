# YOLO-World v2s on CM5 + Hailo-10H

Open-vocabulary object detection module for **reComputer CM5 / Raspberry Pi 5 +
Hailo-10H**. It runs the Hailo Model Zoo YOLO-World v2s HEF with two inputs: a
640x640 RGB frame and an 80x512 CLIP text-embedding matrix. The bundled
vocabulary is the 80 COCO classes.

Features: real-time MJPEG preview, REST prediction, offline batch video
analysis, and a confidence slider. The HEF contains the text-contrast head and
a host-side NMS post-process layer.

---

## 1. Host preparation

Install the Hailo-10H runtime and driver, then verify the device:

```bash
sudo apt update
sudo apt install hailo-h10-all -y
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0
```

Install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

The container uses the bundled `hailort-5.1.1-cp313-cp313-linux_aarch64.whl`
and must bind-mount the matching host `libhailort.so.5.1.1`.

---

## 2. Build and run

```bash
cd src/hailo10h_yolo_world_v2s
sudo docker build -f ../../docker/hailo10h/yolo_world_v2s.dockerfile \
  -t hailo10h-yolo-world-v2s:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  hailo10h-yolo-world-v2s:latest
```

The default command loops `video/test.mp4` and serves the Web UI on
`http://<Board_IP>:8000`.

Camera mode:

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  --device /dev/video0:/dev/video0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  hailo10h-yolo-world-v2s:latest \
  python web_detection.py --model_path model/yolo_world_v2s.hef --camera_id 0
```

---

## 3. Model assets

| File | Size | Source |
|---|---:|---|
| `model/yolo_world_v2s.hef` | 25,092,096 B | [Model Zoo v5.4.0 hailo10h](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo_world_v2s.hef) |
| `model/coco_text_embeddings.npy` | 163,968 B | CLIP ViT-B/32 text embeddings, shape `(1, 80, 512)` |
| `model/coco_classes.txt` | 701 B | 80 COCO class names, one per line |

HEF SHA256:
`489a366562d617ea8428958405e84f37e65775eb4b66e1af112617d1321ef1f0`.

Hailo's public Hailo-10H zero-shot table links the v5.5.0 build (36.3 float
mAP, 34.9 hardware mAP, 42.8 FPS batch 1). This module uses v5.4.0 to stay
aligned with the other Hailo-10H modules in this repository. The HEF format is
v3 and its metadata reports SDK 5.4.0; confirm compatibility with the board's
HailoRT before publishing results. The v5.5.0 HEF is a drop-in alternative if
your runtime is newer.

See [model/README.md](model/README.md) for provenance and caveats.

---

## 4. Vocabulary

The HEF accepts exactly 80 prompts. Row `i` of `coco_text_embeddings.npy`
produces output class `i`; row `i` of `coco_classes.txt` is the displayed label.
The bundled embeddings were generated with `openai/clip-vit-base-patch32`
(`CLIPTextModelWithProjection`, class name only, L2-normalized), matching
Hailo's `create_coco_tfrecord_with_text_embeddings.py`.

To use another vocabulary, precompute exactly 80 embeddings and pass both files:

```bash
python web_detection.py \
  --model_path model/yolo_world_v2s.hef \
  --embeddings_path /path/to/my_embeddings.npy \
  --class_path /path/to/my_classes.txt
```

Unused slots still need valid prompts; an all-zero row can produce a spurious
mid-confidence score in the text-contrast head.

---

## 5. HTTP API

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Web UI |
| GET/POST | `/api/config` | Read/update display thresholds |
| POST | `/api/video/upload` | Upload a video |
| GET | `/api/video/list` | List uploads and outputs |
| POST | `/api/video/analyze` | Start offline analysis |
| GET | `/api/video/status` | Progress and error state |
| GET | `/api/video/download/{filename}` | Download an annotated MP4 |
| POST | `/api/models/yolo_world/predict` | Single-frame prediction |

---

## 6. Implementation notes

- Two inputs are discovered from the HEF: `yolo_world_v2s/input_layer1`
  (UINT8 640x640x3 RGB) and `yolo_world_v2s/input_layer2` (float32 80x512).
- The image is letterboxed to 640x640 and converted to RGB; the HEF's
  normalization divides by 255.
- HailoRT 5.1.1 bindings are created once; both input buffers are updated per
  frame and outputs are requested as FLOAT32.
- The NMS layer uses the Model Zoo config: 80 classes, IoU 0.7, up to 300
  proposals per class, score threshold 0.001. The UI threshold filters rows.

---

## 7. Known limitations

- The vocabulary is fixed at startup; there is no on-board CLIP text encoder.
- HEF/runtime version compatibility (SDK 5.4.0 vs bundled HailoRT 5.1.1) must
  be confirmed on hardware.
- The Web service targets a trusted LAN and has no authentication or TLS.
- Hardware validation is pending; see `TEST_REPORT.md`.