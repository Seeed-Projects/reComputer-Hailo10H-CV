# YOLO-World v2m on CM5 + Hailo-10H

Open-vocabulary object detection module for **reComputer CM5 / Raspberry Pi 5 +
Hailo-10H**. It runs the Hailo Model Zoo YOLO-World v2m HEF with two inputs: a
640x640 RGB frame and an 80x512 CLIP text-embedding matrix. The bundled
vocabulary is the 80 COCO classes.

The v2m variant uses the medium YOLO-World backbone: 28.4M parameters, 85.9G
operations, 42.0 COCO mAP (full precision) and 38.4 hardware mAP on Hailo-10H.
Hailo reports 22.7 FPS at batch size 1.

Features: real-time MJPEG preview, REST prediction, offline batch video
analysis, and a confidence slider. The HEF contains the text-contrast head and
a host-side NMS post-process layer.

---

## 1. Host preparation

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

The container ships the `hailort-5.1.1-cp313-cp313-linux_aarch64.whl` wheel and
bind-mounts the matching host `libhailort.so.5.1.1`.

---

## 2. Build and run

```bash
cd src/hailo10h_yolo_world_v2m
sudo docker build -f ../../docker/hailo10h/yolo_world_v2m.dockerfile \
  -t hailo10h-yolo-world-v2m:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  hailo10h-yolo-world-v2m:latest
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
  hailo10h-yolo-world-v2m:latest \
  python web_detection.py --model_path model/yolo_world_v2m.hef --camera_id 0
```

---

## 3. Model assets

| File | Size | Source |
|---|---:|---|
| `model/yolo_world_v2m.hef` | 47,288,320 B | [Model Zoo v5.5.0 hailo10h](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.5.0/hailo10h/yolo_world_v2m.hef) |
| `model/coco_text_embeddings.npy` | 163,968 B | CLIP ViT-B/32 text embeddings, shape `(1, 80, 512)` |
| `model/coco_classes.txt` | 701 B | 80 COCO class names, one per line |

HEF SHA256:
`37d346e90606544efd7ab9e1a7062f56f876f6f5b56df3d6d9f0aa75efc49255`.

The M variant is not published for v5.4.0; v5.5.0 is the only official
Hailo-10H build. Its metadata reports SDK 5.5.0, while the bundled HailoRT wheel
is 5.1.1. Confirm that the runtime accepts the HEF before publishing results;
if it does not, upgrade the host driver, `libhailort.so`, wheel, and HEF
together.

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
  --model_path model/yolo_world_v2m.hef \
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

- Two inputs are discovered from the HEF: `yolo_world_v2m/input_layer1`
  (UINT8 640x640x3 RGB) and `yolo_world_v2m/input_layer2` (float32 80x512).
- The image is letterboxed to 640x640 and converted to RGB; the HEF's
  normalization divides by 255.
- HailoRT 5.1.1 bindings are created once; both input buffers are updated per
  frame and outputs are requested as FLOAT32.
- The NMS layer uses the Model Zoo config: 80 classes, IoU 0.7, up to 300
  proposals per class, score threshold 0.001. The UI threshold filters rows.

---

## 7. Known limitations

- The vocabulary is fixed at startup; there is no on-board CLIP text encoder.
- The v5.5.0 HEF may require a newer HailoRT than the bundled 5.1.1; verify on
  hardware and upgrade the runtime if needed.
- The Web service targets a trusted LAN and has no authentication or TLS.
- Hardware validation is pending; see `TEST_REPORT.md`.