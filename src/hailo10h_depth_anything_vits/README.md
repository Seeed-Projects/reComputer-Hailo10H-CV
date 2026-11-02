# Depth-Anything-ViTS on CM5 + Hailo-10H

Depth-Anything-ViTS (ViT-S backbone) performs zero-shot monocular depth
estimation on **Hailo-10H** through HailoRT. The HEF outputs a 224x224x1
relative depth map; the app normalizes it, applies an INFERNO colormap, and
alpha-blends onto the original frame. The FastAPI service supports images,
video files, USB cameras, an MJPEG preview, and REST prediction endpoints.

This is the V1 model: same architecture and I/O as V2-ViTS, with a lower AbsRel
(0.13 vs 0.147).

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 224x224x3 RGB (normalize_in_net ImageNet RGB mean/std) |
| Output | Depth map 224x224x1 (relative, zero-shot), FLOAT32 |
| Parameters | 24.2M |
| Operations | 16.7G |
| AbsRel | 0.13 |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

```bash
sudo docker build -f docker/hailo10h/depth_anything_vits.dockerfile \
  -t depth_anything_vits:latest \
  src/hailo10h_depth_anything_vits
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-depth-anything-vits \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/depth_anything_vits:latest \
  python web_detection.py --model_path model/depth_anything_vits.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/depth_anything_vits/predict" \
  -F "file=@test.jpg"
```

The response returns depth statistics (min, max, mean, std).

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/depth_anything_vits/predict` | POST | Depth statistics (JSON) |
| `/api/video_feed` | GET | MJPEG preview stream |
| `/api/config` | GET / POST | Read or change the overlay alpha |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- **Zero-shot relative depth**: the output is relative depth (not metric).
  Higher values = farther. The app min-max normalizes per-frame for
  visualization.
- **Visualization**: INFERNO colormap (dark=close, bright=far) alpha-blended
  onto the original frame. The slider controls the blend alpha.
- `normalize_in_net` with ImageNet RGB mean/std; feed raw uint8 RGB after
  letterboxing.
- No on-chip post-processing — the depth map is the HEF output. The wrapper
  requests FLOAT32 vstreams, so the host receives dequantized values.
- The HEF input size is read from the model at startup (224x224).

## Model source

[Hailo Model Zoo v5.4.0, Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_zero_shot_depth_estimation.rst)
(model `depth_anything_vits`, source: LiheYoung/depth-anything-small-hf).