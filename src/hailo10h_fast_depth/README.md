# FastDepth on CM5 + Hailo-10H

FastDepth performs real-time monocular depth estimation for indoor scenes on
**Hailo-10H** through HailoRT. The HEF takes a 224x224 RGB frame and returns a
224x224x1 depth map. The Model Zoo `fast_depth_postprocessing` returns the
network output unchanged (FastDepth predicts depth on the NYU-v2 scale, not
logits), so the app normalizes each frame with the 2nd/98th percentiles and
renders a pseudo-color (TURBO) overlay. The FastAPI service supports images,
video files, USB cameras, an MJPEG preview, offline video analysis, and a REST
prediction endpoint.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 224x224x3 RGB (`normalize_in_net`, mean 0 / std 255) |
| Output | Depth map 224x224x1 (NYU-v2 scale) |
| Parameters | 1.35M |
| Operations | 0.74G |
| RMSE (float / hardware) | 0.60 / 0.61 |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H (2,015,232 bytes) |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

```bash
sudo docker build -f docker/hailo10h/fast_depth.dockerfile \
  -t fast_depth:latest \
  src/hailo10h_fast_depth
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-fast-depth \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/fast_depth:latest \
  python web_detection.py --model_path model/fast_depth.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/fast_depth/predict" \
  -F "file=@test.jpg"
```

The response keeps the YOLO-style `predictions` list plus a `depth` statistics
object:

```json
{
  "class": "depth",
  "confidence": 1.0,
  "min_depth": 0.12,
  "max_depth": 8.4,
  "mean_depth": 1.9,
  "median_depth": 1.5,
  "valid_pixels": 921600
}
```

Depth values follow the model's NYU-v2 scale (roughly meters), not a calibrated
sensor measurement.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/fast_depth/predict` | POST | Depth statistics (JSON) |
| `/api/video_feed` | GET | MJPEG with pseudo-color depth overlay |
| `/api/config` | GET / POST | Read or change the overlay alpha |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- The HEF output is the depth map itself: the Model Zoo
  `fast_depth_postprocessing` performs no sigmoid or inverse-depth transform.
- The app normalizes each frame with the 2nd/98th percentiles before applying
  the TURBO colormap, then crops the letterbox content area and resizes to the
  original frame so the overlay aligns with the preview.
- `normalize_in_net` (mean 0 / std 255, folded) is compiled into the HEF; feed
  raw uint8 RGB after letterboxing.
- No on-chip post-processing. The wrapper requests FLOAT32 vstreams so the host
  receives dequantized values.
- The HEF input size is read from the model at startup (224x224).
- A non-square output would arrive as (224, 224); the decoder also handles
  transposed or smaller layouts defensively.

## Model source

[Hailo Model Zoo v5.4.0, Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_depth_estimation.rst)
(model `fast_depth`, source: [dwofk/fast-depth](https://github.com/dwofk/fast-depth)).