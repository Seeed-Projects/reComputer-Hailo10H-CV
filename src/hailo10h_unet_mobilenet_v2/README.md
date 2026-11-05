# U-Net MobileNet v2 on CM5 + Hailo-10H

U-Net MobileNet v2 performs semantic segmentation on **Hailo-10H** through
HailoRT. The HEF takes a 256x256 RGB frame and returns a 256x256x3 score map;
the app takes the per-pixel argmax (no argmax/softmax is compiled on chip),
un-letterboxes the class mask, and alpha-blends a palette overlay onto the
original frame. The FastAPI service supports images, video files, USB cameras,
an MJPEG preview, offline video analysis, and a REST prediction endpoint.

The model is the TensorFlow Oxford-IIIT Pet U-Net tutorial network; class
indices follow the tutorial's `mask -= 1` step: 0 = pet, 1 = border,
2 = background.

## Compatibility

| Component | Version |
|---|---|
| Accelerator | Hailo-10H PCIe (`/dev/hailo0`) |
| HailoRT host/runtime | 5.1.1 |
| Python | 3.13, aarch64 |
| Input | 256x256x3 RGB (`normalize_in_net`, mean 0 / std 255) |
| Output | 256x256x3 score map → 256x256 class mask (pet / border / background) |
| Parameters | 10.08M |
| Operations | 28.88G |
| mIoU (float / hardware) | 77.32 / 77.1 |
| HEF | Hailo Model Zoo v5.4.0, Hailo-10H (8,630,272 bytes) |

The host driver, firmware, `libhailort.so`, and Python wheel must use the same
HailoRT major/minor version.

## Build

```bash
sudo docker build -f docker/hailo10h/unet_mobilenet_v2.dockerfile \
  -t unet_mobilenet_v2:latest \
  src/hailo10h_unet_mobilenet_v2
```

## Run the demo video

```bash
sudo docker run --rm \
  --name cm5-hailo10h-unet-mobilenet-v2 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/unet_mobilenet_v2:latest \
  python web_detection.py --model_path model/unet_mobilenet_v2.hef --video_path video/test.mp4
```

Open `http://<BOARD_IP>:8000`.

For a USB camera, mount `/dev/video0` and replace `--video_path ...` with
`--camera_id 0`.

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/unet/predict" \
  -F "file=@test.jpg"
```

The response keeps the YOLO-style `predictions` list; each entry carries the
class name, the fraction of mask pixels it covers (`confidence`), and the raw
pixel count:

```json
{
  "class": "pet",
  "confidence": 0.42,
  "pixels": 27525
}
```

Background pixels are omitted from the list.

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Web preview UI |
| `/api/models/unet/predict` | POST | Segmentation summary (JSON) |
| `/api/video_feed` | GET | MJPEG with palette mask overlay |
| `/api/config` | GET / POST | Read or change runtime thresholds |
| `/api/video/upload` | POST | Upload a source video |
| `/api/video/analyze` | POST | Start offline video analysis |
| `/api/video/status` | GET | Read analysis progress |
| `/api/video/download/{filename}` | GET | Download the annotated result |

## Implementation notes

- The HEF output is a 3-channel per-pixel score map; `argmax` is disabled in
  the compile (`device_pre_post_layers.argmax: false`), so the host runs
  `np.argmax` over the class axis.
- The decoded mask is cropped to the letterbox content area and resized with
  nearest-neighbor so class indices stay integral.
- Palette: pet (green), border (cyan); background is not blended, so the
  original frame shows through. Overlay alpha is 0.4.
- `normalize_in_net` (mean 0 / std 255) is compiled into the HEF; feed raw
  uint8 RGB after letterboxing (pad color 0).
- No on-chip post-processing; the wrapper requests FLOAT32 vstreams so the host
  receives dequantized scores. The HEF input size is read at startup (256x256).

## Model source

[Hailo Model Zoo v5.4.0, Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_semantic_segmentation.rst)
(model `unet_mobilenet_v2`, source:
[TensorFlow image segmentation tutorial](https://www.tensorflow.org/tutorials/images/segmentation),
trained on Oxford-IIIT Pets).