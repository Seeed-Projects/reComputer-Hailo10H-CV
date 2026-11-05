# Test Report — U-Net MobileNet v2 on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | unet_mobilenet_v2 |
| Task | Semantic segmentation (Oxford-IIIT Pet trimap) |
| Source | TensorFlow image segmentation tutorial |
| Parameters | 10.08M |
| Operations | 28.88G |
| mIoU (float / hardware) | 77.32 / 77.1 |
| Model Zoo version | v5.4.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/unet_mobilenet_v2.hef |
| Size | 8,630,272 bytes |
| SHA256 | 16fea8580277cc70d14e0491f52bb769853fb15171b5dc23a9d0ad631d5c7a10 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/unet_mobilenet_v2.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 256x256x3 RGB, uint8 (`normalize_in_net`, mean 0 / std 255) |
| Raw output | 256x256x3 per-pixel score map (no on-chip argmax/softmax) |
| Decode | host `np.argmax` over the class axis → 256x256 class mask |
| Classes | 0 = pet, 1 = border, 2 = background (TF tutorial `mask -= 1`) |
| Visualization | palette (pet green / border cyan, background transparent), alpha 0.4 |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Executor unit tests (mocked HailoRT) | Pass |
| Segmentation post-processing tests | Pass |
| Offline inference_loop harness (synthetic scores) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Raw output layout confirmed (256x256x3) | Pending |
| Class index order correct (pet / border / background) | Pending |
| Demo video: mask overlay aligns with the pet region | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The output is a 3-channel score map; if the HEF exposes a different channel
  order, the argmax indices and palette mapping need to be adjusted together.
- Background is not blended; if the whole frame is covered, check that the
  background class id matches the compiled output.
- The first-inference log prints the raw output shape and the labels present
  (`[UNet Pet] raw output shape=..., decoded mask shape=..., labels=[...]`).