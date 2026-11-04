# Test Report — FastDepth on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | fast_depth |
| Task | Monocular depth estimation (indoor) |
| Source | dwofk/fast-depth |
| Parameters | 1.35M |
| Operations | 0.74G |
| RMSE (float / hardware) | 0.60 / 0.61 |
| Model Zoo version | v5.4.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/fast_depth.hef |
| Size | 2,015,232 bytes |
| SHA256 | 8dead61ac1a6de037d463ce58d1c70bc917992708325bcd72a1bda38be560990 |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/fast_depth.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 224x224x3 RGB, uint8 (`normalize_in_net`, mean 0 / std 255) |
| Raw output | 224x224x1 depth map (NYU-v2 scale) |
| Decode | none — Model Zoo `fast_depth_postprocessing` returns the output unchanged |
| Visualization | per-frame 2nd/98th percentile normalization → TURBO colormap → alpha blend |
| Depth type | Model output scale (roughly meters), not calibrated sensor depth |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Executor unit tests (mocked HailoRT) | Pass |
| FastDepth post-processing tests | Pass |
| Offline inference_loop harness (synthetic output) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Raw output layout confirmed (224x224) | Pending |
| Depth direction correct (near = close, far = bright) | Pending |
| Demo video: depth overlay aligns with scene | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The output is the model's depth prediction, not a calibrated measurement.
  Values are normalized per frame for visualization.
- If the preview is blank or inverted, compare the first-inference log
  (`[FastDepth] raw output shape=...`) with the decode branch.
- The TURBO colormap and the default alpha can be adjusted from the web UI
  slider.