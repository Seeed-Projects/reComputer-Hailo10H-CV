# Test Report — SCDepthV3 on CM5 + Hailo-10H

## Model

| Field | Value |
|---|---|
| Model name | scdepthv3 |
| Task | Monocular relative-depth estimation (indoor) |
| Source | JiawangBian/sc_depth_pl |
| Parameters | 14.8M |
| Operations | 10.7G |
| RMSE (full precision) | 0.481 |
| Model Zoo version | v5.4.0 |
| Compile target | hailo10h |

## HEF artifact

| Field | Value |
|---|---|
| File | model/scdepthv3.hef |
| Size | 11,579,392 bytes |
| SHA256 | c8817322d879bf950038a43d9ed955a05f9a69324779a2ece086fe8b9fc4a37c |
| Source | https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/scdepthv3.hef |

## Input / output

| Field | Value |
|---|---|
| Input | 256x320x3 RGB, uint8 (`normalize_in_net`, mean 114.75 / std 57.375) |
| Raw output | 256x320x1 logits |
| Decode | `depth = 1 / (sigmoid(raw) * 10 + 0.009)` |
| Visualization | per-frame 2nd/98th percentile normalization → TURBO colormap → alpha blend |
| Depth type | Relative (not metric meters) |

## Verification status

| Check | Status |
|---|---|
| Python syntax (py_compile) | Pass |
| Executor unit tests (mocked HailoRT) | Pass |
| SCDepthV3 post-processing tests | Pass |
| Offline inference_loop harness (synthetic logits) | Pass |
| Dockerfile present + CMD references real HEF | Pass |
| CI matrix entry added | Pass |

### Hardware verification — pending

| Check | Status |
|---|---|
| HEF loads | Pending |
| Raw output layout confirmed (256x320 or transposed 320x256) | Pending |
| Depth direction correct (near = close, far = bright) | Pending |
| Demo video: depth overlay aligns with scene | Pending |
| Official GHCR image re-pull + run | Pending |

## Known points to verify on hardware

- The decoded output is **relative depth**, not metric. Values are normalized
  per frame for visualization.
- If the preview looks transposed, compare the first-inference log
  (`[SCDepthV3] raw output shape=...`) with the transpose branch in
  `post_process_hailo`.
- The TURBO colormap and the default alpha 0.65 can be adjusted from the web UI
  slider.