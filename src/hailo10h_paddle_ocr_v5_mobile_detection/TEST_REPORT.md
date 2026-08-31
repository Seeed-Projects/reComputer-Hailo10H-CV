# Test report — PaddleOCR v5 Mobile Detection (Hailo-10H)

## Local static validation

- Python syntax: `web_detection.py`, `py_utils/hailo_executor.py`, and `py_utils/db_postprocess.py` compile successfully.
- Bundled Hailo-10H detection HEF is present (5,160,960 bytes), SHA256 `35bf92cb...fc223ce0`.
- Executor swapped to the HailoRT 5.1.1 `create_infer_model` API (same as the other hailo10h modules); wheel is `hailort-5.1.1-cp313`.
- DB post-processing is unchanged from the validated Hailo-8 module (binarize -> contours -> min-area-rect -> score -> unclip).

## Hardware validation required

Run the Docker command from `README.md` on CM5 + Hailo-10H with HailoRT 5.1.1.
Confirm `/dev/hailo0`, open the MJPEG page, then POST `video/test.png` to the REST endpoint.
Verify the executor-reported output shape matches the Hailo-8 build (heatmap, values in [0, 1]).
