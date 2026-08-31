# Test report — PaddleOCR v5 Mobile Recognition (Hailo-10H)

## Local static validation

- Python syntax: `web_detection.py`, `py_utils/hailo_executor.py`, and `py_utils/ctc_decoder.py` compile successfully.
- Bundled Hailo-10H recognition HEF is present (5,120,000 bytes), SHA256 `5f5b3113...774435e`.
- Executor swapped to the HailoRT 5.1.1 `create_infer_model` API (same as the other hailo10h modules); wheel is `hailort-5.1.1-cp313`.
- CTC decoder already strips double batch dims and returns a dict (fixes ported from the Hailo-8 module).

## Hardware validation required

Run the Docker command from `README.md` on CM5 + Hailo-10H with HailoRT 5.1.1.
Confirm `/dev/hailo0`, open the MJPEG page, then POST `video/test.png` to the REST endpoint.
Verify the decoded text matches the demo line and the confidence is above 0.5.
