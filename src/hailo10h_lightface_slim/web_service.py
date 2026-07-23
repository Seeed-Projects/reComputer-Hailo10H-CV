"""
LightFace Slim - Face Detection Inference Server
==================================================
Ultra-Light-Fast-Generic-Face-Detector-1MB
Input:  240x320 RGB
Output: Face bounding boxes with confidence
Platform: Hailo-10H
"""

import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

try:
    from hailo_platform import (
        VDevice, HEF, ConfigureParams,
        InputVStreamParams, OutputVStreamParams,
        FormatType, InferVStreams
    )
    HAILO_AVAILABLE = True
except ImportError:
    HAILO_AVAILABLE = False
    print("Warning: HailoRT not available, running in mock mode")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "lightface_slim.hef")
INPUT_SIZE = (320, 240)  # (width, height)
CONF_THRESHOLD = 0.5

app = FastAPI(title="LightFace Slim - Face Detection", version="1.0.0")

target = None
network_group = None
infer_streams = None
input_tensor_name = None

if HAILO_AVAILABLE and os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading HEF: {MODEL_PATH}")
    hef = HEF(MODEL_PATH)
    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=target.get_default_streams_interface())
    network_group = target.configure(hef, configure_params)[0]
    input_info = network_group.get_input_vstream_infos()[0]
    input_tensor_name = input_info.name
    output_info = network_group.get_output_vstream_infos()[0]
    print(f"[INFO] Input: {input_tensor_name} ({input_info.shape}), Output: {output_info.name}")
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    infer_streams = InferVStreams(network_group, input_vstreams_params, output_vstreams_params)
    print("[OK] Model loaded")
else:
    if not HAILO_AVAILABLE: print("[WARN] Mock mode")
    if not os.path.exists(MODEL_PATH): print(f"[WARN] Model not found: {MODEL_PATH}")

def preprocess(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.0) / 128.0
    return np.expand_dims(img, axis=0)

def draw_boxes(image: np.ndarray, boxes, confs):
    vis = image.copy()
    for box, conf in zip(boxes, confs):
        if conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return vis

@app.get("/health")
async def health():
    return {"status": "ok", "model": "lightface_slim", "model_loaded": infer_streams is not None}

@app.post("/api/models/lightface/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    h, w = img.shape[:2]
    input_tensor = preprocess(img)

    if infer_streams is not None:
        output = infer_streams.infer({input_tensor_name: input_tensor})
        detections = list(output.values())[0]
        boxes = detections[0][:, :4]
        confs = detections[0][:, 4]
        boxes[:, [0, 2]] *= w / 320
        boxes[:, [1, 3]] *= h / 240
    else:
        boxes = np.array([[w*0.2, h*0.2, w*0.6, h*0.6]])
        confs = np.array([0.85])

    faces = []
    for box, conf in zip(boxes, confs):
        if conf > CONF_THRESHOLD:
            faces.append({
                "x1": float(box[0]), "y1": float(box[1]),
                "x2": float(box[2]), "y2": float(box[3]),
                "confidence": float(conf)
            })

    return {"faces": faces, "count": len(faces)}

@app.post("/api/models/lightface/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    h, w = img.shape[:2]
    input_tensor = preprocess(img)

    if infer_streams is not None:
        output = infer_streams.infer({input_tensor_name: input_tensor})
        detections = list(output.values())[0]
        boxes = detections[0][:, :4]
        confs = detections[0][:, 4]
        boxes[:, [0, 2]] *= w / 320
        boxes[:, [1, 3]] *= h / 240
    else:
        boxes = np.array([[w*0.2, h*0.2, w*0.6, h*0.6]])
        confs = np.array([0.85])

    vis = draw_boxes(img, boxes, confs)
    _, buf = cv2.imencode('.jpg', vis)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)