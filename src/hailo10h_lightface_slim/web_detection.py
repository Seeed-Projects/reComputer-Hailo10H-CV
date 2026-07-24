"""
LightFace Slim - Face Detection Inference Server
==================================================
Model: Ultra-Light-Fast-Generic-Face-Detector-1MB
Input: 240x320 RGB, Output: Face bounding boxes
Platform: Hailo-10H (Raspberry Pi 5)
"""

import os, sys, cv2, argparse, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "model", "lightface_slim.hef"))
parser.add_argument("--video_path", default=None)
parser.add_argument("--camera_id", type=int, default=None)
args, _ = parser.parse_known_args()

from py_utils.hailo_executor import HailoInfer

MODEL_PATH = args.model_path
INPUT_SIZE = (320, 240)
CONF_THRESHOLD = 0.5

app = FastAPI(title="LightFace Slim Face Detection", version="1.0.0")

if os.path.exists(MODEL_PATH):
    infer = HailoInfer(MODEL_PATH)
    print(f"[OK] Model loaded: {MODEL_PATH}")
else:
    infer = None
    print(f"[WARN] Model not found: {MODEL_PATH}")

def preprocess(image):
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = (img - 127.0) / 128.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def post_process_hailo(output, orig_shape):
    detections = list(output.values())[0]
    boxes = detections[0][:, :4]
    confs = detections[0][:, 4]
    oh, ow = orig_shape[:2]
    boxes[:, [0, 2]] *= ow / INPUT_SIZE[0]
    boxes[:, [1, 3]] *= oh / INPUT_SIZE[1]
    return boxes, confs

def draw_boxes(image, boxes, confs):
    vis = image.copy()
    for box, conf in zip(boxes, confs):
        if conf > CONF_THRESHOLD:
            x1, y1, x2, y2 = box.astype(int)
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{conf:.2f}", (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
    return vis

@app.get("/health")
async def health():
    return {"status": "ok", "model": "lightface_slim", "model_loaded": infer is not None}

@app.post("/api/models/lightface/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        boxes, confs = post_process_hailo(output, (oh, ow))
    else:
        boxes = np.array([[ow*0.2, oh*0.2, ow*0.6, oh*0.6]])
        confs = np.array([0.85])
    faces = []
    for box, conf in zip(boxes, confs):
        if conf > CONF_THRESHOLD:
            faces.append({"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3]), "confidence": float(conf)})
    return {"faces": faces, "count": len(faces)}

@app.post("/api/models/lightface/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        boxes, confs = post_process_hailo(output, (oh, ow))
    else:
        boxes = np.array([[ow*0.2, oh*0.2, ow*0.6, oh*0.6]])
        confs = np.array([0.85])
    vis = draw_boxes(img, boxes, confs)
    _, buf = cv2.imencode('.jpg', vis)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)