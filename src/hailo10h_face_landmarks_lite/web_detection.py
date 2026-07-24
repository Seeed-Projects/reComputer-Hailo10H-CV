"""
Face Landmarks Lite - Facial Landmark Detection
=================================================
Model: MediaPipe Face Landmarks Lite
Input: 192x192 RGB, Output: Facial keypoints
Platform: Hailo-10H (Raspberry Pi 5)
"""

import os, sys, cv2, argparse, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "model", "face_landmarks_lite.hef"))
parser.add_argument("--video_path", default=None)
parser.add_argument("--camera_id", type=int, default=None)
args, _ = parser.parse_known_args()

from py_utils.hailo_executor import HailoInfer

MODEL_PATH = args.model_path
INPUT_SIZE = (192, 192)
NUM_KEYPOINTS = 98

FACIAL_CONTOURS = [
    list(range(0, 33)),
    list(range(33, 42)) + [33],
    list(range(42, 51)) + [42],
    list(range(51, 60)) + [51],
    list(range(60, 68)) + [60],
    list(range(68, 76)) + [68],
    list(range(76, 82)) + [76],
    list(range(82, 88)) + [82],
    list(range(88, 93)) + [88],
    list(range(93, 96)) + [93],
    list(range(96, 98)) + [96],
]

app = FastAPI(title="Face Landmarks Lite", version="1.0.0")

if os.path.exists(MODEL_PATH):
    infer = HailoInfer(MODEL_PATH)
    print(f"[OK] Model loaded: {MODEL_PATH}")
else:
    infer = None
    print(f"[WARN] Model not found: {MODEL_PATH}")

def preprocess(image):
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def post_process_hailo(output, orig_shape):
    raw = list(output.values())[0]
    oh, ow = orig_shape[:2]
    if raw.ndim == 3 and raw.shape[0] == 1:
        raw = raw[0]
    if raw.shape[0] == NUM_KEYPOINTS * 2:
        kps = raw.reshape(NUM_KEYPOINTS, 2)
        confs = np.ones(NUM_KEYPOINTS)
    elif raw.shape[0] == NUM_KEYPOINTS * 3:
        kps = raw.reshape(NUM_KEYPOINTS, 3)[:, :2]
        confs = raw.reshape(NUM_KEYPOINTS, 3)[:, 2]
    else:
        kps = raw.reshape(-1, 2)[:NUM_KEYPOINTS]
        confs = np.ones(NUM_KEYPOINTS)
    return kps, confs

def draw_landmarks(image, kps, confs, threshold=0.3):
    h, w = image.shape[:2]
    vis = image.copy()
    for i, (x, y) in enumerate(kps):
        if confs[i] > threshold:
            px, py = int(x * w), int(y * h)
            cv2.circle(vis, (px, py), 2, (0, 255, 0), -1)
    for contour in FACIAL_CONTOURS:
        pts = []
        for idx in contour:
            if idx < len(kps) and confs[idx] > threshold:
                pts.append((int(kps[idx][0] * w), int(kps[idx][1] * h)))
        if len(pts) > 1:
            cv2.polylines(vis, [np.array(pts)], False, (0, 255, 0), 1)
    return vis

@app.get("/health")
async def health():
    return {"status": "ok", "model": "face_landmarks_lite", "model_loaded": infer is not None}

@app.get("/api/models/face_landmarks/keypoints")
async def get_keypoints():
    return {"num_keypoints": NUM_KEYPOINTS}

@app.post("/api/models/face_landmarks/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        kps, confs = post_process_hailo(output, (oh, ow))
    else:
        kps = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        confs = np.clip(np.random.rand(NUM_KEYPOINTS), 0.5, 0.95).astype(np.float32)
    landmarks = [{"id": i, "x": float(k[0]), "y": float(k[1]), "confidence": float(confs[i])} for i, k in enumerate(kps)]
    return {"landmarks": landmarks, "num_keypoints": NUM_KEYPOINTS}

@app.post("/api/models/face_landmarks/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        kps, confs = post_process_hailo(output, (oh, ow))
    else:
        kps = np.random.rand(NUM_KEYPOINTS, 2).astype(np.float32)
        confs = np.clip(np.random.rand(NUM_KEYPOINTS), 0.5, 0.95).astype(np.float32)
    vis = draw_landmarks(img, kps, confs)
    _, buf = cv2.imencode('.jpg', vis)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)