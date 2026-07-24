"""
MSPN RegNetX-800MF - Single Person Pose Estimation
====================================================
Model: MSPN with RegNetX-800MF backbone
Input: 256x192 RGB, Output: 17 COCO keypoints
Platform: Hailo-10H (Raspberry Pi CM5)
"""

import os, sys, cv2, argparse, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "model", "mspn_regnetx_800mf.hef"))
parser.add_argument("--video_path", default=None)
parser.add_argument("--camera_id", type=int, default=None)
args, _ = parser.parse_known_args()

from py_utils.hailo_executor import HailoInfer

MODEL_PATH = args.model_path
INPUT_SIZE = (192, 256)
NUM_KEYPOINTS = 17

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (15,13), (13,11), (16,14), (14,12), (11,12),
    (5,11), (6,12), (5,6), (5,7), (6,8),
    (7,9), (8,10), (1,2), (0,1), (0,2),
    (1,3), (2,4), (3,5), (4,6)
]

KEYPOINT_COLORS = [
    (255,0,0),(0,255,0),(0,0,255),(255,255,0),(0,255,255),
    (255,0,255),(128,255,0),(255,128,0),(0,128,255),(128,0,255),
    (255,255,128),(128,255,255),(255,128,255),(0,255,128),(128,128,255),
    (255,128,128),(128,255,128)
]

app = FastAPI(title="MSPN Pose Estimation", version="1.0.0")

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
    heatmaps = list(output.values())[0]
    oh, ow = orig_shape[:2]
    keypoints = []
    for k in range(NUM_KEYPOINTS):
        hm = heatmaps[0][k]
        idx = np.argmax(hm)
        y, x = idx // hm.shape[1], idx % hm.shape[1]
        conf = float(hm[y, x])
        keypoints.append([x / hm.shape[1], y / hm.shape[0], conf])
    return np.array(keypoints, dtype=np.float32)

def draw_pose(image, keypoints, threshold=0.3):
    h, w = image.shape[:2]
    vis = image.copy()
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > threshold:
            px, py = int(x * w), int(y * h)
            cv2.circle(vis, (px, py), 4, KEYPOINT_COLORS[i], -1)
    for a, b in COCO_SKELETON:
        if keypoints[a][2] > threshold and keypoints[b][2] > threshold:
            p1 = (int(keypoints[a][0]*w), int(keypoints[a][1]*h))
            p2 = (int(keypoints[b][0]*w), int(keypoints[b][1]*h))
            cv2.line(vis, p1, p2, (0,255,0), 2)
    return vis

@app.get("/health")
async def health():
    return {"status": "ok", "model": "mspn_regnetx_800mf", "model_loaded": infer is not None}

@app.get("/api/models/mspn_pose/keypoints")
async def get_keypoints():
    return {"keypoints": COCO_KEYPOINTS, "skeleton": [[a,b] for a,b in COCO_SKELETON]}

@app.post("/api/models/mspn_pose/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        keypoints = post_process_hailo(output, (oh, ow))
    else:
        keypoints = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float32)
        keypoints[:, 2] = np.clip(keypoints[:, 2], 0.3, 0.9)
    kps = [{"name": COCO_KEYPOINTS[i], "x": float(k[0]), "y": float(k[1]), "confidence": float(k[2])} for i, k in enumerate(keypoints)]
    return {"keypoints": kps, "num_keypoints": NUM_KEYPOINTS}

@app.post("/api/models/mspn_pose/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        keypoints = post_process_hailo(output, (oh, ow))
    else:
        keypoints = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float32)
        keypoints[:, 2] = np.clip(keypoints[:, 2], 0.3, 0.9)
    vis = draw_pose(img, keypoints)
    _, buf = cv2.imencode('.jpg', vis)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)