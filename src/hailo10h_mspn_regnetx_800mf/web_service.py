"""
MSPN RegNetX-800MF - Single Person Pose Estimation
====================================================
MSPN (Multi-Stage Pose Network) with RegNetX-800MF backbone
Input:  256x192 RGB
Output: 17 keypoints (COCO format)
Platform: Hailo-10H (Raspberry Pi CM5)
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

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "mspn_regnetx_800mf.hef")
INPUT_SIZE = (192, 256)  # (width, height)
NUM_KEYPOINTS = 17

COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

COCO_SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12),
    (5, 11), (6, 12), (5, 6), (5, 7), (6, 8),
    (7, 9), (8, 10), (1, 2), (0, 1), (0, 2),
    (1, 3), (2, 4), (3, 5), (4, 6)
]

KEYPOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (128, 255, 0), (255, 128, 0), (0, 128, 255), (128, 0, 255),
    (255, 255, 128), (128, 255, 255), (255, 128, 255), (0, 255, 128), (128, 128, 255),
    (255, 128, 128), (128, 255, 128)
]

app = FastAPI(title="MSPN Pose Estimation", version="1.0.0")

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
    print(f"[INFO] Input: {input_tensor_name} ({input_info.shape})")
    print(f"[INFO] Output: {output_info.name} ({output_info.shape})")
    input_vstreams_params = InputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    output_vstreams_params = OutputVStreamParams.make(network_group, format_type=FormatType.FLOAT32)
    infer_streams = InferVStreams(network_group, input_vstreams_params, output_vstreams_params)
    print("[OK] Hailo-10H model loaded")
else:
    if not HAILO_AVAILABLE:
        print("[WARN] HailoRT not installed — mock mode")
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model not found: {MODEL_PATH}")

def preprocess(image: np.ndarray) -> np.ndarray:
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def draw_pose(image: np.ndarray, keypoints: np.ndarray, threshold=0.3) -> np.ndarray:
    h, w = image.shape[:2]
    vis = image.copy()
    for i, (x, y, conf) in enumerate(keypoints):
        if conf > threshold:
            px, py = int(x * w), int(y * h)
            cv2.circle(vis, (px, py), 4, KEYPOINT_COLORS[i], -1)
            cv2.putText(vis, str(i), (px+6, py), cv2.FONT_HERSHEY_SIMPLEX, 0.4, KEYPOINT_COLORS[i], 1)
    for a, b in COCO_SKELETON:
        if keypoints[a][2] > threshold and keypoints[b][2] > threshold:
            p1 = (int(keypoints[a][0] * w), int(keypoints[a][1] * h))
            p2 = (int(keypoints[b][0] * w), int(keypoints[b][1] * h))
            cv2.line(vis, p1, p2, (0, 255, 0), 2)
    return vis

@app.get("/health")
async def health():
    return {"status": "ok", "model": "mspn_regnetx_800mf", "platform": "Hailo-10H", "model_loaded": infer_streams is not None}

@app.get("/api/models/mspn_pose/keypoints")
async def get_keypoints():
    return {"keypoints": COCO_KEYPOINTS, "skeleton": [[a, b] for a, b in COCO_SKELETON]}

@app.post("/api/models/mspn_pose/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    original_h, original_w = img.shape[:2]
    input_tensor = preprocess(img)

    if infer_streams is not None:
        output = infer_streams.infer({input_tensor_name: input_tensor})
        heatmaps = list(output.values())[0]
        keypoints = []
        for k in range(NUM_KEYPOINTS):
            hm = heatmaps[0][k]
            idx = np.argmax(hm)
            y, x = idx // hm.shape[1], idx % hm.shape[1]
            conf = float(hm[y, x])
            keypoints.append([x / hm.shape[1], y / hm.shape[0], conf])
        keypoints = np.array(keypoints, dtype=np.float32)
    else:
        keypoints = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float32)
        keypoints[:, 2] = np.clip(keypoints[:, 2], 0.3, 0.9)

    return {
        "keypoints": [{"name": COCO_KEYPOINTS[i], "x": float(kp[0]), "y": float(kp[1]), "confidence": float(kp[2])} for i, kp in enumerate(keypoints)],
        "num_keypoints": NUM_KEYPOINTS
    }

@app.post("/api/models/mspn_pose/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    input_tensor = preprocess(img)

    if infer_streams is not None:
        output = infer_streams.infer({input_tensor_name: input_tensor})
        heatmaps = list(output.values())[0]
        keypoints = []
        for k in range(NUM_KEYPOINTS):
            hm = heatmaps[0][k]
            idx = np.argmax(hm)
            y, x = idx // hm.shape[1], idx % hm.shape[1]
            keypoints.append([x / hm.shape[1], y / hm.shape[0], float(hm[y, x])])
        keypoints = np.array(keypoints, dtype=np.float32)
    else:
        keypoints = np.random.rand(NUM_KEYPOINTS, 3).astype(np.float32)
        keypoints[:, 2] = np.clip(keypoints[:, 2], 0.3, 0.9)

    vis = draw_pose(img, keypoints)
    _, buf = cv2.imencode('.jpg', vis)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)