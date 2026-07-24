"""
MSPN RegNetX-800MF - Semantic Segmentation Inference Server
==============================================
Model: MSPN RegNetX-800MF (Short-Term Dense Concatenate)
Input: 1024x1920 RGB, Output: 19-class mask (Cityscapes)
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
INPUT_SIZE = (1920, 1024)
NUM_CLASSES = 19

CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

CITYSCAPES_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

app = FastAPI(title="MSPN RegNetX-800MF Semantic Segmentation", version="1.0.0")

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
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img / 255.0 - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def mask_to_color(mask):
    h, w = mask.shape
    cm = np.zeros((h, w, 3), dtype=np.uint8)
    for i, c in enumerate(CITYSCAPES_COLORS):
        cm[mask == i] = c
    return cm

def post_process_hailo(output, orig_shape):
    logits = list(output.values())[0]
    mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    oh, ow = orig_shape[:2]
    if (oh, ow) != (1024, 1920):
        mask = cv2.resize(mask, (ow, oh), interpolation=cv2.INTER_NEAREST)
    return mask

@app.get("/health")
async def health():
    return {"status": "ok", "model": "mspn_regnetx_800mf", "model_loaded": infer is not None}

@app.get("/api/models/mspn_regnetx_800mf/classes")
async def get_classes():
    return {"classes": [{"id": i, "name": n, "color": list(c)} for i, (n, c) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_COLORS))]}

@app.post("/api/models/mspn_regnetx_800mf/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        mask = post_process_hailo(output, (oh, ow))
    else:
        mask = np.random.randint(0, NUM_CLASSES, (oh, ow), dtype=np.uint8)
    unique, counts = np.unique(mask, return_counts=True)
    stats = {CITYSCAPES_CLASSES[int(c)]: float(n / mask.size * 100) for c, n in zip(unique, counts)}
    return {"mask": mask.tolist(), "width": ow, "height": oh, "num_classes": NUM_CLASSES, "stats": stats}

@app.post("/api/models/mspn_regnetx_800mf/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        mask = post_process_hailo(output, (oh, ow))
    else:
        mask = np.random.randint(0, NUM_CLASSES, (oh, ow), dtype=np.uint8)
    blended = cv2.addWeighted(cv2.resize(img, (ow, oh)), 0.5, mask_to_color(mask), 0.5, 0)
    _, buf = cv2.imencode('.jpg', blended)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)