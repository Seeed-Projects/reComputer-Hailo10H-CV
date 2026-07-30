"""
ArcFace MobileFaceNet - Face Recognition
==========================================
Model: ArcFace + MobileFaceNet backbone
Input: 112x112 RGB, Output: 512-dim face embedding
Platform: Hailo-10H (Raspberry Pi CM5)
"""

import os, sys, cv2, argparse, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "model", "arcface_mobilefacenet.hef"))
parser.add_argument("--video_path", default=None)
parser.add_argument("--camera_id", type=int, default=None)
args, _ = parser.parse_known_args()

from py_utils.hailo_executor import HailoInfer

MODEL_PATH = args.model_path
INPUT_SIZE = (112, 112)
EMBEDDING_DIM = 512

app = FastAPI(title="ArcFace Face Recognition", version="1.0.0")

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
    img = (img / 127.5) - 1.0
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def post_process_hailo(output, orig_shape):
    embedding = list(output.values())[0]
    if embedding.ndim == 3:
        embedding = embedding[0]
    return embedding.flatten()

def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

@app.get("/health")
async def health():
    return {"status": "ok", "model": "arcface_mobilefacenet", "model_loaded": infer is not None}

@app.post("/api/models/arcface/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        embedding = post_process_hailo(output, (oh, ow))
    else:
        embedding = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        embedding /= np.linalg.norm(embedding)
    return {"embedding": embedding.tolist(), "dim": len(embedding)}

@app.post("/api/models/arcface/compare")
async def compare(file1: UploadFile = File(...), file2: UploadFile = File(...)):
    img1 = cv2.imdecode(np.frombuffer(await file1.read(), np.uint8), cv2.IMREAD_COLOR)
    img2 = cv2.imdecode(np.frombuffer(await file2.read(), np.uint8), cv2.IMREAD_COLOR)
    if img1 is None or img2 is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    if infer is not None:
        e1 = post_process_hailo(infer.run((preprocess(img1)*255).astype(np.uint8)), (0,0))
        e2 = post_process_hailo(infer.run((preprocess(img2)*255).astype(np.uint8)), (0,0))
    else:
        e1 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
        e2 = np.random.randn(EMBEDDING_DIM).astype(np.float32)
    sim = float(cosine_similarity(e1, e2))
    return {"similarity": sim, "same_person": sim > 0.5, "threshold": 0.5}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)