"""
LightFace Slim - Semantic Segmentation Inference Server
==============================================
Ultra-Lightweight Face Detection
Model: LightFace Slim (Short-Term Dense Concatenate)
Input:  1024x1920 RGB
Output: 1024x1920 segmentation mask (19 classes, Cityscapes)
Platform: Hailo-10H (Raspberry Pi CM5)
"""

import os
import sys
import cv2
import argparse
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "model", "lightface.hef"))
parser.add_argument("--video_path", default=None)
parser.add_argument("--camera_id", type=int, default=None)
args, _ = parser.parse_known_args()

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

app = FastAPI(title="LightFace Slim Semantic Segmentation", version="1.0.0")

target = None
network_group = None
infer_streams = None
input_tensor_name = None

if HAILO_AVAILABLE and os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading HEF model: {MODEL_PATH}")
    hef = HEF(MODEL_PATH)

    target = VDevice()

    configure_params = ConfigureParams.create_from_hef(
        hef, interface=target.get_default_streams_interface()
    )
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    input_info = network_group.get_input_vstream_infos()[0]
    input_tensor_name = input_info.name
    print(f"[INFO] Input tensor: {input_tensor_name} ({input_info.shape})")

    output_info = network_group.get_output_vstream_infos()[0]
    print(f"[INFO] Output tensor: {output_info.name} ({output_info.shape})")

    input_vstreams_params = InputVStreamParams.make(
        network_group, format_type=FormatType.FLOAT32
    )
    output_vstreams_params = OutputVStreamParams.make(
        network_group, format_type=FormatType.FLOAT32
    )

    infer_streams = InferVStreams(
        network_group, input_vstreams_params, output_vstreams_params
    )

    print("[OK] Hailo-10H model loaded and ready")
else:
    if not HAILO_AVAILABLE:
        print("[WARN] HailoRT not installed — running in mock mode")
    if not os.path.exists(MODEL_PATH):
        print(f"[WARN] Model file not found: {MODEL_PATH}")

def preprocess(image: np.ndarray) -> np.ndarray:
    """Resize to 1024x1920 and normalize"""
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img / 255.0 - mean) / std
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """Convert class index mask to color image"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == cls_id] = color
    return color_mask

def post_process_hailo(output, original_shape):
    """Post-process Hailo inference output for semantic segmentation"""
    logits = list(output.values())[0]
    mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    h, w = original_shape[:2]
    if (h, w) != (1024, 1920):
        mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
    return mask

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "lightface",
        "platform": "Hailo-10H",
        "hailo_available": HAILO_AVAILABLE,
        "model_loaded": infer_streams is not None,
        "mode": "hailo" if infer_streams is not None else "mock"
    }

@app.get("/api/models/lightface/classes")
async def get_classes():
    """Return all class names and colors"""
    return {
        "classes": [
            {"id": i, "name": name, "color": list(color)}
            for i, (name, color) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_COLORS))
        ]
    }

@app.post("/api/models/lightface/predict")
async def predict(file: UploadFile = File(...)):
    """
    Perform semantic segmentation on uploaded image
    Return per-pixel class labels
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    original_h, original_w = img.shape[:2]

    input_tensor = preprocess(img)

    if infer_streams is not None:
        input_data = {input_tensor_name: input_tensor}
        output = infer_streams.infer(input_data)
        logits = list(output.values())[0]
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    else:
        mask = np.random.randint(0, NUM_CLASSES, (1024, 1920), dtype=np.uint8)

    if (original_h, original_w) != (1024, 1920):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    unique, counts = np.unique(mask, return_counts=True)
    stats = {
        CITYSCAPES_CLASSES[int(cls_id)]: float(count / mask.size * 100)
        for cls_id, count in zip(unique, counts)
    }

    return {
        "mask": mask.tolist(),
        "width": original_w,
        "height": original_h,
        "num_classes": NUM_CLASSES,
        "stats": stats
    }

@app.post("/api/models/lightface/visualize")
async def visualize(file: UploadFile = File(...)):
    """
    Return visualized segmentation overlay
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    original_h, original_w = img.shape[:2]

    input_tensor = preprocess(img)

    if infer_streams is not None:
        logits = list(infer_streams.infer({input_tensor_name: input_tensor}).values())[0]
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    else:
        mask = np.random.randint(0, NUM_CLASSES, (1024, 1920), dtype=np.uint8)

    if (original_h, original_w) != (1024, 1920):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    color_mask = mask_to_color(mask)
    original = cv2.resize(img, (original_w, original_h))
    blended = cv2.addWeighted(original, 0.5, color_mask, 0.5, 0)

    _, buf = cv2.imencode('.jpg', blended)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)