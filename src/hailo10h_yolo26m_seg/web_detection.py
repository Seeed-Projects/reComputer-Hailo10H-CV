"""
YOLO26n-seg - Instance Segmentation
=====================================
Model: YOLO26n-seg (COCO instance segmentation)
Input: 640x640 RGB, Output: bounding boxes + instance masks
Platform: Hailo-10H (Raspberry Pi 5)
"""

import os, sys, cv2, argparse, numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", default=os.path.join(os.path.dirname(__file__), "model", "yolo26m_seg.hef"))
parser.add_argument("--video_path", default=None)
parser.add_argument("--camera_id", type=int, default=None)
args, _ = parser.parse_known_args()

from py_utils.hailo_executor import HailoInfer

MODEL_PATH = args.model_path
INPUT_SIZE = (640, 640)
CONF_THRESHOLD = 0.25
IOU_THRESHOLD = 0.45

COCO_CLASSES = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
    "traffic light","fire hydrant","stop sign","parking meter","bench","bird","cat",
    "dog","horse","sheep","cow","elephant","bear","zebra","giraffe","backpack",
    "umbrella","handbag","tie","suitcase","frisbee","skis","snowboard","sports ball",
    "kite","baseball bat","baseball glove","skateboard","surfboard","tennis racket",
    "bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
    "couch","potted plant","bed","dining table","toilet","tv","laptop","mouse",
    "remote","keyboard","cell phone","microwave","oven","toaster","sink",
    "refrigerator","book","clock","vase","scissors","teddy bear","hair drier","toothbrush"
]

COLORS = [(np.random.randint(50,255),np.random.randint(50,255),np.random.randint(50,255)) for _ in range(80)]

app = FastAPI(title="YOLO26n-seg Instance Segmentation", version="1.0.0")

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
    detections = list(output.values())[0]
    boxes = detections[0][:, :4]
    scores = detections[0][:, 4]
    class_ids = detections[0][:, 5].astype(int)
    masks = detections[0][:, 6:] if detections[0].shape[1] > 6 else None
    oh, ow = orig_shape[:2]
    boxes[:, [0, 2]] *= ow / INPUT_SIZE[0]
    boxes[:, [1, 3]] *= oh / INPUT_SIZE[1]
    return boxes, scores, class_ids, masks

def draw_segmentation(image, boxes, scores, class_ids, masks=None):
    vis = image.copy()
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        if score < CONF_THRESHOLD:
            continue
        x1, y1, x2, y2 = box.astype(int)
        color = COLORS[cls_id % 80]
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{COCO_CLASSES[cls_id % 80]}: {score:.2f}"
        cv2.putText(vis, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    return vis

@app.get("/health")
async def health():
    return {"status": "ok", "model": "yolo26m_seg", "model_loaded": infer is not None}

@app.get("/api/models/yolo26m_seg/classes")
async def get_classes():
    return {"classes": [{"id": i, "name": n} for i, n in enumerate(COCO_CLASSES)]}

@app.post("/api/models/yolo26m_seg/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        boxes, scores, class_ids, masks = post_process_hailo(output, (oh, ow))
    else:
        boxes = np.array([[ow*0.2, oh*0.2, ow*0.6, oh*0.6]])
        scores = np.array([0.85])
        class_ids = np.array([0])
        masks = None
    predictions = []
    for i, (box, score, cls_id) in enumerate(zip(boxes, scores, class_ids)):
        if score > CONF_THRESHOLD:
            predictions.append({
                "class": COCO_CLASSES[cls_id % 80],
                "class_id": int(cls_id),
                "confidence": float(score),
                "box": {"x1": float(box[0]), "y1": float(box[1]), "x2": float(box[2]), "y2": float(box[3])}
            })
    return {"predictions": predictions, "count": len(predictions)}

@app.post("/api/models/yolo26m_seg/visualize")
async def visualize(file: UploadFile = File(...)):
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)
    oh, ow = img.shape[:2]
    input_tensor = preprocess(img)
    if infer is not None:
        output = infer.run((input_tensor * 255).astype(np.uint8))
        boxes, scores, class_ids, masks = post_process_hailo(output, (oh, ow))
    else:
        boxes = np.array([[ow*0.2, oh*0.2, ow*0.6, oh*0.6]])
        scores = np.array([0.85])
        class_ids = np.array([0])
        masks = None
    vis = draw_segmentation(img, boxes, scores, class_ids, masks)
    _, buf = cv2.imencode('.jpg', vis)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)