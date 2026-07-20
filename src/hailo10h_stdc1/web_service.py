"""
STDC1 - Semantic Segmentation Inference Server
==============================================
Rethinking BiSeNet For Real-time Semantic Segmentation
Model: STDC1 (Short-Term Dense Concatenate)
Input:  1024x1920 RGB
Output: 1024x1920 segmentation mask (19 classes, Cityscapes)
Platform: Hailo-10H (Raspberry Pi CM5)
"""

import os
import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, Response
import uvicorn

# ── 尝试导入 HailoRT ──────────────────────────────
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

# ── 常量 ──────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "stdc1.hef")
INPUT_SIZE = (1920, 1024)  # (width, height)
NUM_CLASSES = 19

# Cityscapes 19 类标签
CITYSCAPES_CLASSES = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation", "terrain",
    "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

# 每类颜色（BGR 格式，用于可视化）
CITYSCAPES_COLORS = [
    (128, 64, 128), (244, 35, 232), (70, 70, 70), (102, 102, 156), (190, 153, 153),
    (153, 153, 153), (250, 170, 30), (220, 220, 0), (107, 142, 35), (152, 251, 152),
    (70, 130, 180), (220, 20, 60), (255, 0, 0), (0, 0, 142), (0, 0, 70),
    (0, 60, 100), (0, 80, 100), (0, 0, 230), (119, 11, 32)
]

app = FastAPI(title="STDC1 Semantic Segmentation", version="1.0.0")

# ── 加载 Hailo-10H 模型 ───────────────────────────
target = None
network_group = None
infer_streams = None
input_tensor_name = None

if HAILO_AVAILABLE and os.path.exists(MODEL_PATH):
    print(f"[INFO] Loading HEF model: {MODEL_PATH}")
    hef = HEF(MODEL_PATH)

    # 创建虚拟设备（Hailo-10H）
    target = VDevice()

    # 配置网络组
    configure_params = ConfigureParams.create_from_hef(
        hef, interface=target.get_default_streams_interface()
    )
    network_group = target.configure(hef, configure_params)[0]
    network_group_params = network_group.create_params()

    # 获取输入张量名称
    input_info = network_group.get_input_vstream_infos()[0]
    input_tensor_name = input_info.name
    print(f"[INFO] Input tensor: {input_tensor_name} ({input_info.shape})")

    # 获取输出张量信息
    output_info = network_group.get_output_vstream_infos()[0]
    print(f"[INFO] Output tensor: {output_info.name} ({output_info.shape})")

    # 创建输入输出流
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

# ── 预处理 ────────────────────────────────────────
def preprocess(image: np.ndarray) -> np.ndarray:
    """Resize 到 1024x1920，归一化"""
    img = cv2.resize(image, INPUT_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    # 标准化：ImageNet 均值/标准差
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img = (img / 255.0 - mean) / std
    # NHWC → NCHW
    img = np.transpose(img, (2, 0, 1))
    return np.expand_dims(img, axis=0)

def mask_to_color(mask: np.ndarray) -> np.ndarray:
    """将类别索引掩码转换为彩色图像"""
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for cls_id, color in enumerate(CITYSCAPES_COLORS):
        color_mask[mask == cls_id] = color
    return color_mask

# ── API 端点 ──────────────────────────────────────
@app.get("/health")
async def health():
    return {
        "status": "ok",
        "model": "stdc1",
        "platform": "Hailo-10H",
        "hailo_available": HAILO_AVAILABLE,
        "model_loaded": infer_streams is not None,
        "mode": "hailo" if infer_streams is not None else "mock"
    }

@app.get("/api/models/stdc1/classes")
async def get_classes():
    """返回所有类别名称和颜色"""
    return {
        "classes": [
            {"id": i, "name": name, "color": list(color)}
            for i, (name, color) in enumerate(zip(CITYSCAPES_CLASSES, CITYSCAPES_COLORS))
        ]
    }

@app.post("/api/models/stdc1/predict")
async def predict(file: UploadFile = File(...)):
    """
    对上传图像进行语义分割
    返回每个像素的类别标签
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    original_h, original_w = img.shape[:2]

    # 预处理
    input_tensor = preprocess(img)

    # 推理
    if infer_streams is not None:
        # Hailo-10H 真实推理
        input_data = {input_tensor_name: input_tensor}
        output = infer_streams.infer(input_data)
        # 输出是 (1, 19, 1024, 1920) 的 logits，取 argmax 得到类别
        logits = list(output.values())[0]
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    else:
        # Mock 模式：返回随机掩码（开发测试用）
        mask = np.random.randint(0, NUM_CLASSES, (1024, 1920), dtype=np.uint8)

    # 缩放回原始尺寸
    if (original_h, original_w) != (1024, 1920):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 统计各类别像素比例
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

@app.post("/api/models/stdc1/visualize")
async def visualize(file: UploadFile = File(...)):
    """
    返回可视化后的分割结果图像
    """
    contents = await file.read()
    img = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    original_h, original_w = img.shape[:2]

    # 预处理 + 推理
    input_tensor = preprocess(img)

    if infer_streams is not None:
        logits = list(infer_streams.infer({input_tensor_name: input_tensor}).values())[0]
        mask = np.argmax(logits[0], axis=0).astype(np.uint8)
    else:
        mask = np.random.randint(0, NUM_CLASSES, (1024, 1920), dtype=np.uint8)

    if (original_h, original_w) != (1024, 1920):
        mask = cv2.resize(mask, (original_w, original_h), interpolation=cv2.INTER_NEAREST)

    # 生成彩色掩码 + 与原图叠加
    color_mask = mask_to_color(mask)
    original = cv2.resize(img, (original_w, original_h))
    blended = cv2.addWeighted(original, 0.5, color_mask, 0.5, 0)

    # 编码为 JPEG 返回
    _, buf = cv2.imencode('.jpg', blended)
    return Response(content=buf.tobytes(), media_type="image/jpeg")

# ── 启动 ──────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)