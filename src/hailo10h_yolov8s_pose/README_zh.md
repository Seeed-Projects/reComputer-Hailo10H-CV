# CM5 + Hailo-10H 上的 YOLOv8s Pose

本模块在 Hailo-10H 加速器上运行单类别人体关键点估计。应用沿用本仓库 YOLO
系列的服务模板（FastAPI、MJPEG 预览、离线视频分析、USB 摄像头），并实现
YOLOv8 Pose 的后处理：片上 NMS 的每一行包含人体框、置信度和 17 个 COCO 关键点。

| 模型 | 路径 | 大小 | 说明 |
|---|---|---:|---|
| YOLOv8s Pose | `model/yolov8s_pose.hef` | 13,799,424 字节 | 默认，速度优先 |

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| HailoRT 宿主/运行时 | 5.1.1 |
| Python | 3.13, aarch64 |
| 输入 | 640x640x3 RGB（letterbox、灰色填充，归一化在 HEF 内） |
| 输出 | 人体框、置信度、17 个 COCO 关键点（片上 NMS） |
| 类别 | 1（`person`） |
| HEF | Hailo Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 与容器内 Python wheel 的 HailoRT 主次版本
必须一致。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/yolov8s_pose.dockerfile \
  -t yolov8s_pose:latest \
  src/hailo10h_yolov8s_pose
```

## 运行内置演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov8s-pose \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov8s_pose:latest \
  python web_detection.py \
    --model_path model/yolov8s_pose.hef \
    --video_path video/test.mp4
```

浏览器打开 `http://<开发板IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成
`--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/yolov8_pose/predict" \
  -F "file=@test.jpg"
```

```json
{
  "success": true,
  "source": "uploaded image",
  "predictions": [
    {
      "class": "person",
      "confidence": 0.91,
      "box": {"x1": 120, "y1": 80, "x2": 360, "y2": 520},
      "keypoints": [{"x": 180, "y": 120, "score": 0.87}]
    }
  ],
  "image": {"width": 1280, "height": 720}
}
```

低于 `KEYPOINT_SCORE_THRESH`（0.30）的关键点不会出现在响应里。同一个 Web 界面
还提供 MJPEG 预览、上传视频离线分析和摄像头模式。

## 后处理

- HEF 在片上完成 NMS，因此 `nms_thresh` 仅保留参数接口、不参与计算。
- 每个 NMS 行为 `[ymin, xmin, ymax, xmax, score, kpt0_y, kpt0_x, kpt0_s, ...]`，
  共 17 个关键点、56 个数值，均为相对网络输入的归一化值。人体框和关键点先换算
  到 640x640，再按 letterbox 参数还原到原图。
- 缓冲区按"每类一个计数 + 若干行"的紧凑格式解析，同时兼容
  `(1, C, N, 56)` / `(C, N, 56)` 稠密布局和 ragged 布局。若扁平缓冲区按 56 宽
  解析失败，会用仅检测的 5 宽再试一次，保证后处理版本差异时人体框仍能解出。
- 首次推理会打印输出类型、shape 和一行样例：
  `[YOLOv8 Pose] raw output type=..., shape=..., dtype=..., row_width=...`。
  可用它确认实机上的关键点顺序：解码按 `(y, x, score)` 读取，返回 `(x, y, score)`。

## 实机验收清单

1. `hailortcli --version` 为 5.1.1，且 `/dev/hailo0` 存在。
2. 首次推理日志中的输出名称、shape 和样例行符合预期（框与关键点都在 `[0, 1]`）。
3. 演示视频中人体框与 COCO 骨架贴合身体。
4. `POST /api/models/yolov8_pose/predict` 能返回人体框和 17 个关键点。
5. USB 摄像头模式下预览持续刷新、无残留帧。

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mock `hailo_platform`，验证 HailoRT 5.1.1
  封装（单套持久绑定、`run([bindings], timeout=10_000)`、FLOAT32 输出缓冲）。
- `tests/test_pose_postprocess.py` 验证 NMS 行解码、关键点顺序、分数过滤和
  letterbox 还原。

两个测试都不需要设备，但不能替代上面的实机验收。