# YOLOv5m on CM5 + Hailo-10H

YOLOv5m 在 CM5 + Hailo-10H 上执行 COCO 80 类目标检测，采用**片上 Hailo HPP
NMS**（zoo `base/yolo.yaml`，`meta_arch=yolo_v5`）：HEF 直接输出解码后的检测结果，
应用只解析 post-NMS 张量。FastAPI 服务提供图像预测、视频与摄像头输入、MJPEG
预览以及离线视频分析。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| HailoRT 运行时 | 5.1.1 |
| Python | 3.13, aarch64 |
| 输入 | 640x640x3 RGB（normalize_in_net mean=0/std=255） |
| 输出 | 片上 NMS 张量，zoo 标注 post-NMS 形状 80x5x80 |
| 类别 | 80（COCO，0 基） |
| 参数量 | 21.78M |
| 运算量 | 52.17G |
| HEF | Hailo Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须使用同一 HailoRT 大版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/yolov5m.dockerfile \
  -t yolov5m:latest \
  src/hailo10h_yolov5m
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov5-m \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov5m:latest \
  python web_detection.py --model_path model/yolov5m.hef --video_path video/test.mp4
```

浏览器打开 `http://<BOARD_IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov5m/predict" \
  -F "file=@bus.jpg" -F "conf=0.25" -F "iou=0.45"
```

| 接口 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 网页预览界面 |
| `/api/models/yolov5m/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |
| `/api/config` | GET / POST | 读取或修改运行时阈值 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注后的结果 |

## 实现说明

- 片上 NMS（HPP，zoo `base/yolo.yaml`）：应用只解析 post-NMS 张量，
  `nms_thresh` 仅保留参数接口。
- 预处理把画面 letterbox 到 640x640x3，填充色为 YOLOv5 的灰色（114），
  BGR 转 RGB 后送入原始 uint8；`/255` 归一化（`normalize_in_net`，
  mean 0 / std 255）已编译进 HEF。
- 解析器兼容 HailoRT 的多种 NMS 布局：紧凑的逐类缓冲区（计数 + 5 值行）、
  `80x5x80` / `80x80x5` 稠密布局，以及 ragged（NMS-by-score）列表。
- `cls_id`（0..79）直接索引标准 COCO 类别表。
- 首次推理会打印一次原始输出类型与形状
  （`[YOLOv5] raw output type=..., shape=...`），便于实机核对。

## 硬件验收清单

1. `hailortcli --version` 为 5.1.1 且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input size: ...`）。
3. 演示视频能显示带类别标签的 COCO 检测框。
4. `POST /api/models/yolov5m/predict` 返回含类别、置信度和框的结果。
5. USB 摄像头模式预览持续刷新，无卡死帧。

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` 用 mock 替代 `hailo_platform`，校验 HailoRT
  5.1.1 封装（持久 binding、`run([bindings], timeout=10_000)`、FLOAT32 输出缓冲）。
- `tests/test_nms_parse.py` 校验 post-NMS 解析（紧凑缓冲区、稠密布局、ragged
  列表、置信度阈值、逆 letterbox）。

以上测试不能替代硬件验收。
