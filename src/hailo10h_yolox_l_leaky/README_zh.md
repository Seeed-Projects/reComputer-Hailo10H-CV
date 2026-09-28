# CM5 + Hailo-10H 上的 YOLOX-L-Leaky

YOLOX-L-Leaky 在 Hailo-10H 上执行 COCO 80 类目标检测，**NMS 在片上完成**（Hailo HPP，
`meta_arch=yolox`）：HEF 直接输出解码后的检测结果，应用只负责解析 post-NMS
张量。FastAPI 服务提供图片推理、视频与摄像头输入、MJPEG 预览和离线视频分析。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| Python | 3.13，aarch64 |
| 输入 | 640x640x3 RGB（normalize_in_net ImageNet RGB 均值/方差） |
| 输出 | 片上 NMS 张量，post-NMS 形状 80x5x100 |
| 类别 | 80（COCO，0 起始） |
| 参数量 | 54.16M |
| 运算量 | 155.48G |
| HEF | Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 与容器内 Python wheel 的 HailoRT 主次版本
必须一致。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/yolox_l_leaky.dockerfile \
  -t yolox_l_leaky:latest \
  src/hailo10h_yolox_l_leaky
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolox-l-leaky \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolox_l_leaky:latest \
  python web_detection.py --model_path model/yolox_l_leaky.hef --video_path video/test.mp4
```

浏览器打开 `http://<开发板IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成
`--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/yolox_l_leaky/predict" \
  -F "file=@bus.jpg" -F "conf=0.25" -F "iou=0.45"
```

## 实现说明

- NMS 在片上完成（HPP，`meta_arch=yolox`）：应用只解析 post-NMS 张量，因此
  `nms_thresh` 仅保留参数接口。
- 预处理 letterbox 到 640x640x3，使用 YOLOX 约定的灰色填充（114），BGR 转
  RGB 后输入原始 uint8；ImageNet 归一化已编译进 HEF。
- 解析器兼容 HailoRT 的几种 NMS 布局：每类"计数 + 5 值行"的紧凑缓冲区、
  `80x5x100` / `80x100x5` 稠密布局，以及 ragged（NMS-by-score）列表。
- `cls_id`（0..79）直接索引标准 COCO 类别表。
- 首次推理会打印一次原始输出类型与 shape（`[YOLOX] raw output type=..., shape=...`），
  便于实机核对布局。

## 实机验收清单

1. `hailortcli --version` 为 5.1.1，且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input size: ...`）。
3. 演示视频中出现带类别标签的 COCO 检测框。
4. `POST /api/models/yolox_l_leaky/predict` 返回类别、置信度和检测框。
5. USB 摄像头模式下预览持续刷新、无残留帧。

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mock `hailo_platform`，验证 HailoRT 5.1.1
  封装（持久绑定、`run([bindings], timeout=10_000)`、FLOAT32 输出缓冲）。
- `tests/test_nms_parse.py` 验证 post-NMS 解析（紧凑缓冲区、稠密布局、ragged
  列表、分数阈值、letterbox 还原）。
