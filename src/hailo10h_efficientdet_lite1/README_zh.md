# CM5 + Hailo-10H 上的 EfficientDet-Lite1

EfficientDet-Lite1 在 Hailo-10H 上执行 COCO 目标检测，**NMS 与 sigmoid 都在片上完成**
（Hailo HPP）：HEF 直接输出解码后的检测结果，应用只负责解析 post-NMS 张量。
FastAPI 服务提供图片推理、视频与摄像头输入、MJPEG 预览和离线视频分析。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| Python | 3.13，aarch64 |
| 输入 | 384x384x3 RGB（归一化已编译进 HEF：mean=127、std=128） |
| 输出 | 片上 NMS 张量，post-NMS 形状 89x5x100 |
| 类别 | 89 个槽位（`labels_offset=1`，对应 COCO 类别 ID 1..89，含 10 个未使用 ID） |
| 参数量 | 4.73M |
| 运算量 | 4G |
| HEF | Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 与容器内 Python wheel 的 HailoRT 主次版本
必须一致。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/efficientdet_lite1.dockerfile \
  -t efficientdet_lite1:latest \
  src/hailo10h_efficientdet_lite1
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-effdet-lite1 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/efficientdet_lite1:latest \
  python web_detection.py --model_path model/efficientdet_lite1.hef --video_path video/test.mp4
```

浏览器打开 `http://<开发板IP>:8000`。使用 USB 摄像头时挂载 `/dev/video0`，并把
`--video_path video/test.mp4` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/efficientdet_lite1/predict" \
  -F "file=@test.jpg"
```

## 实现说明

- HEF 在片上完成 NMS 与 sigmoid（`device_pre_post_layers: nms=true,
  sigmoid=true`、`hpp=true`）；应用只解析 post-NMS 张量，`nms_thresh` 仅保留
  参数接口。
- post-NMS 每行为 `[ymin, xmin, ymax, xmax, score]`，取值范围 [0,1]（相对
  letterbox 后的输入）；应用缩放到像素并还原到原图。
- HEF 内含归一化（mean=127、std=128）且 `padding_color=127`：应用用灰色（127）
  填充做 letterbox，输入原始 uint8 RGB，宿主侧不做归一化。
- 解析器兼容 HailoRT 的几种 NMS 布局：每类"计数 + 5 值行"的紧凑缓冲区、
  `89x5x100` / `89x100x5` 稠密布局，以及 ragged（NMS-by-score）列表。
- 类别映射：`cls_id`（0..88）对应 COCO 类别 ID `cls_id + 1`（`labels_offset=1`），
  10 个未使用的 ID 记为 "N/A"，不参与绘制。
- 首次推理会打印一次原始输出类型与 shape（`[EfficientDet] raw output type=..., shape=...`），
  便于实机核对布局。

## 实机验收清单

1. `hailortcli --version` 为 5.1.1，且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input size: 38484`）。
3. 演示视频中出现带类别标签的 COCO 检测框。
4. `POST /api/models/efficientdet_lite1/predict` 返回类别、置信度和检测框。
5. USB 摄像头模式下预览持续刷新、无残留帧。

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mock `hailo_platform`，验证 HailoRT 5.1.1
  封装（持久绑定、`run([bindings], timeout=10_000)`、FLOAT32 输出缓冲）。
- `tests/test_nms_parse.py` 验证 post-NMS 解析（紧凑缓冲区、稠密布局、ragged
  列表、分数阈值、letterbox 还原、类别映射）。
