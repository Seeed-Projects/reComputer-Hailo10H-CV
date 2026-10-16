# YOLOv10x on CM5 + Hailo-10H

YOLOv10x 在 **Hailo-10H** 上通过 HailoRT 执行 COCO 80 类目标检测。HEF 在片上
完成 NMS（HPP，`meta_arch=yolov8`，one2one 头）并只暴露一个输出 vstream；应用解析
HailoRT 5.1.1 返回的逐类紧凑缓冲区，同时保留其他 HailoRT NMS 布局的处理分支。
FastAPI 服务提供图像预测、视频与摄像头输入、MJPEG 预览以及离线视频分析。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| HailoRT 运行时 | 5.1.1 |
| Python | 3.13, aarch64 |
| 输入 | 640x640x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| 输出 | 片上 NMS，80x5x100（每类最多 100 个框） |
| 类别 | 80（COCO，0 基） |
| 参数量 | 31.72M |
| 运算量 | 160.56G |
| COCO mAP（全精度） | 53.7 |
| HEF | Hailo Model Zoo v5.4.0 (Hailo-10H) |
| 模型许可 | AGPL-3.0 (upstream: THU-MIG/yolov10) |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须使用同一 HailoRT 大版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/yolov10x.dockerfile \
  -t yolov10x:latest \
  src/hailo10h_yolov10x
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov10x \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov10x:latest \
  python web_detection.py --model_path model/yolov10x.hef --video_path video/test.mp4
```

浏览器打开 `http://<BOARD_IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov10x/predict" \
  -F "file=@bus.jpg" -F "conf=0.25"
```

| 接口 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 网页预览界面 |
| `/api/models/yolov10x/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |
| `/api/config` | GET / POST | 读取或修改运行时阈值 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注后的结果 |

## 实现说明

- HEF 在片上完成 NMS：parser 节点是 one2one 头（`one2one_cv2.*` 框、
  `one2one_cv3.*` 类别），主机不做解码也不做 NMS；`nms_thresh` 仅为接口兼容而
  保留，实际不生效。
- HailoRT 5.1.1 的 HPP 输出是扁平的逐类紧凑缓冲区（每类：检测数量 + 该数量条
  `[ymin, xmin, ymax, xmax, score]`，全部归一化到 [0, 1]）。按固定步长读取会让
  第一类之后的每一类错位；`_per_class_iterable` 同时保留 ragged NMS-by-score 与
  `(1,C,5,D)` / `(1,C,D,5)` 稠密布局的处理分支。
- 预处理把画面 letterbox 到 640x640x3，填充灰色（114），BGR 转 RGB 后送入原始
  uint8；`/255` 归一化已编译进 HEF。
- `cls_id`（0..79）直接索引标准 COCO 类别表。
- 首次推理的布局日志在生产中关闭（`_OUTPUT_LOGGED = True`）；验证新 HEF 时改为
  `False` 可打印原始输出的类型/形状/取值范围与逐类数据（SOP §10/§14）。

## 硬件验收清单

1. `hailortcli --version` 为 5.1.1 且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input size: ...`）。
3. 演示视频能显示带类别标签的 COCO 检测框。
4. `POST /api/models/yolov10x/predict` 返回含类别、置信度和框的结果。
5. USB 摄像头模式预览持续刷新，无卡死帧。
