# YOLOv11n_obb on CM5 + Hailo-10H

YOLOv11n_obb 在 **Hailo-10H** 上通过 HailoRT 执行旋转框目标检测（DOTA v1
15 类）。编译出的 HEF **没有片上 NMS**：它暴露九个原始头（每个 stride 一个 64 通道
DFL 框张量、一个 15 通道类别张量、一个 1 通道角度张量），宿主负责解码并运行旋转框
概率 IoU NMS——与 Hailo Model Zoo 复现其官方 DOTA mAP 时使用的解码完全一致。

## 兼容性

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| HailoRT 运行时 | 5.1.1 |
| Python | 3.13, aarch64 |
| 输入 | 1024x1024x3 RGB (normalize_in_net mean 0 / std 255, padding 114) |
| 输出 | 旋转框 `[cx, cy, w, h, theta]` + 每框最佳类别，DOTA 15 类 |
| 参数量 | 3.80M |
| 运算量 | 17.25G |
| DOTA mAP（硬件） | 48.4（全精度 49.7） |
| Hailo-10H 帧率（batch 1） | 166 |
| HEF | Hailo Model Zoo v5.4.0 (Hailo-10H)，6,893,568 字节 |
| 模型许可 | AGPL-3.0 (upstream: ultralytics/ultralytics) |

HEF SHA256：`6188275241433279adc067c9a419c216aa1bc7212e0baf1297731015112f1008`

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须使用同一 HailoRT 大版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/yolov11n_obb.dockerfile \
  -t yolov11n_obb:latest \
  src/hailo10h_yolov11n_obb
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-yolov11n_obb \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolov11n_obb:latest \
  python web_detection.py --model_path model/yolov11n_obb.hef --video_path video/test.mp4
```

浏览器打开 `http://<BOARD_IP>:8000`。

仓库自带的演示视频是街景，而 DOTA 是航拍数据集，因此在演示视频上检测结果会很少；
请用一张航拍图通过下面的 REST API 核对旋转框与角度。使用 USB 摄像头时挂载
`/dev/video0`，并把 `--video_path ...` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<BOARD_IP>:8000/api/models/yolov11n_obb/predict" \
  -F "file=@aerial.jpg" -F "conf=0.25" -F "iou=0.7"
```

| 接口 | 方法 | 用途 |
|---|---|---|
| `/` | GET | 网页预览界面 |
| `/api/models/yolov11n_obb/predict` | POST | 旋转框检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |
| `/api/config` | GET / POST | 读取或修改运行时阈值 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注后的结果 |

每条结果包含 `class`、`confidence` 和
`obb: {x_center, y_center, width, height, angle}`，其中 `angle` 为**度**
（相对 +x 逆时针，范围 [-45, 135)）。

## 实现说明

- 头布局：九个原始张量，按通道数（64 / 15 / 1）识别并按 stride 配对
  （128x128、64x64、32x32）。没有片上 NMS、设备端 sigmoid 也关闭
  （`base/yolo_obb.yaml`），所以下面全部在宿主完成。
- 角度：`theta = (sigmoid(a) - 0.25) * pi`（[-pi/4, 3pi/4)），与 Model Zoo 的
  `YoloOBBPostProc` 及 ultralytics OBB 头一致。
- 框：对 16 个 DFL 箱做 softmax 期望得到 stride 单位的 l/t/r/b 距离；
  `w = l + r`、`h = t + b`，中心为锚点加上旋转后的半轴偏移，距离再乘 stride。
- 选择：每个锚点取最佳类别并按置信度滑块过滤，随后做带类别偏移的概率 IoU
  Fast-NMS（max_wh 7680，候选上限 30000，输出上限 300）——置信度与 IOU 滑块都生效。
- 预处理把画面 letterbox 到 1024x1024x3，填充灰色（114），BGR 转 RGB 后送入原始
  uint8；`/255` 归一化已编译进 HEF。
- 类别 ID 0..14 对应 DOTA v1 列表（plane、ship、storage-tank…），与 Model Zoo 的
  `dotav1_names.json` 一致。
- 首次推理诊断在生产中关闭（`_DET_OUTPUT_LOGGED = True`）；验证新 HEF 时改为
  `False` 可打印头布局、取值范围与解码后的角度范围（SOP §10/§14）。

## 硬件验收清单

1. `hailortcli --version` 为 5.1.1 且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input size: 1024x1024`）。
3. 用一张航拍图调用 `predict`，返回的旋转框角度与物体朝向一致（若整体差 90°
   则归一化 w/h）。
4. 演示视频预览持续刷新，无卡死帧。
5. USB 摄像头模式可用。
