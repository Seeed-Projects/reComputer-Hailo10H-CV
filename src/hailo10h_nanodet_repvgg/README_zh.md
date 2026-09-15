# NanoDet-RepVGG：reComputer CM5 + Hailo-10H 目标检测

本模块在 reComputer CM5 + Hailo-10H 上运行 NanoDet-RepVGG 目标检测（COCO 80 类），
NMS 在片上完成（Hailo HPP）。HEF 在设备上执行 NMS 并输出已解码的检测结果，
应用只解析 NMS 后的张量。FastAPI 服务支持图片、视频文件、USB 摄像头、
MJPEG 预览和 REST 预测接口。

从已验证的 Hailo-8 模块 `rpi5_hailo8_nanodet_repvgg` 移植；执行器替换为
HailoRT 5.1.1 的 `create_infer_model` 多输出封装（输出请求 FLOAT32）。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| Python | 3.13，aarch64 |
| 输入 | 320x320x3（尺寸在启动时从 HEF 读取） |
| 输出 | 片上 NMS 张量，后 NMS shape 80x5x100 |
| 类别 | 80（COCO，0 起索引，无 labels_offset） |
| HEF | Hailo Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 和 Python wheel 必须使用同一
HailoRT 主次版本。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/nanodet_repvgg.dockerfile \
  -t nanodet_repvgg:latest \
  src/hailo10h_nanodet_repvgg
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-nanodet-repvgg \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/nanodet_repvgg:latest \
  python web_detection.py --model_path model/nanodet_repvgg.hef --video_path video/test.mp4
```

浏览器打开 `http://<CM5_IP>:8000`。USB 摄像头模式挂载 `/dev/video0`，
将 `--video_path video/test.mp4` 换成 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<CM5_IP>:8000/api/models/nanodet_repvgg/predict" \
  -F "file=@test.jpg"
```

| 端点 | 方法 | 说明 |
|---|---|---|
| `/health` | GET | 健康检查 |
| `/api/models/nanodet_repvgg/predict` | POST | 检测结果（JSON） |
| `/api/video_feed` | GET | MJPEG 预览流 |

## 实现说明

- HEF 片上执行 NMS，应用只解析 NMS 后张量，`nms_thresh` 仅为接口兼容而保留。
- NMS 后的行格式为 `[ymin, xmin, ymax, xmax, score]`，相对 letterbox 输入归一化；
  应用换算为像素并还原 letterbox。
- 归一化/颜色转换编译进 HEF；应用 letterbox 后直接送原始 uint8 像素。
- HailoRT 可能以逐类别 ragged 列表（NMS-by-score）返回 NMS vstream，
  解析器同时兼容 ragged/object/dense 布局。首次推理会打印原始类型/形状用于核验。
- 执行器输出统一请求 `FormatType.FLOAT32`（由 SDK 去量化）；直接读量化原始值
  会得到万级数值且检测全空。
- 类别映射：`cls_id`（0..79）直接对应标准 COCO 80 类列表。

## 模型来源

`nanodet_repvgg.hef` 为 Hailo Model Zoo v5.4.0 的 Hailo-10H 构建——
URL、SHA256 和大小见 `model/README.md`。
