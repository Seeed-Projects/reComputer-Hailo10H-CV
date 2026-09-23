# CM5 + Hailo-10H 上的 ViTPose-Small

ViTPose-Small 用于单人 2D 姿态估计，输出 17 个 COCO 关键点。HEF 输出单张热图
（64x48x17），应用对每个通道取 argmax 得到关键点坐标，缩放到 256x192 的
网络输入，再按 letterbox 参数还原到原图并绘制 COCO 骨架。FastAPI 服务支持
图片、视频文件、USB 摄像头、MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| Python | 3.13，aarch64 |
| 输入 | 256x192x3 RGB（normalize_in_net ImageNet RGB 均值/方差） |
| 输出 | 热图 64x48x17（17 个 COCO 关键点，argmax 解码，无 NMS） |
| 参数量 | 24.29M |
| 运算量 | 17.17G |
| HEF | Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 与容器内 Python wheel 的 HailoRT 主次版本
必须一致。

## 构建

在仓库根目录执行：

```bash
sudo docker build -f docker/hailo10h/vit_pose_small.dockerfile \
  -t vit_pose_small:latest \
  src/hailo10h_vit_pose_small
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-vit-pose-small \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/vit_pose_small:latest \
  python web_detection.py --model_path model/vit_pose_small.hef --video_path video/test.mp4
```

浏览器打开 `http://<开发板IP>:8000`。

使用 USB 摄像头时挂载 `/dev/video0`，并把 `--video_path ...` 换成
`--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/vit_pose_small/predict" \
  -F "file=@test.jpg"
```

响应包含画面中单人的 17 个关键点（`x`、`y`、`score`）。置信度低于阈值（0.30）
的关键点不会出现在骨架叠加里。

## 实现说明

- **单人模型**：网络假设人物位于画面中心；多人场景需要先用检测器裁剪每个人。
- **预处理**：letterbox 到 256x192，黑色填充，BGR 转 RGB，输入原始 uint8。
  归一化已编译进 HEF（RGB 均值 `[123.675, 116.28, 103.53]`，方差
  `[58.395, 57.12, 57.375]`），宿主侧不再归一化。
- **后处理**：对每个热图通道取 argmax 得到 64x48 上的 `(x, y)`，缩放到
  256x192 输入后按 letterbox 参数还原。没有片上 NMS，也没有 DARK 亚像素细化
  （`nms_thresh` 仅为接口兼容保留）。
- 首次推理会打印一次原始输出 shape（`[ViTPose] raw output shape=...`），便于
  实机核对布局。

## 实机验收清单

1. `hailortcli --version` 为 5.1.1，且 `/dev/hailo0` 存在。
2. 启动日志打印 HEF 输入尺寸（`Model input: 192x256`）。
3. 演示视频中的 COCO 骨架与人物贴合。
4. `POST /api/models/vit_pose_small/predict` 返回 17 个带分数的关键点。
5. USB 摄像头模式下预览持续刷新、无残留帧。

## 测试

```bash
python -m unittest discover -s tests -v
```

- `tests/test_hailo_executor.py` mock `hailo_platform`，验证 HailoRT 5.1.1
  封装（持久绑定、`run([bindings], timeout=10_000)`、FLOAT32 输出缓冲）。
- `tests/test_pose_heatmap.py` 验证热图解码（argmax、布局归一化、坐标缩放与
  letterbox 还原）。

两个测试都不能替代上面的实机验收。
