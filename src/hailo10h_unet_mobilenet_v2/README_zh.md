# U-Net MobileNet v2：CM5 + Hailo-10H 语义分割

U-Net MobileNet v2 通过 HailoRT 在 Hailo-10H 上进行语义分割。HEF 接收 256×256
RGB 图像并输出 256×256×3 的逐像素分数图；片上没有 argmax/softmax，应用在主机侧做
逐像素 argmax 得到类别掩码，去掉 letterbox 填充后按调色板半透明叠加到原图。
FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览、离线视频分析和 REST
预测接口。

模型是 TensorFlow 的 Oxford-IIIT Pet U-Net 教程网络；类别索引沿用教程的
`mask -= 1` 处理：0 = pet，1 = border，2 = background。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| 容器 Python | 3.13，aarch64 |
| 输入 | 256x256x3 RGB（`normalize_in_net`，mean 0 / std 255） |
| 输出 | 256x256x3 分数图 → 256x256 类别掩码（pet / border / background） |
| 参数量 | 10.08M |
| 运算量 | 28.88G |
| mIoU（浮点 / 硬件） | 77.32 / 77.1 |
| HEF | Hailo Model Zoo v5.4.0，Hailo-10H（8,630,272 字节） |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须属于同一 HailoRT 版本组合。

## 构建

```bash
sudo docker build -f docker/hailo10h/unet_mobilenet_v2.dockerfile \
  -t unet_mobilenet_v2:latest \
  src/hailo10h_unet_mobilenet_v2
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-unet-mobilenet-v2 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/unet_mobilenet_v2:latest \
  python web_detection.py --model_path model/unet_mobilenet_v2.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000` 查看分割叠加效果。USB 摄像头需挂载 `/dev/video0`，
并把 `--video_path ...` 改为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/unet/predict" \
  -F "file=@test.jpg"
```

响应保留 YOLO 风格的 `predictions` 列表；每个条目包含类别名、该类像素占掩码的
比例（`confidence`）和原始像素数：

```json
{
  "class": "pet",
  "confidence": 0.42,
  "pixels": 27525
}
```

背景像素不会出现在列表中。

| 端点 | 方法 | 用途 |
|---|---|---|
| `/` | GET | Web 预览界面 |
| `/api/models/unet/predict` | POST | 分割摘要（JSON） |
| `/api/video_feed` | GET | 带调色板掩码叠加的 MJPEG |
| `/api/config` | GET / POST | 读取或修改运行时阈值 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注结果 |

## 实现说明

- HEF 输出是 3 通道逐像素分数图；编译时 `argmax` 关闭
  （`device_pre_post_layers.argmax: false`），所以主机侧对类别轴做 `np.argmax`。
- 解码后的掩码裁掉 letterbox 填充区，并用最近邻缩放，保证类别索引保持整数。
- 调色板：pet（绿色）、border（青色）；background 不参与混合，保留原画面。
  叠加透明度 0.4。
- `normalize_in_net`（mean 0 / std 255）已编译进 HEF；letterbox（pad 0）后直接喂
  原始 uint8 RGB。
- 无片上后处理；封装层请求 FLOAT32 vstream，主机拿到反量化后的分数。启动时从
  HEF 读取实际输入尺寸（256x256）。

## 模型来源

[Hailo Model Zoo v5.4.0，Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_semantic_segmentation.rst)
（模型 `unet_mobilenet_v2`，来源
[TensorFlow 图像分割教程](https://www.tensorflow.org/tutorials/images/segmentation)，
在 Oxford-IIIT Pets 上训练）。