# SCDepthV3：CM5 + Hailo-10H 单目深度估计

SCDepthV3 通过 HailoRT 在 Hailo-10H 上进行室内场景的单目相对深度估计。HEF 接收
256×320 RGB 图像并输出 256×320×1 原始张量；应用使用
`depth = 1 / (sigmoid(raw) * 10 + 0.009)` 解码为相对深度，恢复到原始画幅坐标，
再叠加伪彩色（TURBO）深度图。FastAPI 服务支持图片、视频文件、USB 摄像头、
MJPEG 预览、离线视频分析和 REST 预测接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| 容器 Python | 3.13，aarch64 |
| 输入 | 256x320x3 RGB（`normalize_in_net`，mean 114.75 / std 57.375） |
| 输出 | 相对深度图 256x320x1（由原始 logits 解码） |
| 参数量 | 14.8M |
| 运算量 | 10.7G |
| RMSE（全精度） | 0.481 |
| HEF | Hailo Model Zoo v5.4.0，Hailo-10H（11,579,392 字节） |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须属于同一 HailoRT 版本组合。

## 构建

```bash
sudo docker build -f docker/hailo10h/scdepthv3.dockerfile \
  -t scdepthv3:latest \
  src/hailo10h_scdepthv3
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-scdepthv3 \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/scdepthv3:latest \
  python web_detection.py --model_path model/scdepthv3.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000` 查看 Web 预览。USB 摄像头需挂载 `/dev/video0`，
并把 `--video_path ...` 改为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/scdepthv3/predict" \
  -F "file=@test.jpg"
```

响应保留 YOLO 风格的 `predictions` 列表，并附带 `depth` 统计对象：

```json
{
  "class": "depth",
  "confidence": 1.0,
  "min_depth": 0.12,
  "max_depth": 8.4,
  "mean_depth": 1.9,
  "median_depth": 1.5,
  "valid_pixels": 921600
}
```

深度值是**相对深度**，不是公制米。

| 端点 | 方法 | 用途 |
|---|---|---|
| `/` | GET | Web 预览界面 |
| `/api/models/scdepthv3/predict` | POST | 深度统计（JSON） |
| `/api/video_feed` | GET | 带伪彩色深度叠加的 MJPEG |
| `/api/config` | GET / POST | 读取或修改叠加透明度 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注结果 |

## 实现说明

- **相对深度**：原始输出是 logits。应用按
  `depth = 1 / (sigmoid(raw) * 10 + 0.009)` 解码，并用每帧第 2/98 百分位归一化后
  应用 TURBO 配色。
- 解码后的深度图会裁掉 letterbox 填充区并缩放到原始画幅，保证叠加与预览对齐。
- `normalize_in_net` 的均值/标准差已编译进 HEF；letterbox 后直接喂原始 uint8 RGB。
- 无片上后处理；封装层请求 FLOAT32 vstream，主机拿到反量化后的 logits。
- 启动时从 HEF 读取实际输入尺寸（256x320）。
- 非方形模型的输出可能以 (320, 256) 转置形式返回，解码器会先转置再做 sigmoid 映射。

## 模型来源

[Hailo Model Zoo v5.4.0，Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_depth_estimation.rst)
（模型 `scdepthv3`，来源：[JiawangBian/sc_depth_pl](https://github.com/JiawangBian/sc_depth_pl/)）。