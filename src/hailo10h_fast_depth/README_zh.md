# FastDepth：CM5 + Hailo-10H 单目深度估计

FastDepth 通过 HailoRT 在 Hailo-10H 上进行室内场景的实时单目深度估计。HEF 接收
224×224 RGB 图像并输出 224×224×1 深度图。Model Zoo 的
`fast_depth_postprocessing` 原样返回网络输出（FastDepth 直接预测 NYU-v2 尺度的
深度，不是 logits），应用按每帧第 2/98 百分位归一化后叠加伪彩色（TURBO）深度图。
FastAPI 服务支持图片、视频文件、USB 摄像头、MJPEG 预览、离线视频分析和 REST
预测接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| 容器 Python | 3.13，aarch64 |
| 输入 | 224x224x3 RGB（`normalize_in_net`，mean 0 / std 255） |
| 输出 | 深度图 224x224x1（NYU-v2 尺度） |
| 参数量 | 1.35M |
| 运算量 | 0.74G |
| RMSE（浮点 / 硬件） | 0.60 / 0.61 |
| HEF | Hailo Model Zoo v5.4.0，Hailo-10H（2,015,232 字节） |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须属于同一 HailoRT 版本组合。

## 构建

```bash
sudo docker build -f docker/hailo10h/fast_depth.dockerfile \
  -t fast_depth:latest \
  src/hailo10h_fast_depth
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-fast-depth \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/fast_depth:latest \
  python web_detection.py --model_path model/fast_depth.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000` 查看 Web 预览。USB 摄像头需挂载 `/dev/video0`，
并把 `--video_path ...` 改为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/fast_depth/predict" \
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

深度值使用模型的 NYU-v2 尺度（约等于米），不是经过标定的传感器测量值。

| 端点 | 方法 | 用途 |
|---|---|---|
| `/` | GET | Web 预览界面 |
| `/api/models/fast_depth/predict` | POST | 深度统计（JSON） |
| `/api/video_feed` | GET | 带伪彩色深度叠加的 MJPEG |
| `/api/config` | GET / POST | 读取或修改叠加透明度 |
| `/api/video/upload` | POST | 上传源视频 |
| `/api/video/analyze` | POST | 启动离线视频分析 |
| `/api/video/status` | GET | 读取分析进度 |
| `/api/video/download/{filename}` | GET | 下载标注结果 |

## 实现说明

- HEF 输出就是深度图：Model Zoo 的 `fast_depth_postprocessing` 不做 sigmoid 或
  逆深度变换。
- 应用按每帧第 2/98 百分位归一化后应用 TURBO 配色，再裁掉 letterbox 填充区并
  缩放到原始画幅，保证叠加与预览对齐。
- `normalize_in_net`（mean 0 / std 255，已折叠）编译进 HEF；letterbox 后直接喂
  原始 uint8 RGB。
- 无片上后处理；封装层请求 FLOAT32 vstream，主机拿到反量化后的值。
- 启动时从 HEF 读取实际输入尺寸（224x224）。
- 非方形或较小输出会按转置/缩放分支处理。

## 模型来源

[Hailo Model Zoo v5.4.0，Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_depth_estimation.rst)
（模型 `fast_depth`，来源：[dwofk/fast-depth](https://github.com/dwofk/fast-depth)）。