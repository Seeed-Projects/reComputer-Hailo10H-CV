# Depth-Anything-V2-ViTS：CM5 + Hailo-10H 深度估计

本模块运行 Depth-Anything-V2-Small（ViT-S 主干），通过 HailoRT 在 Hailo-10H 上
进行零样本单目深度估计。HEF 输出 224×224×1 相对深度图，应用进行逐帧归一化、
INFERNO 着色并半透明叠加到原图。FastAPI 服务支持图片、视频文件、USB 摄像头、
MJPEG 预览和 REST 推理接口。

## 兼容环境

| 组件 | 版本 |
|---|---|
| 加速器 | Hailo-10H PCIe（`/dev/hailo0`） |
| 宿主机/运行时 | HailoRT 5.1.1 |
| 容器 Python | 3.13，aarch64 |
| 输入 | 224x224x3 RGB（normalize_in_net ImageNet RGB） |
| 输出 | 深度图 224x224x1（相对深度，零样本），FLOAT32 |
| 参数量 | 24.2M |
| 运算量 | 16.7G |
| AbsRel | 0.147 |
| HEF | Model Zoo v5.4.0，Hailo-10H |

宿主机驱动、固件、`libhailort.so` 与 Python wheel 必须属于同一 HailoRT 版本组合。

## 构建

```bash
sudo docker build -f docker/hailo10h/depth_anything_v2_vits.dockerfile \
  -t depth_anything_v2_vits:latest \
  src/hailo10h_depth_anything_v2_vits
```

## 运行演示视频

```bash
sudo docker run --rm \
  --name cm5-hailo10h-depth-anything-v2-vits \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/depth_anything_v2_vits:latest \
  python web_detection.py --model_path model/depth_anything_v2_vits.hef --video_path video/test.mp4
```

打开 `http://<开发板IP>:8000` 查看 Web 预览。USB 摄像头需挂载 `/dev/video0`，
并把 `--video_path ...` 改为 `--camera_id 0`。

## REST API

```bash
curl -X POST "http://<开发板IP>:8000/api/models/depth_anything_v2_vits/predict" \
  -F "file=@test.jpg"
```

响应返回深度统计值（min、max、mean、std）。

## 实现说明

- **零样本相对深度**：输出是相对深度（非公制），值越大越远。应用按帧
  min-max 归一化用于可视化。
- **可视化**：INFERNO 着色（暗=近，亮=远），alpha 混合到原图。滑块控制混合比例。
- `normalize_in_net`（ImageNet RGB 均值/方差），无 input_conversion → 喂原始 uint8 RGB。
- 无片上后处理，深度图就是 HEF 输出；封装层请求 FLOAT32 vstream，主机拿到反量化后的值。
- 启动时从 HEF 读取实际输入尺寸（224x224）。

## 模型来源

[Hailo Model Zoo v5.4.0，Hailo-10H](https://github.com/hailo-ai/hailo_model_zoo/blob/master/docs/public_models/HAILO10H/HAILO10H_zero_shot_depth_estimation.rst)
（模型 `depth_anything_v2_vits`，源：[Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)）。