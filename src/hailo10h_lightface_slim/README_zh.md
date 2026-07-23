# LightFace Slim - 人脸检测

超轻量人脸检测（0.26M 参数），Hailo-10H 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | Ultra-Light-Fast-Generic-Face-Detector-1MB |
| 输入 | 240×320×3 RGB |
| 输出 | 人脸边界框 |
| 参数量 | 0.26M |
| FPS | 817 |
| 格式 | HEF (Hailo-10H) |

## 快速开始

```bash
docker build -t lightface -f docker/hailo10h/lightface_slim.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  lightface
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models/lightface/predict` | POST | 人脸框 (JSON) |
| `/api/models/lightface/visualize` | POST | 框叠加 (JPEG) |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。