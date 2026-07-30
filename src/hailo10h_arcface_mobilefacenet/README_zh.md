# ArcFace MobileFaceNet - 人脸识别

ArcFace + MobileFaceNet（2.04M 参数），Hailo-10H 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | ArcFace + MobileFaceNet |
| 输入 | 112×112×3 RGB |
| 输出 | 512 维人脸嵌入向量 |
| 参数量 | 2.04M |
| 准确率 | 99.4% (LFW) |
| 格式 | HEF (Hailo-10H) |

## 快速开始

```bash
docker build -t arcface -f docker/hailo10h/arcface_mobilefacenet.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  arcface
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models/arcface/predict` | POST | 人脸嵌入向量 (JSON) |
| `/api/models/arcface/compare` | POST | 人脸比对 (JSON) |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。