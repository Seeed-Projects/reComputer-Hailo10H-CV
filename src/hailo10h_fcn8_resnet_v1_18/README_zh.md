# FCN8 ResNet-18 - 语义分割

FCN-8s + ResNet-18 骨干网络，在 Hailo-10H 上的实时语义分割。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | FCN-8s + ResNet-18 |
| 输入 | 1024×1920×3 RGB |
| 输出 | 19类分割掩码 (Cityscapes) |
| 参数量 | 11.20M |
| mIoU | 69.2% (硬件) |
| 格式 | HEF (Hailo-10H) |

## 快速开始

```bash
docker build -t fcn8 -f docker/hailo10h/fcn8_resnet_v1_18.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  fcn8
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models/fcn8_resnet/predict` | POST | 分割掩码 (JSON) |
| `/api/models/fcn8_resnet/visualize` | POST | 叠加可视化 (JPEG) |
| `/api/models/fcn8_resnet/classes` | GET | Cityscapes 类别列表 |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。