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

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
docker build -t fcn8 -f docker/hailo10h/fcn8_resnet_v1_18.dockerfile src/hailo10h_fcn8_resnet_v1_18

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  fcn8
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/fcn8_resnet/predict` | POST | 分割掩码 (JSON) |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。
