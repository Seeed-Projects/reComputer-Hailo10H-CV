# CenterNet (resnet_v1_18) - COCO 目标检测

CenterNet（ResNet-18 骨干网络）在 Hailo-10H 上的 COCO 80 类目标检测。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | CenterNet + ResNet-18 |
| 输入 | 512×512×3 RGB |
| 输出 | 80 类检测框（3 个输出头：wh、reg、稀疏热力图） |
| 参数量 | 14.22M |
| 硬件 mAP | 25.0% |
| 格式 | HEF (Hailo-10H) |

## 快速开始

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
# 构建镜像
docker build -t centernet_resnet_v1_18_postprocess -f docker/hailo10h/centernet_resnet_v1_18_postprocess.dockerfile src/hailo10h_centernet_resnet_v1_18_postprocess

# 运行（需要 Hailo-10H 硬件）
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centernet_resnet_v1_18_postprocess
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/centernet_resnet_v1_18_postprocess/predict` | POST | 检测框（JSON） |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。
