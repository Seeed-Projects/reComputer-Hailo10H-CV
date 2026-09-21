# YOLOv5m-seg - 实例分割

YOLOv5m-seg（anchor-based 实例分割，32.60M 参数），Hailo-10H 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLOv5m-seg（3 anchors，stride 8/16/32） |
| 输入 | 640×640×3 RGB |
| 输出 | 80 类检测框 + 实例掩码（proto 160x160x32） |
| 参数量 | 32.60M |
| 硬件 mAP | 36.7（COCO，Hailo Model Zoo 参考值） |
| 格式 | HEF (Hailo-10H) |

## 快速开始

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
docker build -t yolov5m_seg -f docker/hailo10h/yolov5m_seg.dockerfile src/hailo10h_yolov5m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov5m_seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/yolov5m_seg/predict` | POST | 检测框 + 实例掩码（JSON） |

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。
