# YOLO26m-seg - 实例分割

YOLO26m-seg（23.6M 参数），Hailo-10H 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLO26m-seg |
| 输入 | 640×640×3 RGB |
| HEF 输出 | 边界框 + 实例掩码张量 (COCO 80类) |
| 参数量 | 23.6M |
| 格式 | HEF (Hailo-10H) |

## 快速开始

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
docker build -t yolo26m-seg -f docker/hailo10h/yolo26m_seg.dockerfile src/hailo10h_yolo26m_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26m-seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/yolo26m_seg/predict` | POST | 框级检测结果 (JSON) |

当前 Web 后处理仅输出框级检测结果；实例掩码解码仍需在目标硬件上验证。仓库内的 YOLO26m HEF 相对所述模型规模异常偏小，发布前需确认工件。

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。
