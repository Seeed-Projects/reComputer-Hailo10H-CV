# YOLOv8n-seg - 实例分割

YOLOv8n-seg（3.4M 参数），Hailo-10H 平台。

## 模型

| 属性 | 值 |
|------|-----|
| 架构 | YOLOv8n-seg（anchor-free，one2many head，stride 8/16/32） |
| 输入 | 640x640x3 RGB（letterbox 填充 114） |
| HEF 输出 | 10 个原始张量：每个 stride 的框距离、80 通道类别分数、32 通道掩码系数，另有 160x160x32 掩码原型。框分支可能是 4 通道折叠布局（l、t / r、b 距离）或 64 通道 DFL 4x16 布局，两者都能解码。 |
| 参数量 | 3.4M |
| 运算量 | 12.04G |
| 参考 mAP | 30.32（COCO 实例分割，全精度，Hailo Model Zoo；非 Hailo-10H 实测值） |
| 许可 | AGPL-3.0（Ultralytics） |
| 格式 | HEF（Hailo-10H） |

## 快速开始

运行基线：Python 3.13 与 HailoRT 5.1.1。构建命令在仓库根目录执行。

```bash
docker build -t yolov8n_seg -f docker/hailo10h/yolov8n_seg.dockerfile src/hailo10h_yolov8n_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolov8n_seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/yolov8n_seg/predict` | POST | 框级检测结果（JSON） |

HEF 输出原始分支头、无片上 NMS（yolov8 检测编译使用 nms_postprocess）：主机侧完成
DFL/折叠框解码、sigmoid 与逐类 NMS、掩码合成（系数 x 原型，按框裁剪）。实例掩码在
MJPEG 预览中绘制；REST 返回框、类别与置信度。首次推理会打印全部张量的名称、形状与
数值范围。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0（Hailo-10H）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolov8n_seg.hef
```
