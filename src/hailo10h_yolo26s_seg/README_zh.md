# YOLO26s-seg - 实例分割

YOLO26s-seg（10.4M 参数），Hailo-10H 平台。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLO26s-seg |
| 输入 | 640×640×3 RGB |
| HEF 输出 | 10 个原始张量：每个 stride 的 4 通道框（l,t / r,b 距离）、80 通道类别 logits、32 通道掩码系数，另加 160x160x32 原型 |
| 参数量 | 10.4M |
| 格式 | HEF (Hailo-10H) |

## 快速开始

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
docker build -t yolo26s-seg -f docker/hailo10h/yolo26s_seg.dockerfile src/hailo10h_yolo26s_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  yolo26s-seg
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/yolo26s_seg/predict` | POST | 框级检测结果 (JSON) |

HEF 输出的是原始头（无片上 NMS）：宿主侧执行 sigmoid、one2one 头的两段式 top-k（post_nms_topk=100，不做 NMS）、regression_length=1 的框解码，以及掩码合成（系数 × 原型，再按框裁剪）。

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0（Hailo-10H）：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo26s_seg.hef
```

大小：18,345,984 字节 · SHA-256：`f9a36c8429a4ccd8663284ff56239e6bb1303f16a3f4fe287fed5b5694fc87b9`
