# YOLO26l-seg - 实例分割

YOLO26l-seg（28M 参数），Hailo-10H 平台。模块会解码原始分割头、组装实例掩码
（掩码系数 × prototype，按框裁剪）并绘制到带标注的预览里。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YOLO26l-seg |
| 输入 | 640x640x3 RGB |
| HEF 输出 | 10 个原始张量：每个 stride 的 4 通道框（l,t / r,b 距离）、80 通道类别 logits、32 通道掩码系数，外加 160x160x32 prototype |
| 参数量 | 28M |
| 运算量 | 150.2G |
| 格式 | HEF (Hailo-10H) |
| HEF 大小 | 40,140,800 字节 |
| HEF SHA256 | `8b4c51b588f5213ff5437100d4da9f6ea0969c723ece562dbd445aed837ec637` |

## 快速开始

运行基线：Python 3.13 与 HailoRT 5.1.1。在仓库根目录执行构建命令。

```bash
docker build -t yolo26l-seg -f docker/hailo10h/yolo26l_seg.dockerfile src/hailo10h_yolo26l_seg

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/yolo26l_seg:latest
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 视频流 |
| `/api/models/yolo26l_seg/predict` | POST | 检测结果（JSON：类别、置信度、框） |

HEF 暴露原始头（没有片上 NMS）：宿主做 sigmoid、one2one 头的两段式 top-k
（post_nms_topk=100，无 NMS）、regression_length=1 的框解码以及掩码组装。
JSON 只携带框；实例掩码会合成到带标注的预览和离线视频输出中。

## 来源

HEF 来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) v5.4.0 (Hailo-10H)：

```text
https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo26l_seg.hef
```

Hailo-10H 的 zoo 编译了 yolo26n_seg / yolo26s_seg / yolo26m_seg / yolo26l_seg（不存在 x_seg 构建）。
