# MSPN RegNetX-800MF - 姿态估计

MSPN（多阶段姿态网络）+ RegNetX-800MF 骨干，在 Hailo-10H 上的单人姿态估计。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | MSPN + RegNetX-800MF |
| 输入 | 256×192×3 RGB |
| 输出 | 17 个关键点 (COCO) |
| 参数量 | 7.17M |
| 格式 | HEF (Hailo-10H) |

## 快速开始

```bash
docker build -t mspn -f docker/hailo10h/mspn_regnetx_800mf.dockerfile .

sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  mspn
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models/mspn_pose/predict` | POST | 关键点 (JSON) |
| `/api/models/mspn_pose/visualize` | POST | 姿态叠加 (JPEG) |
| `/api/models/mspn_pose/keypoints` | GET | 关键点定义 |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。