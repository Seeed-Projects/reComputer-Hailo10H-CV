# STDC1 - 实时语义分割

STDC1（Short-Term Dense Concatenate）在 Hailo-10H 上的实时语义分割。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | STDC1 |
| 输入 | 1024×1920×3 RGB |
| 输出 | 19类分割掩码 (Cityscapes) |
| 参数量 | 8.27M |
| mIoU | 73.7% |
| 格式 | HEF (Hailo-10H) |

## 快速开始

```bash
# 构建镜像
docker build -t stdc1 -f docker/hailo10h/stdc1.dockerfile .

# 运行（需要 Hailo-10H 硬件）
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1
```

## API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/models/stdc1/predict` | POST | 分割掩码 (JSON) |
| `/api/models/stdc1/visualize` | POST | 叠加可视化 (JPEG) |
| `/api/models/stdc1/classes` | GET | Cityscapes 类别列表 |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。