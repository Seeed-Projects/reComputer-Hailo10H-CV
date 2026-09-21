# CenterPose RegNetX-800MF - 多人姿态估计

CenterPose（RegNetX-800MF 骨干网络）在 Hailo-10H 上的多人姿态估计。

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | CenterPose + RegNetX-800MF |
| 输入 | 512×512×3 BGR |
| 输出 | 人体框 + 17 个 COCO 关键点（6 个输出头：hm、wh、hps、reg、hm_hp、hp_offset） |
| 参数量 | 12.31M |
| 硬件 AP | 43.1% |
| 格式 | HEF (Hailo-10H) |

## 快速开始

运行时基线：Python 3.13、HailoRT 5.1.1。请在仓库根目录执行构建命令。

```bash
# 构建镜像
docker build -t centerpose_regnetx_800mf -f docker/hailo10h/centerpose_regnetx_800mf.dockerfile src/hailo10h_centerpose_regnetx_800mf

# 运行（需要 Hailo-10H 硬件）
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  centerpose_regnetx_800mf
```

## API

| 接口 | 方法 | 说明 |
|------|------|------|
| `/` | GET | Web 预览 |
| `/api/video_feed` | GET | MJPEG 流 |
| `/api/models/centerpose_regnetx_800mf/predict` | POST | 人体框 + 17 个关键点（JSON） |

## 来源

HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)。
