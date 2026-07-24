# reComputer-Hailo10H-CV

[English](./README.md) | [中文]

本项目为 reComputer R Series（树莓派 CM5）搭载 **Hailo-10H** 加速器提供工业级计算机视觉方案。模型来自 Hailo Model Zoo，通过 Docker 一键部署。

## 项目架构

每个模型独立管理，包含推理代码、模型文件和 Docker 配置：

```text
reComputer-Hailo10H-CV/
├── .github/workflows/       # CI/CD 自动构建镜像
├── tools/                   # 脚手架脚本
├── docs/                    # 文档
├── docker/hailo10h/         # 各模型 Dockerfile
├── src/
│   └── hailo10h_<model>/
│       ├── web_detection.py # FastAPI 推理服务
│       ├── py_utils/        # HailoInfer + 工具函数
│       ├── hailot-packages/ # HailoRT Python wheel
│       ├── model/           # HEF 模型文件
│       ├── video/           # 演示视频
│       └── requirements.txt
```

## 支持平台

| 平台 | 芯片 | 加速器 | 算力 |
| :--- | :--- | :--- | :--- |
| **reComputer R Series** | 树莓派 CM5 | Hailo-10H | 10 TOPS |

## 已收录模型

| 模型 | 任务 | 参数量 | 输入 | 指标 | HEF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [**STDC1**](src/hailo10h_stdc1/) | 语义分割 | 8.27M | 1024×1920 | 73.7% mIoU | [下载](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/stdc1.hef) |
| [**FCN8 ResNet-18**](src/hailo10h_fcn8_resnet_v1_18/) | 语义分割 | 11.20M | 1024×1920 | 69.2% mIoU | [下载](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/fcn8_resnet_v1_18.hef) |
| [**MSPN RegNetX-800MF**](src/hailo10h_mspn_regnetx_800mf/) | 姿态估计 | 7.17M | 256×192 | 69.8% AP | [下载](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/mspn_regnetx_800mf.hef) |
| [**LightFace Slim**](src/hailo10h_lightface_slim/) | 人脸检测 | 0.26M | 240×320 | 39.3% mAP | [下载](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/lightface_slim.hef) |

## 快速开始

### 1. 安装 Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker && sudo systemctl start docker
```

### 2. 安装 HailoRT (Hailo-10H)

```bash
sudo apt update
sudo apt install hailo-h10-all -y
sudo reboot

# 验证
hailortcli fw-control identify
ls /dev/hailo0
```

### 3. 拉取并运行

```bash
# 从 GHCR 拉取
docker pull ghcr.io/seeed-projects/recomputer-hailo10h-cv/stdc1:latest

# 运行
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/stdc1:latest
```

打开 `http://<设备IP>:8000` 查看网页预览。

### 或本地构建

```bash
cd src/hailo10h_stdc1
docker build -f ../../docker/hailo10h/stdc1.dockerfile -t stdc1 .
```

## REST API

### 预测

```bash
curl -X POST http://<设备IP>:8000/api/models/stdc1/predict \
  -F "file=@test.jpg"
```

### 可视化

```bash
curl -X POST http://<设备IP>:8000/api/models/stdc1/visualize \
  -F "file=@test.jpg" -o result.jpg
```

## 模型来源

所有 HEF 模型来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)，已针对 Hailo-10H 预编译。

## 贡献指南

1. 使用脚手架：`python tools/scaffold_hailo_model.py <模型名> --task <任务类型>`
2. 将 HEF 放入 `model/`
3. 更新 `web_detection.py` 后处理逻辑
4. 将 HailoRT wheel 放入 `hailort-packages/`
5. 在 `recomputer-ai-lab` 中添加模型
6. 详见 [docs/HAILO_MODEL_PORTING_zh.md](docs/HAILO_MODEL_PORTING_zh.md)

## 许可证

开源。各模型许可证见对应目录。