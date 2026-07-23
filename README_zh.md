# reComputer-Hailo10H-CV

[English](./README.md) | [中文]

本项目旨在为 reComputer R Series（树莓派 CM5）搭载 **Hailo-10H** 加速器提供工业级、高性能的计算机视觉（CV）应用方案。目前已支持语义分割，更多模型持续添加中。

## 项目架构

本项目采用多模型独立适配架构，每个模型的代码和环境配置独立管理：

```text
reComputer-Hailo10H-CV/
├── docker/                 # Docker 镜像配置文件
│   └── hailo10h/           # Hailo-10H 专用 Dockerfile
├── src/                    # 源代码目录
│   └── hailo10h_<model>/   # 每个模型的源代码和依赖
│       ├── web_service.py  # FastAPI 推理服务
│       ├── requirements.txt
│       ├── README.md
│       └── model/          # HEF 模型文件
└── .github/workflows/      # GitHub Actions 自动化构建脚本
```

## 支持平台

| 平台 | 芯片 | 加速器 | 算力 |
| :--- | :--- | :--- | :--- |
| **reComputer R Series** | 树莓派 CM5 | Hailo-10H | 10 TOPS |

## 已收录模型

| 模型 | 任务 | 参数量 | 输入尺寸 | mIoU | HEF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [**STDC1**](src/hailo10h_stdc1/) | 语义分割 | 8.27M | 1024×1920 | 73.7% | [下载](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/stdc1.hef) |

## 快速开始

### 1. 安装 Docker

在 reComputer R Series 上执行以下命令安装 Docker：

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. 安装 HailoRT

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

# 验证安装
hailortcli fw-control identify
ls /dev/hailo0
```

### 3. 运行 STDC1

```bash
# 构建镜像
docker build -t stdc1 -f docker/hailo10h/stdc1.dockerfile .

# 运行容器
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1
```

打开 `http://<设备IP>:8000/health` 确认服务运行正常。

## REST API

### 语义分割预测

```bash
curl -X POST http://<设备IP>:8000/api/models/stdc1/predict \
  -F "file=@test.jpg"
```

### 可视化输出

```bash
curl -X POST http://<设备IP>:8000/api/models/stdc1/visualize \
  -F "file=@test.jpg" \
  -o result.jpg
```

### 类别列表

```bash
curl http://<设备IP>:8000/api/models/stdc1/classes
```

## 模型来源

所有 HEF 模型均来自 [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo)，已针对 Hailo-10H 预编译优化。

## 贡献指南

添加新模型的步骤：

1. 新建目录：`src/hailo10h_<模型名>/`
2. 将 HEF 模型文件放入 `model/`
3. 编写 `web_service.py`（FastAPI 推理服务）
4. 添加 `requirements.txt` 和 `README.md`
5. 在 `docker/hailo10h/` 下创建 Dockerfile
6. 在 `recomputer-ai-lab` 项目中添加模型 YAML 配置

## 许可证

本项目开源。各模型的许可证信息见对应目录。