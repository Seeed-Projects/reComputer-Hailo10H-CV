# reComputer-Hailo10H-CV

[English] | [中文](./README_zh.md)

This project provides industrial-grade Computer Vision (CV) solutions for the reComputer R Series (Raspberry Pi 5) with **Hailo-10H** accelerator. Models are pre-compiled from the Hailo Model Zoo and packaged with Docker for one-click deployment.

## Project Architecture

Each model is self-contained with its own inference code, model file, and Docker configuration:

```text
reComputer-Hailo10H-CV/
├── .github/workflows/       # CI/CD auto-build Docker images
├── tools/                   # Scaffolding scripts
├── docs/                    # Documentation
├── docker/hailo10h/         # Dockerfiles per model
├── src/
│   └── hailo10h_<model>/
│       ├── web_detection.py # FastAPI inference service
│       ├── py_utils/        # HailoInfer + utilities
│       ├── hailot-packages/ # HailoRT Python wheel
│       ├── model/           # HEF model file
│       ├── video/           # Demo video
│       └── requirements.txt
```

## Supported Platforms

| Platform | Chip | Accelerator | Computing Power |
| :--- | :--- | :--- | :--- |
| **reComputer R Series** | Raspberry Pi 5 | Hailo-10H | 10 TOPS |

## Available Models

| Model | Task | Params | Input | Metric | HEF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [**STDC1**](src/hailo10h_stdc1/) | Semantic Segmentation | 8.27M | 1024×1920 | 73.7% mIoU | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/stdc1.hef) |
| [**FCN8 ResNet-18**](src/hailo10h_fcn8_resnet_v1_18/) | Semantic Segmentation | 11.20M | 1024×1920 | 69.2% mIoU | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/fcn8_resnet_v1_18.hef) |
| [**MSPN RegNetX-800MF**](src/hailo10h_mspn_regnetx_800mf/) | Pose Estimation | 7.17M | 256×192 | 69.8% AP | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/mspn_regnetx_800mf.hef) |
| [**LightFace Slim**](src/hailo10h_lightface_slim/) | Face Detection | 0.26M | 240×320 | 39.3% mAP | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/lightface_slim.hef) |

## Quick Start

### 1. Install Docker

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker && sudo systemctl start docker
```

### 2. Install HailoRT (Hailo-10H)

```bash
sudo apt update
sudo apt install hailo-h10-all -y
sudo reboot

# Verify
hailortcli fw-control identify
ls /dev/hailo0
```

### 3. Pull and Run

```bash
# Pull from GHCR
docker pull ghcr.io/seeed-projects/recomputer-hailo10h-cv/stdc1:latest

# Run
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/stdc1:latest
```

Open `http://<device_ip>:8000` to view the web preview.

### Or Build Locally

```bash
cd src/hailo10h_stdc1
docker build -f ../../docker/hailo10h/stdc1.dockerfile -t stdc1 .
```

## REST API

### Prediction

```bash
curl -X POST http://<device_ip>:8000/api/models/stdc1/predict \
  -F "file=@test.jpg"
```

### Visualization

```bash
curl -X POST http://<device_ip>:8000/api/models/stdc1/visualize \
  -F "file=@test.jpg" -o result.jpg
```

## Model Sources

All HEF models are from [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo), pre-compiled for Hailo-10H.

## Contributing

1. Use scaffold: `python tools/scaffold_hailo_model.py <model_name> --task <task>`
2. Add HEF to `model/`
3. Update `web_detection.py` post-processing
4. Add HailoRT wheel to `hailort-packages/`
5. Add model to `recomputer-ai-lab`
6. See [docs/HAILO_MODEL_PORTING_zh.md](docs/HAILO_MODEL_PORTING_zh.md) for details

## License

Open source. See individual model directories for license information.