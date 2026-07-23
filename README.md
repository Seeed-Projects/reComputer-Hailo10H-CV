# reComputer-Hailo10H-CV

[English] | [中文](./README_zh.md)

This project aims to provide industrial-grade, high-performance Computer Vision (CV) application solutions for the reComputer R Series (Raspberry Pi CM5) with **Hailo-10H** accelerator. It currently supports semantic segmentation, with more models to be added.

## Project Architecture

The project uses a platform-specific adaptation architecture, with code and configuration managed independently for each model:

```text
reComputer-Hailo10H-CV/
├── docker/                 # Docker image configuration files
│   └── hailo10h/           # Hailo-10H specific Dockerfiles
├── src/                    # Source code directory
│   └── hailo10h_<model>/   # Each model's source code and dependencies
│       ├── web_service.py  # FastAPI inference service
│       ├── requirements.txt
│       ├── README.md
│       └── model/          # HEF model files
└── .github/workflows/      # GitHub Actions automated build scripts
```

## Supported Platforms

| Platform | Chip | Accelerator | Computing Power |
| :--- | :--- | :--- | :--- |
| **reComputer R Series** | Raspberry Pi CM5 | Hailo-10H | 10 TOPS |

## Available Models

| Model | Task | Params | Input | mIoU | HEF |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [**STDC1**](src/hailo10h_stdc1/) | Semantic Segmentation | 8.27M | 1024×1920 | 73.7% | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/stdc1.hef) |
| [**FCN8 ResNet-18**](src/hailo10h_fcn8_resnet_v1_18/) | Semantic Segmentation | 11.20M | 1024×1920 | 69.2% | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/fcn8_resnet_v1_18.hef) |
| [**MSPN RegNetX-800MF**](src/hailo10h_mspn_regnetx_800mf/) | Pose Estimation | 7.17M | 256×192 | — | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/mspn_regnetx_800mf.hef) |

## Quick Start

### 1. Install Docker

Run the following commands on the reComputer R Series to install Docker:

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

### 2. Install HailoRT

```bash
sudo apt update
sudo apt install hailo-all
sudo reboot

# Verify installation
hailortcli fw-control identify
ls /dev/hailo0
```

### 3. Run STDC1

```bash
# Build
docker build -t stdc1 -f docker/hailo10h/stdc1.dockerfile .

# Run
sudo docker run --rm --privileged --net=host \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  stdc1
```

Open `http://<device_ip>:8000/health` to verify the service is running.

## REST API

### Prediction

```bash
curl -X POST http://<device_ip>:8000/api/models/stdc1/predict \
  -F "file=@test.jpg"
```

### Visualization

```bash
curl -X POST http://<device_ip>:8000/api/models/stdc1/visualize \
  -F "file=@test.jpg" \
  -o result.jpg
```

### Class List

```bash
curl http://<device_ip>:8000/api/models/stdc1/classes
```

## Model Sources

All HEF models are pre-compiled from the [Hailo Model Zoo](https://github.com/hailo-ai/hailo_model_zoo) and optimized for Hailo-10H.

## Contributing

To add a new model:

1. Create a new directory: `src/hailo10h_<model_name>/`
2. Add the HEF model file to `model/`
3. Write `web_service.py` with FastAPI endpoints
4. Add `requirements.txt` and `README.md`
5. Create a Dockerfile under `docker/hailo10h/`
6. Add the model to the AI Lab website via `recomputer-ai-lab`

## License

This project is open source. See individual model directories for specific license information.