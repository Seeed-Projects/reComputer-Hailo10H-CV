# reComputer-Hailo10H-CV

Computer Vision models for reComputer R Series (CM5 + Hailo-10H).

## Models

| Model | Task | Params | HEF |
|-------|------|--------|-----|
| [STDC1](src/hailo10h_stdc1/) | Semantic Segmentation | 8.27M | [Download](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/stdc1.hef) |

## Structure

```
reComputer-Hailo10H-CV/
├── docker/
│   └── hailo10h/
│       └── <model>.dockerfile
├── src/
│   └── hailo10h_<model>/
│       ├── web_service.py
│       ├── requirements.txt
│       ├── README.md
│       └── model/
│           └── <model>.hef
└── .github/
    └── workflows/
```

## Platform

- **Board**: reComputer R Series (Raspberry Pi CM5)
- **Accelerator**: Hailo-10H (PCIe)
- **Runtime**: HailoRT
