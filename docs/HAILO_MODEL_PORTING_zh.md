# Hailo-10H Model Zoo 模型迁移与一键镜像生成

这份说明用于把 Hailo Model Zoo 里的 `.hef` 模型搬到本项目，并生成可在
Raspberry Pi 5 + Hailo-10H 上一键部署的 Docker 镜像。

## 1. 先判断模型类型

目前仓库里已有四类可复用模板：

| 模型类型 | 推荐模板 | 适合的 Hailo Zoo 输出 |
|---|---|---|
| 语义分割 | `src/hailo10h_stdc1/` | 输出为逐像素类别分数或 mask |
| 姿态估计 | `src/hailo10h_mspn_regnetx_800mf/` | 输出为关键点 heatmaps |
| 人脸检测 | `src/hailo10h_lightface_slim/` | 输出为边界框 |
| 目标检测 | 参考 `src/hailo10h_lightface_slim/` | 输出为 (1, N, 5) 检测框 |

如果是其他类型模型，按最接近的模板修改 `web_detection.py` 里的
`post_process_hailo()` 和 API 返回格式。

## 2. 从 Hailo Model Zoo 准备 `.hef`

在 Hailo Model Zoo 里找到目标模型，下载面向 `hailo10h` 的 `.hef`。
文件名必须和模型 slug 一致，例如：

```text
stdc1.hef
fcn8_resnet_v1_18.hef
mspn_regnetx_800mf.hef
lightface_slim.hef
```

注意事项：

- `.hef` 必须是 `hailo10h` 目标，不是 `hailo8` 或 `hailo8l`。
- 宿主机 `hailo-h10-all`、容器里的 `hailort` wheel、固件版本必须一致。
- Hailo-10H 的 HailoRT 版本为 5.x（当前项目使用 5.3.0 wheel）。
- 类别顺序必须和模型训练数据一致；非 COCO/VOC 模型建议额外准备 `class_config.txt`。

## 3. 创建项目模块

参考 SOP 文档 `MODEL_DEVELOPMENT_SOP.md`，按以下结构创建：

```text
src/hailo10h_<model_slug>/
├── web_detection.py
├── py_utils/
├── hailot-packages/
│   └── hailort-5.3.0-cp311-cp311-linux_aarch64.whl
├── model/
│   └── <model_slug>.hef
├── video/
│   └── test.mp4
├── requirements.txt
├── README.md
└── README_zh.md

docker/hailo10h/<model_slug>.dockerfile
```

## 4. 检查关键文件

每搬一个模型，重点检查：

| 文件 | 要检查什么 |
|---|---|
| `docker/hailo10h/<model>.dockerfile` | `CMD` 里的 `--model_path model/<model>.hef` 是否正确 |
| `src/hailo10h_<model>/model/` | 是否存在对应 `.hef` |
| `src/hailo10h_<model>/hailort-packages/` | 是否有 `hailort-5.3.0-cp311-cp311-linux_aarch64.whl` |
| `src/hailo10h_<model>/web_detection.py` | `post_process_hailo()` 是否适配该模型输出 |
| `src/hailo10h_<model>/web_detection.py` | API 路径是否是 `/api/models/<model>/predict` |
| `src/hailo10h_<model>/web_detection.py` | 是否使用 `argparse` 和 `HailoInfer` |

## 5. 在 Pi 5 上构建镜像

```bash
cd src/hailo10h_stdc1

sudo docker build -f ../../docker/hailo10h/stdc1.dockerfile \
    -t hailo10h-stdc1:latest .
```

## 6. 一键运行

使用内置测试视频：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    hailo10h-stdc1:latest
```

USB 摄像头：

```bash
sudo docker run --rm --privileged --net=host \
    -e PYTHONUNBUFFERED=1 \
    --device /dev/hailo0:/dev/hailo0 \
    --device /dev/video0:/dev/video0 \
    -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
    hailo10h-stdc1:latest \
    python web_detection.py --model_path model/stdc1.hef --camera_id 0
```

浏览器打开：

```text
http://<Pi5_IP>:8000
```

## 7. 常见问题

| 现象 | 优先检查 |
|---|---|
| 容器启动时报 HailoRT 相关错误 | 宿主机 `hailo-h10-all`、`.whl`（5.3.0）、`libhailort.so` 版本是否一致 |
| 找不到模型文件 | Dockerfile `CMD` 和 `model/` 里的 `.hef` 文件名是否一致 |
| 检测框全错或无结果 | `.hef` 是否带 NMS；不带 NMS 需要重写检测后处理 |
| 分割颜色或类别名错位 | 类别索引是否按训练数据顺序排列 |
| 画面位置偏移 | 模型输入尺寸和输出 resize 是否一致 |
| CI 构建失败 | wheel 是否正确、`.hef` 是否推送 |

## 8. 发布到 GHCR

镜像 tag 统一：

```bash
ghcr.io/seeed-projects/recomputer-hailo10h-cv/<model>:latest
```

例如：

```bash
ghcr.io/seeed-projects/recomputer-hailo10h-cv/stdc1:latest
```