# 基于 CM5 + Hailo-10H 的 YOLO-World v2s

面向 **reComputer CM5 / Raspberry Pi 5 + Hailo-10H** 的开放词表目标检测模块。
它运行 Hailo Model Zoo 的 YOLO-World v2s HEF，网络有两个输入：640×640 RGB 图像和
80×512 的 CLIP 文本嵌入矩阵。仓库内置的词表是 80 个 COCO 类别。

功能包括 MJPEG 实时预览、REST 预测、离线视频批量分析和置信度滑块。HEF 内置
YOLO-World 的文本对比头和 host 侧 NMS 后处理。

---

## 1. 主机准备

安装 Hailo-10H 运行时和驱动，并确认设备：

```bash
sudo apt update
sudo apt install hailo-h10-all -y
sudo reboot

hailortcli fw-control identify
ls /dev/hailo0
```

安装 Docker：

```bash
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh --mirror Aliyun
sudo systemctl enable docker
sudo systemctl start docker
```

容器使用内置的 `hailort-5.1.1-cp313-cp313-linux_aarch64.whl`，运行时必须挂载
匹配的宿主机 `libhailort.so.5.1.1`。

---

## 2. 构建与运行

```bash
cd src/hailo10h_yolo_world_v2s
sudo docker build -f ../../docker/hailo10h/yolo_world_v2s.dockerfile \
  -t hailo10h-yolo-world-v2s:latest .

sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  hailo10h-yolo-world-v2s:latest
```

默认命令循环播放 `video/test.mp4`，Web UI 地址为 `http://<开发板IP>:8000`。

摄像头模式：

```bash
sudo docker run --rm --privileged --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  --device /dev/video0:/dev/video0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  hailo10h-yolo-world-v2s:latest \
  python web_detection.py --model_path model/yolo_world_v2s.hef --camera_id 0
```

---

## 3. 模型资产

| 文件 | 大小 | 来源 |
|---|---:|---|
| `model/yolo_world_v2s.hef` | 25,092,096 B | [Model Zoo v5.4.0 hailo10h](https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/ModelZoo/Compiled/v5.4.0/hailo10h/yolo_world_v2s.hef) |
| `model/coco_text_embeddings.npy` | 163,968 B | CLIP ViT-B/32 文本嵌入，形状 `(1, 80, 512)` |
| `model/coco_classes.txt` | 701 B | 80 个 COCO 类别，每行一个 |

HEF SHA256：
`489a366562d617ea8428958405e84f37e65775eb4b66e1af112617d1321ef1f0`。

Hailo 公开的 Hailo-10H zero-shot 表链接的是 v5.5.0 构建（36.3 float mAP、
34.9 硬件 mAP、batch 1 42.8 FPS）。本模块使用 v5.4.0，以与仓库内其他 Hailo-10H
模块保持一致。HEF 格式为 v3，元数据报告 SDK 5.4.0；发布结果前请在开发板上确认
与 HailoRT 的兼容性。如果运行时版本更新，可以直接替换为 v5.5.0 HEF。

完整来源与注意事项见 [model/README.md](model/README.md)。

---

## 4. 词表

HEF 固定接收 80 条提示。`coco_text_embeddings.npy` 的第 `i` 行对应输出类别 `i`，
`coco_classes.txt` 的第 `i` 行是显示名称。内置嵌入使用
`openai/clip-vit-base-patch32`（`CLIPTextModelWithProjection`，仅类别名，L2 归一化）
生成，与 Hailo 的 `create_coco_tfrecord_with_text_embeddings.py` 一致。

如需其他词表，请预先计算同样形状的 80 条嵌入，并同时传入两个文件：

```bash
python web_detection.py \
  --model_path model/yolo_world_v2s.hef \
  --embeddings_path /path/to/my_embeddings.npy \
  --class_path /path/to/my_classes.txt
```

未使用的槽位仍需填入有效提示；全零行可能让文本对比头给出虚假的中等置信度。

---

## 5. HTTP 接口

| 方法 | 路径 | 用途 |
|---|---|---|
| GET | `/` | Web UI |
| GET/POST | `/api/config` | 读取/更新显示阈值 |
| POST | `/api/video/upload` | 上传视频 |
| GET | `/api/video/list` | 列出上传与输出 |
| POST | `/api/video/analyze` | 启动离线分析 |
| GET | `/api/video/status` | 进度和错误状态 |
| GET | `/api/video/download/{filename}` | 下载标注 MP4 |
| POST | `/api/models/yolo_world/predict` | 单帧预测 |

---

## 6. 实现说明

- 启动时从 HEF 识别两个输入：`yolo_world_v2s/input_layer1`
  （UINT8 640×640×3 RGB）和 `yolo_world_v2s/input_layer2`（float32 80×512）。
- 图像先 letterbox 到 640×640，再转 RGB；HEF 内的 normalization 会除以 255。
- HailoRT 5.1.1 的 bindings 只创建一次；每帧更新两个输入缓冲区，输出统一请求
  FLOAT32。
- NMS 使用 Model Zoo 配置：80 类、IoU 0.7、每类最多 300 个候选、分数阈值 0.001。
  界面阈值只过滤返回结果。

---

## 7. 已知限制

- 词表在启动时固定，板端没有 CLIP 文本编码器。
- HEF 编译版本（SDK 5.4.0）与内置 HailoRT 5.1.1 的兼容性需要在实机确认。
- Web 服务面向可信局域网，默认没有身份验证或 TLS。
- 实机验证待完成，见 `TEST_REPORT.md`。