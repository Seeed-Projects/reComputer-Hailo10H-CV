# Hailo-10H YOLO-World v2s 测试报告

**测试日期**：2026-11-18
**测试模块**：[src/hailo10h_yolo_world_v2s/](./)
**测试结论**：静态检查与合成后处理校验通过；实机验证待完成

---

## 1. 测试环境

| 项 | 值 |
|---|---|
| 开发主机 | Windows 11，Python 3.12，无 Hailo 设备 |
| 目标硬件 | reComputer CM5 / Raspberry Pi 5 + Hailo-10H（未连接） |
| HailoRT | 宿主机未安装（`hailo_platform` 不可导入） |
| 容器 wheel | `hailort-5.1.1-cp313-cp313-linux_aarch64.whl` |
| 模型 | `model/yolo_world_v2s.hef`，25,092,096 B，Model Zoo v5.4.0 |
| HEF SHA256 | `489a366562d617ea8428958405e84f37e65775eb4b66e1af112617d1321ef1f0` |
| 文本嵌入 | `model/coco_text_embeddings.npy`，`(1, 80, 512)` float32 |
| 测试视频 | `video/test.mp4`（随模块复制，未在实机解码） |

---

## 2. 已执行的检查

| 检查 | 命令/方法 | 结果 |
|---|---|---|
| Python 语法 | `python -m py_compile web_detection.py py_utils/hailo_executor.py` | 通过 |
| NMS 对象数组解码 | 合成 80 类 object 数组，类别 0/16/79 各一条 | 通过：按 `ymin,xmin,ymax,xmax` 解码，坐标缩放正确，阈值过滤正确 |
| NMS 稠密数组解码 | 合成 `(1, 80, 100, 5)` float32 | 通过：低分框被过滤，类别正确 |
| 空输出 | 80 类全空 | 通过：返回 `None`，不产生虚假框 |
| letterbox 反映射 | 3840×2160 → 640×640，框 `[100,200,300,500]` | 通过：映射回 `[600,360,1800,2160]` |
| 嵌入形状 | `np.load` | 通过：`(1, 80, 512)`，float32 |
| 嵌入归一化 | 逐行 L2 范数 | 通过：≈1.0 |
| 类别表 | 读取 `coco_classes.txt` | 通过：80 行，第 0/16/79 行为 person/dog/toothbrush |
| 多输入绑定代码 | 静态检查 `create_bindings` / `bindings.input(name)` | 通过：按 HailoRT 5.1.1 绑定签名实现 |
| Dockerfile CMD | 检查引用的 HEF 路径 | 通过：`model/yolo_world_v2s.hef` 存在 |
| CI 矩阵 | 检查 workflow option/include | 通过：`yolo_world_v2s` 已加入 |

合成校验使用与 `web_detection.py` 相同的 `post_process_hailo()` 和
`preprocess_frame()`，不是对真实 HEF 输出的验证。

---

## 3. 待完成的实机验证

以下项目必须在连接 Hailo-10H 的开发板上执行，当前状态均为 **Pending**：

- [ ] HEF 能被 HailoRT 加载，打印真实的输入/输出 vstream 名称与形状。
- [ ] `input_layer1` 接受 UINT8 640×640×3 RGB，`input_layer2` 接受 float32 80×512。
- [ ] 首帧推理成功，NMS 输出布局与 `post_process_hailo()` 匹配。
- [ ] 80 个 COCO 标签与嵌入行一一对应，无错位。
- [ ] 用 `video/test.mp4` 跑通离线分析和 MJPEG 预览。
- [ ] 记录推理 FPS、端到端 FPS 和内存占用。
- [ ] 抽查实际场景的误检/漏检，确认置信度分布合理。

在完成上述项目之前，不得把本模块描述为“硬件验证通过”。

---

## 4. 版本兼容性风险

本模块的 HEF 来自 Model Zoo v5.4.0，元数据报告 SDK 5.4.0；内置 Python wheel
是 HailoRT 5.1.1。Hailo 官方 Hailo-10H zero-shot 表链接的是 v5.5.0 构建。
实机需确认 HailoRT 是否接受更高 SDK 版本编译的 HEF；如果拒绝，应把宿主机驱动、
`libhailort.so`、wheel 和 HEF 一起升级到同一版本，或改用与运行时匹配的 HEF。

---

## 5. 复现命令

```bash
python -m py_compile web_detection.py py_utils/hailo_executor.py
python - <<'PY'
import numpy as np
emb = np.load("model/coco_text_embeddings.npy")
print(emb.shape, emb.dtype)
print(np.linalg.norm(emb[0], axis=-1).min(), np.linalg.norm(emb[0], axis=-1).max())
PY
```

实机验证需要在 Hailo-10H 开发板上运行容器，见 `README.md` 的 Docker 命令。