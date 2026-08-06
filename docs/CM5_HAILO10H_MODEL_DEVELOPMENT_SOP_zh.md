# CM5 + Hailo-10H 模型开发与发布 SOP

本文用于指导 AI 或开发者在 `reComputer-Hailo10H-CV` 中新增、适配、验证和发布 CM5 + Hailo-10H 模型，并把部署入口接入 AI Lab。所有版本号、模型输入输出、镜像名称和运行命令都必须以实际文件或硬件测试结果为准，不得凭经验推测。

## 1. 交付目标

一次完整开发应交付以下内容：

1. GitHub 模型代码、HEF、演示视频、依赖和 Dockerfile。
2. HailoRT 推理封装、预处理、后处理和 Web 服务。
3. 单元测试或最小回归测试。
4. GitHub Actions 镜像构建配置和 GHCR 镜像。
5. AI Lab 模型 YAML、平台 Wiki 和下载统计来源。
6. 在真实 CM5 + Hailo-10H 设备上的运行记录。

## 2. 仓库职责

### 2.1 GitHub：模型和容器

仓库：`Seeed-Projects/reComputer-Hailo10H-CV`

用于保存：

- 模型推理代码；
- Hailo-10H HEF；
- HailoRT Python wheel；
- Dockerfile；
- GitHub Actions 工作流；
- GHCR 容器镜像。

### 2.2 AI Lab：模型卡片和文档

仓库：`recomputer-ai-lab`

用于保存：

- `content/models/<type>/<family>/<slug>.yaml`；
- `content/models/wikis/*.md`；
- 由 `npm run build:models` 生成的模型数据。

不得手工修改 `lib/models-data.generated.ts`。

## 3. 命名规范

统一使用小写下划线模型 slug，例如 `stdc1`、`mspn_regnetx_800mf`。

| 项目 | 规范 | 示例 |
| --- | --- | --- |
| 模型目录 | `src/hailo10h_<slug>/` | `src/hailo10h_stdc1/` |
| Dockerfile | `docker/hailo10h/<slug>.dockerfile` | `docker/hailo10h/stdc1.dockerfile` |
| GHCR 镜像 | `ghcr.io/seeed-projects/recomputer-hailo10h-cv/<slug>:latest` | `.../stdc1:latest` |
| 容器名 | `cm5-hailo10h-<short-name>` | `cm5-hailo10h-stdc1` |
| AI Lab 平台 | `reComputer R Series (CM5 + Hailo-10H)` | 不使用含糊的 `CM5 + Hailo` |
| 平台 Wiki | 沿用仓库既有规则 | `pi5-hailo10h-stdc1.md` |

容器名只影响 Docker 进程的名称，不影响镜像、模型路径或推理代码，但同一台设备上不能同时存在重名容器。

## 4. 标准目录结构

```text
reComputer-Hailo10H-CV/
├── .github/workflows/
│   └── build-ghcr-images.yml
├── docker/hailo10h/
│   └── <slug>.dockerfile
├── docs/
│   └── CM5_HAILO10H_MODEL_DEVELOPMENT_SOP_zh.md
└── src/hailo10h_<slug>/
    ├── web_detection.py
    ├── py_utils/
    │   ├── __init__.py
    │   ├── hailo_executor.py
    │   └── <postprocess_utils>.py
    ├── hailort-packages/
    │   └── hailort-5.1.1-cp313-cp313-linux_aarch64.whl
    ├── model/
    │   └── <slug>.hef
    ├── video/
    │   └── test.mp4
    ├── tests/
    │   └── test_hailo_executor.py
    ├── requirements.txt
    ├── README.md
    └── README_zh.md
```

复制其他模型作为模板时，只复用通用服务结构。模型输入尺寸、张量格式、类别、后处理、可视化和 API 响应必须按目标 HEF 重新实现。

## 5. 固定运行基线

当前项目基线为：

- 硬件：CM5 + Hailo-10H；
- 设备节点：`/dev/hailo0`；
- HailoRT：`5.1.1`；
- 容器 Python：`3.13`；
- wheel：`hailort-5.1.1-cp313-cp313-linux_aarch64.whl`；
- 宿主机动态库：`libhailort.so.5.1.1`。

宿主机驱动、HailoRT 运行库、固件、Python wheel 和容器动态库必须属于兼容的同一版本组合。不要混用：

- Hailo-8 的 HailoRT 4.23.x wheel；
- HailoRT 5.3.x wheel 与 5.1.1 宿主机运行库；
- 不匹配 Python ABI 的 wheel，例如在 Python 3.13 中安装 `cp311`。

## 6. 开发前硬件检查

在 CM5 上执行：

```bash
vcgencmd get_throttled
lspci -nn | grep -i hailo
lsmod | grep -E 'hailo1x_pci|hailo'
modinfo hailo1x_pci | grep -E 'version|vermagic'
hailortcli --version
ls -l /dev/hailo0
sudo hailortcli fw-control identify
apt-cache policy hailo-all hailo-h10-all hailort python3-hailort
```

通过条件：

- PCIe 能识别 Hailo-10H；
- `hailo1x_pci` 驱动已加载；
- `/dev/hailo0` 存在；
- `hailortcli fw-control identify` 成功；
- HailoRT 版本与容器绑定一致；
- `get_throttled` 没有欠压或降频异常。

如果 PCIe 能识别设备但 `/dev/hailo0` 不存在，应先修复驱动、内核或固件问题。此时修改模型代码没有意义。

## 7. 获取并核验 HEF

优先使用 Hailo 官方 Model Zoo 提供的 Hailo-10H 编译产物，并记录：

- 模型来源 URL；
- Model Zoo/DFC/HailoRT 版本；
- 目标硬件确实为 Hailo-10H；
- 输入张量名称、形状、数据类型和布局；
- 所有输出张量名称、形状和量化信息；
- 官方预处理和后处理方法；
- 标签集合和类别数量。

不要根据模型名称猜测输入尺寸。启动时应打印 HEF 实际输入输出信息，并以其为准实现代码。

## 8. HailoRT 5.1.1 推理封装

### 8.1 单输入、单输出同步推理模板

```python
import numpy as np

from hailo_platform import HEF, VDevice


INFERENCE_TIMEOUT_MS = 10_000


class HailoInfer:
    def __init__(self, hef_path):
        self.target = VDevice()
        self.hef = HEF(hef_path)
        self.model = self.target.create_infer_model(hef_path)
        self.input_info = self.model.input()
        self.output_info = self.model.output()

        output_vstream_info = self.hef.get_output_vstream_infos()[0]
        output_format_type = output_vstream_info.format.type
        output_dtype_name = str(output_format_type).split(".")[-1].lower()
        self.output_dtype = np.dtype(output_dtype_name)
        self.output_info.set_format_type(output_format_type)

    def run(self, image):
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)
        if image.ndim == 3:
            image = np.expand_dims(image, axis=0)

        with self.model.configure() as configured_model:
            output_buffer = np.empty(
                self.output_info.shape,
                dtype=self.output_dtype,
            )
            bindings = configured_model.create_bindings(
                output_buffers={self.output_info.name: output_buffer}
            )
            bindings.input().set_buffer(image)
            configured_model.run([bindings], timeout=INFERENCE_TIMEOUT_MS)
            output = bindings.output(self.output_info.name).get_buffer()

        return {self.output_info.name: output}

    def release(self):
        self.target.release()
```

### 8.2 HailoRT 5.1.1 关键规则

1. `ConfiguredInferModel.run()` 的 `bindings` 参数必须是可迭代对象。单帧也要使用 `[bindings]`，不能直接传 `bindings`。
2. `timeout` 的单位是毫秒。10 秒应写为 `timeout=10_000`。
3. 必须先按 HEF 输出格式分配 NumPy 数组，再通过 `create_bindings(output_buffers=...)` 注册输出缓冲区；不能依赖未配置的 output view。
4. HEF 输入为 HWC 且接口需要批次时，把 `(H, W, C)` 扩展为 `(1, H, W, C)`。
5. 缓冲区类型、连续性、尺寸和布局必须与 HEF 一致，不能只靠强制转换掩盖错误。
6. 多输入或多输出模型必须按张量名称分别绑定，不能直接套用单输入单输出模板。
7. 在 `configure()` 上下文有效期间完成推理并读取输出。
8. 服务退出时释放 `VDevice`。

遇到 API 参数争议时，应直接检查镜像中已安装的 5.1.1 Python 绑定签名或 wheel 源码，不要从其他 HailoRT 版本照抄。

检查仓库内是否残留同类调用：

```bash
rg -n 'configured_model\.run' src
```

每个模型都要单独验证，不能因为使用相似封装就默认兼容。

### 8.3 最小回归测试

使用 mock 替代真实设备，至少验证：

- `run()` 收到的是单元素列表；
- 超时值为 `10000` 毫秒；
- 输出缓冲区已通过 `output_buffers` 注册，形状和数据类型来自 HEF；
- 无批次输入被扩展为四维；
- 输出张量键名和缓冲区正确返回。

示例执行：

```bash
python -m unittest discover -s src/hailo10h_<slug>/tests -v
python -m py_compile \
  src/hailo10h_<slug>/py_utils/hailo_executor.py \
  src/hailo10h_<slug>/tests/test_hailo_executor.py
```

mock 测试只能防止 Python API 调用回归，不能替代真实 Hailo-10H 硬件测试。

## 9. 预处理、后处理和 Web 服务

### 9.1 预处理

必须逐项确认：

- RGB 或 BGR；
- resize、letterbox 或 crop；
- 插值方式；
- 是否归一化；
- `uint8`、`float32` 或量化类型；
- HWC/NHWC/NCHW；
- 原图到模型输入的缩放和填充参数。

### 9.2 后处理

必须根据输出语义实现，而不是只根据张量形状猜测。常见内容包括：

- 分类 softmax/top-k；
- 检测框解码、置信度过滤和 NMS；
- 分割 argmax、调色板和掩码缩放；
- 姿态关键点解码、骨架连接和坐标还原。

### 9.3 Web 服务

现有项目通常复用 `web_detection.py` 的视频、摄像头、MJPEG 和 REST API 框架。至少验证：

- 服务监听 `0.0.0.0:8000`；
- `/` 可访问；
- `/api/video_feed` 持续输出；
- 模型 predict API 返回正确结构；
- 视频结束、摄像头断开和服务退出时资源能释放；
- 推理线程异常会被记录，不会出现网页看似启动但后台线程已经退出的假成功。

## 10. Dockerfile

参考结构：

```dockerfile
FROM python:3.13-slim

WORKDIR /app

COPY src/hailo10h_<slug>/requirements.txt ./requirements.txt
COPY src/hailo10h_<slug>/hailort-packages/*.whl /tmp/hailort/

RUN pip install --no-cache-dir /tmp/hailort/*.whl \
    && pip install --no-cache-dir -r requirements.txt

COPY src/hailo10h_<slug>/ /app/

EXPOSE 8000
```

实际 Dockerfile 还应安装 OpenCV、FFmpeg 等系统依赖。构建上下文、COPY 路径和 wheel ABI 必须与工作流一致。

## 11. 标准运行命令

```bash
sudo docker run --rm \
  --name cm5-hailo10h-<short-name> \
  --privileged \
  --net=host \
  -e PYTHONUNBUFFERED=1 \
  --device /dev/hailo0:/dev/hailo0 \
  -v /usr/lib/libhailort.so.5.1.1:/usr/lib/libhailort.so.5.1.1:ro \
  -v /usr/lib/libhailort.so:/usr/lib/libhailort.so:ro \
  ghcr.io/seeed-projects/recomputer-hailo10h-cv/<slug>:latest \
  python web_detection.py \
    --model_path model/<model>.hef \
    --video_path video/test.mp4
```

两个动态库挂载都需要保留：版本化文件提供真实库，`libhailort.so` 提供 Python 绑定或链接器使用的通用名称。

运行后在局域网浏览器访问：

```text
http://<CM5-IP>:8000
```

## 12. GitHub Actions 和 GHCR

模型矩阵应满足：

- 每个模型只出现一次；
- `model`、`dockerfile` 和 GHCR 包名一致；
- 普通模型代码变更只构建受影响模型；
- 工作流文件变更或手动选择 `all` 时构建全部模型；
- 同时发布可追踪提交的 SHA 标签和用于用户部署的 `latest` 标签。

建议路径选择逻辑：

```text
src/hailo10h_<slug>/**       -> <slug>
docker/hailo10h/<slug>.dockerfile -> <slug>
.github/workflows/**         -> all
workflow_dispatch all        -> all
```

注意：首次修改筛选工作流本身时，按规则会构建全部模型一次。GHCR 页面出现多个版本通常表示多次成功发布，不代表之前每个版本都通过了真实硬件验证。

## 13. AI Lab 接入

修改前先阅读：

```text
content/README.md
content/models/README.md
```

### 13.1 平台和部署项必须一致

示例：

```yaml
devices:
  - id: reComputer R Series (CM5 + Hailo-10H)
    label: reComputer R Series (CM5 + Hailo-10H)

deviceEngines:
  reComputer R Series (CM5 + Hailo-10H):
    - id: hailo
      label: HailoRT

deploy:
  - deviceId: reComputer R Series (CM5 + Hailo-10H)
    engineId: hailo
    command: |-
      sudo docker run --rm \
        --name cm5-hailo10h-<short-name> \
        ...
```

`devices[].id`、`deviceEngines` 的键和 `deploy[].deviceId` 必须完全相同。平台显示名称要明确区分 Hailo-8 和 Hailo-10H。

### 13.2 Wiki 路由

同一个模型同时支持 Hailo-8 和 Hailo-10H 时，使用 `wikiPlatforms` 为两个硬件提供独立文档，并把更具体的匹配词放在前面。例如：

```yaml
wikiPlatforms:
  - deviceMatch:
      - Hailo-10H
      - R Series
      - CM5
    doc: wikis/pi5-hailo10h-<slug>.md
  - deviceMatch:
      - Hailo-8
      - R Series
      - CM5
    doc: wikis/pi5-hailo8-<slug>.md
```

每个 `doc` 必须真实存在。不要把 Hailo-8 的驱动、库版本或命令复制到 Hailo-10H 文档。

### 13.3 下载统计

模型 YAML 必须声明 GHCR 包地址，否则镜像即使被下载，AI Lab 也不知道到哪里查询下载量：

```yaml
downloadSources:
  - https://github.com/orgs/Seeed-Projects/packages?repo_name=reComputer-Hailo10H-CV&package=recomputer-hailo10h-cv%2F<slug>
```

规则：

- 没有 `downloadSources`：不会显示，也不会建立该模型与 GHCR 统计的关联；
- 添加来源后：可读取 GitHub 包的累计统计，不是从添加日期重新计数；
- 统计值为 0 时，前端可能隐藏数字；
- GitHub 或站点缓存可能造成显示延迟；
- 同一模型的 Hailo-8 和 Hailo-10H 镜像需要同时列出两个来源，页面逻辑可能合计；
- `yolo26n_seg`、`yolo26m_seg`、`yolo26s_seg` 等独立 slug 应分别配置来源，不能错误共用一个包。

不得编造下载次数。

### 13.4 生成和验证

在 `recomputer-ai-lab` 执行：

```bash
npm run build:models
npm run build
npm run dev
```

默认开发预览端口按仓库说明使用 `3004`，若启动器自动选择其他端口，以终端实际输出为准。检查：

- 模型列表卡片；
- 模型详情页；
- Hailo-8/Hailo-10H 下拉切换；
- 推理引擎和部署命令；
- Wiki 内容；
- 下载量显示或其合理的缓存延迟。

## 14. 真实设备验收

至少完成以下测试：

1. 拉取最新镜像并确认镜像摘要发生更新。
2. 用仓库自带 `test.mp4` 连续运行，不出现推理线程异常。
3. 打开 `http://<CM5-IP>:8000`，视频流持续刷新。
4. 检查结果类别、颜色、坐标和原图比例。
5. 测试一次上传视频或 predict API。
6. 停止后确认容器退出、设备可被下一次运行正常打开。
7. 把完整命令、镜像摘要、HailoRT 版本和结果截图保存到测试记录。

仅看到 `Web Preview started` 不算成功；必须确认推理线程持续工作且页面产生正确结果。

## 15. 常见错误

| 现象 | 原因 | 处理 |
| --- | --- | --- |
| `TypeError: 'Bindings' object is not iterable` | 直接把单个 `Bindings` 传给 5.1.1 的 `run()` | 改为 `run([bindings], timeout=10_000)` |
| 很快触发 timeout | 把毫秒误当成秒，写成 `timeout=10` | 10 秒使用 `10_000` |
| `Trying to get buffer as view ... not configured as view` | 创建 bindings 时没有注册输出数组 | 按 HEF 类型分配数组并传给 `create_bindings(output_buffers=...)` |
| `/dev/hailo0` 不存在 | 驱动、内核、固件或 PCIe 初始化异常 | 先修复宿主机，不修改模型代码 |
| 容器找不到 `libhailort.so.5.1.1` | 只挂载了通用软链接 | 同时挂载版本化库和 `libhailort.so` |
| wheel 无法安装 | Python ABI 或架构不匹配 | Python 3.13/aarch64 使用对应 `cp313` wheel |
| HailoRT 运行时报版本不匹配 | 宿主机、动态库、wheel 或固件混用 | 恢复经过验证的同版本组合 |
| HEF 加载失败 | HEF 为其他 Hailo 芯片或编译版本不兼容 | 下载 Hailo-10H 对应 HEF 并核对来源 |
| 页面启动但无结果 | 后台推理线程已崩溃 | 检查完整容器日志，不能只看 HTTP 启动信息 |
| AI Lab 不显示下载量 | YAML 缺少或写错 `downloadSources` | 配置对应 GHCR 包 URL 后重新生成模型数据 |

## 16. 提交和发布边界

仓库经常存在其他开发者的未提交模型。发布时必须隔离本次模型：

```bash
git status --short
git diff -- src/hailo10h_<slug> docker/hailo10h/<slug>.dockerfile
git add -- \
  src/hailo10h_<slug> \
  docker/hailo10h/<slug>.dockerfile \
  .github/workflows/build-ghcr-images.yml
git diff --cached --check
git diff --cached --stat
git diff --cached
```

原则：

- 不使用 `git add .`；
- 不提交其他模型、临时 wheel、日志或用户未完成的文档；
- 不用 `git reset --hard`、`git clean` 或 checkout 覆盖用户更改；
- 工作区复杂时，使用临时 clone 或独立 worktree 准备提交；
- 推送后检查 GitHub Actions 和 GHCR 包，而不是把 `git push` 成功当作镜像成功；
- AI Lab 变更与模型代码变更分别在各自仓库提交。

## 17. 完成检查表

### GitHub 模型仓库

- [ ] HEF 来源和目标硬件已核验。
- [ ] 输入输出张量已从实际 HEF 确认。
- [ ] 预处理和后处理与模型一致。
- [ ] 输出缓冲区已按 HEF 格式注册，且 `run([bindings], timeout=10_000)` 已使用并测试。
- [ ] wheel、Python ABI、HailoRT 和动态库版本一致。
- [ ] Dockerfile 路径和 CI 矩阵正确且无重复。
- [ ] mock 回归测试和 Python 语法检查通过。
- [ ] 真实 Hailo-10H 视频推理持续运行成功。
- [ ] GHCR `latest` 指向本次验证镜像。

### AI Lab 仓库

- [ ] 平台明确写为 `CM5 + Hailo-10H`。
- [ ] `devices`、`deviceEngines`、`deploy` ID 完全一致。
- [ ] Wiki 文件存在且命令可复制执行。
- [ ] `downloadSources` 指向正确 GHCR 包。
- [ ] 双硬件模型可分别下拉选择 Hailo-8/Hailo-10H。
- [ ] `npm run build:models` 通过。
- [ ] `npm run build` 通过。
- [ ] 本地页面完成视觉和命令检查。

## 18. 可交给其他 AI 的任务模板

```text
请按照 docs/CM5_HAILO10H_MODEL_DEVELOPMENT_SOP_zh.md 开发 <模型名>。

GitHub 仓库：<本地 reComputer-Hailo10H-CV 路径>
AI Lab 仓库：<本地 recomputer-ai-lab 路径>
HEF 来源：<官方 URL>
模型类型：<分类/检测/分割/姿态等>
复用界面模板：<现有模型 slug>

要求：
1. 先阅读两个仓库的说明文件并报告预计修改的文件。
2. 核验 HEF 输入输出，不得猜测张量或性能数据。
3. HailoRT 5.1.1 单帧同步推理必须使用
   create_bindings(output_buffers=...) 注册输出数组，并使用
   configured_model.run([bindings], timeout=10_000)。
4. 平台名必须明确为 reComputer R Series (CM5 + Hailo-10H)。
5. 配置正确的 GHCR downloadSources。
6. 只修改和提交该模型相关文件，不处理其他未提交内容。
7. 运行单元测试、语法检查、AI Lab 模型生成和构建。
8. 给出真实 CM5 + Hailo-10H 测试命令；没有硬件测试时必须明确说明。
9. 完成后报告文件清单、来源、验证结果和仍需人工确认的内容。
```
