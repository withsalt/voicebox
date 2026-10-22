# Qwen3-ASR ONNX 导出项目

这个项目用于将本地 Qwen3-ASR 模型导出为可由 ONNX Runtime 加载的单音频推理图。导出阶段依赖 PyTorch 和 transformers，导出后的运行时目标是使用 ONNX Runtime、tokenizer、feature extractor、NumPy 以及音频读取库完成推理，不再依赖 PyTorch。

## 项目结构

```text
.
├── export_qwen3_asr_onnx.py      # Qwen3-ASR ONNX 导出脚本
├── requirement.txt               # 导出所需 Python 依赖
└── models/
    ├── Qwen3-ASR-0.6B/           # 本地 0.6B 模型目录
    └── Qwen3-ASR-1.7B/           # 本地 1.7B 模型目录
```

## 环境要求

建议使用独立的 Python 3.13 环境。项目依赖见 `requirement.txt`：

安装PyTorch：

```bash
# 替换你的 CUDA 版本，按照命令参考：https://pytorch.org/get-started/locally/
# torch==2.12.x 有bug，会导致导出模型精度下降
pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu130
```

安装依赖：

```bash
pip install -r requirement.txt
```

如果导出大模型或使用 `float16`，建议在具备 CUDA 的机器上运行，并确认 PyTorch、CUDA、ONNX Runtime GPU 版本匹配。

## 快速开始

导出 Qwen3-ASR-0.6B：

```bash
python export_qwen3_asr_onnx.py \
  --model ./models/Qwen3-ASR-0.6B \
  --output-dir ./onnx/Qwen3-ASR-0.6B-fp16 \
  --device cuda:0 \
  --dtype float16 \
  --cache-mode full \
  --overwrite
```

导出 Qwen3-ASR-1.7B：

```bash
python export_qwen3_asr_onnx.py \
  --model ./models/Qwen3-ASR-1.7B \
  --output-dir ./onnx/Qwen3-ASR-1.7B-fp16 \
  --device cuda:0 \
  --dtype float16 \
  --cache-mode full \
  --overwrite
```

如果需要在 CPU 上导出，可将设备和精度改为：

```bash
python export_qwen3_asr_onnx.py \
  --model ./models/Qwen3-ASR-0.6B \
  --output-dir ./onnx/Qwen3-ASR-0.6B-int8 \
  --device cpu \
  --dtype int8 \
  --cache-mode delta \
  --overwrite
```

## 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--model` | 必填 | 本地 Qwen3-ASR 模型目录，例如 `./models/Qwen3-ASR-0.6B` |
| `--output-dir` | 必填 | ONNX 图、元数据和 tokenizer 相关文件输出目录 |
| `--device` | `cuda:0` | 导出设备，例如 `cuda:0` 或 `cpu` |
| `--dtype` | `float16` | 导出精度，可选 `float16`、`float32`、`int8` |
| `--int8-weight-type` | `qint8` | `int8` 动态量化权重类型，可选 `qint8`、`quint8` |
| `--sample-seconds` | `1.0` | 用于 tracing 的虚拟音频长度，单位秒 |
| `--static-cache-len` | `1664` | decoder 固定 KV cache 长度 |
| `--opset` | `20` | ONNX opset 版本 |
| `--overwrite` | 关闭 | 覆盖输出目录中已存在的导出文件 |
| `--cache-mode` | `delta` | KV cache 输出模式，可选 `delta`、`full` |
| `--worklog` | `WORKLOG.md` | 导出完成后追加写入的工作日志路径；如不需要可传空值 |

## 导出产物

导出完成后，输出目录通常包含：

```text
output-dir/
├── encoder.onnx
├── encoder.onnx.data
├── prefill.onnx
├── prefill.onnx.data
├── decode.onnx
├── decode.onnx.data
├── metadata.json
├── tokenizer_config.json
├── preprocessor_config.json
├── vocab.json
├── merges.txt
├── chat_template.json
└── generation_config.json
```

其中：

- `encoder.onnx`：音频编码图，输入音频特征块，输出 audio features。
- `prefill.onnx`：文本 decoder 的 prefill 图，处理 prompt 与音频特征，并输出 logits 和 KV cache。
- `decode.onnx`：逐 token decode 图。仅在 `--cache-mode delta` 下单独导出；`full` 模式下由 `prefill.onnx` 兼任 decode。
- `metadata.json`：记录模型名称、精度、opset、cache 模式、token id、图文件名、隐藏层配置等运行时需要的信息。
- tokenizer / processor 文件：由 `processor.save_pretrained()` 写入，用于运行时构造文本模板和音频特征。

ONNX 模型使用 external data 格式保存，因此 `.onnx.data` 文件必须和对应 `.onnx` 文件放在同一目录。

## Cache 模式说明

脚本支持两种 decoder cache 模式：

### delta 模式

```bash
--cache-mode delta
```

这是默认模式。导出 `prefill.onnx` 和 `decode.onnx` 两张 decoder 图，decode 每步只输出新增的 KV cache。该模式减少每步从设备侧传回主机侧的数据量，适合 CPU 路径或需要较低逐 token 开销的场景。

### full 模式

```bash
--cache-mode full
```

该模式下 `prefill.onnx` 同时承担 prefill 和 decode，`metadata.json` 中的 `decode_model` 会指向 `prefill.onnx`。它更适合 CUDA + IOBinding 这类希望 KV cache 常驻 GPU 的运行时路径，但在 CPU 路径下可能变慢。

## INT8 动态量化

使用 `--dtype int8` 时，脚本会先导出 float32 图，再对权重做 ONNX Runtime 动态量化：

```bash
python export_qwen3_asr_onnx.py \
  --model ./models/Qwen3-ASR-0.6B \
  --output-dir ./onnx/Qwen3-ASR-0.6B-int8 \
  --device cpu \
  --dtype int8 \
  --int8-weight-type qint8 \
  --overwrite
```

在 `delta` 模式下会量化 `encoder.onnx`、`prefill.onnx` 和 `decode.onnx`；在 `full` 模式下会量化 `encoder.onnx` 和 `prefill.onnx`。

## 模型准备

项目当前包含两个本地模型目录：

- `models/Qwen3-ASR-0.6B`
- `models/Qwen3-ASR-1.7B`

如果需要重新下载模型，可使用 ModelScope 或 Hugging Face：

```bash
pip install -U modelscope
modelscope download --model Qwen/Qwen3-ASR-0.6B --local_dir ./models/Qwen3-ASR-0.6B
modelscope download --model Qwen/Qwen3-ASR-1.7B --local_dir ./models/Qwen3-ASR-1.7B
```

或：

```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download Qwen/Qwen3-ASR-0.6B --local-dir ./models/Qwen3-ASR-0.6B
huggingface-cli download Qwen/Qwen3-ASR-1.7B --local-dir ./models/Qwen3-ASR-1.7B
```

## 注意事项

- 默认导出设备是 `cuda:0`，没有 GPU 时需要显式传入 `--device cpu --dtype float32`。
- `float16` 通常适合 GPU 导出和 GPU 推理；CPU 推理一般建议使用 `float32` 或 `int8`。
- 输出目录中已有 `encoder.onnx`、`prefill.onnx`、`decode.onnx` 或 `metadata.json` 时，脚本默认拒绝覆盖，需要添加 `--overwrite`。
- `--static-cache-len` 决定可用的固定 KV cache 长度，过小可能无法容纳较长 prompt 或生成序列，过大会增加显存或内存占用。
- 导出的 `.onnx` 和 `.onnx.data` 是一组文件，移动或部署时需要一起拷贝。

## 报错处理

### 报错 `CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH`

报错示例：

```text
RuntimeError: CUDNN_BACKEND_TENSOR_DESCRIPTOR cudnnFinalize failed
ptrDesc->finalize() cudnn_status: CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH
```

原因：系统安装的 CUDA Toolkit（如 `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.2\bin\x64\`）里带有一套 cuDNN DLL，PyTorch 自带 `torch/lib/` 下也带有一套。两套小版本不一致时，Windows 的 DLL 搜索顺序可能让 `cudnn64_9.dll` 和它的子库（`cudnn_cnn64_9.dll`、`cudnn_ops64_9.dll`、`cudnn_engines_*64_9.dll` 等）分别来自不同版本，cuDNN 内部初始化时就会报子库版本不匹配。

#### 方案 A：临时从 PATH 移除 CUDA Toolkit 路径再跑导出（推荐）

PowerShell 中执行（只影响当前会话）：

```powershell
$env:PATH = ($env:PATH -split ';' | Where-Object { $_ -notmatch 'CUDA\\v13\.2' }) -join ';'
```

这样 PyTorch 只会加载自己 `torch/lib/` 里那套配套的 cuDNN，子库版本就一致了。不会破坏现有 CUDA Toolkit 安装。请把 `CUDA\\v13\.2` 中的版本号替换成你本机实际安装的版本。

#### 方案 B：把 PyTorch 自带 cuDNN 路径插到 PATH 最前面

PowerShell 中执行：

```powershell
$env:PATH = "E:\.packages\conda\envs\qwen3-asr\Lib\site-packages\torch\lib;$env:PATH"
```

把前缀路径替换成你本机 conda 环境下 `torch/lib` 的实际位置。该方案不如方案 A 干净（Windows 上同名 DLL 一旦被任何模块加载，后续 `LoadLibrary` 仍可能返回先前加载的句柄），但通常也能生效。

