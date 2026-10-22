# Confucius4-R2T2 ONNX

将原始 Confucius4-R2T2 转为可独立部署的 ONNX Runtime 模型包。支持 `fp32`、`fp16`、`int8`，使用同一张 decoder 图完成 prefill 和逐 token 解码。运行 C# 时不需要 Python、PyTorch 或 vLLM。

## 环境

建议在独立 conda 环境运行，避免 Windows 用户级 Python 包覆盖环境内的 ONNX Runtime：

```powershell
conda activate confucius4-export
$env:PYTHONNOUSERSITE = '1'
python -s -m pip install -r requirements.txt
```

PyTorch 的 CUDA wheel 需要单独安装并匹配 GPU。验证所用版本会记录在导出包的 `metadata.json`。不要同时安装 CPU 版 `onnxruntime` 和 `onnxruntime-gpu`，它们使用相同的模块路径。

## 导出

```powershell
python -s export_confucius4_onnx.py --model-dir ./Confucius4-R2T2 --out-dir ./outputs/fp32 --dtype fp32
python -s export_confucius4_onnx.py --model-dir ./Confucius4-R2T2 --out-dir ./outputs/fp16 --dtype fp16
python -s export_confucius4_onnx.py --model-dir ./Confucius4-R2T2 --out-dir ./outputs/int8 --dtype int8
```

附加参数：`--device cpu`（默认，避免导出显存峰值）和 `--cache-length 1024`。上下文、音频占位 token 与生成预算之和必须小于缓存容量。较长热词提示可增加缓存，代价是显存和缓存更新开销。

输出目录必须不存在。导出先写临时目录，完成 ONNX checker、ORT 实际加载检查及 SHA-256 清单后才发布目标目录。失败时保留临时目录供诊断，不覆盖原模型或已有部署包。

部署时复制整个目录：

```text
encoder.onnx + encoder.onnx.data
decoder.onnx + decoder.onnx.data
metadata.json
config.json + preprocessor_config.json
tokenizer.json + tokenizer_config.json + vocab.json + merges.txt
chat_template.json + special token / generation configuration
LICENSE + MODEL_LICENSE + MODEL_LICENSE_zh
```

FP16 推荐用于 CUDA；FP32 用于数值基线或 CPU；INT8 为 8 位权重、FP32 激活的分块 MatMul 量化，使用 ONNX Runtime 1.28 的 `com.microsoft.MatMulNBits`（`bits=8`，block size 32）。未量化的 Gemm/卷积、归一化、I/O 和 KV Cache 保持 FP32。实际量化节点数量记录在 metadata。此方案在 CUDA 上验证，CPU 可以执行但本机明显较慢，不推荐用于 CPU 实时对话。INT8 是混合精度部署包，不代表全图整数运算，也不保证比 FP16 更快。

## 流式语义

模型收到新增音频后重新编码当前有界窗口，结合已确认文本前缀生成新内容。decoder 的 KV Cache 在一次自回归解码内复用，不跨发生变化的音频前缀盲目复用。C# 缓存可复用的原始 log-Mel 帧，并重新执行全窗口的归一化。

音频注意力按原始 vLLM/FlashAttention 的分窗边界计算。`qwen-asr==0.0.6` 的 eager 后端没有应用传入的 `cu_seqlens`；Python 独立参考仅为原始 attention 注入对应 mask，其他计算仍使用原始模型。当前测试不等于直接运行了 Linux vLLM。

默认 160ms 更新、首块额外 160ms 前瞻、末尾回退 1 个 BPE token、单次最多生成 4 个 token。每次结果含累计 `Text`、只增加的 `CommittedText` 和可修正的 `UncommittedText`。应用应把 `CommittedText` 用于要求不可撤回的下游操作。

默认每 16 秒强制完成一个窗口，控制音频和缓存的内存及计算量；累计文本会随会话增长。连续语音跨硬边界可能降低识别效果；实时对话建议配合 VoiceBox VAD，以自然语句结束调用 Finish。窗口内无损消耗音频，队列溢出明确报错。请求结束时，即使恰好处于分块边界也会进行最终解码。不足 201 个采样点的独立短输入返回空文本。

## 验证与评测

```powershell
python -s -m pytest tests -q
python -s validate_reference.py --model-dir ./Confucius4-R2T2 --audio ./src/Confucius4-R2T2/resources/test.wav --out-dir ./reports/reference
python -s validate_onnx.py --package-dir ./outputs/fp32 --reference-dir ./reports/reference --provider cuda --report ./reports/onnx-fp32.json
python -s validate_onnx.py --package-dir ./outputs/fp16 --reference-dir ./reports/reference --provider cuda --report ./reports/onnx-fp16.json
python -s validate_onnx.py --package-dir ./outputs/int8 --reference-dir ./reports/reference --provider cuda --report ./reports/onnx-int8.json
```

`validate_reference.py` 比较原始模型与导出包装器的多个音频长度、编码器、前 8 个 decoder logits，并保存 C# 可读取的 Mel、提示词、音频和转写基线。流式与离线分别建立基线，因为分块会影响标点和文本。

`prepare_benchmark.py` 可将现有 VoiceBox evaluation manifest 转为双方使用的 16k float PCM。英文测试可安装 `requirements-test.txt`，运行 `prepare_english_benchmark.py --out-dir reports/english`。`benchmark_python.py` 运行原始模型或 ONNX；`summarize_benchmarks.py` 区分人工标注 CER/WER 和不同运行时的一致率。

完整运行记录见 [VALIDATION.md](VALIDATION.md)。测试数据、原模型和大型部署产物不提交到 Git。模型及其衍生品遵守 `licenses/` 中的原始协议，修改后的模型不代表原权利人背书。
