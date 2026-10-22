# CAM++ Speaker Verification ONNX Export

将 ModelScope [iic/speech_campplus_sv_zh-cn_16k-common](https://modelscope.cn/models/iic/speech_campplus_sv_zh-cn_16k-common)
(CAM++ 说话人验证, 16kHz, 192 维 embedding, 6.9M 参数) 导出为动态长度 ONNX,
供 C# 库 `VoiceBox.Speaker.CampPlus` 推理。

## 环境

conda 环境 `voicebox` (torch / onnx / onnxruntime / modelscope / torchaudio / soundfile)。
模型定义直接来自 `modelscope.models.audio.sv.DTDNN.CAMPPlus`, 无需 funasr。

## 用法

```bash
# 1. 导出 (默认写入 src/VoiceBox.Speaker.CampPlus/runtimes/models/speaker/campplus)
python 01_export_onnx.py --model-dir speech_campplus_sv_zh-cn_16k-common --out-dir <部署目录>

# 2. 真实音频对拍 (官方前处理 + 官方 pipeline 端到端 + 性能)
python 02_verify_parity.py

# 3. 导出黄金向量（供回归验证）
python 03_dump_vectors.py
```

部署目录产物: `campplus.onnx` (单文件, ~28MB fp32) + `config.ini` (特征参数/阈值)。

## 模型契约

```
inputs : feats     [batch, time, 80]   Kaldi fbank 80 维 (25ms/10ms, Povey 窗) + 逐段 CMN
outputs: embedding [batch, 192]        余弦打分, 官方整段验证阈值 0.31
```

时间轴完全动态: 同一张图服务任意长度 (整段注册/验证与流式滑窗共用)。

## 导出坑: seg_pooling 的 AveragePool 语义

原始 `CAMLayer.seg_pooling` 的 `avg_pool1d(ceil_mode=True)` 导出为 ONNX `AveragePool` 后,
最后一个不完整分段窗口的除数语义与 PyTorch 不一致; 且 FCM 头部把时间轴减半, 导致只有
内部长度恰为 seg_len(100) 整数倍的输入才精确 (其他长度 embedding cosine 跌到 0.4~0.98,
TorchScript 与 dynamo 两条导出路径均如此)。

`01_export_onnx.py` 在导出前把 seg_pooling 替换为 **cumsum 分段均值 + index_select 广播**:
数学等价 (脚本内先在 eager 模式证明), 部分窗口除数精确, 全部 shape 计算为动态算子。
替换后 11 个测试长度 (50~1000 帧) 全部 max_diff ≈ 1e-5, cosine = 1.0。

## 流式说明

CAM++ 结尾是全局统计池化 (段级模型, 非因果), 无法做增量状态缓存式流式。
C# 侧采用业界标准的滑动窗口方案: 流式 fbank + 帧环形缓冲, 每 hop (默认 250ms)
对最近 window (默认 2s) 做整窗推理 (CPU ~10ms), 余弦打分 + 指数平滑。

## 对拍结果 (2026-07-12)

- ONNX vs torch (随机输入, 11 长度): max_diff ≤ 1.2e-5, cos = 1.0
- 官方 pipeline vs ONNX (示例 wav): 得分完全一致 (0.6936 / -0.0842)
- C# vs Python 黄金向量: fbank maxErr ~3e-4, embedding cos = 1.0, 得分一致到 1e-4
- 性能 (CPU): 2s 窗 ~11ms, 3s 窗 ~14ms, 10s 整段 ~31ms
