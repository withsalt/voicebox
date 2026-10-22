# Render-aware DTD

这是一个独立的流式 near-end / double-talk detector。模型同时观察原始
`microphone` 和 far-end `render` PCM，每 20 ms 输出一个
`nearend_probability`；它不依赖 DFSMN mask、线性 AEC 输出或 VAD 伪标签。

当前版本只完成训练、固定测试集评估和 ONNX 导出，尚未接入
`VoiceBox.Assistant` 或 `VoiceBox-Assistant-Evaluation-v2`。

## 数据与音频契约

正式数据集位于：

`E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4`

构建数据集时不复制原始音频，只生成确定性 manifest：

- AEC Challenge narrow/fullband stem：near-end、far-end、echo、microphone；
- WenetSpeech4TTS：只作为在线合成的 far-end，避免绑定某个 TTS 组件；
- AISHELL/AISHELL3/THCHS30：near-end；
- DNS/MUSAN/BUT ReverbDB：噪声和 RIR；
- AEC real：real single-talk 监督和最终 unseen 测试。

运行时每个文件都从文件头读取采样率和声道，混合为 mono，并用
`scipy.signal.resample_poly` 重采样到 16 kHz。数据集包含 16/44.1/48/96 kHz
文件，空文件会被 manifest 显式记录并过滤。完整计数和音频契约见
`dataset_summary.json`。

## 模型契约

- sample rate：16 kHz，mono
- 输入：`microphone[1,640]`、`render[1,640]`
- 调用步长：320 samples，即 20 ms；相邻调用保留 320 samples overlap
- 状态：`state_in[3,1,288]`
- 输出：`nearend_probability[1,1,1]`、`state_out[3,1,288]`
- 结构：80 bands、projection 288、GRU hidden 288、3 layers
- 可训练参数：1,656,577
- 计算量：92.7304 MMAC/s

固定 complex filterbank 在 ONNX 图内，部署端不需要复刻 Python STFT。

## 训练

直接运行 `train_dtd.py` 的默认训练配方维护为 v6 candidate-01：

- 230 epochs（最大预算）、12,000 examples/epoch、8 s sequence、batch 128、seed 43；
- 288 projection/hidden、3 层 GRU、80 bands、dropout 0.1；
- learning rate 8e-4、5 epochs warmup、cosine decay、weight decay 2e-4；
- AEC narrow/fullband、Wenet、AEC real 权重为 40%/20%/20%/20%；
- hard-negative/onset/event/reset loss 为 0.15/0.10/0/0；
- reset probability 0.2、hard-far-only probability 0.5；
- `patience=30`、`min_delta=0.0002`、首次运行 6 workers（内存异常时恢复使用 4 workers）。

历史 `candidate-16.json` 保持冻结，不作为可变默认文件。`training_preset.py` 仍按显式
preset 配置运行；使用 preset 入口时输出目录必须显式指定，且不能是已有候选目录。

```powershell
E:\.packages\conda\envs\voicebox\python.exe `
  scripts\render-aware-dtd\train_dtd.py `
  --output-dir E:\Projects\render-aware-dtd\artifacts-v6\candidate-01
```

默认首次运行 6 workers；本轮实测 8 workers 会显著推高 Windows commit memory，恢复时
使用 4 workers。worker 数只影响数据加载，不改变模型或数据契约。训练入口会禁用当前环境中不兼容的
cuDNN 子库，使用 native CUDA kernels。

## 严格评估

旧模型只作为评测基线：

`scripts/render-aware-dtd/artifacts-enriched-v3/best.pt`

candidate-2 与基线在同一固定 manifest、同一随机种子和同一测试样本上比较。
AP、AUC、best F1，以及 0.1%、0.5%、1%、2%、5% FPR 下的 recall 全部要求
candidate 不低于 baseline。

```powershell
E:\.packages\conda\envs\voicebox\python.exe `
  scripts\render-aware-dtd\compare_dtd_models.py `
  scripts\render-aware-dtd\artifacts-enriched-v3\best.pt `
  E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\best.pt `
  --enriched-root E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4 `
  --output E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\comparison.json `
  --batch-size 128 `
  --num-workers 4
```

标准三套测试报告也保存在 `candidate-2/test_metrics.json`。

## ONNX 导出

```powershell
E:\.packages\conda\envs\voicebox\python.exe `
  scripts\render-aware-dtd\export_dtd_onnx.py `
  E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\best.pt `
  --output E:\Projects\render-aware-dtd\artifacts-v4\candidate-2\render_aware_dtd.onnx
```

导出同时产生 `.contract.json` 和 `.parity.json`。当前 parity 结果：

- probability max abs error：`7.45e-8`
- state max abs error：`1.61e-6`

checkpoint 的 `decision_threshold` 只是 validation 上的 frame 工作点。接入时应
保留连续 probability，并针对 VoiceBox 的 fusion/window/Heard 逻辑重新标定。

## 接入边界

训练语义中的 `microphone` 是 AEC 前原始麦克风混合信号，`render` 是与其对应的
far-end reference。当前 Assistant sidechain 使用的音频节点和新模型输入语义
尚未做适配，因此不能只替换 ONNX 文件。后续接入需要单独完成：

1. 选择与训练语义一致的 capture 节点；
2. 维护 640-sample frame、320-sample overlap 和 GRU state；
3. 处理 render 延迟对齐；
4. 保留现有 AEC、fusion、score decay、360 ms accumulator 和 Heard 控制流；
5. 用 Assistant benchmark 做端到端重新标定。

frame 指标不等价于 Heard 端到端指标。
