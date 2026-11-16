# VoiceBox.TTS.VoxCPM

VoxCPM-0.5B(OpenBMB tokenizer-free TTS)的 ONNX 推理实现。16 kHz 单声道,每步产出
80 ms 音频 —— 天然低延迟流式;支持音色克隆(prompt 音频 + 文本)。

架构:自回归 LM(MiniCPM4 24L + residual 6L, KV cache)+ 局部流匹配扩散(10 步
Euler, CFG=2.0)+ AudioVAE。双层循环与判停在宿主,ONNX 只承载 5 个纯张量子图。

```
文本 → BPE(+多字中文拆单字) → prefill(lm.onnx)
  循环: cfm_decode(噪声宿主注入) → stop 头判停 → feat_encoder → lm 单步
  流式: 每步 3-patch 滑窗 audio_vae_decoder, 取尾 80ms
```

## 用法

```csharp
var manager = new AudioServiceManager();
manager.AddVoxCPMTTSService(o =>
{
    o.ModelPath = @"...\voxcpm-0.5b-onnx-int4";      // scripts/voxcpm-tts-onnx-export 导出
    o.ExecutionProvider = OnnxExecutionProvider.Cuda; // CPU 也可实时 (int4 RTF≈1.2)
    o.CloneVoices.Add(new VoxCpmCloneVoice
    {
        Name = "xiaomei",
        PromptWavPath = @"...\prompt.wav",            // 任意采样率, 自动重采样 16k
        PromptText = "prompt 音频对应的文本",
    });
});
var tts = manager.Build().GetStreamTTSServiceInstance();
await foreach (var chunk in tts.GetStreamAudioAsync("你好，世界。",
    new TtsSynthesisOptions { Voice = "xiaomei", Seed = 42 }))
{ /* 每块 1280 样本 (80ms) */ }
```

- 音色 `default` = 无 prompt(模型自由发挥;不固定 seed 时每次韵律不同)。
- `Seed` 固定时 (text, voice, seed) 决定输出(CFM 初始噪声确定)。
- 精度选择 = ModelPath 指向不同导出包:fp32(对拍基准)/ fp16 / int4(体积最小, CPU 最快)。

## 数值验证

fp32 包与 PyTorch 参考(相同噪声注入)e2e SNR 80~100dB;C# 与 python-ORT 参考引擎
逐位一致。导出与 golden 生成见 `scripts/voxcpm-tts-onnx-export/README.md`。
