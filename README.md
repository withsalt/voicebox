# VoiceBox

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

**简体中文** | [English](README.en.md)

## 简介

VoiceBox 是基于 **.NET** 开发的模块化语音交互框架，**基本覆盖从麦克风采集到语音回复播放的完整处理链路**。各组件既可独立使用，也可按需组装：通过 `VoiceBox.Assistant` ，即可构建最小的一问一答，亦可支持自由多轮对话的强大语音助理。

- **组件丰富**：提供录音、播放、高通滤波、回声消除、降噪、语音活动检测、唤醒、双讲检测、说话人验证，以及 ASR、LLM、TTS 组件。
- **按需组装**：通过 `VoiceBox.Assistant` 统一编排，基础组件即可组成简单一问一答链路，按需加入唤醒、连续对话、本人确认、语音打断和工具调用后，又可以构建为强大的、支持任意时刻自由对话的私人语音助理。支持 `AudioServiceManager` 链式注册和 .NET 依赖注入。
- **本地或在线运行**：可使用本地模型构建完全离线运行的语音助手，也可接入在线（目前支持火山引擎豆包，可自行扩展其它云厂商）语音服务和 OpenAI 兼容的 LLM 服务。
- **流式处理**：提供实时音频处理、流式 ASR、LLM 文本输出和 TTS 音频输出。
- **工具扩展**：助手可调用进程内工具及 MCP 服务，适合语音查询与业务系统集成。

## 支持我们

如果您希望为语音助手定制一个专属唤醒词，也欢迎选择 **定制专属唤醒词训练服务**，以此支持项目的持续维护与改进。

- **在线体验**：[点击体验唤醒词效果](https://kws.pub.quarkbook.com/)，先感受，再选择。
- **训练费用**：**300 元 / 次**。
- **下单入口**：[闲鱼 · 唤醒词训练服务](https://m.tb.cn/h.j2Os9hw?tk=3LcHTEIx4nR)。

唤醒词训练代码也已 **完全开源**：[voicebox-kws-train-opensource](https://github.com/withsalt/voicebox-kws-train-opensource)。你可以自行训练，也可以选择由我们协助完成。感谢你以自己的方式支持开源！

## 模型下载

项目所需全部模型下载地址：**[ModelScope · withsalt/voicebox_models](https://www.modelscope.cn/models/withsalt/voicebox_models)**。  

整个模型仓库较大，建议按所选模块按需下载对应模型。当前仓库已包含部分小型模型（比如说VAD、DNR等）与原生库；ASR、TTS、LLM 等模型需按 Demo 配置准备。模型路径一般由各模块的 `ModelPath`、`ModelDir` 等选项指定。

需要自行导出模型时，参考 [scripts](scripts) 中对应目录的说明。

## 模块

下列名称对应 `src` 下的项目，可按需引用。

| 分类 | 名称 | 说明 |
| --- | --- | --- |
| 核心 | `VoiceBox.Core` | 公共服务接口、服务管理 |
| 音频处理 | `VoiceBox.Audio.Common` | PCM、WAV、Opus 读取与流式重采样 |
| 音频后端 | `VoiceBox.Audio.PortAudio` | 基于 PortAudio 的录音、播放、全双工组件 |
| 音频后端 | `VoiceBox.Audio.MiniAudio` | 基于 miniaudio 的录音、播放、全双工组件 |
| HPF（高通滤波） | `VoiceBox.HPF.Butterworth` | Butterworth 高通滤波，抑制直流与低频噪声 |
| APM（音频前处理） | `VoiceBox.APM.WebRtc` | 基于WebRTC的HPF、AEC3（支持神经网络）、噪声抑制、AGC2 与声学延迟估计 |
| AEC（回声消除） | `VoiceBox.AEC.DFSMN` | 线性回声消除与 DFSMN 神经网络残余回声抑制 |
| AEC（回声消除） | `VoiceBox.AEC.DTLN` | DTLN-aec 双阶段流式回声消除 |
| DNR（降噪） | `VoiceBox.DNR.RNNoise` | RNNoise 实时降噪，内置采样率转换 |
| DNR（降噪） | `VoiceBox.DNR.DeepFilterNet` | DeepFilterNet3 低延迟流式降噪 |
| DNR（降噪） | `VoiceBox.DNR.DPDFNet` | DPDFNet 流式降噪，提供多种模型规模 |
| DNR（降噪） | `VoiceBox.DNR.Hush` | Hush / DeepFilterNet3-SE 16 kHz 流式降噪 |
| DTD（双讲检测） | `VoiceBox.DTD.RenderAware` | 结合麦克风与播放参考信号，检测播放期间的用户语音，专为VoiceBox.Assistant训练 |
| VAD（语音活动检测） | `VoiceBox.VAD.FireRed` | FireRed VAD，输出语音起止事件与语音概率 |
| VAD（语音活动检测） | `VoiceBox.VAD.Silero` | Silero VAD，检测语音片段与静音边界 |
| KWS（唤醒词检测） | `VoiceBox.Wakeup.FsmnKws` | FSMN + CTC 流式唤醒，从模型元数据读取关键词，语音唤醒只推荐用这个 |
| KWS（唤醒词检测） | `VoiceBox.Wakeup.SherpaOnnxKws` | 基于 sherpa-onnx Zipformer-Transducer 模型的关键词检测 |
| KWS（唤醒词检测） | `VoiceBox.Wakeup.OpenWakeWord` | 加载 openWakeWord ONNX 唤醒模型，支持多模型共用特征提取 |
| ASR（语音识别） | `VoiceBox.ASR.Qwen3` | Qwen3-ASR 本地流式识别 |
| ASR（语音识别） | `VoiceBox.ASR.Confucius4` | Confucius4-R2T2 本地流式识别 |
| ASR（语音识别） | `VoiceBox.ASR.FireRedASR2` | FireRedASR2 本地识别，不支持流式识别 |
| ASR（语音识别） | `VoiceBox.ASR.DoubaoOnline` | 在线流式识别，使用火山引擎豆包语音服务 |
| 说话人验证 | `VoiceBox.Speaker.CampPlus` | CAM++ 声纹提取、注册、比对与滑动窗口检测 |
| 说话人验证 | `VoiceBox.Speaker.WeSpeaker` | WeSpeaker CN-Celeb ResNet34 声纹提取与验证 |
| LLM（语言模型） | `VoiceBox.LLM.OpenAI` | OpenAI 兼容接口，支持流式文本与结构化工具调用 |
| LLM（语言模型） | `VoiceBox.LLM.LLamaSharp` | 基于 LLamaSharp / llama.cpp 的本地 GGUF 模型推理，支持 grammar 约束的结构化工具调用 |
| LLM（语言模型） | `VoiceBox.LLM.OnnxGenAI` | 基于 ONNX Runtime GenAI 的本地模型推理，不支持工具调用 |
| TTS（语音合成） | `VoiceBox.TTS.Qwen3` | Qwen3-TTS CustomVoice，支持预置音色与流式合成 |
| TTS（语音合成） | `VoiceBox.TTS.Kokoro` | Kokoro 中英文合成，包含文本规范化与音素转换 |
| TTS（语音合成） | `VoiceBox.TTS.MossNano` | MOSS-TTS-Nano 整段与流式合成 |
| TTS（语音合成） | `VoiceBox.TTS.VoxCPM` | VoxCPM 流式合成，支持参考音频与文本进行音色克隆 |
| TTS（语音合成） | `VoiceBox.TTS.DoubaoOnline` | 在线流式语音合成，使用火山引擎豆包语音服务 |
| 助手 | `VoiceBox.Assistant` | 按需编排最小问答或多轮语音助理，支持唤醒、说话人验证与播放打断 |
| 助手 | `VoiceBox.Assistant.Integrations.Mcp` | 助手工具调用，进程内工具、MCP stdio、Streamable HTTP 与 MCP Tasks |

## VoiceBox.Assistant（语音助理）

`VoiceBox.Assistant` 是可按需组装的语音助理组件，负责对话编排，并管理对话历史、播放队列、取消与生命周期。同一套组件可用于两种组合：

- **最小一问一答**：只接入录音、播放、VAD、ASR、LLM 和 TTS，完成“用户说一句，助手答一句”。
- **自由多轮对话**：在基础链路上按需加入唤醒、说话人验证、语音打断与工具调用，结合对话历史和连续对话，让用户连续提问、追问，也能在回复播放期间开口打断。

模型配置：  

助手 Demo 所需模型均可在模型仓库下载。模型路径集中在 [AudioDefaults.cs](src/Demos/VoiceAssistantDemo/AudioDefaults.cs) 中配置：将 `BASE_MODEL_DIR` 指向直接包含 `kws`、`asr`、`tts` 的目录（如下载目录下的 `v1`），并检查各子路径模型是否存在。


基础问答链路：

```text
麦克风 → VAD 分句 → ASR → LLM → TTS → 扬声器
```

回声消除、降噪和高通滤波可按需接入采集端。音频可使用一个全双工服务，也可分别提供录音和播放服务；LLM 与 TTS 使用流式接口。**当前助手在句末取得 ASR 的整句结果，再调用 LLM；回复文本分段合成并播放。**

可选能力通过 Builder 注册：

| 能力 | 注册方式 | 行为与要求 |
| --- | --- | --- |
| 唤醒与连续对话 | `AddWakeup(wakeup)` | 唤醒后开始对话；可连续多轮交流，通过空闲超时或结束词回到待唤醒状态 |
| 说话人验证 | `AddSpeakerVerification(speaker)` | 配合 `Speaker.Confirmation.Enable = true` 启用本人确认；首次使用需注册声纹 |
| 播放期间打断 | `AddBargeIn(options, delayEstimatorFactory)` | 用户开口时可停止当前回复；要求同时采播、播放参考与处理后采集时间线 |
| 工具调用 | `UseMcpTools(...)` | 接入本地工具或 MCP 服务；内置支持的后端为 `VoiceBox.LLM.OpenAI` 与 `VoiceBox.LLM.LLamaSharp` |

使用已创建的服务实例装配最小问答助手：

```csharp
using VoiceBox.Assistant;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.LLMService;
using VoiceBox.Core.Services.TTSService;
using VoiceBox.Core.Services.VadService;

static async Task RunAssistantAsync(
    IAudioDuplexService audio,
    IVoiceActivityDetectService vad,
    IASRService asr,
    IStreamLLMService llm,
    IStreamTTSService tts)
{
    using var assistant = VoiceAssistant.CreateBuilder()
        .UseAudio(audio)
        .UseTurnBoundary(vad)
        .UseSpeechRecognition(asr)
        .UseResponseGeneration(llm)
        .UseSpeechSynthesis(tts)
        .Build();

    assistant.UserSpoke += (_, e) => Console.WriteLine($"用户：{e.Text}");
    assistant.AssistantText += (_, e) =>
    {
        if (!e.IsThinking) Console.Write(e.Delta);
    };

    await assistant.StartAsync();
    Console.ReadLine();
    await assistant.StopAsync();
}
```

服务的创建与配置见 [VoiceAssistantDemo](src/Demos/VoiceAssistantDemo)。使用 .NET Host 时，可通过助手项目内的 `VoiceBox.Assistant.Hosting` 命名空间注册 DI 与托管生命周期；详细设计见 [助手架构文档](docs/voicebox-assistant-architecture.md)。

## 环境要求

| 项目 | 要求 |
| --- | --- |
| 开发环境 | .NET 10 SDK；命令行即可构建，无需特定 IDE |
| 操作系统 | 按后端选择平台；仓库提供 Windows x64、Linux x64 / ARM64 的部分原生依赖 |
| 音频设备 | 实时 Demo 需要可用的麦克风、扬声器及相应设备权限 |
| 本地模型 | 准备所选模块需要的模型；内存、显存和实时速度取决于模型规模、精度与硬件 |
| 在线服务 | 配置对应服务的 API Key、模型或资源 ID，并确保网络可达 |
| Python / C++ | 仅自行导出模型或构建原生依赖时需要，按对应脚本说明安装 |

多数 ONNX 模块仅引用 `Microsoft.ML.OnnxRuntime.Managed`，**运行它们的应用还需提供原生运行时**，版本应与 Managed 包匹配：

| 推理后端 | 应用需引用的 NuGet 包 | 额外要求 |
| --- | --- | --- |
| CPU | `Microsoft.ML.OnnxRuntime` | 模型及算子支持 CPU |
| NVIDIA CUDA | `Microsoft.ML.OnnxRuntime.Gpu` | 匹配的 NVIDIA 驱动、CUDA 与 cuDNN |
| DirectML | `Microsoft.ML.OnnxRuntime.DirectML` | Windows 与兼容的 GPU 驱动 |

`ExecutionProvider` 配置需与运行时匹配。LLamaSharp 另需选择 `LLamaSharp.Backend.*`；ONNX Runtime GenAI 另需选择对应的 `Microsoft.ML.OnnxRuntimeGenAI` 原生包。各 Demo 的 `.csproj` 已给出后端引用示例。

## 快速开始

### 1. 获取源码

```powershell
git clone https://github.com/withsalt/voicebox.git
cd voicebox
```

### 2. 先运行录音 Demo

此示例无需模型或 API Key，用于检查麦克风与音频环境。在仓库根目录执行：

```powershell
dotnet run --project src/Demos/AudioRecordDemo -c Release -- recording.wav
```

程序列出输入设备并使用默认麦克风，按 Enter 或 Ctrl+C 停止，保存为 **16 kHz、单声道、PCM16 WAV**。如需指定设备，在文件名后追加输入设备编号。

### 3. 运行语音助手

当前 [Program.cs](src/Demos/VoiceAssistantDemo/Program.cs) 选择 **Plan04**：miniaudio + WebRTC APM + FireRed VAD + FSMN KWS + WeSpeaker，使用 **Confucius4（CUDA）识别、在线 LLM、VoxCPM（CPU）合成**，并启用本人确认、连续对话、语音打断及日期/天气工具。

运行前完成以下配置：

1. 下载上文列出的模型，修改 [AudioDefaults.cs](src/Demos/VoiceAssistantDemo/AudioDefaults.cs) 的 `BASE_MODEL_DIR` 和各模型路径。
2. 在同一文件中设置 `LlmEndpoint`、`LlmModelName`；默认指向火山方舟，API Key 必须与端点对应。
3. 准备 CUDA/cuDNN 环境。需要更换 ASR、TTS 或音频后端时，修改 [Plan04](src/Demos/VoiceAssistantDemo/Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.cs) 中的服务注册及项目运行时引用。

以下为 PowerShell 命令，从仓库根目录执行。**先进入 Demo 目录**，使 `./voices/*.wav` 音色参考文件的相对路径可用：

```powershell
$env:OPENAI_API_KEY = "你的服务 API Key"
cd src/Demos/VoiceAssistantDemo
dotnet run -c Release
```

首次启动按提示注册本人声纹；完成后说“**小云小云**”唤醒，按 Q 或 Esc 退出。更换 FSMN 模型后，实际唤醒词以其 `kws_meta.json` 为准，单独修改 `WakeupWord` 只会改变 Demo 提示与助手名称。

声纹保存在当前目录的 `tmp/owner-speaker.json`。Demo 默认将送入 ASR 的音频保存到 `tmp/asr-audio/`，可在启动前设置 `$env:VOICEBOX_ASR_AUDIO_DIR = "off"` 关闭。

Demo 还提供以下组合，在 `Program.cs` 中切换 `Plan*.Buid(...)`：

| 方案 | ASR | LLM | TTS | 备注 |
| --- | --- | --- | --- | --- |
| [Plan01：CPU 本地](src/Demos/VoiceAssistantDemo/Plan01_CPU_ALL_LOCAL.cs) | FireRedASR2 | ONNX Runtime GenAI | VoxCPM | 使用CPU推理，完全本地化运行，需要较强的CPU  |
| [Plan02：在线服务](src/Demos/VoiceAssistantDemo/Plan02_CPU_ALL_ONLINE.cs) | 豆包在线 | OpenAI 兼容服务 | 豆包在线 | 使用在线服务，低性能设备首选（火山引擎免费额度很多，个人用量几乎免费）  |
| [Plan03：GPU 本地](src/Demos/VoiceAssistantDemo/Plan03_GPU_16GVRAM_ALL_LOCAL.cs) | Confucius4 | LLamaSharp | VoxCPM |  使用GPU推理（CUDA），完全本地化运行，至少需要16G显存，体验非常好  |
| [Plan04：本地与在线混合（默认）](src/Demos/VoiceAssistantDemo/Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.cs) | Confucius4 / CUDA | OpenAI 兼容服务 | VoxCPM / CPU |  ASR使用GPU推理，TTS使用CPU，至少需要8G显存，体验也很棒 |

使用`VoiceBox.LLM.OnnxGenAI`进行本地推理时，因为不支持工具调用，需移除 `Program.cs` 中的 `UseMcpTools(...)`；`VoiceBox.LLM.LLamaSharp`支持基于 llama.cpp grammar 的结构化调用。两者都需配置对应的模型及原生推理包；

LLM本地推理建议使用ollama或llama.cpp等本地部署模型，然后通过`VoiceBox.LLM.OpenAI`接入。  

Plan02 需要 `VOLC_ASR_API_KEY`、`OPENAI_API_KEY`、`VOLC_TTS_API_KEY`，音频前处理、VAD、唤醒及声纹仍在本地运行。方案文件名中的显存大小仅为方案命名，实际占用请按所选模型实测。

## Demo

示例位于 [src/Demos](src/Demos)。除上面的录音与助手示例外，可按下表选择模块示例；模型路径、设备和执行后端在各 Demo 源码中配置。

| 分类 | 示例 | 内容 |
| --- | --- | --- |
| 音频 | [AudioPlayDemo](src/Demos/AudioPlayDemo)、[AudioRecordDemo](src/Demos/AudioRecordDemo) | 播放仓库内 WAV / PCM / Opus 样本，或录制麦克风 |
| 回声与前处理 | [WebRtcApmLiveDemo](src/Demos/WebRtcApmLiveDemo)、[DTLNAecDemo](src/Demos/DTLNAecDemo) | APM 实时处理与 DTLN 回声消除 |
| 降噪 | [DeepFilterNetLiveDemo](src/Demos/DeepFilterNetLiveDemo)、[DPDFNetLiveDemo](src/Demos/DPDFNetLiveDemo)、[HushDnrLiveDemo](src/Demos/HushDnrLiveDemo) | 麦克风实时降噪 |
| 唤醒 | [FsmnKwsDemo](src/Demos/FsmnKwsDemo)、[SherpaOnnxKwsDemo](src/Demos/SherpaOnnxKwsDemo)、[WakeWordDemo](src/Demos/WakeWordDemo) | 三种唤醒后端的实时检测 |
| 语音识别 | [Qwen3ASRDemo](src/Demos/Qwen3ASRDemo)、[Confucius4AsrDemo](src/Demos/Confucius4AsrDemo)、[FireRedASRDemo](src/Demos/FireRedASRDemo)、[DoubaoOnlineAsrDemo](src/Demos/DoubaoOnlineAsrDemo) | 本地或在线语音识别 |
| 说话人 | [CampPlusSpeakerLiveDemo](src/Demos/CampPlusSpeakerLiveDemo)、[WeSpeakerSpeakerLiveDemo](src/Demos/WeSpeakerSpeakerLiveDemo) | 声纹注册与实时检测 |
| 语言模型 | [OpenAILLMDemo](src/Demos/OpenAILLMDemo)、[LLamaSharpLLMDemo](src/Demos/LLamaSharpLLMDemo)、[OnnxGenAILLMDemo](src/Demos/OnnxGenAILLMDemo) | 在线或本地文本生成、多轮对话 |
| 本地语音合成 | [Qwen3TtsDemo](src/Demos/Qwen3TtsDemo)、[KokoroTtsDemo](src/Demos/KokoroTtsDemo)、[MossNanoTtsDemo](src/Demos/MossNanoTtsDemo)、[VoxCPMTtsDemo](src/Demos/VoxCPMTtsDemo) | 不同 TTS 模型的合成与播放 |
| 在线语音合成 | [DoubaoOnlineTtsDemo](src/Demos/DoubaoOnlineTtsDemo) | 豆包在线语音合成 |
| 完整助手 | [VoiceAssistantDemo](src/Demos/VoiceAssistantDemo) | 唤醒、语音问答、说话人验证、打断与工具调用 |

## 相关文档

- [助手架构与能力组合](docs/voicebox-assistant-architecture.md)
- [MCP 与工具接入](src/VoiceBox.Assistant.Integrations.Mcp/README.md)
- [OpenAI 兼容 LLM](src/VoiceBox.LLM.OpenAI/README.md) · [ONNX Runtime GenAI](src/VoiceBox.LLM.OnnxGenAI/README.md)
- [Confucius4 ASR](src/VoiceBox.ASR.Confucius4/README.md) · [VoxCPM TTS](src/VoiceBox.TTS.VoxCPM/README.md) · [WeSpeaker 声纹](src/VoiceBox.Speaker.WeSpeaker/README.md)
- [双讲检测](src/VoiceBox.DTD.RenderAware/README.md) · [AEC / 打断评测](src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark/README.md) · [降噪评测](src/Tests/VoiceBox.DNR.Benchmark/README.md)
- [WebRTC APM 原生构建](native/webrtc-apm/README.md) · [WeTextProcessing 原生构建](native/wetext-processor/README.md) · [Jetson ONNX Runtime 构建](docs/jetson-onnxruntime-1.29.0-build.md)
- [模型导出与验证脚本](scripts) · [测试与基准项目](src/Tests)

## 感谢

本项目使用或参考了以下项目、模型与 SDK，感谢其作者和社区：

| 用途 | 项目 |
| --- | --- |
| 音频采集与播放 | [PortAudio](https://github.com/PortAudio/portaudio)、[miniaudio](https://github.com/mackron/miniaudio)、[MiniAudioEx](https://www.nuget.org/packages/JAJ.Packages.MiniAudioEx) |
| 编解码与重采样 | [Concentus](https://github.com/lostromb/concentus)、[SpeexDSP](https://github.com/xiph/speexdsp) |
| 音频前处理与回声消除 | [WebRTC](https://webrtc.googlesource.com/src/)、[DFSMN AEC](https://www.modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k)、[DTLN-aec](https://github.com/breizhn/DTLN-aec) |
| 降噪 | [RNNoise](https://github.com/xiph/rnnoise)、[RNNoise.NET](https://github.com/Yellow-Dog-Man/RNNoise.Net)、[DeepFilterNet](https://github.com/Rikorose/DeepFilterNet)、[DPDFNet](https://github.com/ceva-ip/DPDFNet)、[Hush](https://github.com/pulp-vision/Hush) |
| 语音活动检测 | [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD)、[Silero VAD](https://github.com/snakers4/silero-vad) |
| 唤醒 | [FunASR](https://github.com/modelscope/FunASR)、[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx)、[openWakeWord](https://github.com/dscripka/openWakeWord) |
| 语音识别 | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR)、[Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2)、[FireRedASR2](https://github.com/FireRedTeam/FireRedASR2S)、[manyeyes](https://github.com/manyeyes) 的 ONNX ASR 实现 |
| 说话人验证 | [CAM++ / 3D-Speaker](https://github.com/modelscope/3D-Speaker)、[WeSpeaker](https://github.com/wenet-e2e/wespeaker) |
| 语音合成 | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)、[Kokoro](https://huggingface.co/hexgrad/Kokoro-82M)、[kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)、[MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX)、[VoxCPM](https://github.com/OpenBMB/VoxCPM) |
| 文本与音素处理 | [Misaki](https://github.com/hexgrad/misaki)、[WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing)、[DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) |
| 模型推理 | [ONNX Runtime](https://github.com/microsoft/onnxruntime)、[ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai)、[LLamaSharp](https://github.com/SciSharp/LLamaSharp)、[llama.cpp](https://github.com/ggml-org/llama.cpp) |
| 在线模型与工具协议 | [OpenAI .NET SDK](https://github.com/openai/openai-dotnet)、[MCP C# SDK](https://github.com/modelcontextprotocol/csharp-sdk) |

## 许可证

VoiceBox 代码采用 [Apache License 2.0](LICENSE)。第三方代码、原生库、模型权重与音色素材遵循各自的许可证；使用或分发时请同时查看对应项目及模型包的授权说明。
