# VoiceBox

[![.NET](https://img.shields.io/badge/.NET-10.0-512BD4)](https://dotnet.microsoft.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[简体中文](README.md) | **English**

## Introduction

VoiceBox is a modular voice interaction framework built with **.NET**, **covering nearly the entire pipeline from microphone capture to spoken response playback**. Components can be used independently or assembled as needed. With `VoiceBox.Assistant`, you can build anything from a minimal question-and-answer loop to a powerful voice assistant that supports natural, multi-turn conversations.

- **Comprehensive components**: Audio recording and playback, high-pass filtering, echo cancellation, noise reduction, voice activity detection, wake word detection, double-talk detection, speaker verification, ASR, LLM, and TTS.
- **Flexible composition**: `VoiceBox.Assistant` orchestrates the components. Start with a simple question-and-answer pipeline, then add wake word detection, continuous conversation, owner verification, voice interruption, and tool calling to build a personal voice assistant you can talk to at any time. Supports fluent registration through `AudioServiceManager` and .NET dependency injection.
- **Local or online operation**: Build a fully offline voice assistant with local models, or connect to online speech services (currently Volcengine Doubao, with support for extending to other providers) and OpenAI-compatible LLM services.
- **Streaming processing**: Real-time audio processing, streaming ASR, streaming LLM text output, and streaming TTS audio output.
- **Tool integration**: Assistants can call in-process tools and MCP services for voice queries and integration with business systems.

## Support Us

If you would like a custom wake word for your voice assistant, you can support the project's ongoing maintenance and development by ordering the **custom wake word training service**.

- **Try it online**: [Explore the wake word demo](https://kws.pub.quarkbook.com/) before you decide.
- **Training fee**: **CNY 300 per training session**.
- **Place an order**: [Wake word training service on Xianyu](https://m.tb.cn/h.j2Os9hw?tk=3LcHTEIx4nR).

The training code is also **fully open source**: [voicebox-kws-train-opensource](https://github.com/withsalt/voicebox-kws-train-opensource). You can train your own models or have me help with training. Thank you for supporting open source in your own way!

## Model Downloads

All models required by the project are available at **[ModelScope · withsalt/voicebox_models](https://www.modelscope.cn/models/withsalt/voicebox_models)**.

The full model repository is large, so download only the models needed by your selected modules. This code repository already includes some small models (such as VAD and DNR models) and native libraries. Prepare ASR, TTS, LLM, and other models according to the demo configuration. Model paths are typically specified through module options such as `ModelPath` and `ModelDir`.

To export models yourself, see the instructions in the corresponding directories under [scripts](scripts).

## Modules

The names below correspond to projects under `src`. Reference the ones you need.

| Category | Name | Description |
| --- | --- | --- |
| Core | `VoiceBox.Core` | Shared service interfaces and service management |
| Audio processing | `VoiceBox.Audio.Common` | PCM, WAV, and Opus reading, plus streaming resampling |
| Audio backend | `VoiceBox.Audio.PortAudio` | PortAudio-based recording, playback, and full-duplex components |
| Audio backend | `VoiceBox.Audio.MiniAudio` | miniaudio-based recording, playback, and full-duplex components |
| HPF (high-pass filtering) | `VoiceBox.HPF.Butterworth` | Butterworth high-pass filter for suppressing DC and low-frequency noise |
| APM (audio preprocessing) | `VoiceBox.APM.WebRtc` | WebRTC-based HPF, AEC3 (with neural network support), noise suppression, AGC2, and acoustic delay estimation |
| AEC (echo cancellation) | `VoiceBox.AEC.DFSMN` | Linear echo cancellation and DFSMN neural residual echo suppression |
| AEC (echo cancellation) | `VoiceBox.AEC.DTLN` | DTLN-aec two-stage streaming echo cancellation |
| DNR (noise reduction) | `VoiceBox.DNR.RNNoise` | RNNoise real-time noise reduction with built-in sample rate conversion |
| DNR (noise reduction) | `VoiceBox.DNR.DeepFilterNet` | DeepFilterNet3 low-latency streaming noise reduction |
| DNR (noise reduction) | `VoiceBox.DNR.DPDFNet` | DPDFNet streaming noise reduction with multiple model sizes |
| DNR (noise reduction) | `VoiceBox.DNR.Hush` | Hush / DeepFilterNet3-SE streaming noise reduction at 16 kHz |
| DTD (double-talk detection) | `VoiceBox.DTD.RenderAware` | Detects user speech during playback using microphone and playback reference signals; trained specifically for VoiceBox.Assistant |
| VAD (voice activity detection) | `VoiceBox.VAD.FireRed` | FireRed VAD with speech start/end events and speech probabilities |
| VAD (voice activity detection) | `VoiceBox.VAD.Silero` | Silero VAD for detecting speech segments and silence boundaries |
| KWS (keyword spotting) | `VoiceBox.Wakeup.FsmnKws` | FSMN + CTC streaming wake word detection; reads keywords from model metadata; the recommended backend for voice wake-up |
| KWS (keyword spotting) | `VoiceBox.Wakeup.SherpaOnnxKws` | Keyword spotting with sherpa-onnx Zipformer-Transducer models |
| KWS (keyword spotting) | `VoiceBox.Wakeup.OpenWakeWord` | Loads openWakeWord ONNX wake word models, with shared feature extraction across multiple models |
| ASR (speech recognition) | `VoiceBox.ASR.Qwen3` | Local streaming recognition with Qwen3-ASR |
| ASR (speech recognition) | `VoiceBox.ASR.Confucius4` | Local streaming recognition with Confucius4-R2T2 |
| ASR (speech recognition) | `VoiceBox.ASR.FireRedASR2` | Local recognition with FireRedASR2; streaming recognition is not supported |
| ASR (speech recognition) | `VoiceBox.ASR.DoubaoOnline` | Online streaming recognition through Volcengine Doubao speech services |
| Speaker verification | `VoiceBox.Speaker.CampPlus` | CAM++ speaker embedding extraction, enrollment, comparison, and sliding-window detection |
| Speaker verification | `VoiceBox.Speaker.WeSpeaker` | WeSpeaker CN-Celeb ResNet34 speaker embedding extraction and verification |
| LLM (language model) | `VoiceBox.LLM.OpenAI` | OpenAI-compatible API with streaming text and structured tool calling |
| LLM (language model) | `VoiceBox.LLM.LLamaSharp` | Local GGUF model inference through LLamaSharp / llama.cpp, with grammar-constrained structured tool calling |
| LLM (language model) | `VoiceBox.LLM.OnnxGenAI` | Local model inference through ONNX Runtime GenAI; tool calling is not supported |
| TTS (speech synthesis) | `VoiceBox.TTS.Qwen3` | Qwen3-TTS CustomVoice with preset voices and streaming synthesis |
| TTS (speech synthesis) | `VoiceBox.TTS.Kokoro` | Chinese and English synthesis with Kokoro, including text normalization and phoneme conversion |
| TTS (speech synthesis) | `VoiceBox.TTS.MossNano` | MOSS-TTS-Nano synthesis of full utterances and streaming synthesis |
| TTS (speech synthesis) | `VoiceBox.TTS.VoxCPM` | VoxCPM streaming synthesis with voice cloning from reference audio and text |
| TTS (speech synthesis) | `VoiceBox.TTS.DoubaoOnline` | Online streaming speech synthesis through Volcengine Doubao speech services |
| Assistant | `VoiceBox.Assistant` | Composable orchestration for minimal question-and-answer or multi-turn voice assistants, with wake word detection, speaker verification, and playback interruption |
| Assistant | `VoiceBox.Assistant.Integrations.Mcp` | Assistant tool calling with in-process tools, MCP stdio, Streamable HTTP, and MCP Tasks |

## VoiceBox.Assistant (Voice Assistant)

`VoiceBox.Assistant` is a composable voice assistant component that orchestrates conversations and manages conversation history, the playback queue, cancellation, and lifecycle. The same components support two configurations:

- **Minimal question-and-answer loop**: Connect only recording, playback, VAD, ASR, LLM, and TTS to let the user speak and receive a spoken response.
- **Natural multi-turn conversation**: Add wake word detection, speaker verification, voice interruption, and tool calling as needed. Conversation history and continuous conversation allow follow-up questions, and users can interrupt while a response is playing.

Model configuration:

All models required by the assistant demo can be downloaded from the model repository. Model paths are configured centrally in [AudioDefaults.cs](src/Demos/VoiceAssistantDemo/AudioDefaults.cs). Set `BASE_MODEL_DIR` to the directory that directly contains `kws`, `asr`, and `tts` (such as `v1` under your download directory), and verify that the models exist at the configured subpaths.

Basic question-and-answer pipeline:

```text
Microphone → VAD utterance segmentation → ASR → LLM → TTS → Speaker
```

Echo cancellation, noise reduction, and high-pass filtering can be added to the capture pipeline as needed. Audio can use a single full-duplex service or separate recording and playback services. The LLM and TTS use streaming interfaces. **The assistant currently obtains the complete ASR result at the end of each utterance before calling the LLM; response text is synthesized and played in segments.**

Register optional capabilities through the builder:

| Capability | Registration | Behavior and requirements |
| --- | --- | --- |
| Wake word detection and continuous conversation | `AddWakeup(wakeup)` | Starts a conversation after wake-up; supports multiple consecutive turns and returns to waiting for a wake word after an idle timeout or an end phrase |
| Speaker verification | `AddSpeakerVerification(speaker)` | Set `Speaker.Confirmation.Enable = true` to enable owner verification; speaker enrollment is required on first use |
| Interruption during playback | `AddBargeIn(options, delayEstimatorFactory)` | Can stop the current response when the user starts speaking; requires simultaneous capture and playback, a playback reference, and a processed capture timeline |
| Tool calling | `UseMcpTools(...)` | Connects local tools or MCP services; built-in backend support is provided by `VoiceBox.LLM.OpenAI` and `VoiceBox.LLM.LLamaSharp` |

Assemble a minimal question-and-answer assistant using existing service instances:

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

    assistant.UserSpoke += (_, e) => Console.WriteLine($"User: {e.Text}");
    assistant.AssistantText += (_, e) =>
    {
        if (!e.IsThinking) Console.Write(e.Delta);
    };

    await assistant.StartAsync();
    Console.ReadLine();
    await assistant.StopAsync();
}
```

See [VoiceAssistantDemo](src/Demos/VoiceAssistantDemo) for service creation and configuration. When using .NET Host, use the `VoiceBox.Assistant.Hosting` namespace in the assistant project to register dependency injection and hosted lifecycle management. See the [assistant architecture document](docs/voicebox-assistant-architecture.md) for the detailed design.

## Requirements

| Item | Requirement |
| --- | --- |
| Development environment | .NET 10 SDK; builds from the command line without a specific IDE |
| Operating system | Platform support depends on the selected backend; the repository includes some native dependencies for Windows x64 and Linux x64 / ARM64 |
| Audio devices | Real-time demos require a working microphone, speakers, and the corresponding device permissions |
| Local models | Prepare the models required by your selected modules; RAM, VRAM, and real-time performance depend on model size, precision, and hardware |
| Online services | Configure the corresponding API keys and model or resource IDs, and ensure network connectivity |
| Python / C++ | Required only when exporting models or building native dependencies yourself; follow the relevant script instructions |

Most ONNX modules reference only `Microsoft.ML.OnnxRuntime.Managed`. **Applications running these modules must also provide a native runtime**, with a version matching the Managed package:

| Inference backend | NuGet package required by the application | Additional requirements |
| --- | --- | --- |
| CPU | `Microsoft.ML.OnnxRuntime` | The model and its operators must support CPU execution |
| NVIDIA CUDA | `Microsoft.ML.OnnxRuntime.Gpu` | Compatible NVIDIA drivers, CUDA, and cuDNN |
| DirectML | `Microsoft.ML.OnnxRuntime.DirectML` | Windows and a compatible GPU driver |

The `ExecutionProvider` configuration must match the runtime. LLamaSharp also requires a `LLamaSharp.Backend.*` package. ONNX Runtime GenAI requires the corresponding native `Microsoft.ML.OnnxRuntimeGenAI` package. The demo `.csproj` files include examples of backend references.

## Quick Start

### 1. Get the source

```powershell
git clone https://github.com/withsalt/voicebox.git
cd voicebox
```

### 2. Run the recording demo first

This demo requires no models or API keys and helps check your microphone and audio environment. Run it from the repository root:

```powershell
dotnet run --project src/Demos/AudioRecordDemo -c Release -- recording.wav
```

The program lists input devices and uses the default microphone. Press Enter or Ctrl+C to stop and save a **16 kHz, mono, PCM16 WAV** file. To select a device, append its input device index after the filename.

### 3. Run the voice assistant

The current [Program.cs](src/Demos/VoiceAssistantDemo/Program.cs) selects **Plan04**: miniaudio + WebRTC APM + FireRed VAD + FSMN KWS + WeSpeaker, with **Confucius4 (CUDA) recognition, an online LLM, and VoxCPM (CPU) synthesis**. Owner verification, continuous conversation, voice interruption, and date/weather tools are enabled.

Before running:

1. Download the models listed above and update `BASE_MODEL_DIR` and the model paths in [AudioDefaults.cs](src/Demos/VoiceAssistantDemo/AudioDefaults.cs).
2. Set `LlmEndpoint` and `LlmModelName` in the same file. The default endpoint is Volcengine Ark; the API key must match the endpoint.
3. Prepare your CUDA/cuDNN environment. To change the ASR, TTS, or audio backend, update the service registrations in [Plan04](src/Demos/VoiceAssistantDemo/Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.cs) and the project's runtime references.

Run the following PowerShell commands from the repository root. **Enter the demo directory first** so that the relative paths to the `./voices/*.wav` voice reference files resolve correctly:

```powershell
$env:OPENAI_API_KEY = "your service API key"
cd src/Demos/VoiceAssistantDemo
dotnet run -c Release
```

On first launch, follow the prompts to enroll your voice. Then say “**小云小云**” (Mandarin: “Xiaoyun Xiaoyun”) to wake the assistant. Press Q or Esc to exit. If you replace the FSMN model, its `kws_meta.json` determines the actual wake phrase. Changing `WakeupWord` alone only changes the demo prompt and the assistant's name.

The speaker profile is saved to `tmp/owner-speaker.json` in the current directory. By default, the demo saves audio sent to ASR in `tmp/asr-audio/`. Set `$env:VOICEBOX_ASR_AUDIO_DIR = "off"` before starting to disable this.

The demo also provides the following configurations. Switch between them using `Plan*.Buid(...)` in `Program.cs`:

| Configuration | ASR | LLM | TTS | Notes |
| --- | --- | --- | --- | --- |
| [Plan01: Local CPU](src/Demos/VoiceAssistantDemo/Plan01_CPU_ALL_LOCAL.cs) | FireRedASR2 | ONNX Runtime GenAI | VoxCPM | Fully local CPU inference; requires a powerful CPU |
| [Plan02: Online services](src/Demos/VoiceAssistantDemo/Plan02_CPU_ALL_ONLINE.cs) | Doubao online | OpenAI-compatible service | Doubao online | Recommended for less powerful devices; Volcengine's generous free quotas can make personal usage nearly free |
| [Plan03: Local GPU](src/Demos/VoiceAssistantDemo/Plan03_GPU_16GVRAM_ALL_LOCAL.cs) | Confucius4 | LLamaSharp | VoxCPM | Fully local GPU inference (CUDA); requires at least 16 GB of VRAM and provides a very good user experience |
| [Plan04: Hybrid local and online (default)](src/Demos/VoiceAssistantDemo/Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.cs) | Confucius4 / CUDA | OpenAI-compatible service | VoxCPM / CPU | GPU inference for ASR and CPU inference for TTS; requires at least 8 GB of VRAM and also provides a good user experience |

When using `VoiceBox.LLM.OnnxGenAI` for local inference, remove `UseMcpTools(...)` from `Program.cs` because tool calling is not supported. `VoiceBox.LLM.LLamaSharp` supports structured tool calling through llama.cpp grammars. Both require the appropriate models and native inference packages.

For local LLM inference, the recommended approach is to serve a model with ollama or llama.cpp and connect to it through `VoiceBox.LLM.OpenAI`.

Plan02 requires `VOLC_ASR_API_KEY`, `OPENAI_API_KEY`, and `VOLC_TTS_API_KEY`. Audio preprocessing, VAD, wake word detection, and speaker verification still run locally. The VRAM sizes in the configuration filenames are labels for those configurations; measure actual usage with your selected models.

## Demos

Examples are located in [src/Demos](src/Demos). In addition to the recording and assistant demos above, choose a module demo from the table below. Model paths, devices, and execution backends are configured in each demo's source code.

| Category | Examples | Description |
| --- | --- | --- |
| Audio | [AudioPlayDemo](src/Demos/AudioPlayDemo), [AudioRecordDemo](src/Demos/AudioRecordDemo) | Play WAV / PCM / Opus samples included in the repository, or record microphone audio |
| Echo cancellation and preprocessing | [WebRtcApmLiveDemo](src/Demos/WebRtcApmLiveDemo), [DTLNAecDemo](src/Demos/DTLNAecDemo) | Real-time APM processing and DTLN echo cancellation |
| Noise reduction | [DeepFilterNetLiveDemo](src/Demos/DeepFilterNetLiveDemo), [DPDFNetLiveDemo](src/Demos/DPDFNetLiveDemo), [HushDnrLiveDemo](src/Demos/HushDnrLiveDemo) | Real-time microphone noise reduction |
| Wake word detection | [FsmnKwsDemo](src/Demos/FsmnKwsDemo), [SherpaOnnxKwsDemo](src/Demos/SherpaOnnxKwsDemo), [WakeWordDemo](src/Demos/WakeWordDemo) | Real-time detection with three wake word backends |
| Speech recognition | [Qwen3ASRDemo](src/Demos/Qwen3ASRDemo), [Confucius4AsrDemo](src/Demos/Confucius4AsrDemo), [FireRedASRDemo](src/Demos/FireRedASRDemo), [DoubaoOnlineAsrDemo](src/Demos/DoubaoOnlineAsrDemo) | Local or online speech recognition |
| Speaker verification | [CampPlusSpeakerLiveDemo](src/Demos/CampPlusSpeakerLiveDemo), [WeSpeakerSpeakerLiveDemo](src/Demos/WeSpeakerSpeakerLiveDemo) | Speaker enrollment and real-time detection |
| Language models | [OpenAILLMDemo](src/Demos/OpenAILLMDemo), [LLamaSharpLLMDemo](src/Demos/LLamaSharpLLMDemo), [OnnxGenAILLMDemo](src/Demos/OnnxGenAILLMDemo) | Online or local text generation and multi-turn conversations |
| Local speech synthesis | [Qwen3TtsDemo](src/Demos/Qwen3TtsDemo), [KokoroTtsDemo](src/Demos/KokoroTtsDemo), [MossNanoTtsDemo](src/Demos/MossNanoTtsDemo), [VoxCPMTtsDemo](src/Demos/VoxCPMTtsDemo) | Synthesis and playback with different TTS models |
| Online speech synthesis | [DoubaoOnlineTtsDemo](src/Demos/DoubaoOnlineTtsDemo) | Doubao online speech synthesis |
| Complete assistant | [VoiceAssistantDemo](src/Demos/VoiceAssistantDemo) | Wake word detection, voice question-and-answer, speaker verification, interruption, and tool calling |

## Related Documentation

Some linked documents are in Chinese.

- [Assistant architecture and capability combinations](docs/voicebox-assistant-architecture.md)
- [MCP and tool integration](src/VoiceBox.Assistant.Integrations.Mcp/README.md)
- [OpenAI-compatible LLM](src/VoiceBox.LLM.OpenAI/README.md) · [ONNX Runtime GenAI](src/VoiceBox.LLM.OnnxGenAI/README.md)
- [Confucius4 ASR](src/VoiceBox.ASR.Confucius4/README.md) · [VoxCPM TTS](src/VoiceBox.TTS.VoxCPM/README.md) · [WeSpeaker speaker verification](src/VoiceBox.Speaker.WeSpeaker/README.md)
- [Double-talk detection](src/VoiceBox.DTD.RenderAware/README.md) · [AEC / interruption benchmarks](src/Tests/VoiceBox.Assistant.AecChallenge.Benchmark/README.md) · [Noise reduction benchmarks](src/Tests/VoiceBox.DNR.Benchmark/README.md)
- [WebRTC APM native build](native/webrtc-apm/README.md) · [WeTextProcessing native build](native/wetext-processor/README.md) · [Jetson ONNX Runtime build](docs/jetson-onnxruntime-1.29.0-build.md)
- [Model export and validation scripts](scripts) · [Tests and benchmarks](src/Tests)

## Acknowledgments

This project uses or draws on the following projects, models, and SDKs. Thanks to their authors and communities:

| Purpose | Projects |
| --- | --- |
| Audio capture and playback | [PortAudio](https://github.com/PortAudio/portaudio), [miniaudio](https://github.com/mackron/miniaudio), [MiniAudioEx](https://www.nuget.org/packages/JAJ.Packages.MiniAudioEx) |
| Audio coding and resampling | [Concentus](https://github.com/lostromb/concentus), [SpeexDSP](https://github.com/xiph/speexdsp) |
| Audio preprocessing and echo cancellation | [WebRTC](https://webrtc.googlesource.com/src/), [DFSMN AEC](https://www.modelscope.cn/models/iic/speech_dfsmn_aec_psm_16k), [DTLN-aec](https://github.com/breizhn/DTLN-aec) |
| Noise reduction | [RNNoise](https://github.com/xiph/rnnoise), [RNNoise.NET](https://github.com/Yellow-Dog-Man/RNNoise.Net), [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet), [DPDFNet](https://github.com/ceva-ip/DPDFNet), [Hush](https://github.com/pulp-vision/Hush) |
| Voice activity detection | [FireRedVAD](https://github.com/FireRedTeam/FireRedVAD), [Silero VAD](https://github.com/snakers4/silero-vad) |
| Wake word detection | [FunASR](https://github.com/modelscope/FunASR), [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx), [openWakeWord](https://github.com/dscripka/openWakeWord) |
| Speech recognition | [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR), [Confucius4-R2T2](https://github.com/netease-youdao/Confucius4-R2T2), [FireRedASR2](https://github.com/FireRedTeam/FireRedASR2S), ONNX ASR implementations by [manyeyes](https://github.com/manyeyes) |
| Speaker verification | [CAM++ / 3D-Speaker](https://github.com/modelscope/3D-Speaker), [WeSpeaker](https://github.com/wenet-e2e/wespeaker) |
| Speech synthesis | [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS), [Kokoro](https://huggingface.co/hexgrad/Kokoro-82M), [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx), [MOSS-TTS-Nano](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Nano-100M-ONNX), [VoxCPM](https://github.com/OpenBMB/VoxCPM) |
| Text and phoneme processing | [Misaki](https://github.com/hexgrad/misaki), [WeTextProcessing](https://github.com/wenet-e2e/WeTextProcessing), [DeepPhonemizer](https://github.com/as-ideas/DeepPhonemizer) |
| Model inference | [ONNX Runtime](https://github.com/microsoft/onnxruntime), [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai), [LLamaSharp](https://github.com/SciSharp/LLamaSharp), [llama.cpp](https://github.com/ggml-org/llama.cpp) |
| Online models and tool protocols | [OpenAI .NET SDK](https://github.com/openai/openai-dotnet), [MCP C# SDK](https://github.com/modelcontextprotocol/csharp-sdk) |

## License

VoiceBox code is licensed under the [MIT License](LICENSE). Third-party code, native libraries, model weights, and voice assets are subject to their respective licenses. Review the license terms of the relevant projects and model packages when using or distributing them.
