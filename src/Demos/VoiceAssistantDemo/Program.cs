using Microsoft.Extensions.Logging;
using VoiceAssistantDemo.Configuration;
using VoiceAssistantDemo.ConsoleUI;
using VoiceAssistantDemo.Logging;
using VoiceAssistantDemo.Presentation;
using VoiceAssistantDemo.Tools;
using VoiceBox.Assistant;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Diagnostics.Recording;
using VoiceBox.Assistant.Integrations.Mcp;
using VoiceBox.Assistant.Modules.Speaker.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Enrollment;
using VoiceBox.APM.WebRtc;

namespace VoiceAssistantDemo
{
    /// <summary>组装语音服务、日志和控制台交互，管理 Demo 生命周期。</summary>
    internal static class Program
    {
        static async Task Main()
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            // 后创建的资源先释放，日志工厂关闭后再收尾控制台缓存。
            using var display = new ConsoleDisplay(Console.Out, interactive: !Console.IsOutputRedirected);
            var input = new ConsoleInput(Console.In, interactive: !Console.IsInputRedirected);
            using var loggerFactory = DemoLogging.CreateLoggerFactory(
                DemoLoggingConfiguration.FromEnvironment(), display.WriteLog);

            display.WriteLines(string.Empty, "  正在唤醒语音助手, 首次启动需要加载模型, 请稍候...", string.Empty);

            //using var services = Plan01_CPU_ALL_LOCAL.Buid(loggerFactory);
            using var services = Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.Buid(loggerFactory);

            //tts
            var tts = services.Tts;
            var ttsVoiceName = tts.GetDefaultVoice()?.Name;
            var availableVoices = tts.ListVoices();
            display.WriteLines(
                $"  [TTS] 已加载, 输出 {tts.OutputSampleRate}Hz/{tts.OutputChannels}ch, 使用音色: {ttsVoiceName}",
                $"  [TTS] 可选音色: {string.Join(", ", availableVoices.Select(v => v.Name))}");

            // Confirmation.Enable 统一打开唤醒、播放期打断与单讲的本人确认。
            var speakerOptions = new VoiceAssistantSpeakerOptions
            {
                ProfilePath = Path.Combine("tmp", "owner-speaker.json"),
                Confirmation =
                {
                    Enable = true,
                    WindowCount = 3,
                    RequiredMatches = 2,
                    MinimumSpeechMs = 300,
                },
            };
            using var calibrationStimulus = new TtsCalibrationStimulusPlayer(
                services.Audio,
                tts,
                ttsVoiceName,
                AudioDefaults.TtsSpeed);
            if (speakerOptions.Confirmation.Enable)
            {
                // 注册使用与在线验证相同的 identity tap 和前端指纹，包含受控双讲校准。
                var enrollment = new SpeakerEnrollmentCoordinator(
                    services.Audio,
                    services.Vad,
                    services.Speaker,
                    speakerOptions,
                    new SpeakerEnrollmentOptions(),
                    calibrationStimulus,
                    services.AcousticDiagnostics,
                    loggerFactory.CreateLogger<SpeakerEnrollmentCoordinator>());
                var enrollmentInteraction = new SpeakerEnrollmentConsoleInteraction(display, input, AudioDefaults.WakeupWord);
                await enrollmentInteraction.EnsureRegisteredAsync(enrollment, calibrationStimulus);
            }

            var options = new VoiceAssistantOptions
            {
                TtsVoice = ttsVoiceName,
                TtsSpeed = AudioDefaults.TtsSpeed,
                SystemPrompt = AudioDefaults.DefaultAssistantPrompt,
                Conversation = new VoiceAssistantConversationOptions
                {
                    EnableThinking = AudioDefaults.IsEnableThinking,
                    ContinuousConversationEnabled = AudioDefaults.IsContinuousConversationEnabled,
                    ContinuousConversationIdleTimeout = TimeSpan.FromMinutes(1),
                },
                BargeIn = new VoiceAssistantBargeInOptions
                {
                    Enable = true,
                },
                // Idle/连续对话态的监听确认门槛: 过滤 TTS 回声残差 (实测 rms≈0.014) 与环境噪声
                // 触发的 VAD 误报, 否则每次误报都会显示"听到你说话了"并空跑一轮 ASR。
                // 这里写的就是默认值, 列出来便于按环境微调; 都设 0 可关闭门槛。
                Listening = new VoiceAssistantListeningOptions
                {
                    MinDurationMs = 200,
                    MinRms = 0.02f,
                },
                Wakeup = new VoiceAssistantWakeupOptions
                {
                    Enable = true,
                    PreRollMs = 2500,
                    // FSMN-KWS 的 StartTimeMs 是 CTC token 峰值附近的估计, 不是严格起音点;
                    // 多留一点回放余量, 避免唤醒后 ASR 偶发少掉"小木小木"开头的字。
                    ReplayLeadInMs = 400,
                },
                Speaker = speakerOptions,
                // Qwen3-TTS 输出 24kHz, 在此降采样到 16kHz 播放; 其流式块调度 [2,4,8,16,16,16]
                // 后段是大块, 生成耗时可能超过仅 250ms (默认) 的预缓冲领先量, 导致播放队列被抽干、
                // 回调插静音 → 听感上的"破音/咔哒"。实测 250/400ms 会欠载, 500ms 起稳定 (留约 200ms
                // 余量), 与独立 Qwen3TtsDemo 使用的 0.5s 预缓冲一致。播放仍有并发 ASR/LLM/barge-in
                // 抢 GPU, 故取 300ms 留足余量; 若仍偶发可再调大到 500。
                PlaybackPrerollMs = 300,
            };

            // 保存真正送入 ASR 的 pre-roll + 后续音频，便于回听双讲短句误识别。
            // 每个会话最多保留 60 秒；VOICEBOX_ASR_AUDIO_DIR=off 可关闭。
            string? asrAudioDirectory = Environment.GetEnvironmentVariable("VOICEBOX_ASR_AUDIO_DIR");
            using var asrRecording = string.Equals(asrAudioDirectory, "off", StringComparison.OrdinalIgnoreCase)
                ? null
                : new RecognitionAudioRecordingService(
                    services.Asr,
                    string.IsNullOrWhiteSpace(asrAudioDirectory) ? Path.Combine("tmp", "asr-audio") : asrAudioDirectory,
                    AudioDefaults.SampleRate,
                    AudioDefaults.ChannelCount,
                    loggerFactory.CreateLogger<RecognitionAudioRecordingService>());

            VoiceAssistantBuilder builder = VoiceAssistant.CreateBuilder()
                .UseOptions(options)
                .UseAudio(services.Audio)
                .UseTurnBoundary(services.Vad)
                .UseSpeechRecognition(asrRecording ?? services.Asr)
                .UseResponseGeneration(services.Llm)
                .UseMcpTools(mcp =>
                {
                    mcp.AddTool(new CurrentDateTimeTool());
                    mcp.AddTool(new CurrentWeatherTool());
                })
                .UseSpeechSynthesis(services.Tts)
                .AddWakeup(services.Wakeup)
                .AddSpeakerVerification(services.Speaker)
                .AddBargeIn(
                    options.BargeIn,
                    new WebRtcAec3DelayEstimatorFactory())
                .AddAcousticDiagnostics(services.AcousticDiagnostics);
            using var assistant = builder.BuildWithDiagnostics(
                loggerFactory.CreateLogger<VoiceAssistant>()).Assistant;

            using var presenter = new AssistantConsolePresenter(assistant, display, AudioDefaults.WakeupWord);
            try
            {
                await assistant.StartAsync();
                presenter.ShowReady();
                while (true)
                {
                    ConsoleKey key = await input.ReadKeyAsync();
                    if (key is ConsoleKey.Q or ConsoleKey.Escape)
                        break;
                }
            }
            finally
            {
                display.CompleteResponse(string.Empty);
                display.WriteLines(string.Empty, "  正在结束本次对话...");
                await assistant.StopAsync();
            }
            display.WriteLines(string.Empty, "  下次再聊, 再见。", string.Empty);
        }
    }
}
