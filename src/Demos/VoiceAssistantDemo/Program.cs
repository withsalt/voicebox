using Microsoft.Extensions.Logging;
using VoiceAssistantDemo.Logger;
using VoiceAssistantDemo.Tools;
using VoiceBox.Assistant;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Composition.Builder;
using VoiceBox.Assistant.Integrations.Mcp;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.APM.WebRtc;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.Core.Services.AudioService;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.VadService;
using VoiceBox.Core.Onnx;
using System.Runtime.InteropServices;
using VoiceBox.Assistant.Modules.Speaker.Enrollment;
using VoiceBox.Assistant.Diagnostics.Recording;
using VoiceBox.Assistant.Modules.Speaker.Configuration;

namespace VoiceAssistantDemo
{
    /// <summary>
    /// 端到端的本地化语音助手 Demo: Mic → AEC/DNR → VAD → ASR (Qwen3) → LLM → TTS (Qwen3) → 播放。
    /// AEC/DNR 与可打断回复配套启用, 播放 TTS 期间仍可继续监听用户输入。
    /// </summary>
    internal static class Program
    {
        // ─── 控制台 UI 状态 (跨事件线程访问, 走 _consoleLock) ──────────────────
        private static readonly object _consoleLock = new();
        private static string _currentStatus = string.Empty;
        private static bool _streamingAssistant = true;
        private static bool _streamingThinking = true;

        // 助手文本流式输出期间, 低优先级日志 (Debug/Info) 先缓冲, 等这轮 "助手：" 行收尾再统一
        // 落盘, 避免一条 barge-in 调试日志把单条回复劈成两段 "助手："。Warning/Error 仍即时插入。
        private static readonly List<(LogLevel Level, string Ts, string Message)> _pendingLogs = new();

        static async Task Main()
        {
            if (OperatingSystem.IsWindows())
            {
                try { Console.OutputEncoding = System.Text.Encoding.UTF8; } catch { }
                try { Console.InputEncoding = System.Text.Encoding.UTF8; } catch { }
            }

            // 自定义日志 Provider: 走 Program 的 _consoleLock + 状态行逻辑, 让每条日志独占一行、
            // 打在"活动状态行"上方。默认的 AddSimpleConsole 从后台队列直写 stdout, 绕开 _consoleLock,
            // 会把 "22:42:25 info:..." 拼接到 UI 停在行首的状态行上 (如 "小云小云22:42:25 info:")。
            using var loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.SetMinimumLevel(LogLevel.Debug);
                builder.AddProvider(new ConsoleUiLoggerProvider(LogLevel.Debug, WriteLogLine));
            });

            Console.WriteLine();
            Console.WriteLine("  正在唤醒语音助手, 首次启动需要加载模型, 请稍候...");
            Console.WriteLine();

            //using var services = Plan01_CPU_ALL_LOCAL.Buid(loggerFactory);
            using var services = Plan04_MIX_8GVRAM_ASR_GPU_TTS_CPU_LLM_ONLINE.Buid(loggerFactory);

            //tts
            var tts = services.Tts;
            var ttsVoiceName = tts.GetDefaultVoice()?.Name;
            var availableVoices = tts.ListVoices();
            Console.WriteLine($"  [TTS] 已加载, 输出 {tts.OutputSampleRate}Hz/{tts.OutputChannels}ch, 使用音色: {ttsVoiceName}");
            Console.WriteLine($"  [TTS] 可选音色: {string.Join(", ", availableVoices.Select(v => v.Name))}");

            // Confirmation.Enable 统一打开唤醒、播放期打断与单讲的本人确认。
            var speakerOptions = new VoiceAssistantSpeakerOptions
            {
                ProfilePath = Path.Combine("tmp", "owner-speaker.json"),
                Confirmation =
                {
                    Enable = false,
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
            await EnsureOwnerSpeakerRegisteredAsync(
                services.Audio,
                services.Vad,
                services.Speaker,
                speakerOptions,
                calibrationStimulus,
                services.AcousticDiagnostics,
                loggerFactory);

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

            WireUpUi(assistant);

            await assistant.StartAsync();

            PrintBanner();
            SetStatus($"等待唤醒词: {AudioDefaults.WakeupWord}");

            while (true)
            {
                var key = Console.ReadKey(true);
                if (key.Key == ConsoleKey.Q || key.Key == ConsoleKey.Escape) break;
            }

            ClearStatusAndPrint("正在结束本次对话...");
            await assistant.StopAsync();
            Console.WriteLine();
            Console.WriteLine("  下次再聊, 再见。");
            Console.WriteLine();
        }

        private static async Task EnsureOwnerSpeakerRegisteredAsync(
            IAudioDuplexService audio,
            IVoiceActivityDetectService vad,
            ISpeakerService speaker,
            VoiceAssistantSpeakerOptions speakerOptions,
            TtsCalibrationStimulusPlayer calibrationStimulus,
            IAcousticProcessingMetricsSource? acousticDiagnostics,
            ILoggerFactory loggerFactory)
        {
            if (!speakerOptions.Confirmation.Enable)
            {
                return;
            }
            // 注册与在线验证使用同一条 identity tap (采集处理器/AEC 之后、独立 DNR 之前) 与同一
            // 前端指纹 (含 AddAcousticDiagnostics 注册的处理器身份); profile 为 owner-speaker-profile-v2,
            // 按前端指纹分 variant。播放期打断需要双讲 bank, 因此注册包含受控双讲段: 由 TTS 校准语音
            // 经真实播放链路播放。
            var enrollment = new SpeakerEnrollmentCoordinator(
                audio,
                vad,
                speaker,
                speakerOptions,
                new SpeakerEnrollmentOptions(),
                calibrationStimulus,
                acousticDiagnostics,
                loggerFactory.CreateLogger<SpeakerEnrollmentCoordinator>());
            if (enrollment.HasUsableProfile(requireDoubleTalkBank: true))
            {
                Console.WriteLine($"  [Speaker] 已加载本人声纹 (前端指纹 {enrollment.FrontendFingerprint}): {enrollment.ProfilePath}");
                Console.WriteLine();
                return;
            }

            enrollment.ProgressChanged += (_, e) =>
            {
                lock (_consoleLock)
                {
                    switch (e.Kind)
                    {
                        case SpeakerEnrollmentProgressKind.Collecting:
                            Console.Write(
                                $"\r  [Speaker] {DescribeStage(e.Stage)} 第 {e.SegmentIndex + 1}/{e.SegmentCount} 段, " +
                                $"有效语音 {e.Collected.TotalSeconds:F1}/{e.Required.TotalSeconds:F1} 秒   ");
                            break;
                        case SpeakerEnrollmentProgressKind.SegmentOutcome:
                            Console.WriteLine();
                            Console.WriteLine(e.ReasonCode == SpeakerEnrollmentReasonCode.SegmentAccepted
                                ? $"  [Speaker] 第 {e.SegmentIndex + 1} 段已接受 (rms={e.QualitySnapshot?.Rms:F3}, 语音占比={e.QualitySnapshot?.SpeechRatio:P0})。"
                                : $"  [Speaker] 第 {e.SegmentIndex + 1} 段未通过 ({DescribeReason(e.ReasonCode)}), 将重录该段。");
                            break;
                        case SpeakerEnrollmentProgressKind.StageChanged
                            when e.Stage is SpeakerEnrollmentStage.BuildingProfile or SpeakerEnrollmentStage.SavingProfile:
                            Console.WriteLine($"  [Speaker] {DescribeStage(e.Stage)}...");
                            break;
                        case SpeakerEnrollmentProgressKind.Resumed:
                            Console.WriteLine();
                            Console.WriteLine(
                                $"  [Speaker] 从{DescribeResume(e.Stage, e.SegmentIndex)}继续，已通过的段落会保留。");
                            break;
                    }
                }
            };

            bool calibrationPrepared = false;
            while (true)
            {
                try
                {
                    if (!calibrationPrepared)
                    {
                        Console.WriteLine();
                        Console.WriteLine("  [Speaker] 正在准备双讲校准语音（首次使用会加载 TTS 模型），请稍候...");
                        await calibrationStimulus.PrepareAsync(CancellationToken.None);
                        calibrationPrepared = true;
                    }

                    SpeakerEnrollmentResult result = await enrollment.EnrollAsync(
                        new ConsoleEnrollmentPacer(),
                        CancellationToken.None);
                    Console.WriteLine(
                        $"  [Speaker] 注册完成: 清晰段 {result.CleanSegmentsRetained} 段/{result.CleanPrototypeCount} 原型, " +
                        $"双讲段 {result.DoubleTalkSegmentsRetained} 段/{result.DoubleTalkPrototypeCount} 原型, " +
                        $"重录 {result.RetriedSegments} 次, 剔除离群 {result.DiscardedSegments} 段, " +
                        $"独立确认最低得分 {result.ConfirmationMinScore:F4} (阈值 {result.Threshold:F4})。缓存: {result.ProfilePath}");
                    Console.WriteLine();
                    return;
                }
                catch (Exception ex)
                {
                    Console.WriteLine();
                    Console.WriteLine($"  [Speaker] 注册失败：{ex.Message}");
                    if (enrollment.PendingResume is { } resume &&
                        (resume.AcceptedCleanSegments > 0 || resume.AcceptedDoubleTalkSegments > 0 ||
                         resume.Stage is not SpeakerEnrollmentStage.CleanEnrollment || resume.SegmentIndex > 0))
                    {
                        Console.WriteLine(
                            $"  按 R 从{DescribeResume(resume.Stage, resume.SegmentIndex)}继续" +
                            $"（已保留安静环境 {resume.AcceptedCleanSegments} 段、双讲 {resume.AcceptedDoubleTalkSegments} 段），" +
                            "按 N 重新开始，按 Q 或 Esc 退出。");
                    }
                    else
                    {
                        Console.WriteLine("  按 R 重试，按 N 重新开始，按 Q 或 Esc 退出。");
                    }

                    ConsoleKey key = Console.ReadKey(intercept: true).Key;
                    if (key is ConsoleKey.Q or ConsoleKey.Escape)
                        throw new OperationCanceledException("用户取消了本人声纹注册。", ex);
                    if (key == ConsoleKey.N)
                    {
                        enrollment.DiscardPendingEnrollment();
                        Console.WriteLine("  已放弃已录段落，将从头注册。");
                    }
                    else if (key != ConsoleKey.R)
                    {
                        Console.WriteLine("  未识别按键，将从失败处继续。");
                    }
                }
            }
        }

        private static string DescribeResume(SpeakerEnrollmentStage stage, int segmentIndex) => stage switch
        {
            SpeakerEnrollmentStage.CleanEnrollment => $"安静环境第 {segmentIndex + 1} 段",
            SpeakerEnrollmentStage.DoubleTalkEnrollment => $"双讲第 {segmentIndex + 1} 段",
            SpeakerEnrollmentStage.CleanConfirmation => "安静环境确认",
            SpeakerEnrollmentStage.DoubleTalkConfirmation => "双讲确认",
            SpeakerEnrollmentStage.BuildingProfile => "声纹构建",
            SpeakerEnrollmentStage.SavingProfile => "保存声纹",
            _ => DescribeStage(stage),
        };

        private static string DescribeStage(SpeakerEnrollmentStage stage) => stage switch
        {
            SpeakerEnrollmentStage.CleanEnrollment => "安静环境注册",
            SpeakerEnrollmentStage.DoubleTalkEnrollment => "双讲注册",
            SpeakerEnrollmentStage.CleanConfirmation => "安静环境确认",
            SpeakerEnrollmentStage.DoubleTalkConfirmation => "双讲确认",
            SpeakerEnrollmentStage.BuildingProfile => "构建声纹",
            SpeakerEnrollmentStage.SavingProfile => "保存声纹",
            _ => stage.ToString(),
        };

        private static string DescribeReason(SpeakerEnrollmentReasonCode reason) => reason switch
        {
            SpeakerEnrollmentReasonCode.LowEnergy => "音量过低",
            SpeakerEnrollmentReasonCode.Clipping => "音量过大/削波",
            SpeakerEnrollmentReasonCode.DcOffset => "直流偏置异常",
            SpeakerEnrollmentReasonCode.InsufficientSpeech => "有效语音占比不足",
            SpeakerEnrollmentReasonCode.TimelineGap => "采集时间轴中断",
            SpeakerEnrollmentReasonCode.SegmentTimeout => "超时未采到足够语音",
            SpeakerEnrollmentReasonCode.NoRenderDuringDoubleTalk => "双讲期间未检测到播放",
            SpeakerEnrollmentReasonCode.InsufficientNearEndDuringDoubleTalk => "双讲期间近端语音证据不足",
            SpeakerEnrollmentReasonCode.LowWithinSegmentConsistency => "本段声纹前后不一致",
            SpeakerEnrollmentReasonCode.InvalidEmbedding => "声纹提取失败",
            _ => reason.ToString(),
        };

        /// <summary>控制台节拍: 每段开始前提示朗读内容并等待用户按 S。</summary>
        private sealed class ConsoleEnrollmentPacer : ISpeakerEnrollmentPacer
        {
            public Task WaitForSegmentStartAsync(SpeakerEnrollmentProgressEventArgs segment, CancellationToken cancellationToken)
            {
                lock (_consoleLock)
                {
                    Console.WriteLine();
                    switch (segment.Stage)
                    {
                        case SpeakerEnrollmentStage.CleanEnrollment when segment.SegmentIndex == 0 && segment.Attempt == 1:
                            Console.WriteLine("  首次使用需要注册本人声纹, 分为安静环境段与双讲段, 每段约 2.5 秒有效语音。");
                            Console.WriteLine($"  安静环境段 {segment.SegmentIndex + 1}/{segment.SegmentCount}，请在安静环境下自然朗读: {EnrollmentSentences[segment.SegmentIndex % EnrollmentSentences.Length]}");
                            break;
                        case SpeakerEnrollmentStage.CleanEnrollment:
                            Console.WriteLine($"  安静环境段 {segment.SegmentIndex + 1}/{segment.SegmentCount}{(segment.Attempt > 1 ? $" (第 {segment.Attempt} 次)" : string.Empty)}，请自然朗读: {EnrollmentSentences[segment.SegmentIndex % EnrollmentSentences.Length]}");
                            break;
                        case SpeakerEnrollmentStage.DoubleTalkEnrollment:
                            Console.WriteLine($"  双讲段 {segment.SegmentIndex + 1}/{segment.SegmentCount}{(segment.Attempt > 1 ? $" (第 {segment.Attempt} 次)" : string.Empty)}，扬声器会同时播放校准语音, 请不要等它停, 直接朗读: {EnrollmentSentences[(segment.SegmentIndex + 2) % EnrollmentSentences.Length]}");
                            break;
                        case SpeakerEnrollmentStage.CleanConfirmation:
                            Console.WriteLine(
                                $"  安静环境确认，请完整朗读：“{AudioDefaults.WakeupWord}，我想出去玩，可以帮我查一下今天的天气怎么样吗”，用于独立确认刚才注册的声纹。");
                            break;
                        case SpeakerEnrollmentStage.DoubleTalkConfirmation:
                            Console.WriteLine($"  双讲确认，扬声器播放校准语音的同时, 请像打断助手一样说一句话。可以说（可以重复说）：{AudioDefaults.WakeupWord}，我听到你说话了，可以停一下吗");
                            break;
                    }
                    Console.WriteLine("  按 S 键开始收集...");
                }

                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    ConsoleKey key = Console.ReadKey(intercept: true).Key;
                    if (key == ConsoleKey.S)
                        return Task.CompletedTask;
                }
            }
        }

        private static readonly string[] EnrollmentSentences =
        [
            "清晨的阳光照进窗台，我端起一杯温水，准备开始今天的工作。",
            "周末我们打算去郊外走一走，顺便看看山里的枫叶红了没有。",
            "请把明天上午九点的会议提醒改到十点，并通知项目组的同事。",
            "厨房里的汤还在小火慢炖，锅盖边缘不时冒出一缕白色的蒸汽。",
            "这本书讲的是一位老木匠和他的孙女在小镇上度过的一个夏天。",
        ];

        // ──────────────────────────────────────────────────────────────────────
        //  UI 接线
        // ──────────────────────────────────────────────────────────────────────

        private static void WireUpUi(VoiceAssistant assistant)
        {
            assistant.StateChanged += (_, e) =>
            {
                switch (e.To)
                {
                    case AssistantState.Idle:
                        HandleEnterIdle(assistant);
                        break;
                    case AssistantState.Listening:
                        SetStatus("🔴 听到你说话了, 继续...");
                        break;
                    case AssistantState.Recognizing:
                        SetStatus("🔴 听到你说话了, 正在识别...可以继续说话哟~~~");
                        break;
                    case AssistantState.Thinking:
                        SetStatus("💭 组织语言中...");
                        break;
                    case AssistantState.Speaking:
                        // 流式对话内容本身就是状态指示, 不再单独显示状态。
                        SetStatus(string.Empty);
                        break;
                    case AssistantState.Stopped:
                        SetStatus(string.Empty);
                        break;
                }
            };

            assistant.WakeModeChanged += (_, e) =>
            {
                switch (e.To)
                {
                    case AssistantWakeMode.WaitingForWakeWord:
                        SetStatus($"等待唤醒词: {AudioDefaults.WakeupWord}");
                        break;
                    case AssistantWakeMode.ContinuousConversation:
                        SetStatus("已唤醒, 可以继续说");
                        break;
                    case AssistantWakeMode.Disabled:
                        break;
                }
            };

            assistant.WakeupDetected += (_, e) =>
            {
                WriteDialogueLine($"  [唤醒] {e.Wakeup.Keyword}");
            };

            assistant.UserSpoke += (_, e) =>
            {
                WriteDialogueLine($"  我：{e.Text}");
            };

            assistant.BargeInValidation += (_, e) =>
            {
                switch (e.Kind)
                {
                    case BargeInValidationKind.Heard:
                        ShowImmediateStatus("🔴 听到你说话了...");
                        break;
                    case BargeInValidationKind.Rejected:
                        ShowImmediateStatus($"{FormatBargeInValidationReason(e.Reason)}，识别结果无效，继续回答...");
                        break;
                    case BargeInValidationKind.Accepted:
                        // 有效的播放期输入会升级为正常对话回合，后续走 UserSpoke/StateChanged 展示。
                        break;
                }
            };

            assistant.AssistantText += (_, e) =>
            {
                lock (_consoleLock)
                {
                    if (e.IsThinking)
                        WriteThinkingDeltaLocked(e.Delta);
                    else
                        WriteAssistantDeltaLocked(e.Delta);
                }
            };

            assistant.AssistantResponseGenerated += (_, e) => FinishAssistantLine();
            assistant.AssistantReplied += (_, e) => FinishAssistantLine();
        }

        private static string FormatBargeInValidationReason(string? reason)
        {
            return reason switch
            {
                "empty" => "未识别到有效文字",
                "filler" => "只听到短促语气词",
                "asr-start-failed" => "ASR 启动失败",
                "asr-canceled" => "ASR 验证被取消",
                "asr-error" => "ASR 验证失败",
                null or "" => "未知原因",
                _ => reason,
            };
        }

        private static void ShowImmediateStatus(string status)
        {
            lock (_consoleLock)
            {
                _currentStatus = status;

                if (_streamingAssistant || _streamingThinking)
                {
                    FinishAssistantLineLocked();
                    ClearLine();
                    Console.WriteLine("  " + status);
                    return;
                }

                ClearLine();
                if (!string.IsNullOrEmpty(status))
                    Console.Write("  " + status);
            }
        }

        // ──────────────────────────────────────────────────────────────────────
        //  控制台输出原语 (线程安全)
        // ──────────────────────────────────────────────────────────────────────

        private static void PrintBanner()
        {
            lock (_consoleLock)
            {
                Console.WriteLine();
                Console.WriteLine("  ────────────────────────────────────────────");
                Console.WriteLine("            语音助手已上线");
                Console.WriteLine($"    说“{AudioDefaults.WakeupWord}”唤醒, 按 Q 退出");
                Console.WriteLine("  ────────────────────────────────────────────");
                Console.WriteLine();
            }
        }

        /// <summary>
        /// 进入 Idle 时, 把可能因异常残留的"助手：..."未完成行收尾, 再恢复聆听提示。
        /// </summary>
        private static void HandleEnterIdle(VoiceAssistant assistant)
        {
            lock (_consoleLock)
            {
                FinishAssistantLineLocked();
                _currentStatus = assistant.WakeMode == AssistantWakeMode.WaitingForWakeWord
                    ? $"等待唤醒词: {AudioDefaults.WakeupWord}"
                    : "连续对话中...";
                ClearLine();
                Console.Write("  " + _currentStatus);
            }
        }

        /// <summary>
        /// AI 文本完整生成时即把"助手：..."行落到控制台历史; TTS 后续被打断也不会擦掉这轮对话。
        /// </summary>
        private static void FinishAssistantLine()
        {
            lock (_consoleLock)
                FinishAssistantLineLocked();
        }

        private static void FinishAssistantLineLocked()
        {
            CloseThinkingLocked();

            if (_streamingAssistant)
            {
                Console.WriteLine();
                Console.WriteLine();
                _streamingAssistant = false;
            }

            // 流式片段全部收尾后, 把期间缓冲的低优先级日志统一打出 (聚到这轮回复之后)。
            FlushPendingLogsLocked();
        }

        private static void WriteThinkingDeltaLocked(string delta)
        {
            if (string.IsNullOrEmpty(delta)) return;

            if (!_streamingThinking)
            {
                _streamingThinking = true;
                ClearLine();
                Console.ForegroundColor = ConsoleColor.DarkGray;
                Console.Write("  [思考] ");
            }

            Console.Write(delta);
        }

        private static void WriteAssistantDeltaLocked(string delta)
        {
            if (string.IsNullOrEmpty(delta)) return;

            CloseThinkingLocked();
            if (!_streamingAssistant)
            {
                _streamingAssistant = true;
                ClearLine();
                Console.Write("  助手：");
            }

            Console.Write(delta);
        }

        private static void CloseThinkingLocked()
        {
            if (!_streamingThinking) return;

            _streamingThinking = false;
            Console.ResetColor();
            Console.WriteLine();
        }

        private static void SetStatus(string status)
        {
            lock (_consoleLock)
            {
                _currentStatus = status;
                if (_streamingAssistant || _streamingThinking)
                    return;

                ClearLine();
                if (!string.IsNullOrEmpty(status))
                {
                    Console.Write("  " + status);
                }
            }
        }

        /// <summary>
        /// 在当前光标位置擦掉状态行, 输出一行对话, 再把状态行重新打出来。
        /// </summary>
        private static void WriteDialogueLine(string text)
        {
            lock (_consoleLock)
            {
                FinishAssistantLineLocked();
                ClearLine();
                Console.WriteLine(text);
                if (!string.IsNullOrEmpty(_currentStatus))
                    Console.Write("  " + _currentStatus);
            }
        }

        private static void ClearStatusAndPrint(string text)
        {
            lock (_consoleLock)
            {
                FinishAssistantLineLocked();
                _currentStatus = string.Empty;
                ClearLine();
                Console.WriteLine();
                Console.WriteLine("  " + text);
            }
        }

        /// <summary>
        /// 日志输出入口 (注入给 ConsoleUiLoggerProvider)。走 _consoleLock。
        /// 流式输出助手文本/思考期间, 低优先级日志 (Debug/Info) 暂存到 _pendingLogs, 等该行收尾
        /// (FinishAssistantLineLocked) 再统一打出, 避免把单条回复劈成多段 "助手："; Warning/Error
        /// 仍即时插入。诊断内容原样输出 (含 barge-in 长行)。
        /// </summary>
        private static void WriteLogLine(LogLevel level, string message)
        {
            string ts = DateTime.Now.ToString("HH:mm:ss");
            lock (_consoleLock)
            {
                bool streaming = _streamingAssistant || _streamingThinking;
                if (streaming && level < LogLevel.Warning)
                {
                    _pendingLogs.Add((level, ts, message));
                    return;
                }

                // Warning/Error 或非流式态: 收尾当前流式片段后即时打印。
                FinishAssistantLineLocked();
                EmitLogLocked(level, ts, message);
                RedrawStatusLocked();
            }
        }

        /// <summary>把缓冲的低优先级日志按序落盘。调用方需持有 _consoleLock 且已收尾流式行。</summary>
        private static void FlushPendingLogsLocked()
        {
            if (_pendingLogs.Count == 0) return;

            foreach (var (level, ts, message) in _pendingLogs)
                EmitLogLocked(level, ts, message);
            _pendingLogs.Clear();
        }

        /// <summary>打印单条日志 (不含状态行重绘)。调用方需持有 _consoleLock。</summary>
        private static void EmitLogLocked(LogLevel level, string ts, string message)
        {
            ClearLine();
            string lvl = LevelTag(level);

            // 对话内容用默认色, 日志压成暗色/警示色, 让界面层次分明。
            if (level >= LogLevel.Error) Console.ForegroundColor = ConsoleColor.Red;
            else if (level == LogLevel.Warning) Console.ForegroundColor = ConsoleColor.Yellow;
            else Console.ForegroundColor = ConsoleColor.DarkGray;

            Console.WriteLine($"{ts} {lvl}: {message}");
            Console.ResetColor();
        }

        /// <summary>把活动状态行重画到当前光标处。调用方需持有 _consoleLock。</summary>
        private static void RedrawStatusLocked()
        {
            if (!string.IsNullOrEmpty(_currentStatus))
                Console.Write("  " + _currentStatus);
        }

        private static string LevelTag(LogLevel level) => level switch
        {
            LogLevel.Trace => "trce",
            LogLevel.Debug => "dbug",
            LogLevel.Information => "info",
            LogLevel.Warning => "warn",
            LogLevel.Error => "fail",
            LogLevel.Critical => "crit",
            _ => "    ",
        };

        /// <summary>
        /// \r 回到行首, \x1b[2K 清除整行 (VT escape, Win10 1903+ / 现代 *nix 终端都支持)。
        /// 用 ANSI 而不是用空格覆盖, 是为了避免 emoji 的可变宽度让长度计算出错。
        /// </summary>
        private static void ClearLine()
        {
            Console.Write("\r\x1b[2K");
        }
    }
}
