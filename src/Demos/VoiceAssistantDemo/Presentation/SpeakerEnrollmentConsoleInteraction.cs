using VoiceAssistantDemo.ConsoleUI;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Enrollment;
using VoiceBox.Assistant.Modules.Speaker.Models;

namespace VoiceAssistantDemo.Presentation;

/// <summary>声纹注册的进度展示、朗读提示、开始节拍和重试交互。</summary>
internal sealed class SpeakerEnrollmentConsoleInteraction : ISpeakerEnrollmentPacer
{
    private static readonly string[] EnrollmentSentences =
    [
        "清晨的阳光照进窗台，我端起一杯温水，准备开始今天的工作。",
        "周末我们打算去郊外走一走，顺便看看山里的枫叶红了没有。",
        "请把明天上午九点的会议提醒改到十点，并通知项目组的同事。",
        "厨房里的汤还在小火慢炖，锅盖边缘不时冒出一缕白色的蒸汽。",
        "这本书讲的是一位老木匠和他的孙女在小镇上度过的一个夏天。",
    ];

    private readonly ConsoleDisplay _display;
    private readonly ConsoleInput _input;
    private readonly string _wakeupWord;

    public SpeakerEnrollmentConsoleInteraction(ConsoleDisplay display, ConsoleInput input, string wakeupWord)
    {
        ArgumentNullException.ThrowIfNull(display);
        ArgumentNullException.ThrowIfNull(input);
        ArgumentException.ThrowIfNullOrWhiteSpace(wakeupWord);
        _display = display;
        _input = input;
        _wakeupWord = wakeupWord;
    }

    public async Task EnsureRegisteredAsync(
        SpeakerEnrollmentCoordinator enrollment,
        TtsCalibrationStimulusPlayer calibrationStimulus,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(enrollment);
        ArgumentNullException.ThrowIfNull(calibrationStimulus);
        cancellationToken.ThrowIfCancellationRequested();
        if (enrollment.HasUsableProfile(requireDoubleTalkBank: true))
        {
            _display.WriteLines(
                $"  [Speaker] 已加载本人声纹 (前端指纹 {enrollment.FrontendFingerprint}): {enrollment.ProfilePath}",
                string.Empty);
            return;
        }

        enrollment.ProgressChanged += OnProgressChanged;
        try
        {
            bool calibrationPrepared = false;
            while (true)
            {
                try
                {
                    if (!calibrationPrepared)
                    {
                        _display.WriteLines(string.Empty,
                            "  [Speaker] 正在准备双讲校准语音（首次使用会加载 TTS 模型），请稍候...");
                        await calibrationStimulus.PrepareAsync(cancellationToken);
                        calibrationPrepared = true;
                    }

                    SpeakerEnrollmentResult result = await enrollment.EnrollAsync(this, cancellationToken);
                    _display.CompleteResponse(string.Empty);
                    _display.WriteLines(
                        $"  [Speaker] 注册完成: 清晰段 {result.CleanSegmentsRetained} 段/{result.CleanPrototypeCount} 原型, " +
                        $"双讲段 {result.DoubleTalkSegmentsRetained} 段/{result.DoubleTalkPrototypeCount} 原型, " +
                        $"重录 {result.RetriedSegments} 次, 剔除离群 {result.DiscardedSegments} 段, " +
                        $"独立确认最低得分 {result.ConfirmationMinScore:F4} (阈值 {result.Threshold:F4})。缓存: {result.ProfilePath}",
                        string.Empty);
                    return;
                }
                catch (OperationCanceledException)
                {
                    throw;
                }
                catch (Exception ex)
                {
                    _display.CompleteResponse(string.Empty);
                    _display.WriteLines(string.Empty, $"  [Speaker] 注册失败：{ex.Message}", RetryPrompt(enrollment));
                    ConsoleKey key = await _input.ReadKeyAsync(cancellationToken);
                    if (key is ConsoleKey.Q or ConsoleKey.Escape)
                        throw new OperationCanceledException("用户取消了本人声纹注册。", ex, cancellationToken);
                    if (key == ConsoleKey.N)
                    {
                        enrollment.DiscardPendingEnrollment();
                        _display.WriteLines("  已放弃已录段落，将从头注册。");
                    }
                    else if (key != ConsoleKey.R)
                    {
                        _display.WriteLines("  未识别按键，将从失败处继续。");
                    }
                }
            }
        }
        finally
        {
            enrollment.ProgressChanged -= OnProgressChanged;
            _display.CompleteResponse(string.Empty);
        }
    }

    public async Task WaitForSegmentStartAsync(
        SpeakerEnrollmentProgressEventArgs segment,
        CancellationToken cancellationToken)
    {
        cancellationToken.ThrowIfCancellationRequested();
        var lines = new List<string> { string.Empty };
        if (segment.Stage == SpeakerEnrollmentStage.CleanEnrollment && segment.SegmentIndex == 0 && segment.Attempt == 1)
            lines.Add("  首次使用需要注册本人声纹, 分为安静环境段与双讲段, 每段约 2.5 秒有效语音。");

        string attempt = segment.Attempt > 1 ? $" (第 {segment.Attempt} 次)" : string.Empty;
        lines.Add(segment.Stage switch
        {
            SpeakerEnrollmentStage.CleanEnrollment =>
                $"  安静环境段 {segment.SegmentIndex + 1}/{segment.SegmentCount}{attempt}，请自然朗读: {EnrollmentSentences[segment.SegmentIndex % EnrollmentSentences.Length]}",
            SpeakerEnrollmentStage.DoubleTalkEnrollment =>
                $"  双讲段 {segment.SegmentIndex + 1}/{segment.SegmentCount}{attempt}，扬声器会同时播放校准语音, 请不要等它停, 直接朗读: {EnrollmentSentences[(segment.SegmentIndex + 2) % EnrollmentSentences.Length]}",
            SpeakerEnrollmentStage.CleanConfirmation =>
                $"  安静环境确认，请完整朗读：“{_wakeupWord}，我想出去玩，可以帮我查一下今天的天气怎么样吗”，用于独立确认刚才注册的声纹。",
            SpeakerEnrollmentStage.DoubleTalkConfirmation =>
                $"  双讲确认，扬声器播放校准语音的同时, 请像打断助手一样说一句话。可以说（可以重复说）：{_wakeupWord}，我听到你说话了，可以停一下吗",
            _ => $"  [Speaker] {DescribeStage(segment.Stage)}",
        });
        lines.Add("  按 S 键开始收集，按 Q 或 Esc 退出...");
        _display.CompleteResponse(string.Empty);
        _display.WriteLines(lines.ToArray());

        while (true)
        {
            ConsoleKey key = await _input.ReadKeyAsync(cancellationToken);
            if (key == ConsoleKey.S)
                return;
            if (key is ConsoleKey.Q or ConsoleKey.Escape)
                throw new OperationCanceledException("用户取消了本人声纹注册。", cancellationToken);
        }
    }

    private void OnProgressChanged(object? sender, SpeakerEnrollmentProgressEventArgs e)
    {
        if (e.Kind == SpeakerEnrollmentProgressKind.Collecting)
        {
            _display.SetStatus(
                $"[Speaker] {DescribeStage(e.Stage)} 第 {e.SegmentIndex + 1}/{e.SegmentCount} 段, " +
                $"有效语音 {e.Collected.TotalSeconds:F1}/{e.Required.TotalSeconds:F1} 秒");
            return;
        }

        string? message = e.Kind switch
        {
            SpeakerEnrollmentProgressKind.SegmentOutcome => e.ReasonCode == SpeakerEnrollmentReasonCode.SegmentAccepted
                ? $"  [Speaker] 第 {e.SegmentIndex + 1} 段已接受 (rms={e.QualitySnapshot?.Rms:F3}, 语音占比={e.QualitySnapshot?.SpeechRatio:P0})。"
                : $"  [Speaker] 第 {e.SegmentIndex + 1} 段未通过 ({DescribeReason(e.ReasonCode)}), 将重录该段。",
            SpeakerEnrollmentProgressKind.StageChanged
                when e.Stage is SpeakerEnrollmentStage.BuildingProfile or SpeakerEnrollmentStage.SavingProfile =>
                $"  [Speaker] {DescribeStage(e.Stage)}...",
            SpeakerEnrollmentProgressKind.Resumed =>
                $"  [Speaker] 从{DescribeResume(e.Stage, e.SegmentIndex)}继续，已通过的段落会保留。",
            _ => null,
        };
        if (message is not null)
        {
            _display.CompleteResponse(string.Empty);
            _display.WriteLines(message);
        }
    }

    private static string RetryPrompt(SpeakerEnrollmentCoordinator enrollment)
    {
        if (enrollment.PendingResume is { } resume &&
            (resume.AcceptedCleanSegments > 0 || resume.AcceptedDoubleTalkSegments > 0 ||
             resume.Stage != SpeakerEnrollmentStage.CleanEnrollment || resume.SegmentIndex > 0))
        {
            return $"  按 R 从{DescribeResume(resume.Stage, resume.SegmentIndex)}继续" +
                $"（已保留安静环境 {resume.AcceptedCleanSegments} 段、双讲 {resume.AcceptedDoubleTalkSegments} 段），" +
                "按 N 重新开始，按 Q 或 Esc 退出。";
        }
        return "  按 R 重试，按 N 重新开始，按 Q 或 Esc 退出。";
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
}
