using VoiceAssistantDemo.ConsoleUI;
using VoiceBox.Assistant;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;

namespace VoiceAssistantDemo.Presentation;

/// <summary>将助手公开事件映射为显示操作，独立管理订阅生命周期。</summary>
internal sealed class AssistantConsolePresenter : IDisposable
{
    private readonly VoiceAssistant _assistant;
    private readonly ConsoleDisplay _display;
    private readonly string _wakeupWord;
    private volatile bool _disposed;

    public AssistantConsolePresenter(VoiceAssistant assistant, ConsoleDisplay display, string wakeupWord)
    {
        ArgumentNullException.ThrowIfNull(assistant);
        ArgumentNullException.ThrowIfNull(display);
        ArgumentException.ThrowIfNullOrWhiteSpace(wakeupWord);
        _assistant = assistant;
        _display = display;
        _wakeupWord = wakeupWord;
        _assistant.StateChanged += OnStateChanged;
        _assistant.WakeModeChanged += OnWakeModeChanged;
        _assistant.WakeupDetected += OnWakeupDetected;
        _assistant.UserSpoke += OnUserSpoke;
        _assistant.BargeInValidation += OnBargeInValidation;
        _assistant.AssistantText += OnAssistantText;
        _assistant.AssistantResponseGenerated += OnResponseCompleted;
        _assistant.AssistantReplied += OnResponseCompleted;
        _assistant.Faulted += OnFaulted;
    }

    public void ShowReady()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        _display.WriteLines(
            string.Empty,
            "  ────────────────────────────────────────────",
            "            语音助手已上线",
            _assistant.WakeMode == AssistantWakeMode.Disabled
                ? "    可以直接说话, 按 Q 退出"
                : $"    说“{_wakeupWord}”唤醒, 按 Q 退出",
            "  ────────────────────────────────────────────",
            string.Empty);
        _display.SetStatus(IdleStatus());
    }

    public void Dispose()
    {
        if (_disposed)
            return;
        _disposed = true;
        _assistant.StateChanged -= OnStateChanged;
        _assistant.WakeModeChanged -= OnWakeModeChanged;
        _assistant.WakeupDetected -= OnWakeupDetected;
        _assistant.UserSpoke -= OnUserSpoke;
        _assistant.BargeInValidation -= OnBargeInValidation;
        _assistant.AssistantText -= OnAssistantText;
        _assistant.AssistantResponseGenerated -= OnResponseCompleted;
        _assistant.AssistantReplied -= OnResponseCompleted;
        _assistant.Faulted -= OnFaulted;
    }

    private void OnStateChanged(object? sender, AssistantStateChangedEventArgs e)
    {
        if (_disposed)
            return;
        switch (e.To)
        {
            case AssistantState.Idle:
                _display.CompleteResponse(IdleStatus());
                break;
            case AssistantState.Listening:
                _display.SetStatus("🔴 听到你说话了, 继续...");
                break;
            case AssistantState.Recognizing:
                _display.SetStatus("🔴 听到你说话了, 正在识别...可以继续说话哟~~~");
                break;
            case AssistantState.Thinking:
                _display.SetStatus("💭 组织语言中...");
                break;
            case AssistantState.Speaking:
                _display.SetStatus(string.Empty);
                break;
            case AssistantState.Stopped:
                _display.CompleteResponse(string.Empty);
                break;
        }
    }

    private void OnWakeModeChanged(object? sender, AssistantWakeModeChangedEventArgs e)
    {
        if (_disposed)
            return;
        _display.SetStatus(e.To switch
        {
            AssistantWakeMode.WaitingForWakeWord => $"等待唤醒词: {_wakeupWord}",
            AssistantWakeMode.ContinuousConversation => "已唤醒, 可以继续说",
            _ => string.Empty,
        });
    }

    private void OnWakeupDetected(object? sender, AssistantWakeupDetectedEventArgs e)
    {
        if (!_disposed)
            _display.WriteLines($"  [唤醒] {e.Wakeup.Keyword}");
    }

    private void OnUserSpoke(object? sender, UserSpokeEventArgs e)
    {
        if (!_disposed)
            _display.WriteLines($"  我：{e.Text}");
    }

    private void OnBargeInValidation(object? sender, BargeInValidationEventArgs e)
    {
        if (_disposed)
            return;
        switch (e.Kind)
        {
            case BargeInValidationKind.Heard:
                _display.ShowImmediateStatus("🔴 听到你说话了...");
                break;
            case BargeInValidationKind.Rejected:
                _display.ShowImmediateStatus($"{DescribeBargeInReason(e.Reason)}，识别结果无效，继续回答...");
                break;
        }
    }

    private void OnAssistantText(object? sender, AssistantTextEventArgs e)
    {
        if (_disposed)
            return;
        if (e.IsThinking)
            _display.AppendThinking(e.Delta);
        else
            _display.AppendReply(e.Delta);
    }

    private void OnResponseCompleted(object? sender, EventArgs e)
    {
        if (!_disposed)
            _display.CompleteResponse();
    }

    private void OnFaulted(object? sender, AssistantFaultedEventArgs e)
    {
        if (_disposed)
            return;
        _display.CompleteResponse(string.Empty);
        _display.WriteLines($"  助手已停止 ({e.Component}): {e.Exception.Message}");
    }

    private string IdleStatus() => _assistant.WakeMode switch
    {
        AssistantWakeMode.WaitingForWakeWord => $"等待唤醒词: {_wakeupWord}",
        AssistantWakeMode.ContinuousConversation => "连续对话中...",
        _ => "可以直接说话",
    };

    private static string DescribeBargeInReason(string? reason) => reason switch
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
