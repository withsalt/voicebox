using System.Globalization;
using Microsoft.Extensions.Logging;

namespace VoiceAssistantDemo.ConsoleUI;

/// <summary>串行协调状态行、流式文本和日志的显示，不持有助手或日志框架实例。</summary>
internal sealed class ConsoleDisplay : IDisposable
{
    private readonly object _sync = new();
    private readonly TextWriter _output;
    private readonly bool _interactive;
    private readonly int _pendingLogCapacity;
    private readonly Queue<(DateTimeOffset Timestamp, LogLevel Level, string Message)> _pendingLogs = new();
    private string _currentStatus = string.Empty;
    private string _lastPrintedStatus = string.Empty;
    private bool _statusVisible;
    private StreamKind? _stream;
    private int _omittedLogs;
    private bool _disposed;

    public ConsoleDisplay(TextWriter output, bool interactive, int pendingLogCapacity = 256)
    {
        ArgumentNullException.ThrowIfNull(output);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(pendingLogCapacity);
        _output = output;
        _interactive = interactive;
        _pendingLogCapacity = pendingLogCapacity;
    }

    public void WriteLines(params string[] lines)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            CompleteStreamCore();
            ClearStatusCore();
            foreach (string line in lines)
                _output.WriteLine(line);
            RenderStatusCore();
        }
    }

    public void SetStatus(string status)
    {
        ArgumentNullException.ThrowIfNull(status);
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            SetStatusCore(status);
            if (_stream is null)
            {
                ClearStatusCore();
                RenderStatusCore();
            }
        }
    }

    /// <summary>打断当前显示片段；流式输出期间的提示保留在历史中。</summary>
    public void ShowImmediateStatus(string status)
    {
        ArgumentNullException.ThrowIfNull(status);
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            bool wasStreaming = _stream is not null;
            CompleteStreamCore();
            ClearStatusCore();
            SetStatusCore(status);
            if (wasStreaming && status.Length > 0)
            {
                _output.WriteLine("  " + status);
                _lastPrintedStatus = status;
            }
            else
            {
                RenderStatusCore();
            }
        }
    }

    public void AppendThinking(string delta) => AppendText(delta, StreamKind.Thinking);

    public void AppendReply(string delta) => AppendText(delta, StreamKind.Reply);

    /// <summary>生成完成、播放结束、进入空闲或停止时均可调用，重复收尾不会额外换行。</summary>
    public void CompleteResponse(string? status = null)
    {
        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (status is not null)
                SetStatusCore(status);
            CompleteStreamCore();
            ClearStatusCore();
            RenderStatusCore();
        }
    }

    /// <summary>普通日志延后显示，警告和错误即时显示；时间戳来自原始日志事件。</summary>
    public void WriteLog(DateTimeOffset timestamp, LogLevel level, string message)
    {
        if (level == LogLevel.None || string.IsNullOrEmpty(message))
            return;

        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_stream is not null && level < LogLevel.Warning)
            {
                if (_pendingLogs.Count == _pendingLogCapacity)
                {
                    _pendingLogs.Dequeue();
                    _omittedLogs++;
                }
                _pendingLogs.Enqueue((timestamp, level, message));
                return;
            }

            CompleteStreamCore();
            EmitLogCore(timestamp, level, message);
            RenderStatusCore();
        }
    }

    public void Dispose()
    {
        lock (_sync)
        {
            if (_disposed)
                return;
            SetStatusCore(string.Empty);
            ClearStatusCore();
            CompleteStreamCore();
            _output.Flush();
            _disposed = true;
        }
    }

    private void AppendText(string delta, StreamKind kind)
    {
        if (string.IsNullOrEmpty(delta))
            return;

        lock (_sync)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_stream != kind)
            {
                CloseStreamCore();
                ClearStatusCore();
                if (kind == StreamKind.Thinking)
                    SetColorCore("90");
                _output.Write(kind == StreamKind.Thinking ? "  [思考] " : "  助手：");
                _stream = kind;
            }
            _output.Write(delta);
        }
    }

    private void SetStatusCore(string status)
    {
        _currentStatus = status;
        if (status.Length == 0)
            _lastPrintedStatus = string.Empty;
    }

    private void CompleteStreamCore()
    {
        CloseStreamCore();
        if (_omittedLogs > 0)
        {
            ClearStatusCore();
            _output.WriteLine($"  [控制台] 已省略 {_omittedLogs} 条缓冲日志。");
            _omittedLogs = 0;
        }
        while (_pendingLogs.TryDequeue(out var entry))
            EmitLogCore(entry.Timestamp, entry.Level, entry.Message);
    }

    private void CloseStreamCore()
    {
        if (_stream is not { } kind)
            return;
        ResetColorCore();
        _output.WriteLine();
        if (kind == StreamKind.Reply)
            _output.WriteLine();
        _stream = null;
    }

    private void EmitLogCore(DateTimeOffset timestamp, LogLevel level, string message)
    {
        ClearStatusCore();
        SetColorCore(level >= LogLevel.Error ? "31" : level == LogLevel.Warning ? "33" : "90");
        try
        {
            string time = timestamp.ToString("HH:mm:ss", CultureInfo.InvariantCulture);
            _output.WriteLine($"{time} {LevelTag(level)}: {message}");
        }
        finally
        {
            ResetColorCore();
        }
    }

    private void ClearStatusCore()
    {
        if (!_statusVisible)
            return;
        // ANSI 清行避免中文和 emoji 的宽度计算；重定向输出只写完整文本行。
        if (_interactive)
            _output.Write("\r\x1b[2K");
        _statusVisible = false;
    }

    private void RenderStatusCore()
    {
        if (_currentStatus.Length == 0 || _stream is not null || _statusVisible)
            return;
        if (_interactive)
        {
            _output.Write("  " + _currentStatus);
            _statusVisible = true;
        }
        else if (_lastPrintedStatus != _currentStatus)
        {
            _output.WriteLine("  " + _currentStatus);
            _lastPrintedStatus = _currentStatus;
        }
    }

    private void SetColorCore(string color)
    {
        if (_interactive)
            _output.Write($"\x1b[{color}m");
    }

    private void ResetColorCore()
    {
        if (_interactive)
            _output.Write("\x1b[0m");
    }

    private static string LevelTag(LogLevel level) => level switch
    {
        LogLevel.Trace => "trce",
        LogLevel.Debug => "dbug",
        LogLevel.Information => "info",
        LogLevel.Warning => "warn",
        LogLevel.Error => "fail",
        LogLevel.Critical => "crit",
        _ => throw new ArgumentOutOfRangeException(nameof(level)),
    };

    private enum StreamKind { Thinking, Reply }
}
