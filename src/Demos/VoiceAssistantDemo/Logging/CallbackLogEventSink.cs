using System.Globalization;
using Microsoft.Extensions.Logging;
using Serilog.Core;
using Serilog.Events;

namespace VoiceAssistantDemo.Logging;

/// <summary>把日志事件转换为显示回调参数，不参与控制台排版或缓冲。</summary>
internal sealed class CallbackLogEventSink : ILogEventSink
{
    private readonly Action<DateTimeOffset, LogLevel, string> _write;

    public CallbackLogEventSink(Action<DateTimeOffset, LogLevel, string> write)
    {
        ArgumentNullException.ThrowIfNull(write);
        _write = write;
    }

    public void Emit(LogEvent logEvent)
    {
        ArgumentNullException.ThrowIfNull(logEvent);
        string message = logEvent.RenderMessage(CultureInfo.InvariantCulture);
        if (logEvent.Exception is not null)
        {
            message = string.IsNullOrEmpty(message)
                ? logEvent.Exception.ToString()
                : message + Environment.NewLine + logEvent.Exception;
        }

        if (message.Length > 0)
        {
            LogLevel level = logEvent.Level switch
            {
                LogEventLevel.Verbose => LogLevel.Trace,
                LogEventLevel.Debug => LogLevel.Debug,
                LogEventLevel.Information => LogLevel.Information,
                LogEventLevel.Warning => LogLevel.Warning,
                LogEventLevel.Error => LogLevel.Error,
                LogEventLevel.Fatal => LogLevel.Critical,
                _ => throw new ArgumentOutOfRangeException(nameof(logEvent)),
            };
            _write(logEvent.Timestamp, level, message);
        }
    }
}
