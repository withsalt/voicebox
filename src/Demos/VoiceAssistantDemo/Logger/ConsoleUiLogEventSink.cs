using System.Globalization;
using Microsoft.Extensions.Logging;
using Serilog.Core;
using Serilog.Events;

namespace VoiceAssistantDemo.Logger
{
    /// <summary>通过现有控制台锁和状态行逻辑输出 Serilog 日志，避免打断流式回复。</summary>
    internal sealed class ConsoleUiLogEventSink : ILogEventSink
    {
        private readonly Action<LogLevel, string> write;

        public ConsoleUiLogEventSink(Action<LogLevel, string> write)
        {
            this.write = write;
        }

        public void Emit(LogEvent logEvent)
        {
            string message = logEvent.RenderMessage(CultureInfo.InvariantCulture);
            if (logEvent.Exception is not null)
            {
                message = string.IsNullOrEmpty(message)
                    ? logEvent.Exception.ToString()
                    : message + Environment.NewLine + logEvent.Exception;
            }

            if (!string.IsNullOrEmpty(message))
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
                write(level, message);
            }
        }
    }
}
