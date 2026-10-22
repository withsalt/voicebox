using Microsoft.Extensions.Logging;

namespace VoiceAssistantDemo.Logger
{
    /// <summary>
    /// 把日志通过 Program 的控制台锁 + 状态行逻辑输出, 让每条日志独占一行、打在"活动状态行"
    /// 上方, 不再和 UI 用 Console.Write 停在行首的状态行拼接错乱 (原 AddSimpleConsole 从
    /// 后台队列直写 stdout, 绕开了 _consoleLock)。
    /// </summary>
    internal sealed class ConsoleUiLoggerProvider : ILoggerProvider
    {
        private readonly LogLevel _minLevel;
        private readonly Action<LogLevel, string> _write;

        public ConsoleUiLoggerProvider(LogLevel minLevel, Action<LogLevel, string> write)
        {
            _minLevel = minLevel;
            _write = write;
        }

        public ILogger CreateLogger(string categoryName) => new ConsoleUiLogger(_minLevel, _write);

        public void Dispose() { }

        class ConsoleUiLogger : ILogger
        {
            private readonly LogLevel _minLevel;
            private readonly Action<LogLevel, string> _write;

            public ConsoleUiLogger(LogLevel minLevel, Action<LogLevel, string> write)
            {
                _minLevel = minLevel;
                _write = write;
            }

            public IDisposable? BeginScope<TState>(TState state) where TState : notnull => null;

            public bool IsEnabled(LogLevel logLevel) => logLevel >= _minLevel && logLevel != LogLevel.None;

            public void Log<TState>(
                LogLevel logLevel,
                EventId eventId,
                TState state,
                Exception? exception,
                Func<TState, Exception?, string> formatter)
            {
                if (!IsEnabled(logLevel)) return;

                string message = formatter(state, exception);
                if (exception is not null)
                    message = string.IsNullOrEmpty(message) ? exception.ToString() : message + Environment.NewLine + exception;

                if (string.IsNullOrEmpty(message)) return;

                _write(logLevel, message);
            }
        }
    }
}
