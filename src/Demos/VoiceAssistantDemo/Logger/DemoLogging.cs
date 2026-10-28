using System.Text;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Events;
using Serilog.Extensions.Logging;

namespace VoiceAssistantDemo.Logger
{
    internal static class DemoLogging
    {
        public static ILoggerFactory CreateLoggerFactory(Action<LogLevel, string> writeConsole)
        {
            bool debugEnabled = ResolveDebugEnabled();
            var configuration = new LoggerConfiguration()
                .MinimumLevel.Is(debugEnabled ? LogEventLevel.Debug : LogEventLevel.Information)
                .Enrich.FromLogContext()
                .WriteTo.Sink(new ConsoleUiLogEventSink(writeConsole));
            if (debugEnabled)
            {
                string fileName = $"voice-assistant-{DateTime.Now:yyyyMMdd-HHmmss-fffffff}-{Environment.ProcessId}.log";
                configuration.WriteTo.File(
                    Path.Combine("tmp", "log", fileName),
                    outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {SourceContext} {EventId} {Message:lj}{NewLine}{Exception}",
                    buffered: false,
                    encoding: new UTF8Encoding(false));
            }

            return new SerilogLoggerFactory(configuration.CreateLogger(), dispose: true);
        }

        private static bool ResolveDebugEnabled()
        {
#if DEBUG
            const bool defaultEnabled = true;
#else
            const bool defaultEnabled = false;
#endif
            string? value = Environment.GetEnvironmentVariable("VOICEBOX_DEBUG");
            return value?.Trim().ToLowerInvariant() switch
            {
                null or "" => defaultEnabled,
                "true" or "1" or "on" => true,
                "false" or "0" or "off" => false,
                _ => throw new InvalidOperationException(
                    "VOICEBOX_DEBUG must be one of: true, false, 1, 0, on, off."),
            };
        }
    }
}
