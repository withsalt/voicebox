using System.Text;
using Microsoft.Extensions.Logging;
using Serilog;
using Serilog.Events;
using Serilog.Extensions.Logging;

namespace VoiceAssistantDemo.Logging;

internal static class DemoLogging
{
    public static ILoggerFactory CreateLoggerFactory(
        LoggingOptions options,
        Action<DateTimeOffset, LogLevel, string>? writeConsole = null)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!Enum.IsDefined(options.ConsoleMinimumLevel) || !Enum.IsDefined(options.FileMinimumLevel))
            throw new ArgumentOutOfRangeException(nameof(options), "日志级别无效。");

        LogLevel minimum = (LogLevel)Math.Min(
            (int)(writeConsole is null ? LogLevel.None : options.ConsoleMinimumLevel),
            (int)(options.FileEnabled ? options.FileMinimumLevel : LogLevel.None));
        var configuration = new LoggerConfiguration()
            .MinimumLevel.Is(ToSerilogLevel(minimum == LogLevel.None ? LogLevel.Critical : minimum))
            .Enrich.FromLogContext();

        if (writeConsole is not null && options.ConsoleMinimumLevel != LogLevel.None)
        {
            configuration.WriteTo.Sink(
                new CallbackLogEventSink(writeConsole),
                restrictedToMinimumLevel: ToSerilogLevel(options.ConsoleMinimumLevel));
        }

        if (options.FileEnabled && options.FileMinimumLevel != LogLevel.None)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(options.FilePath);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.FileSizeLimitBytes);
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(options.RetainedFileCountLimit);
            configuration.WriteTo.File(
                options.FilePath,
                restrictedToMinimumLevel: ToSerilogLevel(options.FileMinimumLevel),
                outputTemplate: "{Timestamp:yyyy-MM-dd HH:mm:ss.fff zzz} [{Level:u3}] {SourceContext} {EventId} {Message:lj}{NewLine}{Exception}",
                rollingInterval: RollingInterval.Day,
                fileSizeLimitBytes: options.FileSizeLimitBytes,
                rollOnFileSizeLimit: true,
                retainedFileCountLimit: options.RetainedFileCountLimit,
                shared: true,
                buffered: false,
                encoding: new UTF8Encoding(false));
        }

        return new SerilogLoggerFactory(configuration.CreateLogger(), dispose: true);
    }

    private static LogEventLevel ToSerilogLevel(LogLevel level) => level switch
    {
        LogLevel.Trace => LogEventLevel.Verbose,
        LogLevel.Debug => LogEventLevel.Debug,
        LogLevel.Information => LogEventLevel.Information,
        LogLevel.Warning => LogEventLevel.Warning,
        LogLevel.Error => LogEventLevel.Error,
        LogLevel.Critical => LogEventLevel.Fatal,
        _ => throw new ArgumentOutOfRangeException(nameof(level)),
    };
}
