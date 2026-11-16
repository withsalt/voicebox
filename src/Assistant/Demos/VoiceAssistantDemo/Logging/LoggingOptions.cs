using Microsoft.Extensions.Logging;

namespace VoiceAssistantDemo.Logging;

internal sealed record LoggingOptions
{
    public LogLevel ConsoleMinimumLevel { get; init; } = LogLevel.Information;
    public bool FileEnabled { get; init; }
    public LogLevel FileMinimumLevel { get; init; } = LogLevel.Debug;
    public string FilePath { get; init; } = Path.Combine("tmp", "log", "voice-assistant-.log");
    public long FileSizeLimitBytes { get; init; } = 10 * 1024 * 1024;
    public int RetainedFileCountLimit { get; init; } = 10;
}
