using Microsoft.Extensions.Logging;
using VoiceAssistantDemo.Logging;

namespace VoiceAssistantDemo.Configuration;

/// <summary>将 Demo 的环境变量约定转换成日志模块的显式配置。</summary>
internal static class DemoLoggingConfiguration
{
    public static LoggingOptions FromEnvironment()
    {
#if DEBUG
        const bool defaultEnabled = true;
#else
        const bool defaultEnabled = false;
#endif
        string? value = Environment.GetEnvironmentVariable("VOICEBOX_DEBUG");
        bool debugEnabled = value?.Trim().ToLowerInvariant() switch
        {
            null or "" => defaultEnabled,
            "true" or "1" or "on" => true,
            "false" or "0" or "off" => false,
            _ => throw new InvalidOperationException(
                "VOICEBOX_DEBUG must be one of: true, false, 1, 0, on, off."),
        };
        return new LoggingOptions
        {
            ConsoleMinimumLevel = debugEnabled ? LogLevel.Debug : LogLevel.Information,
            FileEnabled = debugEnabled,
        };
    }
}
