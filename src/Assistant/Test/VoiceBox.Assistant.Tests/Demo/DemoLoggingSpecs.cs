using Microsoft.Extensions.Logging;
using VoiceAssistantDemo.ConsoleUI;
using VoiceAssistantDemo.Logging;
using Xunit;

namespace VoiceBox.Assistant.Tests.Demo;

public sealed class DemoLoggingSpecs
{
    [Fact]
    public void CallbackPreservesLevelStructuredValuesAndExceptions()
    {
        var entries = new List<(DateTimeOffset Timestamp, LogLevel Level, string Message)>();
        using var factory = DemoLogging.CreateLoggerFactory(
            new LoggingOptions { ConsoleMinimumLevel = LogLevel.Trace },
            (timestamp, level, message) => entries.Add((timestamp, level, message)));
        var logger = factory.CreateLogger("test");
        DateTimeOffset before = DateTimeOffset.Now;
        logger.LogError(new InvalidOperationException("故障原因"), "处理回合 {TurnId} 失败", 42);

        var entry = Assert.Single(entries);
        Assert.Equal(LogLevel.Error, entry.Level);
        Assert.Contains("处理回合 42 失败", entry.Message);
        Assert.Contains("System.InvalidOperationException: 故障原因", entry.Message);
        Assert.InRange(entry.Timestamp, before, DateTimeOffset.Now);
    }

    [Fact]
    public void FileLogsAreImmediateEvenWhenTheConsoleDefersThemAndEachSinkHasItsOwnLevel()
    {
        string directory = Path.Combine(Path.GetTempPath(), "voicebox-demo-logging-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            using var output = new StringWriter();
            using var display = new ConsoleDisplay(output, interactive: false);
            using (var factory = DemoLogging.CreateLoggerFactory(new LoggingOptions
            {
                ConsoleMinimumLevel = LogLevel.Information,
                FileEnabled = true,
                FileMinimumLevel = LogLevel.Debug,
                FilePath = Path.Combine(directory, "test-.log"),
            }, display.WriteLog))
            {
                var logger = factory.CreateLogger("demo-test");
                display.AppendReply("流式回答");
                logger.LogDebug("仅文件可见");
                logger.LogInformation("两个输出均可见");

                Assert.DoesNotContain("两个输出均可见", output.ToString());
                string file = Assert.Single(Directory.GetFiles(directory, "*.log"));
                using var reader = new StreamReader(new FileStream(file, FileMode.Open, FileAccess.Read, FileShare.ReadWrite));
                string contents = reader.ReadToEnd();
                Assert.Contains("仅文件可见", contents);
                Assert.Contains("两个输出均可见", contents);
                Assert.Contains("demo-test", contents);

                display.CompleteResponse(string.Empty);
                Assert.Contains("两个输出均可见", output.ToString());
                Assert.DoesNotContain("仅文件可见", output.ToString());
            }
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }

    [Fact]
    public void FileLoggingWorksWithoutAnyConsoleCallback()
    {
        string directory = Path.Combine(Path.GetTempPath(), "voicebox-demo-logging-tests", Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(directory);
        try
        {
            using (var factory = DemoLogging.CreateLoggerFactory(new LoggingOptions
            {
                FileEnabled = true,
                FileMinimumLevel = LogLevel.Warning,
                FilePath = Path.Combine(directory, "test-.log"),
            }))
            {
                var logger = factory.CreateLogger("file-only");
                logger.LogInformation("已过滤");
                logger.LogWarning("文件独立记录");
            }

            string contents = File.ReadAllText(Assert.Single(Directory.GetFiles(directory, "*.log")));
            Assert.Contains("文件独立记录", contents);
            Assert.DoesNotContain("已过滤", contents);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
