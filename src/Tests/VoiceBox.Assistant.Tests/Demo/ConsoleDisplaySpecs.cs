using System.Globalization;
using Microsoft.Extensions.Logging;
using VoiceAssistantDemo.ConsoleUI;
using Xunit;

namespace VoiceBox.Assistant.Tests.Demo;

public sealed class ConsoleDisplaySpecs
{
    private static readonly DateTimeOffset LogTime = new(2026, 10, 28, 9, 12, 34, TimeSpan.FromHours(8));

    [Fact]
    public void OrdinaryLogsWaitUntilTheReplyFinishesAndKeepTheirOriginalTimestamp()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: false);
        display.AppendThinking("思考内容");
        display.WriteLog(LogTime, LogLevel.Debug, "诊断");
        display.AppendReply("第一段");
        display.WriteLog(LogTime.AddSeconds(1), LogLevel.Information, "信息");
        display.AppendReply("第二段");

        Assert.DoesNotContain("诊断", output.ToString());
        Assert.DoesNotContain("信息", output.ToString());
        display.CompleteResponse(string.Empty);

        Assert.Equal(
            "  [思考] 思考内容\n  助手：第一段第二段\n\n09:12:34 dbug: 诊断\n09:12:35 info: 信息\n",
            Normalize(output));
    }

    [Fact]
    public void WarningsInterruptTheReplyAndLaterTextStartsANewReplyLine()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: false);
        display.AppendReply("前半段");
        display.WriteLog(LogTime, LogLevel.Debug, "先前日志");
        display.WriteLog(LogTime.AddSeconds(1), LogLevel.Warning, "警告");

        Assert.EndsWith("09:12:35 warn: 警告\n", Normalize(output));
        display.AppendReply("后半段");
        display.CompleteResponse(string.Empty);

        Assert.Equal(
            "  助手：前半段\n\n09:12:34 dbug: 先前日志\n09:12:35 warn: 警告\n  助手：后半段\n\n",
            Normalize(output));
    }

    [Fact]
    public void FinishingAReplyNeverErasesItsLastLineWhenThereIsNoStatus()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: true);
        display.SetStatus("正在思考");
        display.AppendReply("保留在历史中的回答");
        display.CompleteResponse(string.Empty);
        string completed = output.ToString();
        display.CompleteResponse(string.Empty);

        Assert.Contains("\r\x1b[2K  助手：保留在历史中的回答", completed);
        Assert.Equal(completed, output.ToString());
        Assert.DoesNotContain("\x1b[2K", completed[(completed.IndexOf("回答", StringComparison.Ordinal) + 2)..]);
    }

    [Fact]
    public void RedirectedOutputUsesPlainLinesAndDoesNotRepeatUnchangedStatuses()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: false);
        display.SetStatus("等待唤醒");
        display.SetStatus("等待唤醒");
        display.WriteLog(LogTime, LogLevel.Error, "错误");
        display.WriteLines("其他消息");
        display.SetStatus("正在识别");

        Assert.Equal("  等待唤醒\n09:12:34 fail: 错误\n其他消息\n  正在识别\n", Normalize(output));
        Assert.DoesNotContain('\x1b', output.ToString());
        Assert.DoesNotContain('\r', Normalize(output));
    }

    [Fact]
    public void ImmediateStatusPersistsBetweenInterruptedReplyFragments()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: false);
        display.AppendReply("之前");
        display.ShowImmediateStatus("听到你说话了");
        display.AppendReply("之后");
        display.CompleteResponse(string.Empty);

        Assert.Equal("  助手：之前\n\n  听到你说话了\n  助手：之后\n\n", Normalize(output));
    }

    [Fact]
    public void DeferredLogBufferKeepsRecentEntriesAndReportsOmittedEntries()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: false, pendingLogCapacity: 2);
        display.AppendReply("回答");
        display.WriteLog(LogTime, LogLevel.Debug, "较早日志");
        display.WriteLog(LogTime, LogLevel.Debug, "较新日志");
        display.WriteLog(LogTime, LogLevel.Information, "最新日志");
        display.CompleteResponse(string.Empty);

        Assert.DoesNotContain("较早日志", output.ToString());
        Assert.Contains("已省略 1 条缓冲日志", output.ToString());
        Assert.EndsWith("09:12:34 dbug: 较新日志\n09:12:34 info: 最新日志\n", Normalize(output));
    }

    [Fact]
    public void DisposeFlushesDeferredLogsWithoutClosingTheProvidedWriter()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        var display = new ConsoleDisplay(output, interactive: false);
        display.AppendReply("最后的回答");
        display.WriteLog(LogTime, LogLevel.Information, "最后的日志");
        display.Dispose();
        display.Dispose();
        output.WriteLine("仍可写入");

        Assert.Equal("  助手：最后的回答\n\n09:12:34 info: 最后的日志\n仍可写入\n", Normalize(output));
    }

    [Fact]
    public async Task ConcurrentLogWritersProduceCompleteDistinctLines()
    {
        using var output = new StringWriter(CultureInfo.InvariantCulture);
        using var display = new ConsoleDisplay(output, interactive: false);
        await Task.WhenAll(Enumerable.Range(0, 100).Select(index =>
            Task.Run(() => display.WriteLog(LogTime, LogLevel.Information, $"message-{index}"))));

        string[] lines = Normalize(output).Split('\n', StringSplitOptions.RemoveEmptyEntries);
        Assert.Equal(100, lines.Length);
        Assert.Equal(100, lines.Distinct().Count());
        Assert.All(lines, line => Assert.Matches(@"^09:12:34 info: message-\d+$", line));
    }

    private static string Normalize(StringWriter output) => output.ToString().Replace("\r\n", "\n");
}
