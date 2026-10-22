using System.Globalization;
using System.Text.Json;
using VoiceBox.Assistant.Api.Tools.Contracts;
using VoiceBox.Assistant.Api.Tools.Models;

namespace VoiceAssistantDemo.Tools;

/// <summary>
/// Demonstrates the smallest in-process MCP tool integration: implement IAssistantTool and
/// register the instance with UseMcpTools/AddTool.
/// </summary>
internal sealed class CurrentDateTimeTool : IAssistantTool
{
    private static readonly ChineseLunisolarCalendar LunarCalendar = new();
    private readonly TimeProvider _timeProvider;

    public CurrentDateTimeTool(TimeProvider? timeProvider = null)
        => _timeProvider = timeProvider ?? TimeProvider.System;

    public AssistantToolDescriptor Descriptor { get; } = new(
        name: "get_current_date_time",
        displayName: "获取当前日期和时间",
        description: "获取运行 VoiceAssistant 的设备当前本地公历日期、农历日期、时间、星期和时区。",
        inputSchemaJson:
            """
            {
              "type": "object",
              "properties": {},
              "additionalProperties": false
            }
            """,
        outputSchemaJson:
            """
            {
              "type": "object",
              "properties": {
                "date": { "type": "string", "description": "本地日期，格式 yyyy-MM-dd" },
                "lunarDate": { "type": ["string", "null"], "description": "本地日期对应的农历日期，如 2025年闰六月初一；超出支持范围时为 null" },
                "time": { "type": "string", "description": "本地时间，格式 HH:mm:ss" },
                "dayOfWeek": { "type": "string", "description": "中文星期" },
                "localDateTime": { "type": "string", "description": "包含 UTC 偏移的本地 ISO 8601 时间" },
                "utcDateTime": { "type": "string", "description": "UTC ISO 8601 时间" },
                "timeZone": { "type": "string", "description": "设备本地时区标识" },
                "utcOffset": { "type": "string", "description": "UTC 偏移，格式 ±HH:mm" }
              },
              "required": [
                "date",
                "lunarDate",
                "time",
                "dayOfWeek",
                "localDateTime",
                "utcDateTime",
                "timeZone",
                "utcOffset"
              ],
              "additionalProperties": false
            }
            """,
        risk: AssistantToolRisk.ReadOnly,
        executionPolicy: AssistantToolExecutionPolicy.ForegroundOnly,
        timeout: TimeSpan.FromSeconds(2));

    public ValueTask<AssistantToolExecutionResult> ExecuteAsync(
        AssistantToolInvocation invocation,
        CancellationToken cancellationToken)
    {
        ArgumentNullException.ThrowIfNull(invocation);
        cancellationToken.ThrowIfCancellationRequested();

        DateTimeOffset now = _timeProvider.GetLocalNow();
        string date = now.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture);
        string? lunarDate = ToLunarDate(now.Date);
        string time = now.ToString("HH:mm:ss", CultureInfo.InvariantCulture);
        string dayOfWeek = ToChineseDayOfWeek(now.DayOfWeek);
        string utcOffset = FormatUtcOffset(now.Offset);
        string structuredContent = JsonSerializer.Serialize(new
        {
            date,
            lunarDate,
            time,
            dayOfWeek,
            localDateTime = now.ToString("yyyy-MM-dd'T'HH:mm:sszzz", CultureInfo.InvariantCulture),
            utcDateTime = now.UtcDateTime.ToString(
                "yyyy-MM-dd'T'HH:mm:ss'Z'",
                CultureInfo.InvariantCulture),
            timeZone = _timeProvider.LocalTimeZone.Id,
            utcOffset,
        });
        string lunarDateContent = lunarDate is not null
            ? $"农历是 {lunarDate}，"
            : "农历日期超出支持范围，";
        string content =
            $"当前日期是 {date}（{dayOfWeek}），{lunarDateContent}当前时间是 {time}，" +
            $"时区为 {_timeProvider.LocalTimeZone.Id}（UTC{utcOffset}）。";

        return ValueTask.FromResult(
            AssistantToolExecutionResult.Success(content, structuredContent));
    }

    private static string FormatUtcOffset(TimeSpan offset)
    {
        char sign = offset < TimeSpan.Zero ? '-' : '+';
        TimeSpan absolute = offset.Duration();
        return string.Create(
            CultureInfo.InvariantCulture,
            $"{sign}{absolute.Hours:00}:{absolute.Minutes:00}");
    }

    private static string? ToLunarDate(DateTime date)
    {
        if (date < LunarCalendar.MinSupportedDateTime || date > LunarCalendar.MaxSupportedDateTime)
            return null;

        int lunarYear = LunarCalendar.GetYear(date);
        int lunarMonth = LunarCalendar.GetMonth(date);
        int lunarDay = LunarCalendar.GetDayOfMonth(date);
        int leapMonth = LunarCalendar.GetLeapMonth(lunarYear);
        string leapPrefix = lunarMonth == leapMonth ? "闰" : string.Empty;
        // The calendar numbers the leap month and all following months one higher.
        if (leapMonth > 0 && lunarMonth >= leapMonth)
            lunarMonth--;

        const string digits = "一二三四五六七八九十";
        string monthName = lunarMonth switch
        {
            1 => "正月",
            11 => "冬月",
            12 => "腊月",
            _ => $"{digits[lunarMonth - 1]}月",
        };
        string dayName = lunarDay switch
        {
            <= 10 => $"初{digits[lunarDay - 1]}",
            < 20 => $"十{digits[lunarDay - 11]}",
            20 => "二十",
            < 30 => $"廿{digits[lunarDay - 21]}",
            _ => "三十",
        };
        return string.Create(CultureInfo.InvariantCulture, $"{lunarYear}年{leapPrefix}{monthName}{dayName}");
    }

    private static string ToChineseDayOfWeek(DayOfWeek dayOfWeek) => dayOfWeek switch
    {
        DayOfWeek.Monday => "星期一",
        DayOfWeek.Tuesday => "星期二",
        DayOfWeek.Wednesday => "星期三",
        DayOfWeek.Thursday => "星期四",
        DayOfWeek.Friday => "星期五",
        DayOfWeek.Saturday => "星期六",
        DayOfWeek.Sunday => "星期日",
        _ => throw new ArgumentOutOfRangeException(nameof(dayOfWeek)),
    };

    public void Dispose() { }
}
