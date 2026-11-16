using System.Text;

namespace VoiceBox.Assistant.Conversation;

/// <summary>
/// 工具会话自行生成的固定播报文字。这些文字由宿主配置, 属于受信任内容, 不经过模型。
/// 每条模板中的 <c>{0}</c> 会替换为工具的 <c>DisplayName</c>, 也可以不使用占位符; 字面量花括号写成
/// <c>{{</c> 和 <c>}}</c>。默认值为中文。
/// </summary>
public sealed class ConversationToolAnnouncements
{
    /// <summary>
    /// 一批工具转入后台执行、且工具没有自己的 <c>ExecutionPrompt</c> 时的提示。
    /// <c>{0}</c> 为本批工具的显示名, 多个时以 <see cref="NameSeparator"/> 连接。
    /// </summary>
    public string BackgroundStarted { get; set; } = "我正在后台执行{0}任务，等会给你结果。";

    /// <summary>连接多个工具显示名的分隔符。</summary>
    public string NameSeparator { get; set; } = "、";

    /// <summary>后台结果无法由模型总结时的兜底文字: 工具执行成功。</summary>
    public string BackgroundSucceeded { get; set; } = "{0}任务已完成。";

    /// <summary>后台结果无法由模型总结时的兜底文字: 调用未获授权。</summary>
    public string BackgroundDenied { get; set; } = "{0}任务未获授权，未执行。";

    /// <summary>后台结果无法由模型总结时的兜底文字: 调用被取消。</summary>
    public string BackgroundCancelled { get; set; } = "{0}任务已取消。";

    /// <summary>后台结果无法由模型总结时的兜底文字: 有副作用的调用结果未知。</summary>
    public string BackgroundUnknownOutcome { get; set; } = "{0}任务执行状态暂时无法确认，请勿重复操作。";

    /// <summary>后台结果无法由模型总结时的兜底文字: 其他失败。</summary>
    public string BackgroundFailed { get; set; } = "{0}任务执行失败。";

    internal ConversationToolAnnouncements Clone()
    {
        var clone = new ConversationToolAnnouncements
        {
            BackgroundStarted = BackgroundStarted,
            NameSeparator = NameSeparator,
            BackgroundSucceeded = BackgroundSucceeded,
            BackgroundDenied = BackgroundDenied,
            BackgroundCancelled = BackgroundCancelled,
            BackgroundUnknownOutcome = BackgroundUnknownOutcome,
            BackgroundFailed = BackgroundFailed,
        };
        Validate(clone);
        return clone;
    }

    internal static void Validate(ConversationToolAnnouncements announcements)
    {
        ArgumentNullException.ThrowIfNull(announcements);
        ArgumentNullException.ThrowIfNull(announcements.NameSeparator, nameof(NameSeparator));
        ValidateTemplate(announcements.BackgroundStarted, nameof(BackgroundStarted));
        ValidateTemplate(announcements.BackgroundSucceeded, nameof(BackgroundSucceeded));
        ValidateTemplate(announcements.BackgroundDenied, nameof(BackgroundDenied));
        ValidateTemplate(announcements.BackgroundCancelled, nameof(BackgroundCancelled));
        ValidateTemplate(announcements.BackgroundUnknownOutcome, nameof(BackgroundUnknownOutcome));
        ValidateTemplate(announcements.BackgroundFailed, nameof(BackgroundFailed));
    }

    private static void ValidateTemplate(string template, string propertyName)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(template, propertyName);
        try
        {
            if (CompositeFormat.Parse(template).MinimumArgumentCount > 1)
                throw new ArgumentException("An announcement template may only use the {0} placeholder.", propertyName);
        }
        catch (FormatException exception)
        {
            throw new ArgumentException("The announcement template is not a valid format string.", propertyName, exception);
        }
    }
}
