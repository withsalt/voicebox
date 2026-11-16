namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 可选的静态来源元数据。连接建立前就能确定来源别名时实现它, 让宿主在冻结配置阶段
/// 拒绝重复或保留的别名。
/// </summary>
public interface IConversationToolProviderMetadata
{
    /// <summary>Aliases this provider will expose after StartAsync.</summary>
    IReadOnlyList<string> ServerAliases { get; }
}
