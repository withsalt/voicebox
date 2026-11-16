namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 一个由当前进程承载的具体工具。工具内部仍可访问 HTTP、数据库等外部服务。
/// </summary>
/// <remarks>
/// <para>
/// 工具的生命周期由拥有它的宿主管理。宿主在释放工具前应先取消并等待正在执行的调用；
/// <see cref="IDisposable.Dispose"/> 实现应支持重复调用。
/// 需要向会话暴露动态工具目录时, 使用 <see cref="IConversationToolProvider"/>。
/// </para>
/// <para>
/// 工具必须响应取消令牌。消费方可以在 <see cref="ConversationToolDescriptor.Timeout"/> 到期或调用被取消后
/// 不再等待本次调用; 此后才返回的结果会被丢弃, 仍在运行的调用可能与同一工具的下一次调用重叠。
/// </para>
/// </remarks>
public interface IConversationTool : IDisposable
{
    /// <summary>工具的受信任元数据与模型可见 Schema。</summary>
    ConversationToolDescriptor Descriptor { get; }

    /// <summary>执行一次调用。调用 ID、工具身份和参数由消费方构造。</summary>
    ValueTask<ConversationToolResult> ExecuteAsync(ConversationToolCall call, CancellationToken cancellationToken);
}
