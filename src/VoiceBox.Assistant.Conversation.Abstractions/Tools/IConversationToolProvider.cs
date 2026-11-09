namespace VoiceBox.Assistant.Conversation.Abstractions.Tools;

/// <summary>
/// 工具来源的公开窄接口: 提供工具目录, 并按身份调用一次工具。
/// </summary>
/// <remarks>
/// <para>
/// <see cref="StartAsync"/> 建立本次运行所需的连接并完成发现; 读取目录和调用均使用同一来源实例。
/// 启动失败必须清理已经创建的连接。<see cref="StopAsync"/> 结束本次运行但保留对象, 允许随后再次启动;
/// <see cref="IAsyncDisposable.DisposeAsync"/> 释放来源拥有的资源。
/// </para>
/// <para>
/// 来源只负责发现与执行。审批、Schema 校验、并发/重试预算和历史关联由消费该来源的会话决定;
/// 直接调用 <see cref="InvokeAsync"/> 不会自动执行任何默认会话策略。
/// </para>
/// <para>拥有来源的宿主负责协调生命周期, 并在关闭连接前取消并等待自己的活动调用。</para>
/// </remarks>
public interface IConversationToolProvider : IAsyncDisposable
{
    /// <summary>启动本次运行并建立外部连接。</summary>
    Task StartAsync(CancellationToken cancellationToken = default);

    /// <summary>读取本来源当前暴露的工具目录。</summary>
    ValueTask<IReadOnlyList<ConversationToolDefinition>> GetToolsAsync(
        CancellationToken cancellationToken = default);

    /// <summary>执行一次规范化工具调用。参数校验与审批由会话工具运行时负责。</summary>
    ValueTask<ConversationToolResult> InvokeAsync(
        ConversationToolCall call,
        CancellationToken cancellationToken = default);

    /// <summary>停止本次运行并断开连接。</summary>
    Task StopAsync();
}
