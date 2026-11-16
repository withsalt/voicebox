namespace VoiceBox.Assistant.Conversation.Abstractions;

/// <summary>
/// 会话的可选启动/停止能力。需要建立外部连接或运行后台工作域的会话实现该接口;
/// 构造完成即可使用的会话不需要实现它。
/// </summary>
/// <remarks>
/// Stop 取消会话运行并断开临时连接, 保留业务历史; 之后允许再次 Start 运行。
/// </remarks>
public interface IConversationLifecycle
{
    /// <summary>启动本次运行。失败时实现必须自行清理已经启动的部分。</summary>
    Task StartAsync(CancellationToken cancellationToken = default);

    /// <summary>停止本次运行并等待会话拥有的运行任务退出。</summary>
    Task StopAsync();
}
