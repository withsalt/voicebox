using System.Text.Json;
using VoiceBox.Assistant.Api.Tools.Models;

namespace VoiceBox.Assistant.Api.Tools.Contracts;

/// <summary>
/// 一个可由 LLM 结构化调用的进程内工具。实现该接口并通过 UseMcpTools/AddTool 注册即可。
/// </summary>
/// <remarks>
/// 注册后的工具由 VoiceAssistant 管理生命周期：Dispose 时先取消并等待正在执行的调用，
/// 再释放工具。仅 StopAsync 不会释放工具；工具的 Dispose 实现应支持重复调用。
/// </remarks>
public interface IAssistantTool : IDisposable
{
    AssistantToolDescriptor Descriptor { get; }

    ValueTask<AssistantToolExecutionResult> ExecuteAsync(AssistantToolInvocation invocation, CancellationToken cancellationToken);
}
