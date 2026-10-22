using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;

namespace VoiceBox.Assistant.Runtime.Publications.Dispatching
{
    /// <summary>
    /// 集中发布 VoiceAssistant 对外事件，并隔离订阅者异常。
    /// </summary>
    internal sealed class AssistantEventDispatcher
    {
        private object _sender;
        private readonly ILogger _logger;

        public AssistantEventDispatcher(object sender, ILogger logger)
        {
            _sender = sender ?? throw new ArgumentNullException(nameof(sender));
            _logger = logger ?? throw new ArgumentNullException(nameof(logger));
        }

        /// <summary>
        /// Replaces the placeholder identity Composition used before the facade existed. Every
        /// public event carries one sender for the lifetime of the runtime (EO-19), so this may
        /// only be called during composition, before any event can be raised.
        /// </summary>
        internal void BindSender(object sender)
            => _sender = sender ?? throw new ArgumentNullException(nameof(sender));

        public event EventHandler<AssistantStateChangedEventArgs>? StateChanged;
        public event EventHandler<AssistantWakeModeChangedEventArgs>? WakeModeChanged;
        public event EventHandler<AssistantWakeupDetectedEventArgs>? WakeupDetected;
        public event EventHandler<UserSpokeEventArgs>? UserSpoke;
        public event EventHandler<BargeInValidationEventArgs>? BargeInValidation;
        public event EventHandler<AssistantFaultedEventArgs>? Faulted;
        public event EventHandler<AssistantTextEventArgs>? AssistantText;
        public event EventHandler<AssistantResponseGeneratedEventArgs>? AssistantResponseGenerated;
        public event EventHandler<AssistantReplyEventArgs>? AssistantReplied;

        // 订阅者抛出的异常一律吞掉并记日志, 否则会冒泡到回合完成等关键代码,
        // 把 _gate.Open()、_asrStartTask 赋值这些关键收尾步骤跳过, 留下永远无法接收输入的死锁状态。
        // 行为与 FireRedVAD.SafeInvoke 一致。
        public void RaiseStateChanged(AssistantState from, AssistantState to)
        {
            var handler = StateChanged;
            if (handler is null) return;
            var args = new AssistantStateChangedEventArgs { From = from, To = to };
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantStateChangedEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "StateChanged 事件处理器抛出异常 (From={From} To={To})。", from, to);
                }
            }
        }

        public void RaiseWakeModeChanged(AssistantWakeMode from, AssistantWakeMode to)
        {
            var handler = WakeModeChanged;
            if (handler is null) return;
            var args = new AssistantWakeModeChangedEventArgs { From = from, To = to };
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantWakeModeChangedEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "WakeModeChanged 事件处理器抛出异常 (From={From} To={To})。", from, to);
                }
            }
        }

        public void RaiseWakeupDetected(AssistantWakeupDetectedEventArgs args)
        {
            var handler = WakeupDetected;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantWakeupDetectedEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "WakeupDetected 事件处理器抛出异常。");
                }
            }
        }

        public void RaiseUserSpoke(UserSpokeEventArgs args)
        {
            var handler = UserSpoke;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<UserSpokeEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "UserSpoke 事件处理器抛出异常。");
                }
            }
        }

        public void RaiseBargeInValidation(BargeInValidationEventArgs args)
        {
            var handler = BargeInValidation;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<BargeInValidationEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "BargeInValidation 事件处理器抛出异常 (Kind={Kind})。", args.Kind);
                }
            }
        }

        public void RaiseFaulted(AssistantFaultedEventArgs args)
        {
            var handler = Faulted;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantFaultedEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(
                        ex,
                        "Faulted event handler threw (Component={Component}).",
                        args.Component);
                }
            }
        }

        public void RaiseAssistantText(AssistantTextEventArgs args)
        {
            var handler = AssistantText;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantTextEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "AssistantText 事件处理器抛出异常 (IsThinking={IsThinking})。", args.IsThinking);
                }
            }
        }

        public void RaiseAssistantResponseGenerated(AssistantResponseGeneratedEventArgs args)
        {
            var handler = AssistantResponseGenerated;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantResponseGeneratedEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "AssistantResponseGenerated 事件处理器抛出异常。");
                }
            }
        }

        public void RaiseAssistantReplied(AssistantReplyEventArgs args)
        {
            var handler = AssistantReplied;
            if (handler is null) return;
            foreach (var subscriber in Delegate.EnumerateInvocationList(handler))
            {
                try
                {
                    ((EventHandler<AssistantReplyEventArgs>)subscriber).Invoke(_sender, args);
                }
                catch (Exception ex)
                {
                    _logger.LogError(ex, "AssistantReplied 事件处理器抛出异常。");
                }
            }
        }
    }
}
