using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Api.Events;
using VoiceBox.Assistant.Api.State;
using VoiceBox.Assistant.Runtime.Publications;
using VoiceBox.Assistant.Runtime.Publications.Dispatching;
using VoiceBox.Core.Services.WakeupService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class AssistantEventDispatcherCharacterizationTests
{
    [Fact]
    public void TextDelivery_DoesNotAllocateSubscriberArraysAndKeepsSnapshotSemantics()
    {
        var events = new AssistantEventDispatcher(new object(), NullLogger.Instance);
        var args = new AssistantTextEventArgs { Delta = "x", IsThinking = false };
        int deliveries = 0;
        EventHandler<AssistantTextEventArgs> second = (_, _) => deliveries++;
        EventHandler<AssistantTextEventArgs> first = (_, _) =>
        {
            deliveries++;
            events.AssistantText -= second;
        };
        events.AssistantText += first;
        events.AssistantText += second;
        events.RaiseAssistantText(args);
        Assert.Equal(2, deliveries);
        events.AssistantText -= first;
        events.AssistantText += second;
        for (int i = 0; i < 1_000; i++)
            events.RaiseAssistantText(args);

        long before = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 10_000; i++)
            events.RaiseAssistantText(args);
        long allocated = GC.GetAllocatedBytesForCurrentThread() - before;
        Assert.Equal(11_002, deliveries);
        Assert.InRange(allocated, 0, 1_024);
    }

    [Fact]
    public void EveryPublicEvent_IsolatesThrowingSubscriberAndRetainsSender()
    {
        object sender = new();
        var events = new AssistantEventDispatcher(sender, NullLogger.Instance);
        var delivered = new List<string>();

        events.StateChanged += (_, _) => throw new InvalidOperationException("state");
        events.StateChanged += (actual, _) => Record("StateChanged", actual);
        events.WakeModeChanged += (_, _) => throw new InvalidOperationException("wake-mode");
        events.WakeModeChanged += (actual, _) => Record("WakeModeChanged", actual);
        events.WakeupDetected += (_, _) => throw new InvalidOperationException("wakeup");
        events.WakeupDetected += (actual, _) => Record("WakeupDetected", actual);
        events.UserSpoke += (_, _) => throw new InvalidOperationException("user");
        events.UserSpoke += (actual, _) => Record("UserSpoke", actual);
        events.BargeInValidation += (_, _) => throw new InvalidOperationException("barge-in");
        events.BargeInValidation += (actual, _) => Record("BargeInValidation", actual);
        events.Faulted += (_, _) => throw new InvalidOperationException("fault");
        events.Faulted += (actual, _) => Record("Faulted", actual);
        events.AssistantText += (_, _) => throw new InvalidOperationException("text");
        events.AssistantText += (actual, _) => Record("AssistantText", actual);
        events.AssistantResponseGenerated += (_, _) => throw new InvalidOperationException("generated");
        events.AssistantResponseGenerated += (actual, _) => Record("AssistantResponseGenerated", actual);
        events.AssistantReplied += (_, _) => throw new InvalidOperationException("replied");
        events.AssistantReplied += (actual, _) => Record("AssistantReplied", actual);

        events.RaiseStateChanged(AssistantState.Idle, AssistantState.Listening);
        events.RaiseWakeModeChanged(AssistantWakeMode.Disabled, AssistantWakeMode.WaitingForWakeWord);
        events.RaiseWakeupDetected(new AssistantWakeupDetectedEventArgs
        {
            Wakeup = new WakeupEventArgs("wake", 1, 1, 1, 1),
        });
        events.RaiseUserSpoke(new UserSpokeEventArgs { Text = "hello", SpeechDuration = TimeSpan.Zero });
        events.RaiseBargeInValidation(new BargeInValidationEventArgs { Kind = BargeInValidationKind.Heard });
        events.RaiseFaulted(new AssistantFaultedEventArgs { Component = "test", Exception = new InvalidOperationException() });
        events.RaiseAssistantText(new AssistantTextEventArgs { Delta = "reply", IsThinking = false });
        events.RaiseAssistantResponseGenerated(new AssistantResponseGeneratedEventArgs { Text = "reply" });
        events.RaiseAssistantReplied(new AssistantReplyEventArgs { Text = "reply", Elapsed = TimeSpan.Zero });

        Assert.Equal(
            [
                "StateChanged", "WakeModeChanged", "WakeupDetected", "UserSpoke", "BargeInValidation",
                "Faulted", "AssistantText", "AssistantResponseGenerated", "AssistantReplied",
            ],
            delivered);

        void Record(string name, object? actual)
        {
            Assert.Same(sender, actual);
            delivered.Add(name);
        }
    }
}
