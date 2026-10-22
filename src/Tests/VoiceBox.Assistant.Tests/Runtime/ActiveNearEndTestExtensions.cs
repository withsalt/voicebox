using System.Collections.Concurrent;
using System.Reflection;
using VoiceBox.Assistant.Modules.BargeIn.Controllers;
using VoiceBox.Assistant.Modules.Recognition.Contracts;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Test-only driver for recognition-port helpers that are not part of the production
/// <see cref="IRecognitionAudioPort"/> surface. ASR pump and session lifecycle stay owned by
/// RecognitionModule; these helpers keep data-plane tests from putting those methods on
/// ActiveNearEndCoordinator.
/// </summary>
internal interface IRecognitionAudioTestDriver
{
    ValueTask AttachSessionAsync(long turnId, string sessionId);

    bool HasPendingAudio { get; }
}

/// <summary>
/// Test-only access to the recognition audio port.
/// </summary>
internal static class ActiveNearEndTestExtensions
{
    private static readonly ConcurrentDictionary<ActiveNearEndCoordinator, PumpState> Pumps = new();

    public static ValueTask AttachAsrSessionAsync(
        this ActiveNearEndCoordinator coordinator,
        long turnId,
        string sessionId)
        => TestDriver(coordinator).AttachSessionAsync(turnId, sessionId);

    public static bool HasPendingAsrFeedAudio(this ActiveNearEndCoordinator coordinator)
        => TestDriver(coordinator).HasPendingAudio;

    public static ValueTask DrainAsrFeedQueueAsync(this ActiveNearEndCoordinator coordinator)
        => AudioPort(coordinator).DrainAudioAsync();

    public static Task StartAsrFeedPump(
        this ActiveNearEndCoordinator coordinator,
        CancellationToken cancellationToken)
    {
        PumpState state = Pumps.GetOrAdd(coordinator, static _ => new PumpState());
        if (Interlocked.Exchange(ref state.Started, 1) != 0)
            throw new InvalidOperationException("ASR feed pump can only be started once.");
        return AudioPort(coordinator).StartAudioPump(cancellationToken);
    }

    public static void SignalAsrFeedPump(this ActiveNearEndCoordinator coordinator)
    {
        // RecognitionModule signals its own bounded pump when admission occurs, so this test
        // helper intentionally has no state on the near-end owner.
    }

    private static IRecognitionAudioPort AudioPort(ActiveNearEndCoordinator coordinator)
        => (IRecognitionAudioPort)typeof(ActiveNearEndCoordinator)
            .GetField("_recognitionAudio", BindingFlags.Instance | BindingFlags.NonPublic)!
            .GetValue(coordinator)!;

    private static IRecognitionAudioTestDriver TestDriver(ActiveNearEndCoordinator coordinator)
        => AudioPort(coordinator) as IRecognitionAudioTestDriver
            ?? throw new InvalidOperationException(
                "The injected recognition audio port does not expose test session-admission helpers.");

    private sealed class PumpState
    {
        public int Started;
    }
}
