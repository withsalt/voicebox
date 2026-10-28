using System.Collections.Concurrent;
using VoiceBox.Assistant.Modules.Recognition.Contracts;

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Test-only driver for recognition-port helpers that are not part of the production
/// <see cref="IRecognitionAudioPort"/> surface. ASR pump and session lifecycle stay owned by
/// RecognitionModule; these helpers keep data-plane tests from putting those methods on the
/// near-end module.
/// </summary>
internal interface IRecognitionAudioTestDriver
{
    ValueTask AttachSessionAsync(long turnId, string sessionId);

    bool HasPendingAudio { get; }
}

/// <summary>
/// Test-only access to the recognition audio port behind a composed near-end module.
/// </summary>
internal static class ActiveNearEndTestExtensions
{
    private static readonly ConcurrentDictionary<NearEndTestHarness, PumpState> Pumps = new();

    public static ValueTask AttachAsrSessionAsync(
        this NearEndTestHarness harness,
        long turnId,
        string sessionId)
        => TestDriver(harness).AttachSessionAsync(turnId, sessionId);

    public static bool HasPendingAsrFeedAudio(this NearEndTestHarness harness)
        => TestDriver(harness).HasPendingAudio;

    public static ValueTask DrainAsrFeedQueueAsync(this NearEndTestHarness harness)
        => harness.RecognitionAudio.DrainAudioAsync();

    public static Task StartAsrFeedPump(
        this NearEndTestHarness harness,
        CancellationToken cancellationToken)
    {
        PumpState state = Pumps.GetOrAdd(harness, static _ => new PumpState());
        if (Interlocked.Exchange(ref state.Started, 1) != 0)
            throw new InvalidOperationException("ASR feed pump can only be started once.");
        return harness.RecognitionAudio.StartAudioPump(cancellationToken);
    }

    public static void SignalAsrFeedPump(this NearEndTestHarness harness)
    {
        // RecognitionModule signals its own bounded pump when admission occurs, so this test
        // helper intentionally has no state on the near-end module.
    }

    private static IRecognitionAudioTestDriver TestDriver(NearEndTestHarness harness)
        => harness.RecognitionAudio as IRecognitionAudioTestDriver
            ?? throw new InvalidOperationException(
                "The injected recognition audio port does not expose test session-admission helpers.");

    private sealed class PumpState
    {
        public int Started;
    }
}
