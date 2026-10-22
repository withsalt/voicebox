using VoiceBox.Assistant.Modules.BargeIn.Models;
using VoiceBox.Assistant.Modules.BargeIn.Policies;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class BargeInDetectorTests
{
    [Fact]
    public void AddAudio_WaitsForMinimumDurationBeforeConfirming()
    {
        var detector = new BargeInDetector(minDurationMs: 100, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(interruptedTurnId: 7, responseCts: null);

        Assert.False(detector.AddAudio(new float[99]));
        Assert.True(detector.AddAudio([1f]));
        Assert.Equal(100, detector.CandidateSamples);
    }

    [Fact]
    public void AddAudio_RequiresMinimumRms()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0.5f, sampleRate: 16000, channels: 1);
        detector.BeginCandidate(interruptedTurnId: 7, responseCts: null);

        Assert.False(detector.AddAudio([0.1f, 0.1f]));
        Assert.True(detector.AddAudio([1f, 1f]));
        Assert.True(detector.ComputeRms() >= 0.5);
    }

    [Fact]
    public void AddAudio_IgnoresInitialSuppressionWindowForConfirmation()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(
            interruptedTurnId: 7,
            responseCts: null,
            new BargeInConfirmationSettings(minDurationMs: 100, minRms: 0.5f, initialSuppressionMs: 50));

        Assert.Equal(100, detector.CurrentSettings.MinDurationMs);
        Assert.Equal(0.5f, detector.CurrentSettings.MinRms);
        Assert.Equal(50, detector.CurrentSettings.InitialSuppressionMs);
        Assert.False(detector.AddAudio(ConstantSamples(1f, 50)));
        Assert.False(detector.AddAudio(ConstantSamples(0.6f, 99)));
        Assert.True(detector.AddAudio([0.6f]));
        Assert.Equal(150, detector.CandidateSamples);
        Assert.Equal(100, detector.ConfirmationSamples);
        Assert.True(detector.ComputeConfirmationRms() >= 0.5);
    }

    [Fact]
    public void AddAudio_RejectsLongLowEnergyPlaybackResidual()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(
            interruptedTurnId: 7,
            responseCts: null,
            new BargeInConfirmationSettings(minDurationMs: 100, minRms: 0.08f, initialSuppressionMs: 50));

        Assert.False(detector.AddAudio(ConstantSamples(0.05f, 1000)));
        Assert.Equal(950, detector.ConfirmationSamples);
        Assert.True(detector.ComputeConfirmationRms() < 0.08);
    }

    [Fact]
    public void AddAudio_RestartsUnconfirmedCandidateAfterMaximumDuration()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(
            interruptedTurnId: 7,
            responseCts: null,
            new BargeInConfirmationSettings(
                minDurationMs: 100,
                minRms: 0.5f,
                maxCandidateMs: 120,
                recognitionAudioLimitMs: 500));

        Assert.False(detector.AddAudio(ConstantSamples(0.1f, 180)));

        Assert.True(detector.IsCandidateActive);
        Assert.Equal(60, detector.CandidateSamples);
        Assert.Equal(60, detector.ConfirmationSamples);
        Assert.Single(detector.PendingAudio);
        Assert.Equal(60, detector.PendingAudio[0].Length);
    }

    [Fact]
    public void MarkConfirmedKeepingPendingAudio_RetainsCapturedAudioForAsr()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 16000, channels: 1);
        detector.BeginCandidate(interruptedTurnId: 42, responseCts: null);
        detector.AddAudio([0.4f, 0.5f]);

        detector.MarkConfirmedKeepingPendingAudio();

        Assert.False(detector.IsCandidateActive);
        Assert.Single(detector.PendingAudio);
        Assert.Equal([0.4f, 0.5f], detector.PendingAudio[0]);
    }

    [Fact]
    public void MarkConfirmedKeepingPendingAudio_DropsRecognitionSuppressedAudioForAsr()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(
            interruptedTurnId: 42,
            responseCts: null,
            new BargeInConfirmationSettings(
                minDurationMs: 100,
                minRms: 0,
                initialSuppressionMs: 100,
                recognitionSuppressionMs: 100));

        Assert.False(detector.AddAudio(ConstantSamples(0.1f, 60)));
        Assert.False(detector.AddAudio(ConstantSamples(0.2f, 40)));
        Assert.True(detector.AddAudio(ConstantSamples(0.3f, 100)));

        detector.MarkConfirmedKeepingPendingAudio();

        Assert.Single(detector.PendingAudio);
        Assert.Equal(100, detector.PendingAudio[0].Length);
        Assert.All(detector.PendingAudio[0], sample => Assert.Equal(0.3f, sample));
    }

    [Fact]
    public void MarkConfirmedKeepingPendingAudio_RetainsOnlyRecentRecognitionWindow()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(
            interruptedTurnId: 42,
            responseCts: null,
            new BargeInConfirmationSettings(
                minDurationMs: 100,
                minRms: 0.5f,
                recognitionAudioLimitMs: 150));

        Assert.False(detector.AddAudio(ConstantSamples(0.1f, 120)));
        Assert.True(detector.AddAudio(ConstantSamples(1.0f, 100)));

        detector.MarkConfirmedKeepingPendingAudio();

        float[] pending = Flatten(detector.PendingAudio);
        Assert.Equal(150, pending.Length);
        Assert.All(pending[..50], sample => Assert.Equal(0.1f, sample));
        Assert.All(pending[50..], sample => Assert.Equal(1.0f, sample));
    }

    [Fact]
    public void SpeakerPending_RetainsCandidateStartAndStopsAtSafetyLimit()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 1000, channels: 1);
        detector.BeginCandidate(
            interruptedTurnId: 42,
            responseCts: null,
            new BargeInConfirmationSettings(
                minDurationMs: 100,
                minRms: 0,
                maxCandidateMs: 1200,
                recognitionAudioLimitMs: 1000));

        Assert.True(detector.AddAudio(ConstantSamples(0.1f, 100)));
        Assert.True(detector.IsConfirmationMet);
        Assert.True(detector.AddAudio(ConstantSamples(0.2f, 1100)));
        Assert.True(detector.AddAudio(ConstantSamples(0.3f, 200)));

        float[] pending = Flatten(detector.PendingAudio);
        Assert.Equal(1200, detector.CandidateSamples);
        Assert.Equal(1200, pending.Length);
        Assert.All(pending[..100], sample => Assert.Equal(0.1f, sample));
        Assert.All(pending[100..], sample => Assert.Equal(0.2f, sample));
        Assert.DoesNotContain(0.3f, pending);
    }

    [Fact]
    public void Clear_RemovesCandidateAndPendingAudio()
    {
        var detector = new BargeInDetector(minDurationMs: 0, minRms: 0, sampleRate: 16000, channels: 1);
        detector.BeginCandidate(interruptedTurnId: 42, responseCts: null);
        detector.AddAudio([0.4f, 0.5f]);

        detector.Clear();

        Assert.False(detector.IsCandidateActive);
        Assert.Empty(detector.PendingAudio);
        Assert.Equal(0, detector.CandidateSamples);
        Assert.Equal(0, detector.ComputeRms());
    }

    private static float[] ConstantSamples(float value, int count)
    {
        var samples = new float[count];
        Array.Fill(samples, value);
        return samples;
    }

    private static float[] Flatten(IReadOnlyList<float[]> chunks)
    {
        int length = 0;
        foreach (float[] chunk in chunks)
            length += chunk.Length;

        var result = new float[length];
        int offset = 0;
        foreach (float[] chunk in chunks)
        {
            Array.Copy(chunk, 0, result, offset, chunk.Length);
            offset += chunk.Length;
        }

        return result;
    }
}
