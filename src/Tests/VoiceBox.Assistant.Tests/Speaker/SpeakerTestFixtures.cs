using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using VoiceBox.Core.Services.AudioService.Models;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.SpeakerService.Models;

#pragma warning disable CS0067

namespace VoiceBox.Assistant.Tests;

/// <summary>
/// Deterministic speaker provider for identity-path tests. The owner similarity of a window is
/// encoded in the audio itself: an alternating-sign signal of amplitude <c>a</c> yields an embedding
/// whose cosine similarity to the owner axis is exactly <c>a</c>, so a test can script per-window
/// scores without any model. Silence yields nothing (the quality gate rejects it first).
/// </summary>
internal sealed class ScriptedSpeakerService : ISpeakerService
{
    private int _extractCount;

    public ScriptedSpeakerService(int windowSamples = 8000, int hopSamples = 1600, float threshold = 0.5f)
    {
        StreamingWindowSamples = windowSamples;
        StreamingHopSamples = hopSamples;
        Threshold = threshold;
    }

    public int EmbeddingSize => 8;
    public int SampleRate => 16000;
    public float Threshold { get; }
    public int StreamingWindowSamples { get; }
    public int StreamingHopSamples { get; }
    public IReadOnlyCollection<string> EnrolledSpeakers => [];
    public int ExtractCount => Volatile.Read(ref _extractCount);
    public int ExtractDelayMs { get; set; }
    public Exception? ExtractException { get; set; }
    public TaskCompletionSource? ExtractEntered { get; set; }
    public TaskCompletionSource? ExtractBlock { get; set; }
    public List<int> ExtractedLengths { get; } = [];
    public event EventHandler<SpeakerEventArgs>? OnResult;

    public static float[] OwnerAxis()
    {
        var embedding = new float[8];
        embedding[0] = 1f;
        return embedding;
    }

    public static float[] OtherAxis()
    {
        var embedding = new float[8];
        embedding[1] = 1f;
        return embedding;
    }

    /// <summary>Embedding whose cosine similarity to the owner axis equals <paramref name="ownerScore"/>.</summary>
    public static float[] EmbeddingWithOwnerScore(float ownerScore)
    {
        ownerScore = Math.Clamp(ownerScore, -1f, 1f);
        var embedding = new float[8];
        embedding[0] = ownerScore;
        embedding[1] = MathF.Sqrt(Math.Max(0f, 1f - ownerScore * ownerScore));
        return embedding;
    }

    /// <summary>Alternating-sign signal whose amplitude is the scripted owner score.</summary>
    public static float[] OwnerScoreAudio(float ownerScore, int samples)
    {
        var audio = new float[samples];
        for (int i = 0; i < samples; i++)
            audio[i] = (i & 1) == 0 ? ownerScore : -ownerScore;
        return audio;
    }

    public float[] ExtractEmbedding(ReadOnlySpan<float> samples)
    {
        Interlocked.Increment(ref _extractCount);
        lock (ExtractedLengths)
            ExtractedLengths.Add(samples.Length);
        ExtractEntered?.TrySetResult();
        ExtractBlock?.Task.GetAwaiter().GetResult();
        if (ExtractDelayMs > 0)
            Thread.Sleep(ExtractDelayMs);
        if (ExtractException is { } exception)
            throw exception;

        double sum = 0;
        for (int i = 0; i < samples.Length; i++)
            sum += Math.Abs(samples[i]);
        float amplitude = samples.Length == 0 ? 0f : (float)(sum / samples.Length);
        return EmbeddingWithOwnerScore(amplitude);
    }

    public float CompareEmbeddings(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2)
    {
        double dot = 0, norm1 = 0, norm2 = 0;
        for (int i = 0; i < embedding1.Length; i++)
        {
            dot += embedding1[i] * embedding2[i];
            norm1 += embedding1[i] * embedding1[i];
            norm2 += embedding2[i] * embedding2[i];
        }
        return norm1 <= 0 || norm2 <= 0 ? 0f : (float)(dot / Math.Sqrt(norm1 * norm2));
    }

    public float Verify(ReadOnlySpan<float> samples1, ReadOnlySpan<float> samples2)
        => CompareEmbeddings(ExtractEmbedding(samples1), ExtractEmbedding(samples2));

    public void Enroll(string speakerName, ReadOnlySpan<float> samples) { }
    public bool Unenroll(string speakerName) => false;
    public void FeedAudio(ReadOnlySpan<float> samples) { }
    public void Reset() { }
    public void Dispose() { }
}

internal static class SpeakerTestProfiles
{
    public static SpeakerCaptureDescriptor Descriptor(int sampleRate = 16000, bool aec = false)
        => SpeakerCaptureDescriptor.Create(sampleRate, 1, SampleFormat.Float32, false, aec, aec ? "TestApm" : null);

    /// <summary>
    /// Owner bank built around the owner axis with two slightly different prototypes. The second
    /// prototype is perturbed along axis 2, which no scripted embedding uses, so a scripted owner
    /// score <c>s</c> reports exactly <c>s</c> against this bank.
    /// </summary>
    public static SpeakerPrototypeBank OwnerBank()
    {
        float[] centroid = ScriptedSpeakerService.OwnerAxis();
        var second = new float[8];
        second[0] = 0.99f;
        second[2] = MathF.Sqrt(1f - 0.99f * 0.99f);
        return new SpeakerPrototypeBank(
            centroid,
            [new SpeakerPrototype("p1", ScriptedSpeakerService.OwnerAxis(), 1f), new SpeakerPrototype("p2", second, 0.99f)]);
    }

    public static OwnerSpeakerProfileVariant Variant(
        SpeakerCaptureDescriptor descriptor,
        bool includeDoubleTalk = true)
    {
        var banks = new Dictionary<SpeakerPurposeBank, SpeakerPrototypeBank>
        {
            [SpeakerPurposeBank.CleanNearEnd] = OwnerBank(),
        };
        if (includeDoubleTalk)
            banks[SpeakerPurposeBank.DoubleTalkNearEnd] = OwnerBank();
        return new OwnerSpeakerProfileVariant(
            "variant-1",
            descriptor.FrontendFingerprint,
            descriptor.TapId,
            DateTimeOffset.UtcNow,
            banks,
            OwnerSpeakerProfileQuality.Empty);
    }

    public static OwnerSpeakerProfile Profile(
        ISpeakerService speaker,
        SpeakerCaptureDescriptor descriptor,
        bool includeDoubleTalk = true)
        => new(
            OwnerSpeakerProfile.OwnerSpeakerId,
            DateTimeOffset.UtcNow,
            SpeakerProfileStore.DescribeProvider(speaker),
            [Variant(descriptor, includeDoubleTalk)]);

    public static async Task<string> WriteProfileAsync(
        ISpeakerService speaker,
        SpeakerCaptureDescriptor descriptor,
        string? path = null,
        bool includeDoubleTalk = true)
    {
        path ??= Path.Combine(Path.GetTempPath(), $"voicebox-owner-{Guid.NewGuid():N}.json");
        await SpeakerProfileStore.SaveAsync(speaker, path, Profile(speaker, descriptor, includeDoubleTalk));
        return path;
    }

    public static VoiceAssistantSpeakerOptions Options(string profilePath)
        => new()
        {
            ProfilePath = profilePath,
            Threshold = 0.5f,
            Confirmation = { MinimumSpeechMs = 500, InferenceTimeoutMs = 250, WindowCount = 3, RequiredMatches = 2 },
        };

    public static SpeakerVerificationPolicy Policy(
        ISpeakerService speaker,
        VoiceAssistantSpeakerOptions options,
        bool wakeup = true,
        bool bargeIn = true,
        bool singleTalk = false)
        => SpeakerVerificationPolicy.Create(speaker, options, 16000, 1, new SpeakerPurposeSet(wakeup, bargeIn, singleTalk));

    public static SpeakerCaptureFrame Frame(
        long streamEpoch,
        long startSample,
        float[] audio,
        long sequence = 0,
        bool discontinuity = false,
        string fingerprint = "test-fingerprint")
        => new(
            new CaptureRange(streamEpoch, sequence, sequence, startSample, startSample + audio.Length, 1),
            16000,
            1,
            audio,
            discontinuity,
            fingerprint);
}
