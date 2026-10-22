using Microsoft.ML.OnnxRuntime;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Configuration;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.SpeakerService;
using VoiceBox.Core.Services.SpeakerService.Models;
using VoiceBox.DTD.RenderAware.Models;
using Xunit;

#pragma warning disable CS0067

namespace VoiceBox.Assistant.Tests;

public sealed class VoiceAssistantOptionsValidatorTests
{
    [Fact]
    public void Validate_AcceptsDefaultOptions()
    {
        VoiceAssistantOptionsValidator.Validate(new VoiceAssistantOptions(), "options");
    }

    [Fact]
    public void CloneOptions_PreservesPlaybackDuckingSettings()
    {
        var options = new VoiceAssistantOptions();
        options.BargeIn.PlaybackDuckGain = 0.25f;
        options.BargeIn.PlaybackDuckReleaseMs = 300;

        VoiceAssistantOptions clone = options.Clone();

        Assert.Equal(0.25f, clone.BargeIn.PlaybackDuckGain);
        Assert.Equal(300, clone.BargeIn.PlaybackDuckReleaseMs);
    }

    [Theory]
    [InlineData(-0.1f, 200)]
    [InlineData(1.1f, 200)]
    [InlineData(float.NaN, 200)]
    [InlineData(float.PositiveInfinity, 200)]
    [InlineData(0.15f, -1)]
    [InlineData(0.15f, 60001)]
    public void Validate_RejectsInvalidPlaybackDuckingSettings(float gain, int releaseMs)
    {
        var options = new VoiceAssistantOptions();
        options.BargeIn.PlaybackDuckGain = gain;
        options.BargeIn.PlaybackDuckReleaseMs = releaseMs;

        Assert.Throws<ArgumentException>(() => VoiceAssistantOptionsValidator.Validate(options, "options"));
    }

    [Fact]
    public void DefaultWakeupReplayLeadIn_LeavesCtcAlignmentMargin()
    {
        Assert.Equal(400, new VoiceAssistantOptions().Wakeup.ReplayLeadInMs);
    }

    [Fact]
    public void DefaultToolOptions_KeepACappedContextBudget()
    {
        VoiceAssistantToolOptions tools = new VoiceAssistantOptions().Tools;

        // An MCP result may be a megabyte, so unbounded retention would eventually exhaust the
        // model context. Both the window and the per-result cap must stay finite by default.
        Assert.Equal(3, tools.ToolTranscriptRetentionTurns);
        Assert.Equal(8 * 1024, tools.MaxPersistedToolResultBytes);
        Assert.Equal(4, tools.MaxParallelToolCalls);
        Assert.Equal(2, tools.MaxToolRetryAttempts);
        Assert.True(tools.ValidateToolArguments);
    }

    [Theory]
    [InlineData(-1, 8192, 4, 2, 200, 60000)]
    [InlineData(3, 255, 4, 2, 200, 60000)]
    [InlineData(3, 8192, 0, 2, 200, 60000)]
    [InlineData(3, 8192, 65, 2, 200, 60000)]
    [InlineData(3, 8192, 4, -1, 200, 60000)]
    [InlineData(3, 8192, 4, 9, 200, 60000)]
    [InlineData(3, 8192, 4, 2, -1, 60000)]
    [InlineData(3, 8192, 4, 2, 200, 0)]
    public void Validate_RejectsOutOfRangeToolOptions(
        int retentionTurns,
        int maxResultBytes,
        int maxParallel,
        int retryAttempts,
        int retryBaseDelayMs,
        int retryBudgetMs)
    {
        var options = new VoiceAssistantOptions
        {
            Tools = new VoiceAssistantToolOptions
            {
                ToolTranscriptRetentionTurns = retentionTurns,
                MaxPersistedToolResultBytes = maxResultBytes,
                MaxParallelToolCalls = maxParallel,
                MaxToolRetryAttempts = retryAttempts,
                ToolRetryBaseDelayMs = retryBaseDelayMs,
                MaxToolRetryBudgetMs = retryBudgetMs,
            },
        };

        Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.Validate(options, "options"));
    }

    [Fact]
    public void Validate_RejectsNullToolOptions()
    {
        var options = new VoiceAssistantOptions { Tools = null! };

        Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.Validate(options, "options"));
    }

    [Fact]
    public void CloneOptions_PreservesToolOptions()
    {
        var options = new VoiceAssistantOptions();
        options.Tools.ToolTranscriptRetentionTurns = 7;
        options.Tools.MaxParallelToolCalls = 2;
        options.Tools.ValidateToolArguments = false;

        VoiceAssistantOptions clone = options.Clone();

        Assert.Equal(7, clone.Tools.ToolTranscriptRetentionTurns);
        Assert.Equal(2, clone.Tools.MaxParallelToolCalls);
        Assert.False(clone.Tools.ValidateToolArguments);
    }

    [Fact]
    public void CloneOptions_PreservesDtdOnnxExecutionSettings()
    {
        // Composition clones options defensively. The DTD model settings inherit
        // OnnxExecutionOptions, which carries a SessionOptionsFactory delegate and CUDA tuning.
        // A clone that drops them silently downgrades a configured GPU/custom session to the CPU
        // default, and the runner additionally changes its IO-binding strategy based on them.
        // The delegate is only compared, never invoked: building a real SessionOptions needs the
        // ONNX native library, which this assembly does not load.
        static SessionOptions Factory() => throw new NotSupportedException("never invoked");
        var options = new VoiceAssistantOptions();
        options.BargeIn.Dtd.Model.ExecutionProvider = OnnxExecutionProvider.Cuda;
        options.BargeIn.Dtd.Model.GpuDeviceId = 3;
        options.BargeIn.Dtd.Model.ThreadsNum = 5;
        options.BargeIn.Dtd.Model.CudaTuning = new CudaExecutionTuning
        {
            CudnnConvAlgoSearch = "EXHAUSTIVE",
            DoCopyInDefaultStream = false,
        };
        options.BargeIn.Dtd.Model.SessionOptionsFactory = Factory;

        VoiceAssistantOptions clone = options.Clone();

        RenderAwareDtdOptions model = clone.BargeIn.Dtd.Model;
        Assert.Equal(OnnxExecutionProvider.Cuda, model.ExecutionProvider);
        Assert.Equal(3, model.GpuDeviceId);
        Assert.Equal(5, model.ThreadsNum);
        Assert.NotNull(model.CudaTuning);
        Assert.Equal("EXHAUSTIVE", model.CudaTuning!.CudnnConvAlgoSearch);
        Assert.False(model.CudaTuning.DoCopyInDefaultStream);
        // Delegate equality, not reference equality: converting a static local function twice
        // yields distinct wrappers, but an equal delegate provably targets the caller's method.
        Assert.Equal<Func<SessionOptions>?>(Factory, model.SessionOptionsFactory);
    }

    [Fact]
    public void CloneOptions_PreservesCustomSessionFactoryOnEveryClonedOptionsGraph()
    {
        // UseOptions, AddBargeIn and the composition snapshot all clone. Every hop must keep the
        // caller's delegate, otherwise the configured session silently reverts to the default.
        // The sentinel identifies the caller's own method without needing the ONNX native library.
        var marker = new object();
        Func<SessionOptions> factory = () => throw new NotSupportedException(marker.ToString());
        var options = new VoiceAssistantOptions();
        options.BargeIn.Dtd.Model.SessionOptionsFactory = factory;

        VoiceAssistantOptions first = options.Clone();
        VoiceAssistantOptions second = first.Clone();

        // Repeated cloning must not erode the delegate: the composition snapshot clones again
        // after the builder already cloned once.
        Assert.Same(factory, first.BargeIn.Dtd.Model.SessionOptionsFactory);
        Assert.Same(factory, second.BargeIn.Dtd.Model.SessionOptionsFactory);
        // The source options must be left exactly as the caller supplied them.
        Assert.Same(factory, options.BargeIn.Dtd.Model.SessionOptionsFactory);
    }

    [Fact]
    public void CloneOptions_IsReadOnlyWithRespectToTheSourceUnderConcurrentClones()
    {
        // A hosted container hands the same IOptions<VoiceAssistantOptions>.Value instance to every
        // consumer, so two composition roots can clone one options object at the same time. Cloning
        // must therefore never mutate the source, not even transiently: a concurrent clone that
        // observed a half-detached graph would silently produce a copy without the factory.
        Func<SessionOptions> factory = () => throw new NotSupportedException("never invoked");
        var options = new VoiceAssistantOptions();
        options.BargeIn.Dtd.Model.SessionOptionsFactory = factory;

        var clones = new VoiceAssistantOptions[64];
        Parallel.For(0, clones.Length, index => clones[index] = options.Clone());

        Assert.Same(factory, options.BargeIn.Dtd.Model.SessionOptionsFactory);
        Assert.All(clones, clone =>
            Assert.Same(factory, clone.BargeIn.Dtd.Model.SessionOptionsFactory));
    }

    [Theory]
    [MemberData(nameof(InvalidOptions))]
    public void Validate_RejectsInvalidOptions(VoiceAssistantOptions options, string expectedMessage)
    {
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.Validate(options, "options"));

        Assert.Contains(expectedMessage, ex.Message);
    }

    public static IEnumerable<object[]> InvalidOptions()
    {
        yield return [new VoiceAssistantOptions { MinSentenceChars = 0 }, "MinSentenceChars"];
        yield return [new VoiceAssistantOptions { MaxHistoryTurns = 0 }, "MaxHistoryTurns"];
        yield return [new VoiceAssistantOptions { MaxResponseTokens = 0 }, "MaxResponseTokens"];
        yield return [new VoiceAssistantOptions { PlaybackTailBufferMs = -1 }, "PlaybackTailBufferMs"];
        yield return [new VoiceAssistantOptions { PlaybackPrerollMs = -1 }, "PlaybackPrerollMs"];
        yield return [new VoiceAssistantOptions { Conversation = null! }, "Conversation"];
        yield return [new VoiceAssistantOptions { Conversation = { ContinuousConversationIdleTimeout = TimeSpan.FromMilliseconds(-1) } }, "Conversation.ContinuousConversationIdleTimeout"];
        yield return [new VoiceAssistantOptions { Conversation = { EndConversationKeywords = { " " } } }, "Conversation.EndConversationKeywords"];
        yield return [new VoiceAssistantOptions { Wakeup = null! }, "Wakeup"];
        yield return [new VoiceAssistantOptions { Wakeup = { PreRollMs = -1 } }, "Wakeup.PreRollMs"];
        yield return [new VoiceAssistantOptions { Wakeup = { ReplayLeadInMs = -1 } }, "Wakeup.ReplayLeadInMs"];
        yield return [new VoiceAssistantOptions { Speaker = null! }, "Speaker"];
        yield return [new VoiceAssistantOptions { Speaker = { ProfilePath = "" } }, "Speaker.ProfilePath"];
        yield return [new VoiceAssistantOptions { Speaker = { AudioRetentionMs = 999 } }, "Speaker.AudioRetentionMs"];
        yield return [new VoiceAssistantOptions { Speaker = { AudioRetentionMs = 60001 } }, "Speaker.AudioRetentionMs"];
        yield return [new VoiceAssistantOptions { Speaker = { InferenceQueueCapacity = 0 } }, "Speaker.InferenceQueueCapacity"];
        yield return [new VoiceAssistantOptions { Speaker = { MaxRealtimeWindowMs = 99 } }, "Speaker.MaxRealtimeWindowMs"];
        yield return [new VoiceAssistantOptions { Speaker = { PrefetchMaxWindows = 0 } }, "Speaker.PrefetchMaxWindows"];
        yield return [new VoiceAssistantOptions { Speaker = { MinFrameRms = -0.1f } }, "Speaker.MinFrameRms"];
        yield return [new VoiceAssistantOptions { Speaker = { PrefixHighConfidenceMs = -1 } }, "Speaker.PrefixHighConfidenceMs"];
        yield return [new VoiceAssistantOptions { Speaker = { WakeupPostTriggerEvidenceMs = -1 } }, "Speaker.WakeupPostTriggerEvidenceMs"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { RequiredMatches = 0 } } }, "Speaker.Confirmation.RequiredMatches"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { WindowCount = 65 } } }, "Speaker.Confirmation.WindowCount"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { MinimumSpeechMs = -1 } } }, "Speaker.Confirmation.MinimumSpeechMs"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { InferenceTimeoutMs = 60001 } } }, "Speaker.Confirmation.InferenceTimeoutMs"];
        yield return [new VoiceAssistantOptions { Speaker = { Threshold = float.NaN } }, "Speaker.Threshold"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { Threshold = float.NaN } } }, "Speaker.Confirmation.Threshold"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { Threshold = 1.1f } } }, "Speaker.Confirmation.Threshold"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { IndeterminateScoreMargin = -0.01f } } }, "Speaker.Confirmation.IndeterminateScoreMargin"];
        yield return [new VoiceAssistantOptions { Speaker = { Confirmation = { RecentOwnerPriorMs = 60001 } } }, "Speaker.Confirmation.RecentOwnerPriorMs"];
        yield return [new VoiceAssistantOptions { Speaker = { WakeupStrongScoreMargin = 101 } }, "Speaker.WakeupStrongScoreMargin"];
        yield return [new VoiceAssistantOptions
        {
            Speaker = { Confirmation = { Threshold = 0.6f, HighConfidenceThreshold = 0.6f } },
        }, "Speaker.Confirmation.HighConfidenceThreshold"];
        yield return [new VoiceAssistantOptions { TtsSpeed = 0 }, "TtsSpeed"];
        yield return [new VoiceAssistantOptions { TtsSpeed = float.NaN }, "TtsSpeed"];
        yield return [new VoiceAssistantOptions { Listening = null! }, "Listening"];
        yield return [new VoiceAssistantOptions { Listening = { MinDurationMs = -1 } }, "Listening.MinDurationMs"];
        yield return [new VoiceAssistantOptions { Listening = { MinRms = -1 } }, "Listening.MinRms"];
        yield return [new VoiceAssistantOptions { Listening = { MinRms = float.NaN } }, "Listening.MinRms"];
        yield return [new VoiceAssistantOptions { BargeIn = null! }, "BargeIn"];
        yield return [new VoiceAssistantOptions { BargeIn = { MinDurationMs = -1 } }, "BargeIn.MinDurationMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { MinRms = -1 } }, "BargeIn.MinRms"];
        yield return [new VoiceAssistantOptions { BargeIn = { MinRms = float.PositiveInfinity } }, "BargeIn.MinRms"];
        yield return [new VoiceAssistantOptions { BargeIn = { PlaybackCandidateSuppressionMs = -1 } }, "BargeIn.PlaybackCandidateSuppressionMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { PlaybackMaxCandidateMs = -1 } }, "BargeIn.PlaybackMaxCandidateMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { PlaybackMaxCandidateMs = 60001 } }, "BargeIn.PlaybackMaxCandidateMs"];
        yield return [new VoiceAssistantOptions
        {
            Speaker = { Confirmation = { Enable = true } },
            BargeIn = { Enable = true, PlaybackMaxCandidateMs = 0 },
        }, "BargeIn.PlaybackMaxCandidateMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { PlaybackRecognitionAudioLimitMs = -1 } }, "BargeIn.PlaybackRecognitionAudioLimitMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { PlaybackStartupGuardMs = -1 } }, "BargeIn.PlaybackStartupGuardMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { ActiveNearEndWindowMs = 0 } }, "BargeIn.ActiveNearEndWindowMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { ActiveNearEndPreRollMs = -1 } }, "BargeIn.ActiveNearEndPreRollMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { ActiveNearEndEndSilenceMs = 0 } }, "BargeIn.ActiveNearEndEndSilenceMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { ActiveNearEndAudioDrainTimeoutMs = 0 } }, "BargeIn.ActiveNearEndAudioDrainTimeoutMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { ActiveNearEndCooldownMs = -1 } }, "BargeIn.ActiveNearEndCooldownMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { Dtd = null! } }, "BargeIn.Dtd"];
        yield return [new VoiceAssistantOptions { BargeIn = { Dtd = { Model = null! } } }, "BargeIn.Dtd.Model"];
        yield return [new VoiceAssistantOptions { BargeIn = { Dtd = { DecisionHistoryMs = 99 } } }, "BargeIn.Dtd.DecisionHistoryMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { Dtd = { AcousticPathChangeGuardMs = -1 } } }, "BargeIn.Dtd.AcousticPathChangeGuardMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = null! } }, "BargeIn.RenderAlignment"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { DelayHintMaxAgeMs = 0 } } }, "BargeIn.RenderAlignment.DelayHintMaxAgeMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { EstimateHoldoverMs = 0 } } }, "BargeIn.RenderAlignment.EstimateHoldoverMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { MaxTrackedDelayMs = -1 } } }, "BargeIn.RenderAlignment.MaxTrackedDelayMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { PathChangeThresholdMs = 301 } } }, "BargeIn.RenderAlignment.PathChangeThresholdMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { DelayUpdateIntervalMs = 0 } } }, "BargeIn.RenderAlignment.DelayUpdateIntervalMs"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { MaximumDriftPpm = double.NaN } } }, "BargeIn.RenderAlignment.MaximumDriftPpm"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { MaximumDelaySlewPerSecond = 0 } } }, "BargeIn.RenderAlignment.MaximumDelaySlewPerSecond"];
        yield return [new VoiceAssistantOptions { BargeIn = { RenderAlignment = { MinimumAlignmentConfidence = double.NaN } } }, "BargeIn.RenderAlignment.MinimumAlignmentConfidence"];
    }

    private static readonly SpeakerPurposeSet WakeupAndBargeIn = new(true, true, false);

    [Fact]
    public void ValidateSpeakerGeometry_RejectsShortPreRoll()
    {
        var options = new VoiceAssistantOptions
        {
            Wakeup = { Enable = true, PreRollMs = 499 },
            Speaker = { Confirmation = { MinimumSpeechMs = 500 } },
        };
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
                options,
                new GeometrySpeakerService(window: 8000, hop: 1600),
                new SpeakerPurposeSet(true, false, false),
                "options"));
        Assert.Contains("Wakeup.PreRollMs", ex.Message);
    }

    [Fact]
    public void ValidateSpeakerGeometry_RejectsShortBargeInCandidate()
    {
        var options = new VoiceAssistantOptions
        {
            Wakeup = { Enable = true },
            Speaker = { Confirmation = { MinimumSpeechMs = 600, WindowCount = 3 } },
            BargeIn = { Enable = true, PlaybackMaxCandidateMs = 700 },
        };
        // 600ms window + 2 hops of 100ms needs 800ms of candidate; 700ms is too short.
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
                options,
                new GeometrySpeakerService(window: 9600, hop: 1600),
                WakeupAndBargeIn,
                "options"));
        Assert.Contains("BargeIn.PlaybackMaxCandidateMs", ex.Message);
    }

    [Fact]
    public void ValidateSpeakerGeometry_RejectsTooManyBargeInWindows()
    {
        var options = new VoiceAssistantOptions
        {
            Wakeup = { Enable = true },
            Speaker =
            {
                Confirmation = { MinimumSpeechMs = 500, WindowCount = 3, RequiredMatches = 2 },
            },
            BargeIn = { Enable = true, PlaybackMaxCandidateMs = 1200 },
        };
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
                options,
                new GeometrySpeakerService(window: 8000, hop: 16),
                WakeupAndBargeIn,
                "options"));
        Assert.Contains("no more than 64 windows", ex.Message);
    }

    [Fact]
    public void ValidateSpeakerGeometry_RejectsRealtimePurposeOnLongProviderWindow()
    {
        var options = new VoiceAssistantOptions { Wakeup = { Enable = true } };
        ArgumentException ex = Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
                options,
                new GeometrySpeakerService(window: 32000, hop: 1600),
                new SpeakerPurposeSet(true, false, false),
                "options"));
        Assert.Contains("MaxRealtimeWindowMs", ex.Message);
    }

    [Fact]
    public void ValidateSpeakerGeometry_IgnoresDisabledPurposesAndRetentionCoversPreRoll()
    {
        var options = new VoiceAssistantOptions
        {
            Wakeup = { Enable = true, PreRollMs = 2500 },
            Speaker = { AudioRetentionMs = 3000 },
        };
        // 700ms coverage + 2500ms pre-roll exceeds 3000ms retention.
        Assert.Throws<ArgumentException>(() =>
            VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
                options,
                new GeometrySpeakerService(window: 8000, hop: 1600),
                new SpeakerPurposeSet(true, false, false),
                "options"));
        options.Speaker.AudioRetentionMs = 6000;
        VoiceAssistantOptionsValidator.ValidateSpeakerGeometry(
            options,
            new GeometrySpeakerService(window: 8000, hop: 1600),
            new SpeakerPurposeSet(true, false, false),
            "options");
    }

    private sealed class GeometrySpeakerService(int window, int hop) : ISpeakerService
    {
        public int EmbeddingSize => 4;
        public int SampleRate => 16000;
        public float Threshold => 0.5f;
        public int StreamingWindowSamples { get; } = window;
        public int StreamingHopSamples { get; } = hop;
        public IReadOnlyCollection<string> EnrolledSpeakers => [];
        public event EventHandler<SpeakerEventArgs>? OnResult;

        public float[] ExtractEmbedding(ReadOnlySpan<float> samples) => new float[EmbeddingSize];
        public float CompareEmbeddings(ReadOnlySpan<float> embedding1, ReadOnlySpan<float> embedding2) => 0;
        public float Verify(ReadOnlySpan<float> samples1, ReadOnlySpan<float> samples2) => 0;
        public void Enroll(string speakerName, ReadOnlySpan<float> samples) { }
        public bool Unenroll(string speakerName) => false;
        public void FeedAudio(ReadOnlySpan<float> samples) { }
        public void Reset() { }
        public void Dispose() { }
    }
}
