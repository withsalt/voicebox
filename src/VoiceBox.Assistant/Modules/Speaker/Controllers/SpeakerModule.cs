using Microsoft.Extensions.Logging;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Composition.Plans;
using VoiceBox.Assistant.Modules.AudioFrontend.Models;
using VoiceBox.Assistant.Modules.Speaker.Contracts;
using VoiceBox.Assistant.Modules.Speaker.Diagnostics;
using VoiceBox.Assistant.Modules.Speaker.Models;
using VoiceBox.Assistant.Modules.Speaker.Policies;
using VoiceBox.Assistant.Modules.Speaker.Profiles;
using VoiceBox.Assistant.Modules.Speaker.State;
using VoiceBox.Core.Services.SpeakerService;

namespace VoiceBox.Assistant.Modules.Speaker.Controllers;

/// <summary>
/// Online owner-identity module. It consumes the identity tap only, owns the
/// <see cref="SpeechGatedSpeakerVerifier"/> (ring, prefetch, scheduler, cache) and the VAD-gate
/// candidate session, and exposes nothing but typed evidence results to the rest of the runtime.
/// </summary>
internal sealed class SpeakerModule : ISpeakerModule
{
    private readonly SpeechGatedSpeakerVerifier _verifier;
    private readonly SpeakerVerificationState _verificationState;
    private readonly SpeakerCaptureDescriptor _descriptor;
    private int _disposed;

    public SpeakerModule(
        ISpeakerService speaker,
        VoiceAssistantSpeakerOptions options,
        SpeakerPurposeSet effectivePurposes,
        SpeakerCaptureDescriptor descriptor,
        int audioSampleRate,
        int audioChannels,
        ILogger logger)
    {
        ArgumentNullException.ThrowIfNull(speaker);
        ArgumentNullException.ThrowIfNull(options);
        _descriptor = descriptor ?? throw new ArgumentNullException(nameof(descriptor));
        ArgumentNullException.ThrowIfNull(logger);
        if (!effectivePurposes.Any)
        {
            throw new InvalidOperationException(
                "SpeakerModule must not be created without an effective online purpose; Confirmation.Enable-off compositions skip it.");
        }

        SpeakerVerificationPolicy policy = SpeakerVerificationPolicy.Create(
            speaker,
            options,
            audioSampleRate,
            audioChannels,
            effectivePurposes);
        if (policy.SampleRate != descriptor.SampleRate || descriptor.Channels != 1)
        {
            throw new InvalidOperationException(
                $"Identity tap format {descriptor.SampleRate}Hz/{descriptor.Channels}ch does not match the speaker service " +
                $"{policy.SampleRate}Hz/1ch.");
        }

        string profilePath = options.ResolveProfilePath();
        OwnerSpeakerProfile profile = SpeakerProfileStore.Load(speaker, profilePath);
        OwnerSpeakerProfileVariant variant = SpeakerProfileStore.RequireVariant(
            profile,
            descriptor,
            policy.RequiredBanks,
            profilePath);

        _verifier = new SpeechGatedSpeakerVerifier(speaker, policy, variant, logger);
        _verificationState = new SpeakerVerificationState(_verifier);
    }

    public AssistantTimelineRequirement TimelineRequirement => AssistantTimelineRequirement.SpeakerCaptureTimeline;

    public SpeakerCaptureDescriptor Descriptor => _descriptor;

    public SpeakerVerificationPolicy Policy => _verifier.Policy;

    internal SpeakerVerificationDiagnostics Diagnostics => _verifier.Diagnostics;

    public void SetPlaybackActive(bool active) => _verifier.SetPlaybackActive(active);

    /// <summary>Exposed for diagnostics and tests; never for the control plane.</summary>
    internal SpeechGatedSpeakerVerifier Verifier => _verifier;

    public void Start()
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref _disposed) != 0, this);
        _verifier.Start();
    }

    public Task StopAsync(CancellationToken cancellationToken = default)
        => _verifier.StopAsync(cancellationToken);

    public void OnSpeakerCapture(in SpeakerCaptureFrame frame) => _verifier.OnSpeakerCapture(in frame);

    public SpeakerCandidateId NewCandidateId() => _verifier.NewCandidateId();

    public SpeakerVerificationResult Evaluate(in SpeakerEvidenceQuery query) => _verifier.Evaluate(in query);

    public Task<SpeakerVerificationResult> WaitForDecisionAsync(
        SpeakerEvidenceQuery query,
        TimeSpan timeout,
        CancellationToken cancellationToken = default)
        => _verifier.WaitForDecisionAsync(query, timeout, cancellationToken);

    public void Cancel(SpeakerCandidateId candidateId) => _verifier.Cancel(candidateId);

    public TimeSpan GetInferenceTimeout(SpeakerVerificationPurpose purpose) => _verifier.GetInferenceTimeout(purpose);

    bool ISpeakerVerificationStatePort.IsCandidateActive => _verificationState.IsCandidateActive;

    long ISpeakerVerificationStatePort.CurrentVersion => _verificationState.CurrentVersion;

    void ISpeakerVerificationStatePort.BeginCandidate(SpeakerVerificationPurpose purpose)
        => _verificationState.BeginCandidate(purpose);

    void ISpeakerVerificationStatePort.Reset() => _verificationState.Reset();

    bool ISpeakerVerificationStatePort.TryBuildQuery(
        long version,
        long streamEpoch,
        long startSample,
        long endSample,
        long queryTimestampTicks,
        out SpeakerEvidenceQuery query)
        => _verificationState.TryBuildQuery(version, streamEpoch, startSample, endSample, queryTimestampTicks, out query);

    bool ISpeakerVerificationStatePort.TryCompleteQuery(long version, long queryTimestampTicks, out SpeakerEvidenceQuery query)
        => _verificationState.TryCompleteQuery(version, queryTimestampTicks, out query);

    bool ISpeakerVerificationStatePort.IsCurrent(long version) => _verificationState.IsCurrent(version);

    bool ISpeakerVerificationStatePort.TryResolve(long version) => _verificationState.TryResolve(version);

    public void Dispose()
    {
        if (Interlocked.Exchange(ref _disposed, 1) != 0)
            return;
        _verifier.Dispose();
    }
}
