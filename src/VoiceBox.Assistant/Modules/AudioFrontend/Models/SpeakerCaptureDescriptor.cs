using System.Text;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Assistant.Modules.AudioFrontend.Models;

/// <summary>
/// Immutable, Assistant-private description of the identity audio tap. Enrollment and online
/// verification must share the same value so a profile variant is only ever compared against audio
/// that went through the same acoustic front end. The independent DNR that follows the tap is
/// deliberately not part of the fingerprint.
/// </summary>
internal sealed record SpeakerCaptureDescriptor(
    string TapId,
    int SampleRate,
    int Channels,
    string CaptureProcessorId,
    string CaptureProcessorDigest,
    string ModelDigest,
    string FrontendFingerprint)
{
    public const string PostCaptureProcessorTapId = "post-capture-processor-pre-independent-dnr-v1";

    public const string Unknown = "unknown";

    public const string NoCaptureProcessor = "none";

    /// <summary>
    /// Builds the descriptor from facts the duplex provider exposes today. Fields that cannot be
    /// obtained through the public provider surface are marked <see cref="Unknown"/>; they only
    /// ever match a variant that carries the same marker.
    /// </summary>
    public static SpeakerCaptureDescriptor Create(
        int sampleRate,
        int channels,
        SampleFormat sampleFormat,
        bool highPassFilterEnabled,
        bool captureProcessorEnabled,
        string? captureProcessorId,
        string? captureProcessorDigest = null,
        string? modelDigest = null)
    {
        if (sampleRate <= 0)
            throw new ArgumentOutOfRangeException(nameof(sampleRate));
        if (channels <= 0)
            throw new ArgumentOutOfRangeException(nameof(channels));

        string processorId = captureProcessorEnabled
            ? Normalize(captureProcessorId)
            : NoCaptureProcessor;
        string processorDigest = captureProcessorEnabled
            ? Normalize(captureProcessorDigest)
            : NoCaptureProcessor;
        string model = captureProcessorEnabled ? Normalize(modelDigest) : NoCaptureProcessor;

        var fingerprint = new StringBuilder(128)
            .Append(PostCaptureProcessorTapId)
            .Append('|').Append(sampleRate)
            .Append('|').Append(channels == 1 ? "mono" : channels + "ch")
            .Append('|').Append(sampleFormat.ToString().ToLowerInvariant())
            .Append("|hpf=").Append(highPassFilterEnabled ? "on" : "off")
            .Append("|aec=").Append(processorId)
            .Append("|aecDigest=").Append(processorDigest)
            .Append("|model=").Append(model)
            .ToString();

        return new SpeakerCaptureDescriptor(
            PostCaptureProcessorTapId,
            sampleRate,
            channels,
            processorId,
            processorDigest,
            model,
            fingerprint);
    }

    /// <summary>
    /// The identity tap of a duplex provider is its after-capture-processor signal. The fingerprint
    /// is frozen from the provider facts reachable through the public duplex surface; the acoustic
    /// diagnostics registration, when present, is the only stable processor identity available to
    /// the Assistant. Enrollment and the runtime frontend must both go through this method.
    /// </summary>
    public static SpeakerCaptureDescriptor FromDuplexService(
        Core.Services.AudioService.IAudioDuplexService audio,
        Core.Services.AcousticProcessing.IAcousticProcessingMetricsSource? acousticDiagnostics)
    {
        ArgumentNullException.ThrowIfNull(audio);
        int sampleRate = audio.Options.SampleRate is > 0 and <= int.MaxValue
            ? (int)audio.Options.SampleRate
            : 1;
        int channels = audio.Options.ChannelCount is > 0 and <= int.MaxValue
            ? (int)audio.Options.ChannelCount
            : 1;
        return Create(
            sampleRate,
            channels,
            audio.Options.SampleFormat,
            audio.EnableHighPassFilter,
            audio.EnableAec,
            acousticDiagnostics?.GetType().Name,
            captureProcessorDigest: null,
            modelDigest: null);
    }

    private static string Normalize(string? value)
        => string.IsNullOrWhiteSpace(value) ? Unknown : value.Trim();
}
