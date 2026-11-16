using System.Collections.Concurrent;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;

namespace VoiceBox.Assistant.Diagnostics.Recording;

/// <summary>
/// Records the PCM passed to an ASR provider, including bootstrap and directly fed audio, so
/// recognition errors can be reproduced without a microphone. This decorator does not own the
/// provider. File I/O and signal statistics run at the endpoint, never on the capture callback.
/// </summary>
public sealed class RecognitionAudioRecordingService : IASRService
{
    private readonly IASRService _provider;
    private readonly string _directory;
    private readonly int _sampleRate;
    private readonly int _channels;
    private readonly int _maxSamples;
    private readonly ILogger _logger;
    private readonly ConcurrentDictionary<string, Recording> _recordings = new(StringComparer.Ordinal);

    public RecognitionAudioRecordingService(
        IASRService provider,
        string directory,
        int sampleRate,
        int channels,
        ILogger logger,
        int maxDurationSeconds = 60)
    {
        ArgumentNullException.ThrowIfNull(provider);
        ArgumentException.ThrowIfNullOrWhiteSpace(directory);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(sampleRate);
        if (channels is < 1 or > 32)
            throw new ArgumentOutOfRangeException(nameof(channels));
        ArgumentNullException.ThrowIfNull(logger);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxDurationSeconds);
        _provider = provider;
        _directory = Path.GetFullPath(directory);
        _sampleRate = sampleRate;
        _channels = channels;
        _maxSamples = checked(sampleRate * channels * maxDurationSeconds);
        _logger = logger;
    }

    public async Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
    {
        SessionInfo session = await _provider.StartSessionAsync(ct).ConfigureAwait(false);
        _recordings[session.SessionId] = new Recording();
        return session;
    }

    public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData)
    {
        if (!_recordings.TryGetValue(sessionId, out Recording? recording))
        {
            _provider.FeedAudio(sessionId, audioData);
            return;
        }

        lock (recording)
        {
            _provider.FeedAudio(sessionId, audioData);
            if (recording.Closed || audioData.IsEmpty)
                return;

            recording.ForwardedSamples += audioData.Length;
            recording.FeedCalls++;
            int take = Math.Min(audioData.Length, _maxSamples - recording.SavedSamples);
            if (take > 0)
            {
                recording.Chunks.Add(audioData[..take].ToArray());
                recording.SavedSamples += take;
            }
        }
    }

    public async Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default)
    {
        _recordings.TryRemove(sessionId, out Recording? recording);
        if (recording is not null)
        {
            lock (recording)
                recording.Closed = true;
        }

        AsrResult? result = null;
        string status = "failed";
        try
        {
            result = await _provider.FinishSessionAsync(sessionId, ct).ConfigureAwait(false);
            status = "completed";
            return result;
        }
        catch (OperationCanceledException)
        {
            status = "cancelled";
            throw;
        }
        finally
        {
            if (recording is { SavedSamples: > 0 })
            {
                // Cancellation/failure is useful evidence too. Diagnostic I/O must neither
                // replace the provider's exception nor turn a good transcription into a fault.
                await Task.Run(() => Save(sessionId, recording, result, status)).ConfigureAwait(false);
            }
        }
    }

    public async Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
    {
        _recordings.TryRemove(sessionId, out Recording? recording);
        if (recording is not null)
        {
            lock (recording)
                recording.Closed = true;
        }
        try
        {
            await _provider.CancelSessionAsync(sessionId, ct).ConfigureAwait(false);
        }
        finally
        {
            if (recording is { SavedSamples: > 0 })
                await Task.Run(() => Save(sessionId, recording, null, "cancelled")).ConfigureAwait(false);
        }
    }

    /// <summary>Releases unfinished diagnostic buffers; the caller retains provider ownership.</summary>
    public void Dispose() => _recordings.Clear();

    private void Save(string sessionId, Recording recording, AsrResult? result, string status)
    {
        try
        {
            Directory.CreateDirectory(_directory);
            // Provider session IDs are opaque and may contain path separators.
            string stem = $"{recording.StartedAt:yyyyMMdd-HHmmss-fff}-{Guid.NewGuid():N}";
            string path = Path.Combine(_directory, stem + ".wav");
            double energy = 0;
            double peak = 0;
            int nonFiniteSamples = 0;
            int clippedSamples = 0;
            using (var writer = new BinaryWriter(File.Create(path), Encoding.UTF8))
            {
                // WAVE_FORMAT_IEEE_FLOAT preserves the exact samples supplied to FeedAudio,
                // including low-level speech and over-range samples; no gain or PCM16 clipping.
                int bytes = checked(recording.SavedSamples * sizeof(float));
                writer.Write("RIFF"u8);
                writer.Write(checked(50 + bytes));
                writer.Write("WAVEfmt "u8);
                writer.Write(18);
                writer.Write((ushort)3);
                writer.Write((ushort)_channels);
                writer.Write(_sampleRate);
                writer.Write(checked(_sampleRate * _channels * sizeof(float)));
                writer.Write((ushort)(_channels * sizeof(float)));
                writer.Write((ushort)32);
                writer.Write((ushort)0);
                writer.Write("fact"u8);
                writer.Write(4);
                writer.Write(recording.SavedSamples / _channels);
                writer.Write("data"u8);
                writer.Write(bytes);
                foreach (float[] chunk in recording.Chunks)
                {
                    foreach (float sample in chunk)
                    {
                        writer.Write(sample);
                        if (!float.IsFinite(sample))
                        {
                            nonFiniteSamples++;
                            continue;
                        }
                        double magnitude = Math.Abs((double)sample);
                        peak = Math.Max(peak, magnitude);
                        energy += (double)sample * sample;
                        if (magnitude >= 1)
                            clippedSamples++;
                    }
                }
            }

            double durationMs = recording.SavedSamples * 1000d / _sampleRate / _channels;
            double rms = Math.Sqrt(energy / Math.Max(1, recording.SavedSamples - nonFiniteSamples));
            bool truncated = recording.ForwardedSamples != recording.SavedSamples;
            var metadata = new
            {
                sessionId,
                provider = _provider.GetType().FullName,
                recording.StartedAt,
                status,
                text = result?.Text,
                sampleRate = _sampleRate,
                channels = _channels,
                recording.ForwardedSamples,
                recording.SavedSamples,
                recording.FeedCalls,
                savedChunkSamples = recording.Chunks.Select(static chunk => chunk.Length).ToArray(),
                durationMs,
                rms,
                peak,
                clippedSamples,
                nonFiniteSamples,
                truncated,
            };
            File.WriteAllText(Path.ChangeExtension(path, ".json"),
                JsonSerializer.Serialize(metadata, new JsonSerializerOptions { WriteIndented = true }));
            _logger.LogInformation(
                "ASR 输入音频已保存: sessionId={SessionId}, path={Path}, samples={Samples}, " +
                "durationMs={DurationMs:F0}, rms={Rms:F6}, peak={Peak:F4}, truncated={Truncated}, status={Status}.",
                sessionId, path, recording.SavedSamples, durationMs, rms, peak, truncated, status);
        }
        catch (Exception ex)
        {
            _logger.LogWarning(ex, "保存 ASR 输入音频失败: sessionId={SessionId}.", sessionId);
        }
    }

    private sealed class Recording
    {
        public DateTimeOffset StartedAt { get; } = DateTimeOffset.Now;
        public List<float[]> Chunks { get; } = [];
        public int SavedSamples { get; set; }
        public long ForwardedSamples { get; set; }
        public long FeedCalls { get; set; }
        public bool Closed { get; set; }
    }
}
