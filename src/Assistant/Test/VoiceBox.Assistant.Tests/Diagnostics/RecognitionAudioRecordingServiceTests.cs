using System.Text.Json;
using Microsoft.Extensions.Logging.Abstractions;
using VoiceBox.Assistant.Diagnostics.Recording;
using VoiceBox.Core.Services.ASRService;
using VoiceBox.Core.Services.ASRService.Models;
using Xunit;

namespace VoiceBox.Assistant.Tests;

public sealed class RecognitionAudioRecordingServiceTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "voicebox-asr-" + Guid.NewGuid().ToString("N"));

    [Fact]
    public async Task RecordsExactProviderInputAcrossBootstrapAndReusedCaptureBuffers()
    {
        var provider = new Provider();
        using var recorder = Create(provider);
        string session = (await recorder.StartSessionAsync()).SessionId;
        float[] bootstrap = [0.000001f, -0.25f, 1.2f];
        recorder.FeedAudio(session, bootstrap);
        Array.Fill(bootstrap, 9f);
        float[] tail = [-0.125f, 0.5f];
        recorder.FeedAudio(session, tail);
        Array.Clear(tail);

        AsrResult? result = await recorder.FinishSessionAsync(session);

        Assert.Equal("换一个", result?.Text);
        float[] expected = [0.000001f, -0.25f, 1.2f, -0.125f, 0.5f];
        Assert.Equal(expected, provider.Audio[session]);
        Assert.Equal(expected, ReadWave(Assert.Single(Directory.GetFiles(_directory, "*.wav"))));
        using JsonDocument metadata = ReadMetadata();
        Assert.Equal(session, metadata.RootElement.GetProperty("sessionId").GetString());
        Assert.Equal("换一个", metadata.RootElement.GetProperty("text").GetString());
        Assert.Equal(5, metadata.RootElement.GetProperty("ForwardedSamples").GetInt32());
        Assert.Equal(1, metadata.RootElement.GetProperty("clippedSamples").GetInt32());
        Assert.False(metadata.RootElement.GetProperty("truncated").GetBoolean());
        Assert.Equal([3, 2], metadata.RootElement.GetProperty("savedChunkSamples").EnumerateArray().Select(v => v.GetInt32()));
    }

    [Fact]
    public async Task DiagnosticLimitDoesNotTruncateProviderAudio()
    {
        var provider = new Provider();
        using var recorder = Create(provider, sampleRate: 10, maxSeconds: 1);
        string session = (await recorder.StartSessionAsync()).SessionId;
        recorder.FeedAudio(session, Enumerable.Range(0, 8).Select(i => (float)i).ToArray());
        recorder.FeedAudio(session, Enumerable.Range(8, 8).Select(i => (float)i).ToArray());
        await recorder.FinishSessionAsync(session);

        Assert.Equal(16, provider.Audio[session].Count);
        Assert.Equal(Enumerable.Range(0, 10).Select(i => (float)i), ReadWave(Assert.Single(Directory.GetFiles(_directory, "*.wav")), 10));
        using JsonDocument metadata = ReadMetadata();
        Assert.True(metadata.RootElement.GetProperty("truncated").GetBoolean());
        Assert.Equal(16, metadata.RootElement.GetProperty("ForwardedSamples").GetInt32());
        Assert.Equal(10, metadata.RootElement.GetProperty("SavedSamples").GetInt32());
    }

    [Theory]
    [InlineData(false, "failed")]
    [InlineData(true, "cancelled")]
    public async Task ProviderFailureStillSavesInputAndPreservesException(bool cancelled, string status)
    {
        Exception failure = cancelled ? new OperationCanceledException() : new InvalidOperationException("provider failure");
        var provider = new Provider { Failure = failure };
        using var recorder = Create(provider);
        string session = (await recorder.StartSessionAsync()).SessionId;
        recorder.FeedAudio(session, [0.1f]);

        Exception? actual = await Record.ExceptionAsync(() => recorder.FinishSessionAsync(session));

        Assert.Same(failure, actual);
        Assert.Equal([0.1f], ReadWave(Assert.Single(Directory.GetFiles(_directory, "*.wav"))));
        using JsonDocument metadata = ReadMetadata();
        Assert.Equal(status, metadata.RootElement.GetProperty("status").GetString());
    }

    [Fact]
    public async Task RecordingFailureDoesNotReplaceRecognitionResultOrDisposeProvider()
    {
        // A regular file cannot be used as the output directory.
        await File.WriteAllTextAsync(_directory, "occupied");
        var provider = new Provider();
        using (var recorder = Create(provider))
        {
            string session = (await recorder.StartSessionAsync()).SessionId;
            recorder.FeedAudio(session, [0.1f]);
            Assert.Equal("换一个", (await recorder.FinishSessionAsync(session))?.Text);
        }
        Assert.False(provider.Disposed);
    }

    [Fact]
    public async Task OverlappingSessionsRemainSeparate()
    {
        var provider = new Provider();
        using var recorder = Create(provider);
        string first = (await recorder.StartSessionAsync()).SessionId;
        string second = (await recorder.StartSessionAsync()).SessionId;
        recorder.FeedAudio(first, [0.1f]);
        recorder.FeedAudio(second, [0.2f]);
        recorder.FeedAudio(first, [0.3f]);
        await Task.WhenAll(recorder.FinishSessionAsync(first), recorder.FinishSessionAsync(second));

        Assert.Equal(2, Directory.GetFiles(_directory, "*.json").Length);
        foreach (string path in Directory.GetFiles(_directory, "*.json"))
        {
            using JsonDocument metadata = JsonDocument.Parse(await File.ReadAllTextAsync(path));
            string id = metadata.RootElement.GetProperty("sessionId").GetString()!;
            Assert.Equal(provider.Audio[id], ReadWave(Path.ChangeExtension(path, ".wav")));
        }
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task ExplicitCancellation_ForwardsAndPreservesDiagnosticAudio(bool failCancellation)
    {
        var failure = new InvalidOperationException("cancel failed");
        var provider = new CancellableProvider { CancellationFailure = failCancellation ? failure : null };
        using var recorder = Create(provider);
        IASRService port = recorder;
        string session = (await recorder.StartSessionAsync()).SessionId;
        recorder.FeedAudio(session, [0.1f, 0.2f]);

        Exception? actual = await Record.ExceptionAsync(() => port.CancelSessionAsync(session));

        Assert.Same(failCancellation ? failure : null, actual);
        Assert.Equal(1, provider.CancelCount);
        Assert.False(provider.Disposed);
        Assert.Equal([0.1f, 0.2f], ReadWave(Assert.Single(Directory.GetFiles(_directory, "*.wav"))));
        using JsonDocument metadata = ReadMetadata();
        Assert.Equal("cancelled", metadata.RootElement.GetProperty("status").GetString());
    }

    private sealed class CancellableProvider : Provider
    {
        public Exception? CancellationFailure { get; init; }
        public int CancelCount { get; private set; }
        public override Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
        {
            CancelCount++;
            return CancellationFailure is { } failure ? Task.FromException(failure) : Task.CompletedTask;
        }
    }

    private RecognitionAudioRecordingService Create(Provider provider, int sampleRate = 16000, int maxSeconds = 60)
        => new(provider, _directory, sampleRate, 1, NullLogger.Instance, maxSeconds);

    private JsonDocument ReadMetadata()
        => JsonDocument.Parse(File.ReadAllText(Assert.Single(Directory.GetFiles(_directory, "*.json"))));

    private static float[] ReadWave(string path, int sampleRate = 16000)
    {
        using var reader = new BinaryReader(File.OpenRead(path));
        Assert.Equal("RIFF", new string(reader.ReadChars(4)));
        Assert.Equal(reader.BaseStream.Length - 8, reader.ReadInt32());
        Assert.Equal("WAVE", new string(reader.ReadChars(4)));
        while (reader.BaseStream.Position < reader.BaseStream.Length)
        {
            string id = new(reader.ReadChars(4));
            int length = reader.ReadInt32();
            long next = reader.BaseStream.Position + length;
            if (id == "fmt ")
            {
                Assert.Equal(3, reader.ReadUInt16());
                Assert.Equal(1, reader.ReadUInt16());
                Assert.Equal(sampleRate, reader.ReadInt32());
                Assert.Equal(sampleRate * 4, reader.ReadInt32());
                Assert.Equal(4, reader.ReadUInt16());
                Assert.Equal(32, reader.ReadUInt16());
            }
            else if (id == "data")
            {
                float[] samples = new float[length / 4];
                for (int i = 0; i < samples.Length; i++) samples[i] = reader.ReadSingle();
                Assert.Equal(reader.BaseStream.Length, reader.BaseStream.Position);
                return samples;
            }
            reader.BaseStream.Position = next + (length % 2);
        }
        throw new InvalidDataException("Missing WAVE data chunk.");
    }

    public void Dispose()
    {
        if (Directory.Exists(_directory)) Directory.Delete(_directory, recursive: true);
        if (File.Exists(_directory)) File.Delete(_directory);
    }

    private class Provider : IASRService
    {
        private int _nextId;
        public Dictionary<string, List<float>> Audio { get; } = [];
        public Exception? Failure { get; init; }
        public bool Disposed { get; private set; }
        public Task<SessionInfo> StartSessionAsync(CancellationToken ct = default)
        {
            // Include a separator to prove opaque provider IDs never become file names.
            string id = $"session/{++_nextId}";
            Audio[id] = [];
            return Task.FromResult(new SessionInfo(id));
        }
        public void FeedAudio(string sessionId, ReadOnlySpan<float> audioData)
            => Audio[sessionId].AddRange(audioData.ToArray());
        public Task<AsrResult?> FinishSessionAsync(string sessionId, CancellationToken ct = default)
            => Failure is { } failure
                ? Task.FromException<AsrResult?>(failure)
                : Task.FromResult<AsrResult?>(new AsrResult { Text = "换一个" });
        public virtual Task CancelSessionAsync(string sessionId, CancellationToken ct = default)
        {
            ct.ThrowIfCancellationRequested();
            return Task.CompletedTask;
        }
        public void Dispose() => Disposed = true;
    }
}
