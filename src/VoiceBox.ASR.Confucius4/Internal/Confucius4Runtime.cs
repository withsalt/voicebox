using System.Text.Json;
using System.Security.Cryptography;
using Microsoft.Extensions.Logging;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.ASR.Confucius4.Internal;
using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.Confucius4.Internal;

/// <summary>Confucius4 tensor runtime with package validation and exact token rollback.</summary>
internal sealed class Confucius4Runtime : IConfucius4Runtime
{
    private readonly Confucius4AsrOptions _options;
    private readonly Confucius4AsrMetadata _metadata;
    private readonly Confucius4AsrTokenizer _tokenizer;
    private readonly Confucius4AsrSessions _sessions;
    private readonly Confucius4AsrAudioEncoder _encoder;
    private readonly Confucius4AsrDecoder _decoder;

    public Confucius4Runtime(Confucius4AsrOptions options, ILogger logger)
    {
        _options = options;
        string json = File.ReadAllText(Path.Combine(options.ModelPath, "metadata.json"));
        using var doc = JsonDocument.Parse(json);
        var root = doc.RootElement;
        if (root.GetProperty("schema_version").GetInt32() != 1 ||
            root.GetProperty("model_family").GetString() != "confucius4-r2t2")
            throw new InvalidDataException("Unsupported Confucius4 package schema.");
        if (options.Language is not null && !root.GetProperty("languages").EnumerateArray()
            .Any(x => string.Equals(x.GetString(), options.Language, StringComparison.Ordinal)))
            throw new ArgumentException($"Unsupported language: {options.Language}. Use a name from metadata.json.");
        _metadata = JsonSerializer.Deserialize<Confucius4AsrMetadata>(json)!;
        if (!_metadata.DynamicChunkFrames || _metadata.NWindow != 50 || _metadata.AudioOutputDim != 2048 ||
            root.GetProperty("sample_rate").GetInt32() != 16000 || _metadata.DecodeCacheMode != "full" ||
            _metadata.StaticCacheLen < 128 || _metadata.EosTokenIds.Count == 0)
            throw new InvalidDataException("Invalid or incompatible Confucius4 tensor contract.");
        // ORT's cuDNN frontend disables all tensor-core plans when use_tf32=0,
        // including FP16 plans. Preserve FP32 accuracy without penalizing FP16.
        options.CudaTuning ??= new CudaExecutionTuning();
        options.CudaTuning.AdditionalOptions ??= new();
        options.CudaTuning.AdditionalOptions.TryAdd("use_tf32", _metadata.Dtype == "float16" ? "1" : "0");
        foreach (string graph in new[]
        {
            _metadata.EncoderModel,
            _metadata.ConvEmbedModel,
            _metadata.AudioStackModel,
            _metadata.ResolvePrefillModel(),
            _metadata.ResolveDecodeModel(),
        }.Where(graph => !string.IsNullOrWhiteSpace(graph)).Select(graph => graph!))
            if (Path.GetFileName(graph) != graph)
                throw new InvalidDataException("Model filenames must be relative basenames.");
        foreach (var file in root.GetProperty("files").EnumerateObject())
        {
            if (Path.GetFileName(file.Name) != file.Name) throw new InvalidDataException("Invalid package filename.");
            var info = new FileInfo(Path.Combine(options.ModelPath, file.Name));
            if (!info.Exists || info.Length != file.Value.GetProperty("bytes").GetInt64())
                throw new InvalidDataException($"Missing or truncated package asset: {file.Name}");
        }
        if (_metadata.Dtype == "int8" && options.ExecutionProvider != OnnxExecutionProvider.Cpu &&
            (!root.TryGetProperty("quantization", out var quantization) ||
             quantization.GetProperty("method").GetString() != "weight_only_blockwise"))
            throw new NotSupportedException("Dynamic INT8 packages require CPU. Use FP16 for CUDA.");
        _tokenizer = new Confucius4AsrTokenizer(options.ModelPath);
        var preprocessor = JsonSerializer.Deserialize<PreprocessorConfig>(
            File.ReadAllText(Path.Combine(options.ModelPath, "preprocessor_config.json")))!;
        _sessions = new Confucius4AsrSessions(options.ModelPath, _metadata, purpose =>
        {
            SessionOptions sessionOptions = OnnxSessionFactory.Create(options, purpose);
            if (options.SuppressOrtWarnings && Environment.GetEnvironmentVariable("ORT_LOG_VERBOSE") != "1")
                sessionOptions.LogSeverityLevel = Microsoft.ML.OnnxRuntime.OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR;
            return sessionOptions;
        });
        try
        {
            _encoder = new Confucius4AsrAudioEncoder(_metadata, _sessions, new WhisperFeatureExtractor(preprocessor));
            _decoder = new Confucius4AsrDecoder(_metadata, _sessions,
                options.EnableIoBinding && options.ExecutionProvider == OnnxExecutionProvider.Cuda,
                options.GpuDeviceId, logger);
        }
        catch
        {
            _encoder?.Dispose();
            _sessions.Dispose();
            throw;
        }
    }

    public string Rollback(string text, int count)
    {
        if (count == 0) return text;
        int[] tokens = _tokenizer.Encode(text);
        for (int keep = Math.Max(0, tokens.Length - count); keep > 0; keep--)
        {
            string prefix = _tokenizer.Decode(tokens.AsSpan(0, keep).ToArray());
            if (!prefix.Contains('\uFFFD')) return prefix;
        }
        return "";
    }

    public string Decode(ReadOnlySpan<float> audio, string prefix, int maxTokens,
        IncrementalMelState melState, CancellationToken cancellationToken, bool final = false)
    {
        cancellationToken.ThrowIfCancellationRequested();
        AudioFeatures features = _encoder.Encode(audio, melState);
        if (features.IsEmpty) return prefix;
        int[] prompt = _tokenizer.BuildPromptTokens(_metadata.AudioToken, features.Rows,
            _options.Context, _options.Language, prefix);
        cancellationToken.ThrowIfCancellationRequested();
        if (prompt.Length + maxTokens > _metadata.StaticCacheLen)
            throw new InvalidOperationException($"Prompt ({prompt.Length}) + decode budget ({maxTokens}) exceeds " +
                $"package cache capacity ({_metadata.StaticCacheLen}). Reduce context/window or export a larger cache.");
        var result = _decoder.Decode(prompt, features, maxTokens, cancellationToken);
        cancellationToken.ThrowIfCancellationRequested();
        if (final && result.Tokens.Length == maxTokens)
            throw new InvalidOperationException("Final decoding exhausted its token budget. Increase FinalMaxNewTokens or shorten the audio window.");
        return prefix + _tokenizer.Decode(result.Tokens);
    }

    public void Dispose()
    {
        _decoder.Dispose();
        _encoder.Dispose();
        _sessions.Dispose();
    }
}
