using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.TTSService.Models;

namespace VoiceBox.TTS.Hojo.Internal;

internal interface IHojoPipeline : IDisposable
{
    IReadOnlyList<TtsVoice> Voices { get; }
    IEnumerable<float[]> Generate(string text, HojoRequest request, CancellationToken ct);
}

internal sealed class HojoPipeline : IHojoPipeline
{
    private readonly HojoModelBundle _model;
    private readonly HojoTokenizer _tokenizer;
    private readonly HojoTextNormalizer _normalizer;
    private readonly HojoIstft _istft = new();
    private readonly InferenceSession _lm = null!, _fine = null!, _codec = null!;
    private readonly string[] _lmInputs, _lmOutputs;
    private readonly int _chunkFrames;
    public IReadOnlyList<TtsVoice> Voices => _model.Voices;

    public HojoPipeline(HojoTTSOptions options)
    {
        _model = new(options.ModelPath);
        _model.VoiceIndex(options.DefaultVoice);
        _tokenizer = new(Path.Combine(options.ModelPath, "tokenizer.json"), _model.VocabularySize);
        _normalizer = new(options.NormalizeText);
        _chunkFrames = options.ChunkFrames;
        _lmInputs = new string[2 + _model.Layers * 2]; _lmOutputs = new string[_lmInputs.Length];
        _lmInputs[0] = "inputs_embeds"; _lmInputs[1] = "position_ids";
        _lmOutputs[0] = "logits"; _lmOutputs[1] = "last_hidden_state";
        for (int layer = 0; layer < _model.Layers; layer++)
        {
            _lmInputs[2 + layer * 2] = $"past_key_values.{layer}.key";
            _lmInputs[3 + layer * 2] = $"past_key_values.{layer}.value";
            _lmOutputs[2 + layer * 2] = $"present.{layer}.key";
            _lmOutputs[3 + layer * 2] = $"present.{layer}.value";
        }
        try
        {
            using var sessionOptions = OnnxSessionFactory.Create(options);
            if (options.SessionOptionsFactory is null)
            {
                // Three sessions run sequentially. Idle worker spinning would consume all four E-cores
                // even with ThreadsNum=2, competing with ASR and the active session.
                sessionOptions.AddSessionConfigEntry("session.intra_op.allow_spinning", "0");
                sessionOptions.AddSessionConfigEntry("session.inter_op.allow_spinning", "0");
            }
            _lm = new(Path.Combine(options.ModelPath, "lm.onnx"), sessionOptions);
            _fine = new(Path.Combine(options.ModelPath, "fine.onnx"), sessionOptions);
            _codec = new(Path.Combine(options.ModelPath, "decoder.onnx"), sessionOptions);
            Check(_lm, _lmInputs, _lmOutputs);
            Check(_fine, ["hidden_states", "coarse_embeddings", "speaker_embedding", "valid_mask"], ["binary_logits"]);
            Check(_codec, ["bits"], ["mag", "phase"]);
        }
        catch { Dispose(); throw; }
    }

    private static void Check(InferenceSession session, string[] inputs, string[] outputs)
    {
        if (!session.InputMetadata.Keys.ToHashSet().SetEquals(inputs) || !session.OutputMetadata.Keys.ToHashSet().SetEquals(outputs))
            throw new InvalidDataException("Hojo ONNX input/output contract mismatch.");
        foreach (var (name, metadata) in session.InputMetadata)
        {
            Type expected = name == "position_ids" ? typeof(long) : name == "valid_mask" ? typeof(bool) : typeof(float);
            if (metadata.ElementType != expected) throw new InvalidDataException($"Unsupported tensor precision for {name}; run the model preparation script.");
        }
    }

    public IEnumerable<float[]> Generate(string text, HojoRequest request, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        int voice = _model.VoiceIndex(request.Voice);
        string normalized = _normalizer.Normalize(text, request.Language);
        if (string.IsNullOrWhiteSpace(normalized)) yield break;
        if (normalized.Length > 16384) throw new ArgumentException("Normalized text is too long; split into shorter utterances.", nameof(text));
        int[] tokens = _tokenizer.EncodePrompt(normalized, _model.VoicePrompt(voice));
        if (tokens.Length >= HojoModelBundle.MaxPositions - 10)
            throw new ArgumentException("Text exceeds the Hojo context window; split into shorter utterances.", nameof(text));
        using var run = new RunOptions();
        using var registration = ct.Register(() => run.Terminate = true);
        using var lm = new HojoLmGeneration(_lm, _model, run, _lmInputs, _lmOutputs);
        var decoder = new HojoStreamDecoder(_fine, _codec, run, _model, _istft, voice, _chunkFrames);
        var sampler = new HojoSampler(request, _model.VocabularySize);
        ct.ThrowIfCancellationRequested();
        lm.Prefill(tokens, voice);
        int limit = Math.Min(request.MaxFrames, HojoModelBundle.MaxPositions - tokens.Length);
        for (int i = 0; i < limit; i++)
        {
            ct.ThrowIfCancellationRequested();
            int token = sampler.Next(lm.Logits);
            if (token == _tokenizer.SpeechEndId)
            {
                if (decoder.TotalFrames == 0) throw new InvalidDataException("Hojo ended without generating audio.");
                float[] tail = decoder.Flush(final: true, ct);
                if (tail.Length > 0) yield return tail;
                yield break;
            }
            if (_tokenizer.IsAudio(token))
            {
                decoder.Add(token, lm.Hidden);
                if (decoder.IsReady) yield return decoder.Flush(final: false, ct);
            }
            if (i + 1 < limit) lm.Step(token);
        }
        throw new InvalidOperationException($"Hojo reached its {limit}-token safety limit before EOS; split the text into shorter utterances.");
    }

    public void Dispose()
    {
        _codec?.Dispose(); _fine?.Dispose(); _lm?.Dispose(); _normalizer.Dispose();
    }
}
