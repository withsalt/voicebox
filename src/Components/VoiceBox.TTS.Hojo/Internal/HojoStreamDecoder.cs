using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.Hojo.Internal;

/// <summary>Bounded overlapping windows; retains no utterance-sized hidden-state or waveform arrays.</summary>
internal sealed class HojoStreamDecoder
{
    private const int History = 60, Context = 10;
    private static readonly string[] FineInputs = ["hidden_states", "coarse_embeddings", "speaker_embedding", "valid_mask"];
    private static readonly string[] FineOutputs = ["binary_logits"];
    private static readonly string[] CodecInputs = ["bits"], CodecOutputs = ["mag", "phase"];
    private readonly InferenceSession _fine, _codec;
    private readonly RunOptions _run;
    private readonly HojoModelBundle _model;
    private readonly HojoIstft _istft;
    private readonly float[] _hidden, _coarse, _bits;
    private readonly float[] _speaker;
    private readonly bool[] _validMask;
    private readonly int _chunkFrames;
    private int _frames, _processed, _bitFrames, _emitted, _totalFrames, _bitStart;
    public int TotalFrames => _totalFrames;

    public HojoStreamDecoder(InferenceSession fine, InferenceSession codec, RunOptions run, HojoModelBundle model, HojoIstft istft, int voice, int chunkFrames)
    {
        _fine = fine; _codec = codec; _run = run; _model = model; _istft = istft;
        _chunkFrames = chunkFrames;
        _hidden = new float[(History + chunkFrames + Context) * 512];
        _coarse = new float[_hidden.Length];
        _bits = new float[(chunkFrames + 2 * Context) * 128];
        _speaker = model.SpeakerVector(voice);
        _validMask = new bool[History + chunkFrames + Context];
        Array.Fill(_validMask, true);
    }

    public void Add(int token, ReadOnlySpan<float> hidden)
    {
        hidden.CopyTo(_hidden.AsSpan(_frames * 512, 512));
        _model.TokenEmbedding(token).CopyTo(_coarse.AsSpan(_frames * 512, 512));
        _frames++; _totalFrames++;
    }

    public bool IsReady => _totalFrames >= _emitted + _chunkFrames + Context;

    public float[] Flush(bool final, CancellationToken ct)
    {
        ct.ThrowIfCancellationRequested();
        int pending = _frames - _processed;
        if (pending > 0)
        {
            using var hidden = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _hidden.AsMemory(0, _frames * 512), [1, _frames, 512]);
            using var coarse = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _coarse.AsMemory(0, _frames * 512), [1, _frames, 512]);
            using var speaker = OrtValue.CreateTensorValueFromMemory(_speaker, [1, _model.SpeakerDimension]);
            using var valid = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _validMask.AsMemory(0, _frames), [1, _frames]);
            using var output = _fine.Run(_run, FineInputs, [hidden, coarse, speaker, valid], FineOutputs);
            var logits = output[0].GetTensorDataAsSpan<float>();
            if (logits.Length != _frames * 128) throw new InvalidDataException("FineLocal returned an unexpected shape.");
            for (int i = 0; i < pending * 128; i++)
            {
                float value = logits[_processed * 128 + i];
                if (!float.IsFinite(value)) throw new InvalidDataException("FineLocal returned non-finite logits.");
                _bits[_bitFrames * 128 + i] = value > 0 ? 1 : -1;
            }
            _bitFrames += pending;
        }
        int stable = final ? _totalFrames : _totalFrames - Context;
        int count = (stable - _emitted) * HojoIstft.Hop;
        if (count == 0) return [];
        using var bits = OrtValue.CreateTensorValueFromMemory(OrtMemoryInfo.DefaultInstance, _bits.AsMemory(0, _bitFrames * 128), [_bitFrames, 128]);
        using var decoded = _codec.Run(_run, CodecInputs, [bits], CodecOutputs);
        float[] audio = _istft.Decode(decoded[0].GetTensorDataAsSpan<float>(), decoded[1].GetTensorDataAsSpan<float>(),
            _bitFrames, (_emitted - _bitStart) * HojoIstft.Hop, count, ct);
        _emitted = stable;
        // Keep enough raw input for all four FineLocal layers, and ten bits before the next PCM boundary.
        int keep = Math.Min(History, _frames);
        Array.Copy(_hidden, (_frames - keep) * 512, _hidden, 0, keep * 512);
        Array.Copy(_coarse, (_frames - keep) * 512, _coarse, 0, keep * 512);
        _frames = _processed = keep;
        int nextStart = Math.Max(0, _emitted - Context);
        int discard = nextStart - _bitStart;
        Array.Copy(_bits, discard * 128, _bits, 0, (_bitFrames - discard) * 128);
        _bitFrames -= discard; _bitStart = nextStart;
        return audio;
    }
}
