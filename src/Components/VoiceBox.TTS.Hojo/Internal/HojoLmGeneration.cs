using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.Hojo.Internal;

/// <summary>Owns one utterance's native KV buffers. Present tensors are fed back without managed copies.</summary>
internal sealed class HojoLmGeneration : IDisposable
{
    private readonly InferenceSession _session;
    private readonly HojoModelBundle _model;
    private readonly RunOptions _run;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly OrtValue[] _inputs;
    private readonly OrtValue[] _empty;
    private IDisposableReadOnlyCollection<OrtValue>? _outputs;
    private int _position;
    private readonly float[] _stepEmbedding = new float[512];
    private readonly long[] _stepPosition = new long[1];
    private readonly OrtValue _stepValues = null!, _stepPositions = null!;
    public ReadOnlySpan<float> Logits => _outputs![0].GetTensorDataAsSpan<float>()[^_model.VocabularySize..];
    public ReadOnlySpan<float> Hidden => _outputs![1].GetTensorDataAsSpan<float>()[^512..];

    public HojoLmGeneration(InferenceSession session, HojoModelBundle model, RunOptions run, string[] inputNames, string[] outputNames)
    {
        _session = session; _model = model; _run = run; _inputNames = inputNames; _outputNames = outputNames;
        _inputs = new OrtValue[2 + model.Layers * 2];
        _empty = new OrtValue[model.Layers * 2];
        try
        {
            // These two buffers have fixed shapes and are owned by this utterance.
            _stepValues = OrtValue.CreateTensorValueFromMemory(_stepEmbedding, [1, 1, 512]);
            _stepPositions = OrtValue.CreateTensorValueFromMemory(_stepPosition, [1, 1]);
            for (int i = 0; i < _empty.Length; i++) _empty[i] = OrtValue.CreateTensorValueFromMemory(Array.Empty<float>(), [1, 1, 0, 128]);
        }
        catch { Dispose(); throw; }
    }

    public void Prefill(int[] tokens, int voice)
    {
        var embedding = new float[tokens.Length * 512];
        var positions = new long[tokens.Length];
        for (int i = 0; i < tokens.Length; i++)
        {
            _model.TokenEmbedding(tokens[i]).CopyTo(embedding.AsSpan(i * 512, 512));
            positions[i] = i;
        }
        // Prompt ends with 16 speaker slots, spk_end, target_speech_start.
        if (_model.UsesSpeakerSlots)
            _model.Speaker(voice).CopyTo(embedding.AsSpan((tokens.Length - 18) * 512, 16 * 512));
        using var values = OrtValue.CreateTensorValueFromMemory(embedding, [1, tokens.Length, 512]);
        using var position = OrtValue.CreateTensorValueFromMemory(positions, [1, tokens.Length]);
        Run(values, position, tokens.Length);
    }

    public void Step(int token)
    {
        _model.TokenEmbedding(token).CopyTo(_stepEmbedding);
        _stepPosition[0] = _position;
        Run(_stepValues, _stepPositions, 1);
    }

    private void Run(OrtValue values, OrtValue position, int length)
    {
        _inputs[0] = values; _inputs[1] = position;
        for (int i = 0; i < _empty.Length; i++) _inputs[i + 2] = _outputs is null ? _empty[i] : _outputs[i + 2];
        var next = _session.Run(_run, _inputNames, _inputs, _outputNames);
        _outputs?.Dispose();
        _outputs = next;
        _position += length;
    }

    public void Dispose()
    {
        _outputs?.Dispose(); _outputs = null;
        foreach (var value in _empty) value?.Dispose();
        _stepPositions?.Dispose();
        _stepValues?.Dispose();
    }
}
