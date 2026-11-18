using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.DNR.GRENet.Models;

namespace VoiceBox.DNR.GRENet.Internal;

/// <summary>
/// Stateful one-frame GRE-Net ONNX runner. The model exposes every temporal cache
/// as an input/output tensor. Each hop consumes the preceding run's cache tensors directly.
/// </summary>
internal sealed class GRENetCpuOnnxRunner : IGRENetOnnxRunner
{
    private const int ExpectedFrequencyBins = 201;

    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private readonly RunOptions _runOptions;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly float[] _ampInput;
    private readonly float[] _phaseInput;
    private readonly float[][] _cacheBuffers;
    private readonly OrtValue _ampValue;
    private readonly OrtValue _phaseValue;
    private readonly OrtValue[] _cacheValues;
    private readonly OrtValue[] _inputValues;
    private readonly int[] _outputLengths;
    private readonly AttentionRing?[] _attentionRings;
    private IDisposableReadOnlyCollection<OrtValue>? _previousOutputs;

    public GRENetCpuOnnxRunner(string modelPath, GRENetOptions onnxOptions)
    {
        ArgumentNullException.ThrowIfNull(onnxOptions);
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"GRE-Net model file not found: {modelPath}", modelPath);
        }

        _sessionOptions = OnnxSessionFactory.Create(onnxOptions, SessionPurpose.StaticShapes);
        try
        {
            _session = new InferenceSession(modelPath, _sessionOptions);
            _runOptions = new RunOptions();
            var inputs = _session.InputMetadata;
            var outputs = _session.OutputMetadata;
            // The bundled graph uses 35 inputs with per-layer FFN histories. Older exports use 29, 31 or 35.
            if (inputs.Count is not (29 or 31 or 35) || outputs.Count != inputs.Count + 1)
            {
                throw new InvalidOperationException($"Unsupported GRE-Net signature: {inputs.Count} inputs, {outputs.Count} outputs.");
            }

            _inputNames = inputs.Keys.ToArray();
            _outputNames = outputs.Keys.ToArray();
            RequireName(_inputNames[0], "noisy_amp_t");
            RequireName(_inputNames[1], "noisy_pha_t");
            RequireName(_outputNames[0], "denoised_amp_t");
            RequireName(_outputNames[1], "denoised_pha_t");

            FrequencyBins = ResolveFrequencyBins(inputs[_inputNames[0]]);
            if (FrequencyBins != ExpectedFrequencyBins)
            {
                throw new InvalidOperationException(
                    $"GRE-Net model must expose {ExpectedFrequencyBins} frequency bins, got {FrequencyBins}.");
            }

            _ampInput = new float[FrequencyBins];
            _phaseInput = new float[FrequencyBins];
            _ampValue = OrtValue.CreateTensorValueFromMemory(_ampInput, new long[] { 1, FrequencyBins, 1 });
            _phaseValue = OrtValue.CreateTensorValueFromMemory(_phaseInput, new long[] { 1, FrequencyBins, 1 });

            _cacheBuffers = new float[_inputNames.Length - 2][];
            _cacheValues = new OrtValue[_cacheBuffers.Length];
            _outputLengths = new int[_cacheBuffers.Length];
            _attentionRings = new AttentionRing?[_cacheBuffers.Length];
            for (int i = 0; i < _cacheBuffers.Length; i++)
            {
                NodeMetadata metadata = inputs[_inputNames[i + 2]];
                long[] shape = ResolveShape(metadata, FrequencyBins);
                int length = checked((int)shape.Aggregate(1L, (current, dimension) => current * dimension));
                _cacheBuffers[i] = new float[length];
                _cacheValues[i] = OrtValue.CreateTensorValueFromMemory(_cacheBuffers[i], shape);
                _outputLengths[i] = length;
                string inputName = _inputNames[i + 2];
                string outputName = _outputNames[i + 3];
                if (outputName == $"{inputName}_frame")
                {
                    long[] frame = ResolveShape(outputs[outputName], FrequencyBins);
                    if (shape.Length != 3 || frame.Length != 2 || frame[0] != shape[0] || frame[1] != shape[2])
                    {
                        throw new InvalidOperationException($"Invalid GRE-Net attention ring shape for '{inputName}'.");
                    }

                    _attentionRings[i] = new AttentionRing(_cacheBuffers[i], (int)shape[1], (int)shape[2]);
                    _outputLengths[i] = checked((int)(frame[0] * frame[1]));
                }
            }

            _inputValues = new OrtValue[2 + _cacheValues.Length];
            _inputValues[0] = _ampValue;
            _inputValues[1] = _phaseValue;
            Array.Copy(_cacheValues, 0, _inputValues, 2, _cacheValues.Length);
        }
        catch
        {
            DisposePartiallyInitialized();
            throw;
        }
    }

    public int FrequencyBins { get; }

    public void Run(ReadOnlySpan<float> amplitude, ReadOnlySpan<float> phase, Span<float> outputAmplitude, Span<float> outputPhase)
    {
        if (amplitude.Length != FrequencyBins || phase.Length != FrequencyBins ||
            outputAmplitude.Length != FrequencyBins || outputPhase.Length != FrequencyBins)
        {
            throw new ArgumentException($"GRE-Net spectra must contain {FrequencyBins} bins.");
        }

        amplitude.CopyTo(_ampInput);
        phase.CopyTo(_phaseInput);
        var results = _session.Run(_runOptions, _inputNames, _inputValues, _outputNames);
        try
        {
            results[0].GetTensorDataAsSpan<float>()[..FrequencyBins].CopyTo(outputAmplitude);
            results[1].GetTensorDataAsSpan<float>()[..FrequencyBins].CopyTo(outputPhase);
            for (int i = 0; i < _cacheValues.Length; i++)
            {
                if (results[i + 3].GetTensorDataAsSpan<float>().Length != _outputLengths[i])
                {
                    throw new InvalidOperationException($"GRE-Net cache {i} length mismatch.");
                }
            }
        }
        catch
        {
            results.Dispose();
            throw;
        }

        for (int i = 0; i < _cacheValues.Length; i++)
        {
            if (_attentionRings[i] is AttentionRing ring)
            {
                ring.Append(results[i + 3].GetTensorDataAsSpan<float>());
            }
            else
            {
                _inputValues[i + 2] = results[i + 3];
            }
        }

        _previousOutputs?.Dispose();
        _previousOutputs = results;
    }

    public void Reset()
    {
        foreach (AttentionRing? ring in _attentionRings)
        {
            ring?.Clear();
        }

        Array.Copy(_cacheValues, 0, _inputValues, 2, _cacheValues.Length);
        _previousOutputs?.Dispose();
        _previousOutputs = null;
    }

    public void Dispose()
    {
        _previousOutputs?.Dispose();
        _ampValue.Dispose();
        _phaseValue.Dispose();
        foreach (OrtValue cache in _cacheValues)
        {
            cache.Dispose();
        }

        _runOptions.Dispose();
        _session.Dispose();
        _sessionOptions.Dispose();
    }

    private sealed class AttentionRing(float[] buffer, int history, int width)
    {
        private int _slot;

        public void Append(ReadOnlySpan<float> frame)
        {
            int destination = _slot * width;
            for (int source = 0; source < frame.Length; source += width)
            {
                frame.Slice(source, width).CopyTo(buffer.AsSpan(destination, width));
                destination += history * width;
            }

            _slot = (_slot + 1) % history;
        }

        public void Clear()
        {
            Array.Clear(buffer);
            _slot = 0;
        }
    }

    private static void RequireName(string actual, string expected)
    {
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"GRE-Net model tensor '{expected}' was not found at the expected position (got '{actual}').");
        }
    }

    private static int ResolveFrequencyBins(NodeMetadata metadata)
    {
        int[] dimensions = metadata.Dimensions;
        return dimensions.Length >= 2 && dimensions[1] > 0 ? dimensions[1] : ExpectedFrequencyBins;
    }

    private static long[] ResolveShape(NodeMetadata metadata, int frequencyBins)
    {
        int[] dimensions = metadata.Dimensions;
        if (dimensions.Length == 0 || dimensions.Any(dimension => dimension == 0 || dimension < -1))
        {
            throw new InvalidOperationException("GRE-Net cache tensor has an invalid shape.");
        }

        long[] shape = new long[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            shape[i] = dimensions[i] > 0 ? dimensions[i] : i == 1 ? frequencyBins : 1;
        }

        return shape;
    }

    private void DisposePartiallyInitialized()
    {
        if (_cacheValues is not null)
        {
            foreach (OrtValue value in _cacheValues)
            {
                value?.Dispose();
            }
        }

        _ampValue?.Dispose();
        _phaseValue?.Dispose();
        _runOptions?.Dispose();
        _session?.Dispose();
        _sessionOptions?.Dispose();
    }
}
