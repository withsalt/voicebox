using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace Misaki.NET.Lang.English.Neural;

/// <summary>
/// Wraps the DeepPhonemizer ForwardTransformer ONNX session. Takes encoded text ids,
/// runs inference, returns argmax phoneme ids per position.
/// </summary>
internal sealed class DeepPhonemizerSession : IDisposable
{
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;
    private readonly string _inputName;
    private readonly string _outputName;
    private readonly object _lock = new();
    private bool _disposed;

    public DeepPhonemizerSession(string onnxModelPath)
    {
        if (!File.Exists(onnxModelPath))
            throw new FileNotFoundException($"DeepPhonemizer ONNX model not found: {onnxModelPath}");

        var so = new SessionOptions
        {
            GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL,
            InterOpNumThreads = 1,
            IntraOpNumThreads = Math.Max(1, Environment.ProcessorCount / 2),
        };

        _session = new InferenceSession(onnxModelPath, so);
        _runOptions = new RunOptions();
        _inputName = _session.InputMetadata.Keys.First();
        _outputName = _session.OutputMetadata.Keys.First();
    }

    /// <summary>
    /// Runs the model on a single word's token ids, returns argmax phoneme ids per
    /// position. Thread-safe: ORT session is reentrant, but argmax buffers aren't, so
    /// we serialize. G2P is called per-token; contention is minimal in practice.
    /// </summary>
    public int[] Predict(long[] tokenIds)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        long[] shape = [1L, tokenIds.Length];
        lock (_lock)
        {
            using var inputTensor = OrtValue.CreateTensorValueFromMemory(tokenIds, shape);
            var inputs = new Dictionary<string, OrtValue> { [_inputName] = inputTensor };

            using var results = _session.Run(_runOptions, inputs, [_outputName]);
            var logits = results[0];
            var info = logits.GetTensorTypeAndShape();
            var dims = info.Shape;
            if (dims.Length != 3 || dims[0] != 1)
                throw new InvalidDataException($"Unexpected DeepPhonemizer output shape: [{string.Join(',', dims)}]");

            int seqLen = (int)dims[1];
            int vocab = (int)dims[2];
            var flat = logits.GetTensorDataAsSpan<float>();

            var argmax = new int[seqLen];
            for (int t = 0; t < seqLen; t++)
            {
                int baseIdx = t * vocab;
                int best = 0;
                float bestVal = flat[baseIdx];
                for (int v = 1; v < vocab; v++)
                {
                    float val = flat[baseIdx + v];
                    if (val > bestVal) { bestVal = val; best = v; }
                }
                argmax[t] = best;
            }
            return argmax;
        }
    }

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _runOptions.Dispose();
        _session.Dispose();
    }
}
