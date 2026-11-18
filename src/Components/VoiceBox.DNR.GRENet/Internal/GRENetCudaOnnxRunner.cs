using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using VoiceBox.Core.Onnx;
using VoiceBox.DNR.GRENet.Models;

namespace VoiceBox.DNR.GRENet.Internal;

/// <summary>
/// Keeps attention rings on the GPU and alternates banks for full-state caches.
/// Only the two 201-bin frame outputs return to the CPU for ISTFT.
/// </summary>
internal sealed class GRENetCudaOnnxRunner : IGRENetOnnxRunner
{
    private const int ExpectedFrequencyBins = 201;

    private readonly InferenceSession _session;
    private readonly SessionOptions _sessionOptions;
    private readonly RunOptions _firstBankOptions;
    private readonly RunOptions _secondBankOptions;
    private readonly nint _context;
    private readonly OrtMemoryInfo _memoryInfo;
    private readonly OrtAllocator _allocator;
    private readonly OrtIoBinding[] _bindings;
    private readonly int _cacheCount;
    private readonly OrtValue[][] _cacheBanks;
    private readonly OrtValue _ampValue;
    private readonly OrtValue _phaseValue;
    private readonly OrtValue _outputAmplitudeValue;
    private readonly OrtValue _outputPhaseValue;
    private readonly OrtValue _outputComplexValue;
    private readonly string[] _inputNames;
    private readonly string[] _outputNames;
    private readonly float[] _hostOutputAmplitude = new float[ExpectedFrequencyBins];
    private readonly float[] _hostOutputPhase = new float[ExpectedFrequencyBins];
    private readonly ulong _ampPointer;
    private readonly ulong _phasePointer;
    private readonly ulong _outputAmplitudePointer;
    private readonly ulong _outputPhasePointer;
    private readonly ulong[][] _cachePointers;
    private readonly bool[] _frameUpdates;
    private readonly CudaDeviceMemory.RingCopy[] _ringCopies;
    private int _bank;

    public GRENetCudaOnnxRunner(string modelPath, GRENetOptions options)
    {
        ArgumentNullException.ThrowIfNull(options);
        if (!File.Exists(modelPath))
        {
            throw new FileNotFoundException($"GRE-Net model file not found: {modelPath}", modelPath);
        }

        _sessionOptions = CreateSessionOptions(options);
        try
        {
            _session = new InferenceSession(modelPath, _sessionOptions);
            _firstBankOptions = new RunOptions();
            _secondBankOptions = new RunOptions();
            // Graph 0 and graph 1 capture different cache addresses. The CUDA EP ignores IDs when capture is off.
            _secondBankOptions.AddRunConfigEntry("gpu_graph_id", "1");
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

            _memoryInfo = new OrtMemoryInfo("Cuda", OrtAllocatorType.DeviceAllocator, options.GpuDeviceId, OrtMemType.Default);
            _allocator = new OrtAllocator(_session, _memoryInfo);
            _ampValue = Allocate(_allocator, inputs[_inputNames[0]], FrequencyBins);
            _phaseValue = Allocate(_allocator, inputs[_inputNames[1]], FrequencyBins);
            _outputAmplitudeValue = Allocate(_allocator, outputs[_outputNames[0]], FrequencyBins);
            _outputPhaseValue = Allocate(_allocator, outputs[_outputNames[1]], FrequencyBins);
            _outputComplexValue = Allocate(_allocator, outputs[_outputNames[2]], FrequencyBins);

            _cacheCount = _inputNames.Length - 2;

            _frameUpdates = new bool[_cacheCount];
            for (int i = 0; i < _cacheCount; i++)
            {
                string output = _outputNames[i + 3];
                if (output.EndsWith("_frame", StringComparison.Ordinal))
                {
                    string input = _inputNames[i + 2];
                    RequireName(output, input + "_frame");
                    int[] ring = inputs[input].Dimensions;
                    int[] frame = outputs[output].Dimensions;
                    if (ring.Length != 3 || ring.Any(size => size <= 0) || frame.Length != 2 || frame[0] != ring[0] || frame[1] != ring[2])
                    {
                        throw new InvalidOperationException($"Invalid GRE-Net attention ring shape for '{input}'.");
                    }

                    _frameUpdates[i] = true;
                }
            }

            _cacheBanks = new OrtValue[2][];
            for (int bank = 0; bank < _cacheBanks.Length; bank++)
            {
                _cacheBanks[bank] = new OrtValue[_cacheCount];
                for (int i = 0; i < _cacheCount; i++)
                {
                    // For frame updates bank 0 owns the persistent ring and bank 1 the small output frame.
                    var metadata = _frameUpdates[i] && bank == 1 ? outputs[_outputNames[i + 3]] : inputs[_inputNames[i + 2]];
                    _cacheBanks[bank][i] = Allocate(_allocator, metadata, FrequencyBins);
                }
            }

            _ampPointer = CudaDeviceMemory.GetDevicePointer(_ampValue);
            _context = CudaDeviceMemory.GetAllocationContext(_ampPointer);
            _phasePointer = CudaDeviceMemory.GetDevicePointer(_phaseValue);
            _outputAmplitudePointer = CudaDeviceMemory.GetDevicePointer(_outputAmplitudeValue);
            _outputPhasePointer = CudaDeviceMemory.GetDevicePointer(_outputPhaseValue);
            _cachePointers = new ulong[2][];
            for (int bank = 0; bank < 2; bank++)
            {
                _cachePointers[bank] = new ulong[_cacheCount];
                for (int i = 0; i < _cacheCount; i++)
                {
                    _cachePointers[bank][i] = CudaDeviceMemory.GetDevicePointer(_cacheBanks[bank][i]);
                }
            }

            var copies = new List<CudaDeviceMemory.RingCopy>();
            for (int i = 0; i < _cacheCount; i++)
            {
                if (_frameUpdates[i])
                {
                    int[] shape = inputs[_inputNames[i + 2]].Dimensions;
                    copies.Add(new(_cachePointers[1][i], _cachePointers[0][i], shape[0], shape[1], shape[2]));
                }
            }

            _ringCopies = copies.ToArray();
            _bindings = new OrtIoBinding[2];
            for (int bank = 0; bank < 2; bank++)
            {
                _bindings[bank] = CreateBinding(bank, 1 - bank);
            }

            Reset();
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

        using var scope = new CudaDeviceMemory.ContextScope(_context);
        CudaDeviceMemory.CopyHostToDevice(_ampPointer, amplitude);
        CudaDeviceMemory.CopyHostToDevice(_phasePointer, phase);
        OrtIoBinding binding = _bindings[_bank];
        _session.RunWithBinding(_bank == 0 ? _firstBankOptions : _secondBankOptions, binding);
        binding.SynchronizeBoundOutputs();
        foreach (ref var copy in _ringCopies.AsSpan())
        {
            copy.Append();
        }

        CudaDeviceMemory.CopyDeviceToHost(_hostOutputAmplitude, _outputAmplitudePointer);
        CudaDeviceMemory.CopyDeviceToHost(_hostOutputPhase, _outputPhasePointer);
        _hostOutputAmplitude.AsSpan().CopyTo(outputAmplitude);
        _hostOutputPhase.AsSpan().CopyTo(outputPhase);
        _bank = 1 - _bank;
    }

    public void Reset()
    {
        using var scope = new CudaDeviceMemory.ContextScope(_context);
        for (int bank = 0; bank < 2; bank++)
        {
            for (int i = 0; i < _cacheCount; i++)
            {
                CudaDeviceMemory.Clear(_cachePointers[bank][i], _cacheBanks[bank][i].GetTensorSizeInBytes());
            }
        }

        CudaDeviceMemory.Clear(_ampPointer, _ampValue.GetTensorSizeInBytes());
        CudaDeviceMemory.Clear(_phasePointer, _phaseValue.GetTensorSizeInBytes());
        CudaDeviceMemory.FinishClearing();
        foreach (ref var copy in _ringCopies.AsSpan())
        {
            copy.Reset();
        }

        _bank = 0;
    }

    public void Dispose()
    {
        foreach (OrtIoBinding binding in _bindings)
        {
            binding.Dispose();
        }

        DisposeValues(_cacheBanks);
        _outputComplexValue.Dispose();
        _outputPhaseValue.Dispose();
        _outputAmplitudeValue.Dispose();
        _phaseValue.Dispose();
        _ampValue.Dispose();
        _allocator.Dispose();
        _memoryInfo.Dispose();
        _firstBankOptions.Dispose();
        _secondBankOptions.Dispose();
        _session.Dispose();
        _sessionOptions.Dispose();
    }

    private static SessionOptions CreateSessionOptions(GRENetOptions options)
    {
        if (options.SessionOptionsFactory is not null || options.CudaTuning is not null)
        {
            return OnnxSessionFactory.Create(options, SessionPurpose.StaticShapes);
        }

        var tuning = new CudaExecutionTuning
        {
            AdditionalOptions = new()
            {
                ["enable_cuda_graph"] = "1",
            },
        };
        return OnnxSessionFactory.CreateCuda(options.GpuDeviceId, options.ThreadsNum, tuning, SessionPurpose.StaticShapes);
    }

    private OrtIoBinding CreateBinding(int inputBank, int outputBank)
    {
        var binding = _session.CreateIoBinding();
        binding.BindInput(_inputNames[0], _ampValue);
        binding.BindInput(_inputNames[1], _phaseValue);
        for (int i = 0; i < _cacheCount; i++)
        {
            binding.BindInput(_inputNames[i + 2], _cacheBanks[_frameUpdates[i] ? 0 : inputBank][i]);
        }

        binding.BindOutput(_outputNames[0], _outputAmplitudeValue);
        binding.BindOutput(_outputNames[1], _outputPhaseValue);
        binding.BindOutput(_outputNames[2], _outputComplexValue);
        for (int i = 0; i < _cacheCount; i++)
        {
            binding.BindOutput(_outputNames[i + 3], _cacheBanks[_frameUpdates[i] ? 1 : outputBank][i]);
        }

        return binding;
    }

    private static OrtValue Allocate(OrtAllocator allocator, NodeMetadata metadata, int frequencyBins)
    {
        long[] shape = ResolveShape(metadata, frequencyBins);
        return OrtValue.CreateAllocatedTensorValue(allocator, TensorElementType.Float, shape);
    }

    private static int ResolveFrequencyBins(NodeMetadata metadata)
    {
        int[] dimensions = metadata.Dimensions;
        return dimensions.Length >= 2 && dimensions[1] > 0 ? dimensions[1] : ExpectedFrequencyBins;
    }

    private static long[] ResolveShape(NodeMetadata metadata, int frequencyBins)
    {
        int[] dimensions = metadata.Dimensions;
        long[] shape = new long[dimensions.Length];
        for (int i = 0; i < dimensions.Length; i++)
        {
            shape[i] = dimensions[i] > 0 ? dimensions[i] : i == 1 ? frequencyBins : 1;
        }

        return shape;
    }

    private static void RequireName(string actual, string expected)
    {
        if (!string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"GRE-Net model tensor '{expected}' was not found at the expected position (got '{actual}').");
        }
    }

    private static void DisposeValues(IEnumerable<OrtValue[]> banks)
    {
        foreach (OrtValue[] bank in banks)
        {
            if (bank is null)
            {
                continue;
            }

            foreach (OrtValue value in bank)
            {
                value?.Dispose();
            }
        }
    }

    private void DisposePartiallyInitialized()
    {
        if (_bindings is not null)
        {
            foreach (OrtIoBinding binding in _bindings)
            {
                binding?.Dispose();
            }
        }

        if (_cacheBanks is not null)
        {
            DisposeValues(_cacheBanks);
        }

        _outputComplexValue?.Dispose();
        _outputPhaseValue?.Dispose();
        _outputAmplitudeValue?.Dispose();
        _phaseValue?.Dispose();
        _ampValue?.Dispose();
        _allocator?.Dispose();
        _memoryInfo?.Dispose();
        _firstBankOptions?.Dispose();
        _secondBankOptions?.Dispose();
        _session?.Dispose();
        _sessionOptions?.Dispose();
    }
}
