using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;
using VoiceBox.DTD.RenderAware.Models;

namespace VoiceBox.DTD.RenderAware.Internal;

/// <summary>
/// Fixed-shape ONNX runner with reusable input/output tensors and double-buffered GRU state.
/// The owning service serializes all calls.
/// </summary>
internal sealed class RenderAwareDtdOnnxRunner : IDisposable
{
    internal const string MicrophoneInputName = "microphone";
    internal const string RenderInputName = "render";
    internal const string StateInputName = "state_in";
    internal const string ProbabilityOutputName = "nearend_probability";
    internal const string StateOutputName = "state_out";

    internal const int WindowLengthSamples = 640;
    private static readonly long[] AudioShape = [1, WindowLengthSamples];
    private static readonly long[] ProbabilityShape = [1, 1, 1];

    private readonly SessionOptions _sessionOptions;
    private readonly InferenceSession _session;
    private readonly RunOptions _runOptions;
    private readonly float[] _stateA;
    private readonly float[] _stateB;
    private readonly float[] _probability;
    private readonly OrtValue _microphoneValue;
    private readonly OrtValue _renderValue;
    private readonly OrtValue _stateAValue;
    private readonly OrtValue _stateBValue;
    private readonly OrtValue _probabilityValue;
    private readonly string[] _inputNames =
        [MicrophoneInputName, RenderInputName, StateInputName];
    private readonly string[] _outputNames =
        [ProbabilityOutputName, StateOutputName];
    private readonly OrtValue[] _inputValues;
    private readonly OrtValue[] _outputValues;
    private readonly OrtIoBinding? _bindingAtoB;
    private readonly OrtIoBinding? _bindingBtoA;
    private bool _currentStateIsA = true;
    private bool _disposed;

    public RenderAwareDtdOnnxRunner(
        string modelPath,
        RenderAwareDtdOptions options,
        float[] microphoneWindow,
        float[] renderWindow)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelPath);
        ArgumentNullException.ThrowIfNull(options);
        ArgumentNullException.ThrowIfNull(microphoneWindow);
        ArgumentNullException.ThrowIfNull(renderWindow);

        if (microphoneWindow.Length != WindowLengthSamples)
        {
            throw new ArgumentException(
                $"Microphone window length must be {WindowLengthSamples}.",
                nameof(microphoneWindow));
        }

        if (renderWindow.Length != WindowLengthSamples)
        {
            throw new ArgumentException(
                $"Render window length must be {WindowLengthSamples}.",
                nameof(renderWindow));
        }

        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"Render-aware DTD ONNX model not found: {modelPath}", modelPath);

        SessionOptions? sessionOptions = null;
        InferenceSession? session = null;
        RunOptions? runOptions = null;
        OrtValue? microphoneValue = null;
        OrtValue? renderValue = null;
        OrtValue? stateAValue = null;
        OrtValue? stateBValue = null;
        OrtValue? probabilityValue = null;
        OrtIoBinding? bindingAtoB = null;
        OrtIoBinding? bindingBtoA = null;

        try
        {
            sessionOptions = OnnxSessionFactory.Create(options, SessionPurpose.StaticShapes);
            session = new InferenceSession(modelPath, sessionOptions);
            int[] stateContractShape = ValidateContract(session, modelPath);
            runOptions = new RunOptions();

            StateLength = checked(stateContractShape[0] * stateContractShape[2]);
            long[] stateShape = [stateContractShape[0], stateContractShape[1], stateContractShape[2]];
            var stateA = new float[StateLength];
            var stateB = new float[StateLength];
            var probability = new float[1];

            microphoneValue = OrtValue.CreateTensorValueFromMemory(microphoneWindow, AudioShape);
            renderValue = OrtValue.CreateTensorValueFromMemory(renderWindow, AudioShape);
            stateAValue = OrtValue.CreateTensorValueFromMemory(stateA, stateShape);
            stateBValue = OrtValue.CreateTensorValueFromMemory(stateB, stateShape);
            probabilityValue = OrtValue.CreateTensorValueFromMemory(probability, ProbabilityShape);

            // Persistent CPU bindings see changes to the pinned managed buffers and eliminate
            // per-run marshalling. With a GPU/custom session, ORT may copy mismatched memory only
            // at bind time; use the regular OrtValue Run API there so every hop is current.
            if (options.ExecutionProvider == OnnxExecutionProvider.Cpu &&
                options.SessionOptionsFactory is null)
            {
                bindingAtoB = session.CreateIoBinding();
                Bind(bindingAtoB, microphoneValue, renderValue, stateAValue, probabilityValue, stateBValue);
                bindingBtoA = session.CreateIoBinding();
                Bind(bindingBtoA, microphoneValue, renderValue, stateBValue, probabilityValue, stateAValue);
            }

            _sessionOptions = sessionOptions;
            _session = session;
            _runOptions = runOptions;
            _stateA = stateA;
            _stateB = stateB;
            _probability = probability;
            _microphoneValue = microphoneValue;
            _renderValue = renderValue;
            _stateAValue = stateAValue;
            _stateBValue = stateBValue;
            _probabilityValue = probabilityValue;
            _inputValues = [microphoneValue, renderValue, stateAValue];
            _outputValues = [probabilityValue, stateBValue];
            _bindingAtoB = bindingAtoB;
            _bindingBtoA = bindingBtoA;
        }
        catch
        {
            bindingBtoA?.Dispose();
            bindingAtoB?.Dispose();
            probabilityValue?.Dispose();
            stateBValue?.Dispose();
            stateAValue?.Dispose();
            renderValue?.Dispose();
            microphoneValue?.Dispose();
            runOptions?.Dispose();
            session?.Dispose();
            sessionOptions?.Dispose();
            throw;
        }
    }

    public float RunInPlace()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        if (_bindingAtoB is not null)
        {
            _session.RunWithBinding(
                _runOptions,
                _currentStateIsA ? _bindingAtoB : _bindingBtoA!);
        }
        else
        {
            _session.Run(_runOptions, _inputNames, _inputValues, _outputNames, _outputValues);
        }

        float probability = _probability[0];
        if (!float.IsFinite(probability))
            throw new InvalidOperationException("Render-aware DTD model returned a non-finite probability.");

        float[] nextState = _currentStateIsA ? _stateB : _stateA;
        for (int i = 0; i < nextState.Length; i++)
        {
            if (!float.IsFinite(nextState[i]))
                throw new InvalidOperationException($"Render-aware DTD model returned non-finite state at index {i}.");
        }

        (_inputValues[2], _outputValues[1]) = (_outputValues[1], _inputValues[2]);
        _currentStateIsA = !_currentStateIsA;
        return Math.Clamp(probability, 0f, 1f);
    }

    public void Reset()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        Array.Clear(_stateA);
        Array.Clear(_stateB);
        Array.Clear(_probability);
        _inputValues[2] = _stateAValue;
        _outputValues[1] = _stateBValue;
        _currentStateIsA = true;
    }

    internal int StateLength { get; }

    internal void CopyCurrentStateTo(Span<float> destination)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        if (destination.Length != StateLength)
        {
            throw new ArgumentException(
                $"State destination length must be {StateLength}.",
                nameof(destination));
        }

        (_currentStateIsA ? _stateA : _stateB).CopyTo(destination);
    }

    public void Dispose()
    {
        if (_disposed)
            return;

        _disposed = true;
        _bindingBtoA?.Dispose();
        _bindingAtoB?.Dispose();
        _probabilityValue.Dispose();
        _stateBValue.Dispose();
        _stateAValue.Dispose();
        _renderValue.Dispose();
        _microphoneValue.Dispose();
        _runOptions.Dispose();
        _session.Dispose();
        _sessionOptions.Dispose();
    }

    private static void Bind(
        OrtIoBinding binding,
        OrtValue microphone,
        OrtValue render,
        OrtValue stateInput,
        OrtValue probability,
        OrtValue stateOutput)
    {
        binding.BindInput(MicrophoneInputName, microphone);
        binding.BindInput(RenderInputName, render);
        binding.BindInput(StateInputName, stateInput);
        binding.BindOutput(ProbabilityOutputName, probability);
        binding.BindOutput(StateOutputName, stateOutput);
    }

    private static int[] ValidateContract(InferenceSession session, string modelPath)
    {
        if (session.InputMetadata.Count != 3 || session.OutputMetadata.Count != 2)
        {
            throw new InvalidOperationException(
                $"Unexpected render-aware DTD signature in '{modelPath}': expected 3 inputs and 2 outputs, " +
                $"got {session.InputMetadata.Count} and {session.OutputMetadata.Count}.");
        }

        RequireFloatTensor(session.InputMetadata, MicrophoneInputName, [1, 640], "input", modelPath);
        RequireFloatTensor(session.InputMetadata, RenderInputName, [1, 640], "input", modelPath);
        // The recurrent-state width is model-internal scale. The ONNX graph is the
        // single source of truth so retrained models of a different size load without
        // code changes; C# only requires the streaming form [layers, 1, hidden].
        int[] stateShape = RequireStateShape(session.InputMetadata, StateInputName, "input", modelPath);
        RequireFloatTensor(session.OutputMetadata, ProbabilityOutputName, [1, 1, 1], "output", modelPath);
        RequireFloatTensor(
            session.OutputMetadata,
            StateOutputName,
            stateShape,
            "output",
            modelPath);

        IReadOnlyDictionary<string, string> metadata = session.ModelMetadata.CustomMetadataMap;
        RequireMetadata(metadata, "model_type", "render-aware-nearend-dtd", modelPath);
        RequireMetadata(metadata, "sample_rate", "16000", modelPath);
        RequireMetadata(metadata, "frame_length", "640", modelPath);
        RequireMetadata(metadata, "hop_length", "320", modelPath);
        RequireMetadata(
            metadata,
            "state_shape",
            string.Join(",", stateShape),
            modelPath);
        return stateShape;
    }

    private static int[] RequireStateShape(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string name,
        string kind,
        string modelPath)
    {
        if (!metadata.TryGetValue(name, out NodeMetadata? node))
        {
            throw new InvalidOperationException(
                $"Render-aware DTD {kind} '{name}' is missing from '{modelPath}'.");
        }

        int[] dimensions = node.Dimensions;
        if (node.ElementType != typeof(float) ||
            dimensions.Length != 3 ||
            dimensions[0] < 1 ||
            dimensions[1] != 1 ||
            dimensions[2] < 1)
        {
            throw new InvalidOperationException(
                $"Render-aware DTD {kind} '{name}' must be a float32 streaming state " +
                $"[layers,1,hidden], got {node.ElementType.Name} " +
                $"[{string.Join(",", dimensions)}] in '{modelPath}'.");
        }

        return dimensions;
    }

    private static void RequireFloatTensor(
        IReadOnlyDictionary<string, NodeMetadata> metadata,
        string name,
        int[] expectedShape,
        string kind,
        string modelPath)
    {
        if (!metadata.TryGetValue(name, out NodeMetadata? node))
        {
            throw new InvalidOperationException(
                $"Render-aware DTD {kind} '{name}' is missing from '{modelPath}'.");
        }

        if (node.ElementType != typeof(float) || !node.Dimensions.SequenceEqual(expectedShape))
        {
            throw new InvalidOperationException(
                $"Render-aware DTD {kind} '{name}' must be float32 " +
                $"[{string.Join(",", expectedShape)}], got {node.ElementType.Name} " +
                $"[{string.Join(",", node.Dimensions)}] in '{modelPath}'.");
        }
    }

    private static void RequireMetadata(
        IReadOnlyDictionary<string, string> metadata,
        string key,
        string expected,
        string modelPath)
    {
        if (!metadata.TryGetValue(key, out string? actual) ||
            !string.Equals(actual, expected, StringComparison.Ordinal))
        {
            throw new InvalidOperationException(
                $"Render-aware DTD metadata '{key}' must be '{expected}', got " +
                $"'{actual ?? "<missing>"}' in '{modelPath}'.");
        }
    }
}
