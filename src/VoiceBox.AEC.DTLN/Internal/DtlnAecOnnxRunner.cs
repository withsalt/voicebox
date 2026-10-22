using Microsoft.ML.OnnxRuntime;
using VoiceBox.AEC.DTLN.Models;
using VoiceBox.Core.Onnx;

namespace VoiceBox.AEC.DTLN.Internal
{
    internal sealed class DtlnAecOnnxRunner : IDisposable
    {
        private readonly InferenceSession _part1;
        private readonly InferenceSession _part2;
        private readonly SessionOptions _part1Options;
        private readonly SessionOptions _part2Options;
        private readonly RunOptions _runOptions;

        private readonly float[] _micMag;
        private readonly float[] _lpbMag;
        private readonly float[] _mask;
        private readonly float[] _estimatedBlock;
        private readonly float[] _lpbBlock;
        private readonly float[] _enhancedBlock;

        private readonly float[] _state1;
        private readonly float[] _state2;
        private readonly float[] _state1Out;
        private readonly float[] _state2Out;

        private readonly OrtValue _micMagValue;
        private readonly OrtValue _lpbMagValue;
        private readonly OrtValue _state1Value;
        private readonly OrtValue _maskValue;
        private readonly OrtValue _state1OutValue;
        private readonly OrtValue _estimatedBlockValue;
        private readonly OrtValue _lpbBlockValue;
        private readonly OrtValue _state2Value;
        private readonly OrtValue _enhancedBlockValue;
        private readonly OrtValue _state2OutValue;

        private readonly string[] _part1InputNames;
        private readonly string[] _part1OutputNames;
        private readonly string[] _part2InputNames;
        private readonly string[] _part2OutputNames;
        private readonly OrtValue[] _part1Inputs;
        private readonly OrtValue[] _part1Outputs;
        private readonly OrtValue[] _part2Inputs;
        private readonly OrtValue[] _part2Outputs;
        private bool _disposed;

        public int Units { get; }
        internal float[] MicMagnitudeBuffer => _micMag;
        internal float[] LpbMagnitudeBuffer => _lpbMag;
        internal float[] MaskBuffer => _mask;
        internal float[] EstimatedBlockBuffer => _estimatedBlock;
        internal float[] LpbBlockBuffer => _lpbBlock;
        internal float[] EnhancedBlockBuffer => _enhancedBlock;

        public DtlnAecOnnxRunner(string modelPart1Path, string modelPart2Path, DtlnAecOptions options)
        {
            if (!File.Exists(modelPart1Path))
                throw new FileNotFoundException($"DTLN AEC part-1 ONNX model not found: {modelPart1Path}", modelPart1Path);
            if (!File.Exists(modelPart2Path))
                throw new FileNotFoundException($"DTLN AEC part-2 ONNX model not found: {modelPart2Path}", modelPart2Path);

            SessionOptions? part1Options = null;
            SessionOptions? part2Options = null;
            InferenceSession? part1 = null;
            InferenceSession? part2 = null;
            RunOptions? runOptions = null;
            OrtValue? micMagValue = null;
            OrtValue? lpbMagValue = null;
            OrtValue? state1Value = null;
            OrtValue? maskValue = null;
            OrtValue? state1OutValue = null;
            OrtValue? estimatedBlockValue = null;
            OrtValue? lpbBlockValue = null;
            OrtValue? state2Value = null;
            OrtValue? enhancedBlockValue = null;
            OrtValue? state2OutValue = null;

            try
            {
                part1Options = OnnxSessionFactory.Create(options, SessionPurpose.StaticShapes);
                part2Options = OnnxSessionFactory.Create(options, SessionPurpose.StaticShapes);
                part1 = new InferenceSession(modelPart1Path, part1Options);
                part2 = new InferenceSession(modelPart2Path, part2Options);
                runOptions = new RunOptions();

                string p1Mic = RequireTensor(part1.InputMetadata, "mic_mag", DtlnAecOptions.FrequencyBins, fallbackOrdinal: 0);
                string p1Lpb = RequireTensor(part1.InputMetadata, "lpb_mag", DtlnAecOptions.FrequencyBins, fallbackOrdinal: 1);
                string p1StateIn = RequireState(part1.InputMetadata, "state1_in", out int units1);
                string p1Mask = RequireTensor(part1.OutputMetadata, "mask", DtlnAecOptions.FrequencyBins, fallbackOrdinal: 0);
                string p1StateOut = RequireState(part1.OutputMetadata, "state1_out", out int units1Out);

                string p2Estimated = RequireTensor(part2.InputMetadata, "estimated_block", DtlnAecOptions.BlockLength, fallbackOrdinal: 0);
                string p2Lpb = RequireTensor(part2.InputMetadata, "lpb_block", DtlnAecOptions.BlockLength, fallbackOrdinal: 1);
                string p2StateIn = RequireState(part2.InputMetadata, "state2_in", out int units2);
                string p2Enhanced = RequireTensor(part2.OutputMetadata, "enhanced_block", DtlnAecOptions.BlockLength, fallbackOrdinal: 0);
                string p2StateOut = RequireState(part2.OutputMetadata, "state2_out", out int units2Out);

                if (units1 != units1Out || units1 != units2 || units1 != units2Out)
                    throw new InvalidOperationException($"DTLN state unit mismatch: p1={units1}/{units1Out}, p2={units2}/{units2Out}.");

                int units = units1;
                var micMag = new float[DtlnAecOptions.FrequencyBins];
                var lpbMag = new float[DtlnAecOptions.FrequencyBins];
                var mask = new float[DtlnAecOptions.FrequencyBins];
                var estimatedBlock = new float[DtlnAecOptions.BlockLength];
                var lpbBlock = new float[DtlnAecOptions.BlockLength];
                var enhancedBlock = new float[DtlnAecOptions.BlockLength];

                int stateSize = 1 * 2 * units * 2;
                var state1 = new float[stateSize];
                var state2 = new float[stateSize];
                var state1Out = new float[stateSize];
                var state2Out = new float[stateSize];
                micMagValue = OrtValue.CreateTensorValueFromMemory(micMag, new long[] { 1, 1, DtlnAecOptions.FrequencyBins });
                lpbMagValue = OrtValue.CreateTensorValueFromMemory(lpbMag, new long[] { 1, 1, DtlnAecOptions.FrequencyBins });
                state1Value = OrtValue.CreateTensorValueFromMemory(state1, new long[] { 1, 2, units, 2 });
                maskValue = OrtValue.CreateTensorValueFromMemory(mask, new long[] { 1, 1, DtlnAecOptions.FrequencyBins });
                state1OutValue = OrtValue.CreateTensorValueFromMemory(state1Out, new long[] { 1, 2, units, 2 });
                estimatedBlockValue = OrtValue.CreateTensorValueFromMemory(estimatedBlock, new long[] { 1, 1, DtlnAecOptions.BlockLength });
                lpbBlockValue = OrtValue.CreateTensorValueFromMemory(lpbBlock, new long[] { 1, 1, DtlnAecOptions.BlockLength });
                state2Value = OrtValue.CreateTensorValueFromMemory(state2, new long[] { 1, 2, units, 2 });
                enhancedBlockValue = OrtValue.CreateTensorValueFromMemory(enhancedBlock, new long[] { 1, 1, DtlnAecOptions.BlockLength });
                state2OutValue = OrtValue.CreateTensorValueFromMemory(state2Out, new long[] { 1, 2, units, 2 });

                string[] part1InputNames = new[] { p1Mic, p1StateIn, p1Lpb };
                string[] part1OutputNames = new[] { p1Mask, p1StateOut };
                string[] part2InputNames = new[] { p2Estimated, p2StateIn, p2Lpb };
                string[] part2OutputNames = new[] { p2Enhanced, p2StateOut };
                OrtValue[] part1Inputs = new[] { micMagValue, state1Value, lpbMagValue };
                OrtValue[] part1Outputs = new[] { maskValue, state1OutValue };
                OrtValue[] part2Inputs = new[] { estimatedBlockValue, state2Value, lpbBlockValue };
                OrtValue[] part2Outputs = new[] { enhancedBlockValue, state2OutValue };

                _part1Options = part1Options;
                _part2Options = part2Options;
                _part1 = part1;
                _part2 = part2;
                _runOptions = runOptions;
                _micMag = micMag;
                _lpbMag = lpbMag;
                _mask = mask;
                _estimatedBlock = estimatedBlock;
                _lpbBlock = lpbBlock;
                _enhancedBlock = enhancedBlock;
                _state1 = state1;
                _state2 = state2;
                _state1Out = state1Out;
                _state2Out = state2Out;
                _micMagValue = micMagValue;
                _lpbMagValue = lpbMagValue;
                _state1Value = state1Value;
                _maskValue = maskValue;
                _state1OutValue = state1OutValue;
                _estimatedBlockValue = estimatedBlockValue;
                _lpbBlockValue = lpbBlockValue;
                _state2Value = state2Value;
                _enhancedBlockValue = enhancedBlockValue;
                _state2OutValue = state2OutValue;
                _part1InputNames = part1InputNames;
                _part1OutputNames = part1OutputNames;
                _part2InputNames = part2InputNames;
                _part2OutputNames = part2OutputNames;
                _part1Inputs = part1Inputs;
                _part1Outputs = part1Outputs;
                _part2Inputs = part2Inputs;
                _part2Outputs = part2Outputs;
                Units = units;
            }
            catch
            {
                state2OutValue?.Dispose();
                enhancedBlockValue?.Dispose();
                state2Value?.Dispose();
                lpbBlockValue?.Dispose();
                estimatedBlockValue?.Dispose();
                state1OutValue?.Dispose();
                maskValue?.Dispose();
                state1Value?.Dispose();
                lpbMagValue?.Dispose();
                micMagValue?.Dispose();
                runOptions?.Dispose();
                part2?.Dispose();
                part1?.Dispose();
                part2Options?.Dispose();
                part1Options?.Dispose();
                throw;
            }
        }

        public void RunPart1(ReadOnlySpan<float> micMag, ReadOnlySpan<float> lpbMag, Span<float> mask)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (micMag.Length != _micMag.Length || lpbMag.Length != _lpbMag.Length || mask.Length != _mask.Length)
                throw new ArgumentException("DTLN part-1 tensor length mismatch.");

            micMag.CopyTo(_micMag);
            lpbMag.CopyTo(_lpbMag);

            RunPart1InPlace();

            _mask.AsSpan().CopyTo(mask);
        }

        public void RunPart1InPlace()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _part1.Run(_runOptions, _part1InputNames, _part1Inputs, _part1OutputNames, _part1Outputs);
            SwapStateBuffers(_part1Inputs, inputIndex: 1, _part1Outputs, outputIndex: 1);
        }

        public void RunPart2(ReadOnlySpan<float> estimatedBlock, ReadOnlySpan<float> lpbBlock, Span<float> enhancedBlock)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (estimatedBlock.Length != _estimatedBlock.Length || lpbBlock.Length != _lpbBlock.Length || enhancedBlock.Length != _enhancedBlock.Length)
                throw new ArgumentException("DTLN part-2 tensor length mismatch.");

            estimatedBlock.CopyTo(_estimatedBlock);
            lpbBlock.CopyTo(_lpbBlock);

            RunPart2InPlace();

            _enhancedBlock.AsSpan().CopyTo(enhancedBlock);
        }

        public void RunPart2InPlace()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            _part2.Run(_runOptions, _part2InputNames, _part2Inputs, _part2OutputNames, _part2Outputs);
            SwapStateBuffers(_part2Inputs, inputIndex: 1, _part2Outputs, outputIndex: 1);
        }

        public void Reset()
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            Array.Clear(_state1);
            Array.Clear(_state1Out);
            Array.Clear(_state2);
            Array.Clear(_state2Out);
            _part1Inputs[1] = _state1Value;
            _part1Outputs[1] = _state1OutValue;
            _part2Inputs[1] = _state2Value;
            _part2Outputs[1] = _state2OutValue;
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _micMagValue.Dispose();
            _lpbMagValue.Dispose();
            _state1Value.Dispose();
            _maskValue.Dispose();
            _state1OutValue.Dispose();
            _estimatedBlockValue.Dispose();
            _lpbBlockValue.Dispose();
            _state2Value.Dispose();
            _enhancedBlockValue.Dispose();
            _state2OutValue.Dispose();
            _runOptions.Dispose();
            _part1.Dispose();
            _part2.Dispose();
            _part1Options.Dispose();
            _part2Options.Dispose();
        }

        private static string RequireTensor(IReadOnlyDictionary<string, NodeMetadata> metadata, string preferredName, int lastDim, int fallbackOrdinal)
        {
            if (metadata.TryGetValue(preferredName, out var named) && IsTensorShape(named.Dimensions, lastDim))
                return preferredName;

            int ordinal = 0;
            foreach (var kv in metadata)
            {
                if (!IsTensorShape(kv.Value.Dimensions, lastDim))
                    continue;
                if (ordinal++ == fallbackOrdinal)
                    return kv.Key;
            }

            throw new InvalidOperationException($"Could not identify DTLN tensor [1,1,{lastDim}] ordinal {fallbackOrdinal} among: {string.Join(", ", metadata.Keys)}.");
        }

        private static string RequireState(IReadOnlyDictionary<string, NodeMetadata> metadata, string preferredName, out int units)
        {
            if (metadata.TryGetValue(preferredName, out var named) && TryGetUnits(named.Dimensions, out units))
                return preferredName;

            foreach (var kv in metadata)
                if (TryGetUnits(kv.Value.Dimensions, out units))
                    return kv.Key;

            throw new InvalidOperationException($"Could not identify DTLN state [1,2,U,2] among: {string.Join(", ", metadata.Keys)}.");
        }

        private static bool IsTensorShape(IReadOnlyList<int> dims, int lastDim)
            => dims.Count == 3 && dims[0] == 1 && dims[1] == 1 && dims[2] == lastDim;

        private static bool TryGetUnits(IReadOnlyList<int> dims, out int units)
        {
            units = 0;
            if (dims.Count != 4 || dims[0] != 1 || dims[1] != 2 || dims[3] != 2 || dims[2] <= 0)
                return false;
            units = dims[2];
            return true;
        }

        private static void SwapStateBuffers(OrtValue[] inputs, int inputIndex, OrtValue[] outputs, int outputIndex)
            => (inputs[inputIndex], outputs[outputIndex]) = (outputs[outputIndex], inputs[inputIndex]);
    }
}
