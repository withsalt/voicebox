using System;
using System.Buffers;
using System.Numerics.Tensors;
using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.ASR.Confucius4.Internal
{
    /// <summary>
    /// Wraps the audio encoder branch of Confucius4 ASR. Accepts a raw waveform, computes
    /// a Whisper-style log-mel spectrogram, splits it into <c>n_window * 2</c>-wide
    /// chunks and runs the encoder. The resulting hidden states are returned as
    /// <c>(rows, audio_output_dim)</c> tensor data ready to be flattened into the
    /// decoder prefill input.
    /// </summary>
    internal sealed class Confucius4AsrAudioEncoder : IDisposable
    {
        private readonly Confucius4AsrMetadata _metadata;
        private readonly Confucius4AsrSessions _sessions;
        private readonly WhisperFeatureExtractor _features;
        private readonly bool _useFloat16;
        private readonly RunOptions _runOptions = new();

        // Persistent feature-input scratch buffers. Grown on demand; the encoder is
        // gated by the recognition semaphore so there is at most one Encode call at
        // a time per runtime instance. Reusing these eliminates a per-call
        // ~chunkCount·nMels·windowFrames allocation (≈400 KB at typical AIShell sizes).
        private Float16[]? _featureFp16Scratch;
        private Float16[]? _hiddenFp16Scratch; // split-encoder intermediate
        private float[]? _hiddenFp32Scratch; // split-encoder intermediate

        public int FeatureSize => _features.FeatureSize;
        public int AudioOutputDim => _metadata.AudioOutputDim;

        public Confucius4AsrAudioEncoder(Confucius4AsrMetadata metadata, Confucius4AsrSessions sessions, WhisperFeatureExtractor features)
        {
            _metadata = metadata;
            _sessions = sessions;
            _features = features;
            _useFloat16 = Confucius4AsrDtypeParser.Parse(metadata.Dtype) == Confucius4AsrIoDtype.Float16;
        }

        /// <summary>
        /// Encode the waveform into a flat <c>(rows, dim)</c> tensor in the encoder's
        /// native dtype (Float16 or float). Downstream (decoder prefill) wraps the
        /// returned buffer directly without any dtype conversion — when both encoder
        /// and decoder are FP16, this skips a round-trip Float16→float→Float16 that
        /// is exact but wastes work proportional to <c>rows*dim</c>.
        /// </summary>
        /// <param name="melState">
        /// Optional per-session incremental mel cache. When supplied, leading frames
        /// whose STFT windows are unaffected by new audio are reused from the cache;
        /// pass <c>null</c> for a one-shot/stateless encode.
        /// </param>
        public AudioFeatures Encode(ReadOnlySpan<float> waveform, IncrementalMelState? melState = null)
        {
            var sw = System.Diagnostics.Stopwatch.StartNew();
            LogMelSpectrogram mel = melState is null
                ? _features.Compute(waveform)
                : _features.ComputeIncremental(waveform, melState);
            double melSeconds = sw.Elapsed.TotalSeconds;
            AudioFeatures features = EncodeFromMel(mel, waveform.Length);
            double encodeSeconds = sw.Elapsed.TotalSeconds;
            return features.WithTiming(encodeSeconds, melSeconds);
        }

        private AudioFeatures EncodeFromMel(LogMelSpectrogram mel, int waveformLength)
        {
            int nWindow = _metadata.NWindow;
            int windowFrames = nWindow * 2;
            int featureLen = mel.NumFrames;
            if (featureLen == 0) return AudioFeatures.Empty(AudioOutputDim);
            if (_metadata.DynamicChunkFrames) windowFrames = Math.Min(windowFrames, featureLen);
            int chunkCount = (featureLen + windowFrames - 1) / windowFrames;
            if (chunkCount == 0)
                return AudioFeatures.Empty(AudioOutputDim);

            // Build padded_feature [chunk_count, 1, feature_size, windowFrames] in NCHW
            // order. Pool the scratch buffer; ArrayPool gives us a >= size array which
            // we slice. The buffer is returned in the finally block of the caller via
            // CreateFeatureValue's lifecycle (see below for Float16 path).
            int elementsPerChunk = mel.NumMels * windowFrames;
            int paddedTotal = chunkCount * elementsPerChunk;
            float[] padded = ArrayPool<float>.Shared.Rent(paddedTotal);
            long[] chunkLengths = ArrayPool<long>.Shared.Rent(chunkCount);
            try
            {
                // Pooled buffers may contain stale data beyond the prior tenant's
                // logical size; zero the new region so chunk-tail padding stays at 0.
                Array.Clear(padded, 0, paddedTotal);

                for (int c = 0; c < chunkCount; c++)
                {
                    int srcStart = c * windowFrames;
                    int dstChunkBase = c * elementsPerChunk;
                    int copy = Math.Min(windowFrames, featureLen - srcStart);
                    if (copy <= 0) continue;
                    for (int m = 0; m < mel.NumMels; m++)
                    {
                        int srcRow = m * featureLen + srcStart;
                        int dstRow = dstChunkBase + m * windowFrames;
                        Array.Copy(mel.Data, srcRow, padded, dstRow, copy);
                    }
                }

                for (int c = 0; c < chunkCount; c++) chunkLengths[c] = windowFrames;
                chunkLengths[chunkCount - 1] = featureLen - windowFrames * (chunkCount - 1);

                AudioFeatures features = _sessions.HasFusedEncoder
                    ? RunFusedEncoder(padded, paddedTotal, chunkLengths, chunkCount, mel.NumMels, windowFrames)
                    : RunSplitEncoder(padded, paddedTotal, chunkLengths, chunkCount, mel.NumMels, windowFrames);

                double audioSeconds = waveformLength / (double)_features.SamplingRate;
                return features.WithAudioSeconds(audioSeconds);
            }
            finally
            {
                ArrayPool<float>.Shared.Return(padded);
                ArrayPool<long>.Shared.Return(chunkLengths);
            }
        }

        private AudioFeatures RunFusedEncoder(
            float[] padded, int paddedLen, long[] chunkLengths,
            int chunkCount, int nMels, int windowFrames)
        {
            ReadOnlySpan<long> featureShape = stackalloc long[] { chunkCount, 1, nMels, windowFrames };
            ReadOnlySpan<long> lengthShape = stackalloc long[] { chunkCount };

            using OrtValue featureValue = CreateFeatureValue(padded, paddedLen, featureShape);
            using OrtValue lengthValue = OrtValue.CreateTensorValueFromMemory<long>(
                OrtMemoryInfo.DefaultInstance, chunkLengths.AsMemory(0, chunkCount), lengthShape.ToArray());

            var inputs = new Dictionary<string, OrtValue>
            {
                ["padded_feature"] = featureValue,
                ["chunk_lengths"] = lengthValue,
            };

            using var outputs = _sessions.Encoder!.Run(_runOptions, inputs, new[] { "audio_features" });
            return BuildAudioFeatures(outputs[0]);
        }

        private AudioFeatures RunSplitEncoder(
            float[] padded, int paddedLen, long[] chunkLengths,
            int chunkCount, int nMels, int windowFrames)
        {
            ReadOnlySpan<long> featureShape = stackalloc long[] { chunkCount, 1, nMels, windowFrames };
            using OrtValue featureValue = CreateFeatureValue(padded, paddedLen, featureShape);

            using var convOutputs = _sessions.ConvEmbed!.Run(
                _runOptions,
                new Dictionary<string, OrtValue> { ["padded_feature"] = featureValue },
                new[] { "padded_embed" });

            using OrtValue paddedEmbedValue = convOutputs[0];
            var embedShape = paddedEmbedValue.GetTensorTypeAndShape().Shape;
            int embedFrames = (int)embedShape[1];
            int embedDim = (int)embedShape[2];

            int[] lengthsAfter = new int[chunkCount];
            int totalRows = 0;
            for (int c = 0; c < chunkCount; c++)
            {
                lengthsAfter[c] = FeatureOutputLength((int)chunkLengths[c]);
                totalRows += lengthsAfter[c];
            }

            if (_useFloat16)
            {
                // Keep the intermediate in Float16 — saves an N×D unpack+repack pair.
                Float16[] hiddenStates = EnsureFp16Scratch(ref _hiddenFp16Scratch, totalRows * embedDim);
                int writeOffset = 0;
                ReadOnlySpan<Float16> source = paddedEmbedValue.GetTensorDataAsSpan<Float16>();
                for (int c = 0; c < chunkCount; c++)
                {
                    int copy = lengthsAfter[c];
                    int chunkOffset = c * embedFrames * embedDim;
                    int elems = copy * embedDim;
                    Span<Float16> dst = hiddenStates.AsSpan(writeOffset * embedDim, elems);
                    source.Slice(chunkOffset, elems).CopyTo(dst);
                    writeOffset += copy;
                }

                ReadOnlySpan<long> hiddenShape = stackalloc long[] { totalRows, embedDim };
                long[] shapeArr = hiddenShape.ToArray();
                using OrtValue hiddenValue = OrtValue.CreateTensorValueFromMemory<Float16>(
                    OrtMemoryInfo.DefaultInstance,
                    hiddenStates.AsMemory(0, totalRows * embedDim),
                    shapeArr);
                using var stackOutputs = _sessions.AudioStack!.Run(
                    _runOptions,
                    new Dictionary<string, OrtValue> { ["hidden_states"] = hiddenValue },
                    new[] { "audio_features" });

                return BuildAudioFeatures(stackOutputs[0]);
            }
            else
            {
                float[] hiddenStates = EnsureFp32Scratch(totalRows * embedDim);
                int writeOffset = 0;
                ReadOnlySpan<float> source = paddedEmbedValue.GetTensorDataAsSpan<float>();
                for (int c = 0; c < chunkCount; c++)
                {
                    int copy = lengthsAfter[c];
                    int chunkOffset = c * embedFrames * embedDim;
                    source.Slice(chunkOffset, copy * embedDim)
                          .CopyTo(hiddenStates.AsSpan(writeOffset * embedDim));
                    writeOffset += copy;
                }

                ReadOnlySpan<long> hiddenShape = stackalloc long[] { totalRows, embedDim };
                using OrtValue hiddenValue = OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance,
                    hiddenStates.AsMemory(0, totalRows * embedDim),
                    hiddenShape.ToArray());
                using var stackOutputs = _sessions.AudioStack!.Run(
                    _runOptions,
                    new Dictionary<string, OrtValue> { ["hidden_states"] = hiddenValue },
                    new[] { "audio_features" });

                return BuildAudioFeatures(stackOutputs[0]);
            }
        }

        private OrtValue CreateFeatureValue(float[] padded, int paddedLen, ReadOnlySpan<long> shape)
        {
            long[] shapeArr = shape.ToArray();
            if (!_useFloat16)
            {
                return OrtValue.CreateTensorValueFromMemory<float>(
                    OrtMemoryInfo.DefaultInstance, padded.AsMemory(0, paddedLen), shapeArr);
            }

            Float16[] target = EnsureFp16Scratch(ref _featureFp16Scratch, paddedLen);
            PackFromFloat(padded.AsSpan(0, paddedLen), target.AsSpan(0, paddedLen));
            return OrtValue.CreateTensorValueFromMemory<Float16>(
                OrtMemoryInfo.DefaultInstance, target.AsMemory(0, paddedLen), shapeArr);
        }

        private void PackFromFloat(ReadOnlySpan<float> src, Span<Float16> dst)
        {
            if (src.Length > dst.Length)
                throw new ArgumentException("Destination span smaller than source.", nameof(dst));
            // Float16 and Half are both 2-byte sequential structs holding the same
            // binary16 bits — reinterpret to use TensorPrimitives' SIMD path.
            Span<Half> halfDst = MemoryMarshal.Cast<Float16, Half>(dst.Slice(0, src.Length));
            TensorPrimitives.ConvertToHalf(src, halfDst);
        }

        private static Float16[] EnsureFp16Scratch(ref Float16[]? buf, int minLen)
        {
            if (buf is null || buf.Length < minLen)
            {
                // Grow with headroom (×2 typical) to amortise across calls with
                // similar-but-not-identical sizes.
                int newCap = buf is null ? minLen : Math.Max(minLen, buf.Length * 2);
                buf = new Float16[newCap];
            }
            return buf;
        }

        private float[] EnsureFp32Scratch(int minLen)
        {
            if (_hiddenFp32Scratch is null || _hiddenFp32Scratch.Length < minLen)
            {
                int newCap = _hiddenFp32Scratch is null ? minLen : Math.Max(minLen, _hiddenFp32Scratch.Length * 2);
                _hiddenFp32Scratch = new float[newCap];
            }
            return _hiddenFp32Scratch;
        }

        private AudioFeatures BuildAudioFeatures(OrtValue value)
        {
            var shape = value.GetTensorTypeAndShape().Shape;
            int rows = (int)shape[0];
            int dim = (int)shape[1];
            int total = rows * dim;
            if (_useFloat16)
            {
                Float16[] data = new Float16[total];
                ReadOnlySpan<Float16> src = value.GetTensorDataAsSpan<Float16>();
                src.Slice(0, total).CopyTo(data);
                return AudioFeatures.CreateFp16(data, rows, dim);
            }
            else
            {
                float[] data = new float[total];
                value.GetTensorDataAsSpan<float>().Slice(0, total).CopyTo(data);
                return AudioFeatures.CreateFp32(data, rows, dim);
            }
        }

        // Mirror of Python feature_output_lengths(lengths) for a single value.
        private static int FeatureOutputLength(int len)
        {
            int leave = len % 100;
            if (leave == 0) return (len / 100) * 13;
            int featLen = (leave - 1) / 2 + 1;
            return ((featLen - 1) / 2 + 1 - 1) / 2 + 1 + (len / 100) * 13;
        }

        public void Dispose() => _runOptions.Dispose();
    }

    /// <summary>
    /// Encoder output ready to be consumed by the decoder. Holds the (rows, dim)
    /// tensor in the encoder's native dtype (either <c>Float16</c> or <c>float</c>),
    /// so the decoder can wrap the buffer as an <see cref="OrtValue"/> without any
    /// further dtype conversion. <see cref="EncodeSeconds"/> covers the full
    /// feature-extract + ONNX encode pass; <see cref="MelSeconds"/> is the log-mel
    /// feature subset.
    /// </summary>
    internal readonly struct AudioFeatures
    {
        private readonly Float16[]? _fp16Data;
        private readonly float[]? _fp32Data;

        public int Rows { get; }
        public int Dim { get; }
        public double AudioSeconds { get; }
        public double EncodeSeconds { get; }
        public double MelSeconds { get; }

        public bool IsFloat16 => _fp16Data is not null;
        public bool IsEmpty => Rows == 0;

        private AudioFeatures(
            Float16[]? fp16, float[]? fp32, int rows, int dim,
            double audioSeconds, double encodeSeconds, double melSeconds)
        {
            _fp16Data = fp16;
            _fp32Data = fp32;
            Rows = rows;
            Dim = dim;
            AudioSeconds = audioSeconds;
            EncodeSeconds = encodeSeconds;
            MelSeconds = melSeconds;
        }

        public static AudioFeatures CreateFp16(Float16[] data, int rows, int dim)
            => new(data, null, rows, dim, 0.0, 0.0, 0.0);

        public static AudioFeatures CreateFp32(float[] data, int rows, int dim)
            => new(null, data, rows, dim, 0.0, 0.0, 0.0);

        public static AudioFeatures Empty(int dim)
            => new(null, Array.Empty<float>(), 0, dim, 0.0, 0.0, 0.0);

        internal AudioFeatures WithTiming(double encodeSeconds, double melSeconds)
            => new(_fp16Data, _fp32Data, Rows, Dim, AudioSeconds, encodeSeconds, melSeconds);

        internal AudioFeatures WithAudioSeconds(double audioSeconds)
            => new(_fp16Data, _fp32Data, Rows, Dim, audioSeconds, EncodeSeconds, MelSeconds);

        /// <summary>
        /// Wrap the (rows, dim) tensor as an <see cref="OrtValue"/> in its native
        /// dtype. Caller owns the lifetime of the returned <see cref="OrtValue"/>
        /// and must dispose it. The underlying buffer is pinned for as long as the
        /// OrtValue is alive.
        /// </summary>
        public OrtValue CreateOrtValue()
        {
            long[] shape = { Rows, Dim };
            if (_fp16Data is not null)
                return OrtValue.CreateTensorValueFromMemory<Float16>(_fp16Data, shape);
            return OrtValue.CreateTensorValueFromMemory<float>(_fp32Data!, shape);
        }
    }
}
