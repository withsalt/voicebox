using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Diagnostics;
using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.ASR.FireRedASR2.Internal.Utils;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// AED (attention encoder-decoder) inference engine for FireRedASR. The three Run*
    /// methods follow the same template — build named tensors keyed off the session input
    /// metadata, execute, then materialise outputs — driven by per-call binders so the
    /// shared structure lives in one place.
    /// </summary>
    internal sealed class AedInferenceEngine : IAsrInferenceEngine
    {
        private readonly InferenceSession _encoder;
        private readonly InferenceSession? _decoder;
        private readonly InferenceSession? _ctc;
        private readonly AsrModelMetadata _metadata;

        public AedInferenceEngine(AsrInferenceSessions sessions, AsrModelMetadata metadata)
        {
            ArgumentNullException.ThrowIfNull(sessions);
            ArgumentNullException.ThrowIfNull(metadata);

            _encoder = sessions.Encoder;
            _decoder = sessions.Decoder;
            _ctc = sessions.Ctc;
            _metadata = metadata;
        }

        public AsrModelMetadata Metadata => _metadata;
        public bool HasDecoder => _decoder is not null;
        public bool HasCtc => _ctc is not null;

        public EncoderOutputEntity RunEncoder(IReadOnlyList<AsrInputEntity> modelInputs)
        {
            int batchSize = modelInputs.Count;
            int featureDim = _metadata.FeatureDim;
            long[] inputLengths = new long[batchSize];
            for (int i = 0; i < batchSize; i++)
                inputLengths[i] = modelInputs[i].SpeechLength / featureDim;

            float[] padded = PadHelper.PadSequence(modelInputs);

            return RunSession(_encoder, container =>
            {
                BindByName(_encoder, container, "input", padded,
                    new[] { batchSize, padded.Length / featureDim / batchSize, featureDim });
                BindByName(_encoder, container, "input_lengths", inputLengths, new[] { batchSize });
            }, results =>
            {
                var arr = results.ToArray();
                return new EncoderOutputEntity
                {
                    Output = CopyToArray(arr[0].AsTensor<float>()),
                    OutputLengths = CopyToArray(arr[1].AsTensor<long>()),
                    Mask = CopyToArray(arr[2].AsTensor<bool>()),
                };
            }) ?? new EncoderOutputEntity();
        }

        public DecoderOutputEntity RunDecoderStep(
            long[] tokens,
            int tokensLen,
            float[] encoderOutputs,
            bool[] srcMask,
            IReadOnlyList<float[]> cacheList)
        {
            Debug.Assert(_decoder is not null, "decoder session missing");
            int batchSize = 1; // FireRed always uses batch=1 in offline mode.
            int encoderHidden = _metadata.EncoderHiddenDim;

            var result = RunSession(_decoder!, container =>
            {
                // Borrow `tokens` memory directly — no per-step ToArray()/copy.
                if (_decoder!.InputMetadata.ContainsKey("ys"))
                {
                    var ysTensor = new DenseTensor<long>(tokens.AsMemory(0, tokensLen), new int[] { batchSize, tokensLen });
                    container.Add(NamedOnnxValue.CreateFromTensor("ys", ysTensor));
                }

                BindByName(_decoder!, container, "encoder_outputs", encoderOutputs,
                    new[] { batchSize, encoderOutputs.Length / encoderHidden / batchSize, encoderHidden });

                BindByName(_decoder!, container, "src_mask", srcMask,
                    new[] { batchSize, 1, srcMask.Length / batchSize });

                for (int i = 0; i < _metadata.DecoderCacheLayers; i++)
                {
                    string name = "cache_" + i;
                    if (!_decoder!.InputMetadata.ContainsKey(name)) continue;
                    BindByName(_decoder!, container, name, cacheList[i],
                        new[] { batchSize, cacheList[i].Length / encoderHidden / batchSize, encoderHidden });
                }
            }, results =>
            {
                var arr = results.ToArray();
                var logits = arr[0].AsTensor<float>();

                int rows = logits.Dimensions[0];
                int cols = logits.Dimensions[1];

                // softmax is monotonic and dividing by a positive temperature preserves order,
                // so argmax(logits) == argmax(softmax(logits / T)). Read the raw logits through
                // the dense-tensor buffer to avoid the per-element indexer.
                ReadOnlySpan<float> logitsSpan = (logits is DenseTensor<float> dense)
                    ? dense.Buffer.Span
                    : logits.ToArray().AsSpan();
                var tokensOut = new List<List<long>>(rows);
                for (int i = 0; i < rows; i++)
                {
                    int best = ArgMax(logitsSpan.Slice(i * cols, cols));
                    tokensOut.Add(new List<long> { best });
                }

                var caches = new List<float[]>(arr.Length - 1);
                for (int k = 1; k < arr.Length; k++)
                    caches.Add(CopyToArray(arr[k].AsTensor<float>()));

                return new DecoderOutputEntity { TokensList = tokensOut, CacheList = caches };
            });

            return result ?? new DecoderOutputEntity();
        }

        public CtcOutputEntity RunCtc(float[] encoderOutputs, int batchSize = 1)
        {
            Debug.Assert(_ctc is not null, "ctc session missing");
            int encoderHidden = _metadata.EncoderHiddenDim;

            try
            {
                return RunSession(_ctc!, container =>
                {
                    BindByName(_ctc!, container, "encoder_outputs", encoderOutputs,
                        new[] { batchSize, encoderOutputs.Length / encoderHidden / batchSize, encoderHidden });
                }, results =>
                {
                    var arr = results.ToArray();
                    var logits = arr[0].AsTensor<float>();

                    int b = logits.Dimensions[0];
                    int t = logits.Dimensions[1];
                    int n = logits.Dimensions[2];

                    return new CtcOutputEntity
                    {
                        Logits = CopyToArray(logits),
                        Batch = b,
                        FrameCount = t,
                        VocabSize = n,
                    };
                }) ?? new CtcOutputEntity();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch
            {
                // Original code swallowed CTC failure to fall back to attention-only output.
                return new CtcOutputEntity();
            }
        }

        public void Dispose()
        {
            // Sessions are owned by AsrInferenceSessions and disposed there.
        }

        /// <summary>
        /// Template: build the input container, run the session, map disposable outputs to a
        /// caller-defined result. Disposes the ORT result automatically.
        /// </summary>
        private static TResult? RunSession<TResult>(
            InferenceSession session,
            Action<List<NamedOnnxValue>> bind,
            Func<IDisposableReadOnlyCollection<DisposableNamedOnnxValue>, TResult> map)
            where TResult : class
        {
            var container = new List<NamedOnnxValue>();
            bind(container);

            using IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results = session.Run(container);
            return results is null ? null : map(results);
        }

        private static void BindByName<T>(InferenceSession session, List<NamedOnnxValue> container, string name, T[] data, int[] dims)
        {
            if (!session.InputMetadata.ContainsKey(name)) return;
            container.Add(NamedOnnxValue.CreateFromTensor(name, new DenseTensor<T>(data, dims, false)));
        }

        /// <summary>
        /// Materialise a tensor into a freshly allocated array via the dense-tensor backing
        /// span when possible (one bulk copy, no per-element indexer dispatch).
        /// </summary>
        private static T[] CopyToArray<T>(Tensor<T> tensor)
        {
            if (tensor is DenseTensor<T> dense)
            {
                var dst = new T[dense.Length];
                dense.Buffer.Span.CopyTo(dst);
                return dst;
            }
            return tensor.ToArray();
        }

        private static int ArgMax(ReadOnlySpan<float> row)
        {
            // Tie-break: first index wins, matching PyTorch / NumPy argmax. The previous
            // form `!(row[best] > row[k])` flipped ties to the higher index, which can
            // diverge from the PyTorch reference run on quantised models or wherever
            // softmax saturation produces exact equalities — especially at low token ids
            // where blank / SOS / EOS live.
            int best = 0;
            for (int k = 1; k < row.Length; k++)
                if (row[k] > row[best]) best = k;
            return best;
        }
    }
}
