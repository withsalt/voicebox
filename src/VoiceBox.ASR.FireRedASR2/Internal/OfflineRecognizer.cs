using VoiceBox.ASR.FireRedASR2.Internal.Model;
using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Orchestrates end-to-end offline recognition: encoder → decoder loop → optional CTC
    /// timestamps → text assembly. Owns the loaded model and the inference engine; streams
    /// hold per-session state.
    /// </summary>
    internal sealed class OfflineRecognizer : IDisposable
    {
        private readonly AsrModel? _model;
        private readonly IAsrInferenceEngine _engine;
        private readonly ITokenAssembler _assembler;
        private readonly string _mvnFilePath;
        private bool _disposed;

        public OfflineRecognizer(
            string encoderFilePath,
            string? decoderFilePath,
            string mvnFilePath,
            string tokensFilePath,
            string? ctcFilePath = null,
            OnnxExecutionOptions? onnxOptions = null)
        {
            ArgumentException.ThrowIfNullOrEmpty(encoderFilePath);
            ArgumentException.ThrowIfNullOrEmpty(mvnFilePath);
            ArgumentException.ThrowIfNullOrEmpty(tokensFilePath);

            _mvnFilePath = mvnFilePath;
            _model = new AsrModel.Builder()
                .WithEncoder(encoderFilePath)
                .WithDecoder(decoderFilePath)
                .WithCtc(ctcFilePath)
                .WithOnnxOptions(onnxOptions)
                .Build();

            _engine = new AedInferenceEngine(_model.Sessions, _model.Metadata);

            string[] vocabulary = File.ReadAllLines(tokensFilePath);
            _assembler = new FireRedTokenAssembler(vocabulary);
        }

        internal OfflineRecognizer(IAsrInferenceEngine engine, ITokenAssembler assembler, string mvnFilePath)
        {
            _engine = engine;
            _assembler = assembler;
            _mvnFilePath = mvnFilePath;
        }

        public OfflineStream CreateOfflineStream() => new(_mvnFilePath, _engine.Metadata);

        public OfflineRecognizerResultEntity GetResult(OfflineStream stream, CancellationToken ct = default)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(stream);

            ct.ThrowIfCancellationRequested();
            RunInference(stream, ct);
            ct.ThrowIfCancellationRequested();
            return _assembler.Assemble(stream.Tokens, stream.Timestamps);
        }

        private void RunInference(OfflineStream stream, CancellationToken ct)
        {
            float[]? speech = stream.GetDecodeChunk();
            if (speech is null) return;

            var modelInputs = new[]
            {
                new AsrInputEntity { Speech = speech, SpeechLength = speech.Length },
            };

            try
            {
                ct.ThrowIfCancellationRequested();
                EncoderOutputEntity encoderOutput = _engine.RunEncoder(modelInputs);
                ct.ThrowIfCancellationRequested();
                float[] encoderOutputs = encoderOutput.Output!;
                bool[] srcMask = encoderOutput.Mask!;
                int maxLen = encoderOutputs.Length / _engine.Metadata.EncoderHiddenDim; // batch=1

                long eosId = _engine.Metadata.EosId;
                List<long> hyp = stream.Tokens;

                // Initial caches: 16 empty arrays — equivalent to the legacy stack_states(initialStates)
                // when the model has no pre-existing decoder cache.
                IReadOnlyList<float[]> caches = stream.InitialDecoderCaches;

                if (_engine.HasDecoder)
                {
                    // Single growable buffer for the hypothesis tokens. The previous code
                    // called hyp.ToArray() at every step → O(maxLen²) copying. With a backing
                    // long[] + Memory view we hand the same memory to ONNX without any per-step
                    // allocation.
                    int initialLen = hyp.Count;
                    long[] hypBuf = new long[maxLen + initialLen];
                    for (int i = 0; i < initialLen; i++) hypBuf[i] = hyp[i];
                    int hypLen = initialLen;

                    for (int i = 0; i < maxLen; i++)
                    {
                        ct.ThrowIfCancellationRequested();
                        DecoderOutputEntity step = _engine.RunDecoderStep(hypBuf, hypLen, encoderOutputs, srcMask, caches);
                        ct.ThrowIfCancellationRequested();
                        long newToken = step.TokensList[0][0]; // batch=1, one new token per step
                        hyp.Add(newToken);
                        hypBuf[hypLen++] = newToken;
                        caches = step.CacheList;
                        if (newToken == eosId) break;
                    }
                }

                // Strip leading SOS, drop everything after EOS (keep EOS itself for the stop sentinel).
                List<int> attentionTokens = new(Math.Max(0, hyp.Count - 1));
                for (int i = 1; i < hyp.Count; i++) attentionTokens.Add((int)hyp[i]);
                int eosIdx = attentionTokens.IndexOf((int)eosId);
                if (eosIdx >= 0 && eosIdx + 1 < attentionTokens.Count)
                    attentionTokens.RemoveRange(eosIdx + 1, attentionTokens.Count - eosIdx - 1);

                List<int[]> timestamps;
                if (_engine.HasCtc)
                {
                    ct.ThrowIfCancellationRequested();
                    CtcOutputEntity ctcOutput = _engine.RunCtc(encoderOutputs, batchSize: 1);
                    ct.ThrowIfCancellationRequested();
                    if (ctcOutput.HasData)
                    {
                        float[] logits = ctcOutput.Logits!;
                        int frameCount = ctcOutput.FrameCount;
                        int vocabSize = ctcOutput.VocabSize;

                        if (attentionTokens.Count == 0)
                        {
                            attentionTokens = CtcGreedyDecoder.Decode(logits, frameCount, vocabSize, _engine.Metadata.BlankId, ct);
                            // Keep stream tokens in sync with the CTC fallback path.
                            foreach (int t in attentionTokens) hyp.Add(t);
                        }

                        timestamps = CtcTimestampAligner.Align(logits, frameCount, vocabSize, attentionTokens,
                            _engine.Metadata.BlankId, _engine.Metadata.EncoderFrameShiftSeconds, ct);
                        // Align returns fewer entries than attentionTokens when alignment fails
                        // (short audio with many tokens -> tokens.Count > frameCount, "Not enough
                        // frames", or the forced-align sanity check / back-pointer mismatch).
                        // Without padding, stream.Timestamps keeps only its SOS slot while
                        // stream.Tokens has 1 + N entries, so the assembler collapses
                        // count = min(tokens, timestamps) to 1 and the entire transcription is
                        // dropped. Pad to equal length so the text survives (timestamps are lost
                        // only for the unaligned tail), matching the non-CTC branches below.
                        if (timestamps.Count < attentionTokens.Count)
                        {
                            var padded = new List<int[]>(attentionTokens.Count);
                            for (int j = 0; j < attentionTokens.Count; j++)
                                padded.Add(j < timestamps.Count ? timestamps[j] : new int[2]);
                            timestamps = padded;
                        }
                    }
                    else
                    {
                        timestamps = new List<int[]>(attentionTokens.Count);
                        for (int j = 0; j < attentionTokens.Count; j++) timestamps.Add(new int[2]);
                    }
                }
                else
                {
                    timestamps = new List<int[]>(attentionTokens.Count);
                    for (int j = 0; j < attentionTokens.Count; j++) timestamps.Add(new int[2]);
                }

                ct.ThrowIfCancellationRequested();
                stream.Timestamps.AddRange(timestamps);
                stream.RemoveDecodedChunk();
            }
            catch (OperationCanceledException)
            {
                throw;
            }
            catch (Exception ex)
            {
                throw new InvalidOperationException("Offline recognition failed.", ex);
            }
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _engine.Dispose();
            _model?.Dispose();
        }
    }
}
