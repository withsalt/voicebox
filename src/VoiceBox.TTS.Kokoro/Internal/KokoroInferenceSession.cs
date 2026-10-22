using System.Buffers;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;

namespace VoiceBox.TTS.Kokoro.Internal
{
    internal sealed class KokoroInferenceSession : IDisposable
    {
        public const int SampleRate = 24000;

        private readonly InferenceSession _session;
        private readonly RunOptions _runOptions;
        private readonly string _tokensInputName;
        private readonly bool _speedIsFloat;
        private readonly string[] _outputNames;
        private bool _disposed;

        public KokoroInferenceSession(string modelPath, OnnxExecutionOptions options)
        {
            var sessionOptions = OnnxSessionFactory.Create(options);
            _session = new InferenceSession(modelPath, sessionOptions);
            _runOptions = new RunOptions();

            _tokensInputName = _session.InputMetadata.ContainsKey("input_ids") ? "input_ids" : "tokens";

            // 按 speed 输入的真实元素类型决定写入 float 还是 int，避免 (int)0.8 被截成 0 之类的 bug。
            _speedIsFloat = !_session.InputMetadata.TryGetValue("speed", out var speedMeta)
                || speedMeta.ElementType == typeof(float);

            _outputNames = [.. _session.OutputMetadata.Keys];
        }

        public void Warmup(float[] dummyVoiceStyle, int[] dummyStyleShape)
        {
            int[] tokens = [1, 2, 3, 4, 5];
            Synthesize(tokens, dummyVoiceStyle, dummyStyleShape, 1.0f);
        }

        public float[] Synthesize(int[] tokens, float[] voiceStyle, int[] styleShape, float speed)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);

            int paddedLen = tokens.Length + 2;
            long[] paddedTokens = ArrayPool<long>.Shared.Rent(paddedLen);
            try
            {
                paddedTokens[0] = 0;
                for (int i = 0; i < tokens.Length; i++)
                    paddedTokens[i + 1] = tokens[i];
                paddedTokens[paddedLen - 1] = 0;

                long[] styleShape64 = new long[styleShape.Length];
                for (int i = 0; i < styleShape.Length; i++) styleShape64[i] = styleShape[i];

                using var tokensValue = OrtValue.CreateTensorValueFromMemory(
                    paddedTokens, [1L, paddedLen]);
                using var styleValue = OrtValue.CreateTensorValueFromMemory(
                    voiceStyle, styleShape64);

                OrtValue speedValue;
                if (_speedIsFloat)
                {
                    float[] speedData = [speed];
                    speedValue = OrtValue.CreateTensorValueFromMemory(speedData, [1L]);
                }
                else
                {
                    int[] speedData = [(int)MathF.Round(speed)];
                    speedValue = OrtValue.CreateTensorValueFromMemory(speedData, [1L]);
                }

                using (speedValue)
                {
                    var inputs = new Dictionary<string, OrtValue>(3)
                    {
                        [_tokensInputName] = tokensValue,
                        ["style"] = styleValue,
                        ["speed"] = speedValue,
                    };

                    using var outputs = _session.Run(_runOptions, inputs, _outputNames);
                    var span = outputs[0].GetTensorDataAsSpan<float>();
                    float[] result = new float[span.Length];
                    span.CopyTo(result);
                    return result;
                }
            }
            finally
            {
                ArrayPool<long>.Shared.Return(paddedTokens);
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
}
