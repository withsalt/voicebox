using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using VoiceBox.Core.Onnx;

namespace VoiceBox.VAD.FireRed
{
    /// <summary>
    /// FireRed Stream-VAD ONNX 模型封装。
    /// 输入：feat (1, T, 80) float32, caches_in (8, 1, 128, 19) float32。
    /// 输出：probs (1, T) 或 (T,) float32, caches_out (8, 1, 128, 19) float32。
    /// </summary>
    internal sealed class FireRedVadOnnxRunner : IDisposable
    {
        public const int NumBlocks = 8;
        public const int P = 128;
        public const int LookbackPadding = 19;
        public const int FbankDim = 80;

        private readonly InferenceSession _session;
        private readonly SessionOptions _sessionOptions;
        private readonly float[] _caches;

        public FireRedVadOnnxRunner(string modelPath, OnnxExecutionOptions onnxOptions)
        {
            _sessionOptions = OnnxSessionFactory.Create(onnxOptions);
            _session = new InferenceSession(modelPath, _sessionOptions);
            _caches = new float[NumBlocks * 1 * P * LookbackPadding];
        }

        public void ResetCaches()
        {
            Array.Clear(_caches, 0, _caches.Length);
        }

        /// <summary>
        /// 运行一次推理：输入若干帧 80 维特征，返回每帧的语音概率。
        /// 内部 caches 状态会被滚动更新。
        /// </summary>
        public float[] Run(float[][] fbankFrames)
        {
            int t = fbankFrames.Length;
            if (t == 0) return Array.Empty<float>();

            var featData = new float[t * FbankDim];
            for (int i = 0; i < t; i++)
            {
                Array.Copy(fbankFrames[i], 0, featData, i * FbankDim, FbankDim);
            }

            var featTensor = new DenseTensor<float>(featData, new[] { 1, t, FbankDim });
            var cachesTensor = new DenseTensor<float>(_caches, new[] { NumBlocks, 1, P, LookbackPadding });

            var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("feat", featTensor),
                NamedOnnxValue.CreateFromTensor("caches_in", cachesTensor),
            };

            using var outputs = _session.Run(inputs);

            Tensor<float>? probsT = null;
            Tensor<float>? cachesOutT = null;
            foreach (var o in outputs)
            {
                var tensor = o.AsTensor<float>();
                if (cachesOutT == null && tensor.Length == _caches.Length)
                {
                    cachesOutT = tensor;
                }
                else
                {
                    probsT = tensor;
                }
            }
            if (probsT == null || cachesOutT == null)
            {
                throw new InvalidOperationException("Unexpected ONNX output layout from FireRedVAD model.");
            }

            // 滚动更新 caches
            var cachesArr = cachesOutT.ToArray();
            Array.Copy(cachesArr, _caches, _caches.Length);

            return probsT.ToArray();
        }

        public void Dispose()
        {
            _session?.Dispose();
            _sessionOptions?.Dispose();
        }
    }
}
