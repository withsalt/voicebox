using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// AudioVAE 编解码 (恒 fp32)。decoder 是流式热路径 (每步 3-patch 滑窗);
    /// encoder 仅音色克隆 prompt 使用 (每音色一次, 结果缓存于 VoiceStore)。
    /// </summary>
    internal sealed class AudioVaeCodec : IDisposable
    {
        private static readonly string[] DecoderInNames = ["latents"];
        private static readonly string[] DecoderOutNames = ["wav"];
        private static readonly string[] EncoderInNames = ["wav"];
        private static readonly string[] EncoderOutNames = ["latents"];

        private readonly InferenceSession _decoder;
        private readonly InferenceSession? _encoder;   // 惰性需求: 无克隆音色时可缺
        private readonly RunOptions _run = new();
        private readonly int _Dz;
        private readonly int _chunk;
        private readonly long[] _decodeShape;
        private readonly long[] _encodeShape = [1, 1, 0];

        public AudioVaeCodec(string decoderPath, string? encoderPath, SessionOptions so, VoxCpmModelConfig cfg)
        {
            _decoder = new InferenceSession(decoderPath, so);
            _encoder = encoderPath is null ? null : new InferenceSession(encoderPath, so);
            _Dz = cfg.FeatDim;
            _chunk = cfg.VaeChunkSize;
            _decodeShape = [1, _Dz, 0];
        }

        /// <summary>latents [Dz*T] 行主序 ([Dz,T]) → wav [T*chunk]。</summary>
        public float[] Decode(float[] latents, int t)
        {
            _decodeShape[2] = t;
            var own = new List<OrtValue>(1);
            try
            {
                var inputs = new[] { OnnxIo.MakeFloat(OrtDtype.Float32, latents, _decodeShape, own) };
                using var outputs = _decoder.Run(_run, DecoderInNames, inputs, DecoderOutNames);
                return outputs[0].GetTensorDataAsSpan<float>().ToArray();
            }
            finally { OnnxIo.Dispose(own); }
        }

        /// <summary>wav [S] (16k 单声道; S 必须是 chunk 的整数倍, 调用方负责左填充) → latents [Dz*(S/chunk)]。</summary>
        public float[] Encode(float[] wav)
        {
            if (_encoder is null)
                throw new InvalidOperationException("部署包缺少 audio_vae_encoder.onnx, 无法使用音色克隆。");
            if (wav.Length % _chunk != 0)
                throw new ArgumentException($"VAE encode 输入长度必须是 {_chunk} 的整数倍 (实际 {wav.Length})。", nameof(wav));
            _encodeShape[2] = wav.Length;
            var own = new List<OrtValue>(1);
            try
            {
                var inputs = new[] { OnnxIo.MakeFloat(OrtDtype.Float32, wav, _encodeShape, own) };
                using var outputs = _encoder.Run(_run, EncoderInNames, inputs, EncoderOutNames);
                return outputs[0].GetTensorDataAsSpan<float>().ToArray();
            }
            finally { OnnxIo.Dispose(own); }
        }

        public void Dispose()
        {
            _run.Dispose();
            _encoder?.Dispose();
            _decoder.Dispose();
        }
    }
}
