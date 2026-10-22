using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// feat_encoder.onnx: latent patch 序列 [1,T,Pz,Dz] → LM 嵌入 [1,T,H]
    /// (VoxCPMLocEnc + enc_to_lm_proj)。prefill 时 T=prompt patch 数, 每步 T=1。
    /// </summary>
    internal sealed class FeatEncoder : IDisposable
    {
        private static readonly string[] InNames = ["feat"];
        private static readonly string[] OutNames = ["feat_embed"];

        private readonly InferenceSession _session;
        private readonly RunOptions _run = new();
        private readonly OrtDtype _dtype;
        private readonly int _Dz, _Pz;
        private readonly long[] _featShape;

        public FeatEncoder(string modelPath, SessionOptions so, VoxCpmModelConfig cfg)
        {
            _session = new InferenceSession(modelPath, so);
            _dtype = OnnxIo.Detect(_session, "feat");
            _Dz = cfg.FeatDim;
            _Pz = cfg.PatchSize;
            _featShape = [1, 0, _Pz, _Dz];
        }

        /// <summary>feat [T*Pz*Dz] 行主序 ([T,Pz,Dz]) → feat_embed [T*H]。</summary>
        public float[] Encode(float[] feat, int t)
        {
            _featShape[1] = t;
            var own = new List<OrtValue>(1);
            try
            {
                var inputs = new[] { OnnxIo.MakeFloat(_dtype, feat, _featShape, own) };
                using var outputs = _session.Run(_run, InNames, inputs, OutNames);
                return OnnxIo.ReadFloat(outputs[0], _dtype);
            }
            finally { OnnxIo.Dispose(own); }
        }

        public void Dispose()
        {
            _run.Dispose();
            _session.Dispose();
        }
    }
}
