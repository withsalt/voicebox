using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    /// <summary>
    /// cfm_decode.onnx: 一次 Run 完成一个 patch 的完整 10 步 Euler 流匹配采样
    /// (CFG batch=2 + cfg-zero-star 已在图内)。噪声由宿主注入 (可复现/可对拍)。
    /// 全静态形状 → 每 80ms 音频恰好一次 Run。
    /// </summary>
    internal sealed class CfmDecoder : IDisposable
    {
        private static readonly string[] InNames = ["dit_hidden", "cond", "noise", "cfg_value"];
        private static readonly string[] OutNames = ["pred_feat"];

        private readonly InferenceSession _session;
        private readonly RunOptions _run = new();
        private readonly OrtDtype _dtype;
        private readonly int _H, _Dz, _Pz;
        private readonly long[] _ditHiddenShape;
        private readonly long[] _featureShape;
        private readonly float[] _cfgValueInput = new float[1];
        private static readonly long[] ScalarShape = [1];

        public CfmDecoder(string modelPath, SessionOptions so, VoxCpmModelConfig cfg)
        {
            _session = new InferenceSession(modelPath, so);
            _dtype = OnnxIo.Detect(_session, "dit_hidden");
            _H = cfg.HiddenSize;
            _Dz = cfg.FeatDim;
            _Pz = cfg.PatchSize;
            _ditHiddenShape = [1, _H];
            _featureShape = [1, _Dz, _Pz];
        }

        /// <summary>
        /// 生成一个 patch。ditHidden [H], cond/noise [Dz*Pz] (行主序 [Dz,Pz])。
        /// 返回 pred_feat [Dz*Pz]。
        /// </summary>
        public float[] Decode(float[] ditHidden, float[] cond, float[] noise, float cfgValue)
        {
            _cfgValueInput[0] = cfgValue;
            var own = new List<OrtValue>(4);
            try
            {
                var inputs = new[]
                {
                    OnnxIo.MakeFloat(_dtype, ditHidden, _ditHiddenShape, own),
                    OnnxIo.MakeFloat(_dtype, cond, _featureShape, own),
                    OnnxIo.MakeFloat(_dtype, noise, _featureShape, own),
                    OnnxIo.MakeFloat(_dtype, _cfgValueInput, ScalarShape, own),
                };
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
