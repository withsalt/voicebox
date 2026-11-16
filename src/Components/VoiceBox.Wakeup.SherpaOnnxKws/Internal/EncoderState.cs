using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.Wakeup.SherpaOnnxKws.Internal
{
    /// <summary>
    /// Encoder 的运行状态 (跨 chunk 累积的 KV / 卷积缓存等)。
    /// 用纯托管缓冲区保存, 在每次 RunEncoder 时临时打包为 OrtValue。
    /// </summary>
    internal sealed class EncoderState
    {
        // 36 个 float 状态 (按 sherpa-onnx 输入顺序排布) + embed_states 浮点 + processed_lens (int64)
        // 我们用两个并行数组保存: 前 NumFloatStates 个是 float, 最后一个是 int64
        public float[][] FloatBuffers = Array.Empty<float[]>();
        public long[][] FloatShapes = Array.Empty<long[]>();
        public long[] ProcessedLens = new long[1];
        public long[] ProcessedLensShape = new long[] { 1 };

        public int NumFloatStates => FloatBuffers.Length;
    }

    /// <summary>
    /// 解析自 encoder ONNX 模型 metadata 的形状参数。对应 sherpa-onnx zipformer2 transducer。
    /// </summary>
    internal sealed class EncoderMetadata
    {
        public int T;                       // chunk_size, 每次 encoder 输入的特征帧数
        public int DecodeChunkLen;          // chunk_shift, 每次 encoder 向前推进的帧数
        public int FeatureDim = 80;
        public int[] NumEncoderLayers = Array.Empty<int>();
        public int[] EncoderDims = Array.Empty<int>();
        public int[] QueryHeadDims = Array.Empty<int>();
        public int[] ValueHeadDims = Array.Empty<int>();
        public int[] NumHeads = Array.Empty<int>();
        public int[] CnnModuleKernels = Array.Empty<int>();
        public int[] LeftContextLen = Array.Empty<int>();

        public int TotalLayers
        {
            get
            {
                int sum = 0;
                for (int i = 0; i < NumEncoderLayers.Length; i++) sum += NumEncoderLayers[i];
                return sum;
            }
        }

        public int NumStateTensors => TotalLayers * 6 + 2;

        public int EmbedDim => (((FeatureDim - 1) / 2) - 1) / 2;
    }
}
