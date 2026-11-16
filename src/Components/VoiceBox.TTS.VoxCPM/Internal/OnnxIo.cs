using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.VoxCPM.Internal
{
    internal enum OrtDtype { Float32, Float16 }

    /// <summary>
    /// float[] 与 ORT 张量在 fp32 / fp16 之间的桥接 (源自 Qwen3 OnnxIo, 去掉本包不用的 bf16)。
    /// Float16 ⇔ System.Half 同内存布局 (reinterpret cast)。增长中的 KV cache 不经过这里:
    /// LmDecoder 直接把上一步的 present OrtValue 回喂为 past (零拷贝)。
    /// </summary>
    internal static class OnnxIo
    {
        public static OrtDtype Detect(InferenceSession session, string inputName)
        {
            if (session.InputMetadata.TryGetValue(inputName, out var m) && m.ElementType == typeof(Float16))
                return OrtDtype.Float16;
            return OrtDtype.Float32;
        }

        public static OrtValue MakeFloat(OrtDtype dt, float[] data, long[] shape, List<OrtValue> own)
        {
            OrtValue v = dt switch
            {
                OrtDtype.Float16 => OrtValue.CreateTensorValueFromMemory(PackHalf(data), shape),
                _ => OrtValue.CreateTensorValueFromMemory(data, shape),
            };
            own.Add(v);
            return v;
        }

        public static OrtValue MakeLong(long[] data, long[] shape, List<OrtValue> own)
        {
            var v = OrtValue.CreateTensorValueFromMemory(data, shape);
            own.Add(v);
            return v;
        }

        /// <summary>int32 张量 (GQA 图的 seqlens_k / total_sequence_length; shape=[] 为标量)。</summary>
        public static OrtValue MakeInt(int[] data, long[] shape, List<OrtValue> own)
        {
            var v = OrtValue.CreateTensorValueFromMemory(data, shape);
            own.Add(v);
            return v;
        }

        public static float[] ReadFloat(OrtValue v, OrtDtype dt) => dt switch
        {
            OrtDtype.Float16 => UnpackHalf(v.GetTensorDataAsSpan<Float16>()),
            _ => v.GetTensorDataAsSpan<float>().ToArray(),
        };

        public static void Dispose(List<OrtValue> own)
        {
            foreach (var v in own) v.Dispose();
            own.Clear();
        }

        // ---- 转换 ----

        internal static Float16[] PackHalf(float[] data)
        {
            var f16 = new Float16[data.Length];
            Span<Half> half = MemoryMarshal.Cast<Float16, Half>(f16.AsSpan());
            for (int i = 0; i < data.Length; i++) half[i] = (Half)data[i];
            return f16;
        }

        internal static float[] UnpackHalf(ReadOnlySpan<Float16> src)
        {
            ReadOnlySpan<Half> half = MemoryMarshal.Cast<Float16, Half>(src);
            var outp = new float[half.Length];
            for (int i = 0; i < half.Length; i++) outp[i] = (float)half[i];
            return outp;
        }
    }
}
