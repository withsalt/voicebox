using System.Runtime.InteropServices;
using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.TTS.Qwen3.Internal
{
    internal enum OrtDtype { Float32, Float16, BFloat16 }

    /// <summary>
    /// Bridges float[] tensors to ORT inputs/outputs across fp32 / fp16 / bf16.
    /// Float16 ⇔ System.Half share a layout (reinterpret cast). BFloat16 is the top 16 bits of the
    /// fp32 representation (round-to-nearest-even), so it has the SAME exponent range as fp32 — which
    /// is why bf16 (the model's training precision) stays numerically stable on CUDA where fp16 drifts.
    /// </summary>
    internal static class OnnxIo
    {
        public static OrtDtype Detect(InferenceSession session, string inputName, string cfgDtype)
        {
            if (session.InputMetadata.TryGetValue(inputName, out var m))
            {
                if (m.ElementType == typeof(Float16)) return OrtDtype.Float16;
                if (m.ElementType == typeof(BFloat16)) return OrtDtype.BFloat16;
                if (m.ElementType == typeof(float)) return OrtDtype.Float32;
            }
            return cfgDtype.ToLowerInvariant() switch
            {
                "fp16" or "float16" => OrtDtype.Float16,
                "bf16" or "bfloat16" => OrtDtype.BFloat16,
                _ => OrtDtype.Float32,
            };
        }

        public static OrtValue MakeFloat(OrtDtype dt, float[] data, long[] shape, List<OrtValue> own)
        {
            OrtValue v = dt switch
            {
                OrtDtype.Float16 => OrtValue.CreateTensorValueFromMemory(PackHalf(data), shape),
                OrtDtype.BFloat16 => OrtValue.CreateTensorValueFromMemory(PackBf16(data), shape),
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

        public static OrtValue MakeScalarLong(long value, List<OrtValue> own)
            => MakeLong([value], Array.Empty<long>(), own);

        public static float[] ReadFloat(OrtValue v, OrtDtype dt) => dt switch
        {
            OrtDtype.Float16 => UnpackHalf(v.GetTensorDataAsSpan<Float16>()),
            OrtDtype.BFloat16 => UnpackBf16(v.GetTensorDataAsSpan<BFloat16>()),
            _ => v.GetTensorDataAsSpan<float>().ToArray(),
        };

        // ---- KV cache held in native element type (no per-step conversion of the large, growing KV) ----

        public static Array EmptyKv(OrtDtype dt) => dt switch
        {
            OrtDtype.Float16 => Array.Empty<Float16>(),
            OrtDtype.BFloat16 => Array.Empty<BFloat16>(),
            _ => Array.Empty<float>(),
        };

        public static OrtValue MakeKv(OrtDtype dt, Array data, long[] shape, List<OrtValue> own)
        {
            OrtValue v = dt switch
            {
                OrtDtype.Float16 => OrtValue.CreateTensorValueFromMemory((Float16[])data, shape),
                OrtDtype.BFloat16 => OrtValue.CreateTensorValueFromMemory((BFloat16[])data, shape),
                _ => OrtValue.CreateTensorValueFromMemory((float[])data, shape),
            };
            own.Add(v);
            return v;
        }

        public static Array ReadKv(OrtDtype dt, OrtValue v) => dt switch
        {
            OrtDtype.Float16 => v.GetTensorDataAsSpan<Float16>().ToArray(),
            OrtDtype.BFloat16 => v.GetTensorDataAsSpan<BFloat16>().ToArray(),
            _ => v.GetTensorDataAsSpan<float>().ToArray(),
        };

        /// <summary>Wrap a KV array as an OrtValue the CALLER owns (not tracked in an `own` list).
        /// Used for the IO-binding path's persistent empty-past tensors.</summary>
        public static OrtValue MakeKvOwned(OrtDtype dt, Array data, long[] shape) => dt switch
        {
            OrtDtype.Float16 => OrtValue.CreateTensorValueFromMemory((Float16[])data, shape),
            OrtDtype.BFloat16 => OrtValue.CreateTensorValueFromMemory((BFloat16[])data, shape),
            _ => OrtValue.CreateTensorValueFromMemory((float[])data, shape),
        };

        public static void Dispose(List<OrtValue> own)
        {
            foreach (var v in own) v.Dispose();
            own.Clear();
        }

        // ---- conversions ----

        private static Float16[] PackHalf(float[] data)
        {
            var f16 = new Float16[data.Length];
            Span<Half> half = MemoryMarshal.Cast<Float16, Half>(f16.AsSpan());
            for (int i = 0; i < data.Length; i++) half[i] = (Half)data[i];
            return f16;
        }

        private static float[] UnpackHalf(ReadOnlySpan<Float16> src)
        {
            ReadOnlySpan<Half> half = MemoryMarshal.Cast<Float16, Half>(src);
            var outp = new float[half.Length];
            for (int i = 0; i < half.Length; i++) outp[i] = (float)half[i];
            return outp;
        }

        private static BFloat16[] PackBf16(float[] data)
        {
            var bf = new BFloat16[data.Length];
            Span<ushort> u = MemoryMarshal.Cast<BFloat16, ushort>(bf.AsSpan());
            for (int i = 0; i < data.Length; i++)
            {
                uint bits = BitConverter.SingleToUInt32Bits(data[i]);
                uint rounding = 0x7FFFu + ((bits >> 16) & 1u);   // round-to-nearest-even
                u[i] = (ushort)((bits + rounding) >> 16);
            }
            return bf;
        }

        private static float[] UnpackBf16(ReadOnlySpan<BFloat16> src)
        {
            ReadOnlySpan<ushort> u = MemoryMarshal.Cast<BFloat16, ushort>(src);
            var outp = new float[u.Length];
            for (int i = 0; i < u.Length; i++) outp[i] = BitConverter.UInt32BitsToSingle((uint)u[i] << 16);
            return outp;
        }
    }
}
