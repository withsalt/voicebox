using System;

namespace VoiceBox.ASR.Qwen3.Internal
{
    /// <summary>
    /// Element type of the encoder/decoder ONNX I/O tensors. Weight-only quantized
    /// exports (int8/uint8 dynamic-quant) keep their I/O in float32 — only the matmul
    /// weights are quantized — so they share the <see cref="Float32"/> path. Genuine
    /// fp16 exports use <see cref="Float16"/> with the <c>Float16Bridge</c> packing.
    /// </summary>
    internal enum Qwen3AsrIoDtype
    {
        Float32,
        Float16,
    }

    internal static class Qwen3AsrDtypeParser
    {
        public static Qwen3AsrIoDtype Parse(string? dtype)
        {
            if (string.IsNullOrWhiteSpace(dtype))
                return Qwen3AsrIoDtype.Float32;

            return dtype.Trim().ToLowerInvariant() switch
            {
                "float32" or "fp32" or "f32" => Qwen3AsrIoDtype.Float32,
                "float16" or "fp16" or "f16" or "half" => Qwen3AsrIoDtype.Float16,
                // Weight-only dynamic quantization (ORT / optimum-cli output): tensors
                // travel fp32, weights are stored int8/uint8 inside the graph.
                "int8" or "uint8" or "q8" or "qint8" or "quint8" => Qwen3AsrIoDtype.Float32,
                _ => throw new InvalidOperationException(
                    $"Unsupported metadata dtype \"{dtype}\". Expected float32, float16, or int8."),
            };
        }
    }
}
