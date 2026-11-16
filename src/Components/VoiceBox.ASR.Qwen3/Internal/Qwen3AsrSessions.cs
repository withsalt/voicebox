using System;
using System.IO;
using Microsoft.ML.OnnxRuntime;
using VoiceBox.Core.Onnx;

namespace VoiceBox.ASR.Qwen3.Internal
{
    /// <summary>
    /// Loads the Qwen3-ASR ONNX sessions described by <see cref="Qwen3AsrMetadata"/>.
    /// Some exports ship a fused <c>encoder.onnx</c>; older / split exports ship
    /// <c>conv_embed.onnx</c> + <c>audio_stack.onnx</c>. Either path is supported.
    /// The prefill and decode decoder sessions may be the same physical file
    /// (shared graph) or two distinct files.
    /// </summary>
    /// <remarks>
    /// Each ONNX graph is loaded with its own <see cref="SessionOptions"/> produced by
    /// <paramref name="optionsFactory"/>. Encoder / conv_embed / audio_stack / prefill
    /// are dynamic-shape (chunk count + prompt len vary) so they ask for
    /// <see cref="SessionPurpose.Dynamic"/>; the autoregressive decode session has
    /// fully static shapes (single-token input + fixed KV cache) and asks for
    /// <see cref="SessionPurpose.StaticShapes"/> so the factory can enable
    /// <c>EnableMemoryPattern</c> and choose a tighter arena strategy.
    /// </remarks>
    internal sealed class Qwen3AsrSessions : IDisposable
    {
        public InferenceSession? Encoder { get; }
        public InferenceSession? ConvEmbed { get; }
        public InferenceSession? AudioStack { get; }
        public InferenceSession Prefill { get; }
        public InferenceSession Decode { get; }
        public bool SharedDecoder { get; }
        public bool HasFusedEncoder => Encoder is not null;

        public Qwen3AsrSessions(string modelDir, Qwen3AsrMetadata metadata, Func<SessionPurpose, SessionOptions> optionsFactory)
        {
            ArgumentException.ThrowIfNullOrWhiteSpace(modelDir);
            ArgumentNullException.ThrowIfNull(metadata);
            ArgumentNullException.ThrowIfNull(optionsFactory);
            try
            {
                if (!string.IsNullOrEmpty(metadata.EncoderModel))
                {
                    Encoder = LoadSession(modelDir, metadata.EncoderModel!, optionsFactory, SessionPurpose.Dynamic);
                }
                else
                {
                    if (string.IsNullOrEmpty(metadata.ConvEmbedModel) || string.IsNullOrEmpty(metadata.AudioStackModel))
                        throw new InvalidOperationException(
                            "metadata.json must define either encoder_model or both conv_embed_model and audio_stack_model.");
                    ConvEmbed = LoadSession(modelDir, metadata.ConvEmbedModel!, optionsFactory, SessionPurpose.Dynamic);
                    AudioStack = LoadSession(modelDir, metadata.AudioStackModel!, optionsFactory, SessionPurpose.Dynamic);
                }

                string prefillFile = metadata.ResolvePrefillModel();
                string decodeFile = metadata.ResolveDecodeModel();
                SharedDecoder = string.Equals(prefillFile, decodeFile, StringComparison.OrdinalIgnoreCase);

                // Prefill takes a variable-length prompt + audio_features tensor — dynamic shape.
                Prefill = LoadSession(modelDir, prefillFile, optionsFactory, SessionPurpose.Dynamic);

                // Decode is the hot loop and is fully static (input_ids=[1,1], cache_position=[1,1],
                // KV cache fixed at static_cache_len). Ask for StaticShapes so the factory can
                // enable MemoryPattern + a tighter arena. When shared with prefill, we must
                // reuse the same session — re-loading would create a second copy of the same
                // ~1.5 GB graph on the GPU.
                Decode = SharedDecoder ? Prefill : LoadSession(modelDir, decodeFile, optionsFactory, SessionPurpose.StaticShapes);
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        private static InferenceSession LoadSession(string modelDir, string fileName, Func<SessionPurpose, SessionOptions> optionsFactory, SessionPurpose purpose)
        {
            string path = Path.Combine(modelDir, fileName);
            if (!File.Exists(path))
                throw new FileNotFoundException($"ONNX model file not found: {path}", path);
            // SessionOptions is consumed at session construction; the native runtime keeps
            // its own copy of the relevant config so it is safe to dispose immediately
            // after — and necessary, because each call builds a fresh SessionOptions and
            // a leaked SessionOptions also leaks its appended provider options.
            using var so = optionsFactory(purpose);
            return new InferenceSession(path, so);
        }

        public void Dispose()
        {
            try { Encoder?.Dispose(); } catch { }
            try { ConvEmbed?.Dispose(); } catch { }
            try { AudioStack?.Dispose(); } catch { }
            try
            {
                if (SharedDecoder) Prefill?.Dispose();
                else { Prefill?.Dispose(); Decode?.Dispose(); }
            }
            catch { }
        }
    }
}
