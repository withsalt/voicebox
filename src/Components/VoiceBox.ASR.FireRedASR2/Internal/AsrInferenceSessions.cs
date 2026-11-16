using Microsoft.ML.OnnxRuntime;

namespace VoiceBox.ASR.FireRedASR2.Internal
{
    /// <summary>
    /// Owns the three ONNX inference sessions (encoder / decoder / CTC). CTC and decoder
    /// sessions are optional — null when the model does not provide that path.
    /// </summary>
    internal sealed class AsrInferenceSessions : IDisposable
    {
        public InferenceSession Encoder { get; }
        public InferenceSession? Decoder { get; }
        public InferenceSession? Ctc { get; }

        public AsrInferenceSessions(InferenceSession encoder, InferenceSession? decoder, InferenceSession? ctc)
        {
            ArgumentNullException.ThrowIfNull(encoder);
            Encoder = encoder;
            Decoder = decoder;
            Ctc = ctc;
        }

        public void Dispose()
        {
            Encoder.Dispose();
            Decoder?.Dispose();
            Ctc?.Dispose();
        }
    }
}
