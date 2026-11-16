using VoiceBox.ASR.FireRedASR2.Internal.Model;

namespace VoiceBox.ASR.FireRedASR2.Internal.Utils
{
    internal static class PadHelper
    {
        /// <summary>
        /// Zero-pad each input to the longest <c>Speech</c> length and concatenate them into a
        /// single flat buffer of shape [batch, maxLen]. With batch == 1 this just returns the
        /// (live) feature buffer; with larger batches each row is padded with zeros.
        /// </summary>
        public static float[] PadSequence(IReadOnlyList<AsrInputEntity> modelInputs)
        {
            ArgumentNullException.ThrowIfNull(modelInputs);
            if (modelInputs.Count == 0) return Array.Empty<float>();

            // Fast path: batch == 1 means no padding is needed — just hand back the live buffer.
            if (modelInputs.Count == 1)
                return modelInputs[0].Speech ?? Array.Empty<float>();

            int maxLen = 0;
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? speech = modelInputs[i].Speech;
                if (speech is null) continue;
                if (speech.Length > maxLen) maxLen = speech.Length;
            }

            float[] output = new float[maxLen * modelInputs.Count];
            for (int i = 0; i < modelInputs.Count; i++)
            {
                float[]? speech = modelInputs[i].Speech;
                if (speech is null || speech.Length == 0) continue;
                Array.Copy(speech, 0, output, i * maxLen, speech.Length);
                // Remaining positions stay at 0 (default float[] initialisation).
            }
            return output;
        }
    }
}
