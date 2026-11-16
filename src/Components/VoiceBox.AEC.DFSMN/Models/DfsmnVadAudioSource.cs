namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>
    /// Selects which near-end audio representation is emitted by the VAD events.
    /// </summary>
    public enum DfsmnVadAudioSource
    {
        /// <summary>
        /// Emit the neural AEC output. This is the recommended production path for ASR.
        /// </summary>
        ProcessedCapture = 0,

        /// <summary>
        /// Emit the raw microphone capture. Useful for diagnostics and offline model evaluation.
        /// </summary>
        RawCapture = 1,

        /// <summary>
        /// Emit the linear AEC residual before neural masking.
        /// </summary>
        LinearResidual = 2,
    }
}
