namespace VoiceBox.AEC.DFSMN.Models
{
    /// <summary>
    /// Configuration for the combined DFSMN AEC + near-end VAD service.
    /// A single ONNX invocation produces both the spectral mask and VAD probability.
    /// </summary>
    public sealed class DfsmnAecVadOptions : DfsmnAecOptions
    {
        /// <summary>
        /// Near-end VAD eventing settings.
        /// </summary>
        public DfsmnVadOptions Vad { get; set; } = new DfsmnVadOptions();

        internal new void Validate()
        {
            base.Validate();
            if (Vad is null)
                throw new ArgumentNullException(nameof(Vad), "VAD options cannot be null.");
            Vad.Validate();
        }
    }
}
