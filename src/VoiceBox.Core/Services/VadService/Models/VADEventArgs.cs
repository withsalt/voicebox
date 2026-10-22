namespace VoiceBox.Core.Services.VadService.Models
{
    public class VADEventArgs : EventArgs
    {
        public float[] Data { get; }

        public TimeSpan? Duration { get; }

        public VADEventDataType EventDataType { get; } = VADEventDataType.None;

        public VADEventArgs(float[] audioData, VADEventDataType eventDataType, TimeSpan? ts = null)
        {
            Data = audioData;
            EventDataType = eventDataType;
            Duration = ts;
        }
    }
}
