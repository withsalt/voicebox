namespace VoiceBox.ASR.FireRedASR2.Internal.Model
{
    internal sealed class FrontendConfEntity
    {
        public int SampleRate { get; init; } = 16000;

        public string Window { get; init; } = "povey";

        public int NumMels { get; init; } = 80;

        public int FrameLength { get; init; } = 25;

        public int FrameShift { get; init; } = 10;

        public float Dither { get; init; }

        public int LfrM { get; init; } = 7;

        public int LfrN { get; init; } = 6;

        public bool SnipEdges { get; init; } = true;
    }
}
