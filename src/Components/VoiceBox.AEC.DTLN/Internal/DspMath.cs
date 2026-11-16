namespace VoiceBox.AEC.DTLN.Internal
{
    internal static class DspMath
    {
        public static float Clamp(float value, float min, float max)
            => value < min ? min : value > max ? max : value;
    }
}
