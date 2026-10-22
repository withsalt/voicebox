namespace VoiceBox.TTS.Kokoro.Internal
{
    internal static class AudioTrimmer
    {
        private const int DefaultFrameLength = 2048;
        private const int DefaultHopLength = 512;
        private const float DefaultTopDb = 60f;

        public static float[] TrimSilence(float[] audio)
        {
            if (audio.Length == 0) return audio;

            var (start, end) = FindNonSilentRegion(audio, DefaultFrameLength, DefaultHopLength, DefaultTopDb);
            if (start >= end) return Array.Empty<float>();
            if (start == 0 && end == audio.Length) return audio;

            float[] trimmed = new float[end - start];
            Array.Copy(audio, start, trimmed, 0, trimmed.Length);
            return trimmed;
        }

        private static (int Start, int End) FindNonSilentRegion(float[] audio, int frameLength, int hopLength, float topDb)
        {
            bool[] nonSilent = ComputeNonSilentFrames(audio, frameLength, hopLength, topDb);

            int firstNonSilent = -1;
            int lastNonSilent = -1;
            for (int i = 0; i < nonSilent.Length; i++)
            {
                if (nonSilent[i])
                {
                    if (firstNonSilent < 0) firstNonSilent = i;
                    lastNonSilent = i;
                }
            }

            if (firstNonSilent < 0)
                return (0, 0);

            int start = firstNonSilent * hopLength;
            int end = Math.Min(audio.Length, (lastNonSilent + 1) * hopLength);
            return (start, end);
        }

        private static bool[] ComputeNonSilentFrames(float[] audio, int frameLength, int hopLength, float topDb)
        {
            float[] rms = ComputeRms(audio, frameLength, hopLength);
            float[] db = AmplitudeToDb(rms);

            float maxDb = float.MinValue;
            for (int i = 0; i < db.Length; i++)
            {
                if (db[i] > maxDb) maxDb = db[i];
            }

            bool[] nonSilent = new bool[db.Length];
            float threshold = maxDb - topDb;
            for (int i = 0; i < db.Length; i++)
            {
                nonSilent[i] = db[i] > threshold;
            }
            return nonSilent;
        }

        private static float[] ComputeRms(float[] audio, int frameLength, int hopLength)
        {
            int paddedLength = audio.Length + frameLength;
            float[] padded = new float[paddedLength];
            int halfFrame = frameLength / 2;
            Array.Copy(audio, 0, padded, halfFrame, audio.Length);

            int numFrames = (padded.Length - frameLength) / hopLength + 1;
            float[] rms = new float[numFrames];

            for (int i = 0; i < numFrames; i++)
            {
                int offset = i * hopLength;
                float sum = 0f;
                for (int j = 0; j < frameLength; j++)
                {
                    float val = padded[offset + j];
                    sum += val * val;
                }
                rms[i] = MathF.Sqrt(sum / frameLength);
            }
            return rms;
        }

        private static float[] AmplitudeToDb(float[] amplitude)
        {
            const float amin = 1e-5f;
            const float topDb = 80f;

            float[] db = new float[amplitude.Length];
            float maxDb = float.MinValue;

            for (int i = 0; i < amplitude.Length; i++)
            {
                float val = MathF.Max(amin, amplitude[i]);
                float power = val * val;
                db[i] = 10f * MathF.Log10(MathF.Max(1e-10f, power));
                if (db[i] > maxDb) maxDb = db[i];
            }

            for (int i = 0; i < db.Length; i++)
            {
                db[i] = MathF.Max(db[i], maxDb - topDb);
            }

            return db;
        }
    }
}
