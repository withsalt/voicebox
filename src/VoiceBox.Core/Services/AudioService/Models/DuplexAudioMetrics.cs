namespace VoiceBox.Core.Services.AudioService.Models
{
    /// <summary>
    /// 全双工音频回调中的轻量统计快照。所有电平均为 [-1,1] float 音频的 RMS/peak。
    /// </summary>
    public readonly record struct DuplexAudioMetrics(
        long TimestampTicks,
        uint SampleRate,
        uint Channels,
        uint FrameCount,
        double RenderRms,
        double RenderPeak,
        double CaptureInputRms,
        double CaptureInputPeak,
        double CaptureOutputRms,
        double CaptureOutputPeak,
        double RenderCaptureCorrelation,
        bool CaptureProcessed,
        double RenderCaptureInputCorrelation = 0,
        double RenderCaptureOutputCorrelation = 0,
        double CaptureInputOutputCorrelation = 0,
        double CaptureInputToOutputSuppressionDb = 0,
        double CaptureOutputToInputRmsRatio = 1,
        long SequenceNumber = 0,
        long StreamEpoch = 0,
        long StartSample = 0,
        bool HasDiscontinuity = false,
        bool CorrelationAligned = false)
    {
        public static DuplexAudioMetrics FromFrame(in DuplexAudioFrame frame)
        {
            DuplexSignalStats render = ComputeSignalStats(frame.RenderReference.Span);
            DuplexSignalStats captureInput = ComputeSignalStats(frame.CaptureBeforeAec.Span);
            DuplexSignalStats captureOutput = ComputeSignalStats(frame.CaptureAfterAec.Span);
            double inputCorrelation = ComputeCorrelation(frame.RenderReference.Span, frame.CaptureBeforeAec.Span);
            double outputCorrelation = ComputeCorrelation(frame.RenderReference.Span, frame.CaptureAfterAec.Span);
            double inputOutputCorrelation = ComputeCorrelation(frame.CaptureBeforeAec.Span, frame.CaptureAfterAec.Span);
            double outputToInputRatio = ComputeRatio(captureOutput.Rms, captureInput.Rms);

            return new DuplexAudioMetrics(
                TimestampTicks: frame.TimestampTicks,
                SampleRate: frame.SampleRate,
                Channels: frame.Channels,
                FrameCount: frame.FrameCount,
                RenderRms: render.Rms,
                RenderPeak: render.Peak,
                CaptureInputRms: captureInput.Rms,
                CaptureInputPeak: captureInput.Peak,
                CaptureOutputRms: captureOutput.Rms,
                CaptureOutputPeak: captureOutput.Peak,
                RenderCaptureCorrelation: outputCorrelation,
                CaptureProcessed: frame.CaptureProcessed,
                RenderCaptureInputCorrelation: inputCorrelation,
                RenderCaptureOutputCorrelation: outputCorrelation,
                CaptureInputOutputCorrelation: inputOutputCorrelation,
                CaptureInputToOutputSuppressionDb: ComputeSuppressionDb(captureInput.Rms, captureOutput.Rms),
                CaptureOutputToInputRmsRatio: outputToInputRatio,
                SequenceNumber: frame.SequenceNumber,
                StreamEpoch: frame.StreamEpoch,
                StartSample: frame.StartSample,
                HasDiscontinuity: frame.HasDiscontinuity);
        }

        private static DuplexSignalStats ComputeSignalStats(ReadOnlySpan<float> samples)
        {
            if (samples.IsEmpty)
                return default;

            double energy = 0;
            double peak = 0;
            for (int i = 0; i < samples.Length; i++)
            {
                double sample = samples[i];
                energy += sample * sample;
                peak = Math.Max(peak, Math.Abs(sample));
            }

            return new DuplexSignalStats(
                Rms: Math.Sqrt(energy / samples.Length),
                Peak: peak);
        }

        private static double ComputeCorrelation(ReadOnlySpan<float> left, ReadOnlySpan<float> right)
        {
            int count = Math.Min(left.Length, right.Length);
            if (count <= 0)
                return 0;

            double leftEnergy = 0;
            double rightEnergy = 0;
            double dot = 0;
            for (int i = 0; i < count; i++)
            {
                double l = left[i];
                double r = right[i];
                leftEnergy += l * l;
                rightEnergy += r * r;
                dot += l * r;
            }

            double denominator = Math.Sqrt(leftEnergy * rightEnergy);
            return denominator > 1e-12 ? Math.Clamp(dot / denominator, -1.0, 1.0) : 0;
        }

        private static double ComputeRatio(double numerator, double denominator)
        {
            if (!double.IsFinite(numerator) || numerator < 0)
                numerator = 0;
            if (!double.IsFinite(denominator) || denominator <= 1e-12)
                return numerator <= 1e-12 ? 1 : double.PositiveInfinity;

            return numerator / denominator;
        }

        private static double ComputeSuppressionDb(double inputRms, double outputRms)
        {
            if (!double.IsFinite(inputRms) || inputRms <= 1e-12)
                return 0;
            if (!double.IsFinite(outputRms) || outputRms <= 1e-12)
                return 80;

            return Math.Max(0, 20.0 * Math.Log10(inputRms / outputRms));
        }

        private readonly record struct DuplexSignalStats(double Rms, double Peak);
    }
}
