namespace VoiceBox.Core.Services.DNRService
{
    public static class DigitalNoiseReductionServiceExtensions
    {
        /// <summary>
        /// Processes one independent audio stream and returns exactly one source-backed output
        /// sample for every input sample. Zero input drives buffered real samples through the
        /// pipeline; output belonging to that zero tail is discarded.
        /// </summary>
        public static float[] ProcessAll(
            this IDigitalNoiseReductionService service,
            ReadOnlySpan<float> input,
            int chunkSize = 0)
        {
            ArgumentNullException.ThrowIfNull(service);
            if (chunkSize < 0)
                throw new ArgumentOutOfRangeException(nameof(chunkSize));

            int frameSize = service.FrameSize;
            if (frameSize <= 0)
                throw new InvalidOperationException("DNR FrameSize must be positive.");

            int processingChunkSize = chunkSize == 0 ? frameSize : chunkSize;
            var output = new float[input.Length];
            var work = new float[processingChunkSize];
            int outputCount = 0;

            service.Reset();
            try
            {
                for (int offset = 0; offset < input.Length; offset += processingChunkSize)
                {
                    int length = Math.Min(processingChunkSize, input.Length - offset);
                    input.Slice(offset, length).CopyTo(work);
                    int produced = ProcessChecked(service, work.AsSpan(0, length));
                    if (produced > input.Length - outputCount)
                        throw new InvalidOperationException("DNR produced more source-backed output than input.");

                    work.AsSpan(0, produced).CopyTo(output.AsSpan(outputCount));
                    outputCount += produced;
                }

                long drainBudget = Math.Max(0, service.DelaySamples) + 4L * frameSize;
                long drainedInput = 0;
                while (outputCount < output.Length)
                {
                    work.AsSpan().Clear();
                    int produced = ProcessChecked(service, work);
                    int copyLength = Math.Min(produced, output.Length - outputCount);
                    work.AsSpan(0, copyLength).CopyTo(output.AsSpan(outputCount));
                    outputCount += copyLength;
                    drainedInput += work.Length;

                    if (drainedInput > drainBudget && outputCount < output.Length)
                    {
                        throw new InvalidOperationException(
                            "DNR did not release its source-backed tail within the declared delay bound.");
                    }
                }

                return output;
            }
            finally
            {
                service.Reset();
            }
        }

        private static int ProcessChecked(IDigitalNoiseReductionService service, Span<float> buffer)
        {
            int produced = service.Process(buffer);
            if ((uint)produced > (uint)buffer.Length)
            {
                throw new InvalidOperationException(
                    $"DNR '{service.GetType().FullName}' returned {produced} samples for a " +
                    $"{buffer.Length}-sample input.");
            }

            return produced;
        }
    }
}
