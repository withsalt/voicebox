using VoiceBox.AEC.DTLN.Models;

namespace VoiceBox.AEC.DTLN.Internal
{
    internal sealed class DtlnAecProcessor : IDisposable
    {
        private readonly DtlnAecOnnxRunner _runner;
        private readonly RealFft _fft = new(DtlnAecOptions.BlockLength);

        private readonly float[] _micWindow = new float[DtlnAecOptions.BlockLength];
        private readonly float[] _lpbWindow;
        private readonly float[] _outBuffer = new float[DtlnAecOptions.BlockLength];
        private readonly float[] _micRe = new float[DtlnAecOptions.FrequencyBins];
        private readonly float[] _micIm = new float[DtlnAecOptions.FrequencyBins];
        private readonly float[] _lpbRe = new float[DtlnAecOptions.FrequencyBins];
        private readonly float[] _lpbIm = new float[DtlnAecOptions.FrequencyBins];
        private readonly float[] _micMag;
        private readonly float[] _lpbMag;
        private readonly float[] _mask;
        private readonly float[] _estimatedRe = new float[DtlnAecOptions.FrequencyBins];
        private readonly float[] _estimatedIm = new float[DtlnAecOptions.FrequencyBins];
        private readonly float[] _estimatedBlock;
        private readonly float[] _enhancedBlock;
        private readonly float[] _zeroHop = new float[DtlnAecOptions.HopLength];
        private readonly float[] _discardHop = new float[DtlnAecOptions.HopLength];

        private readonly int _warmupHops;
        private bool _disposed;

        public DtlnAecProcessor(DtlnAecOnnxRunner runner, int warmupHops)
        {
            _runner = runner ?? throw new ArgumentNullException(nameof(runner));
            _warmupHops = Math.Max(0, warmupHops);
            _micMag = runner.MicMagnitudeBuffer;
            _lpbMag = runner.LpbMagnitudeBuffer;
            _mask = runner.MaskBuffer;
            _estimatedBlock = runner.EstimatedBlockBuffer;
            _lpbWindow = runner.LpbBlockBuffer;
            _enhancedBlock = runner.EnhancedBlockBuffer;
            Reset();
        }

        public void ProcessHop(ReadOnlySpan<float> captureHop, ReadOnlySpan<float> renderHop, Span<float> outputHop)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (captureHop.Length != DtlnAecOptions.HopLength ||
                renderHop.Length != DtlnAecOptions.HopLength ||
                outputHop.Length != DtlnAecOptions.HopLength)
                throw new ArgumentException($"DTLN AEC hop length must be {DtlnAecOptions.HopLength}.");

            ShiftAppend(_micWindow, captureHop);
            ShiftAppend(_lpbWindow, renderHop);
            RunCurrentWindow(outputHop);
        }

        public void Reset()
        {
            Array.Clear(_micWindow);
            Array.Clear(_lpbWindow);
            Array.Clear(_outBuffer);
            _runner.Reset();
            for (int i = 0; i < _warmupHops; i++)
                ProcessWarmupHop();
        }

        private void ProcessWarmupHop()
        {
            ShiftAppend(_micWindow, _zeroHop);
            ShiftAppend(_lpbWindow, _zeroHop);
            RunCurrentWindow(_discardHop);
        }

        private void RunCurrentWindow(Span<float> outputHop)
        {
            _fft.ForwardMagnitude(_micWindow, _micRe, _micIm, _micMag);
            _fft.ForwardMagnitude(_lpbWindow, _lpbRe, _lpbIm, _lpbMag);

            _runner.RunPart1InPlace();

            for (int i = 0; i < DtlnAecOptions.FrequencyBins; i++)
            {
                _estimatedRe[i] = _micRe[i] * _mask[i];
                _estimatedIm[i] = _micIm[i] * _mask[i];
            }
            _fft.Inverse(_estimatedRe, _estimatedIm, _estimatedBlock);
            _runner.RunPart2InPlace();

            ShiftOutputAndOverlap(_enhancedBlock, outputHop);
        }

        private void ShiftOutputAndOverlap(ReadOnlySpan<float> enhancedBlock, Span<float> outputHop)
        {
            int hop = DtlnAecOptions.HopLength;
            int overlap = DtlnAecOptions.BlockLength - hop;
            for (int i = 0; i < overlap; i++)
            {
                float sample = _outBuffer[i + hop] + enhancedBlock[i];
                _outBuffer[i] = sample;
                if (i < hop)
                    outputHop[i] = sample;
            }

            enhancedBlock[overlap..].CopyTo(_outBuffer.AsSpan(overlap));
        }

        private static void ShiftAppend(Span<float> window, ReadOnlySpan<float> hop)
        {
            int shift = hop.Length;
            window[shift..].CopyTo(window);
            hop.CopyTo(window[^shift..]);
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            _runner.Dispose();
        }
    }
}
