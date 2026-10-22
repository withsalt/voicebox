using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Models;
using VoiceBox.Core.Services.AcousticProcessing;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Estimation;

internal sealed class CompositeRenderDelayEstimator : IRenderDelayEstimator
{
    // The correlation fallback is the only stream-local estimator that can correct a
    // wrong external delay hint before the primary estimator refines. A 40 ms window is too short to
    // separate speech autocorrelation lobes, so the analysis window grows with the
    // available history up to 200 ms while keeping the original 40 ms floor.
    private const int MinimumCorrelationAnalysisWindow = 640;

    private const int MaximumCorrelationAnalysisWindow = 3200;

    private readonly IAcousticPathDelayEstimator _primary;

    private readonly float[] _renderHistory;

    private readonly float[] _captureHistory;

    private readonly int _maxDelaySamples;

    private int _writeIndex;

    private int _count;

    private long _totalSamples;

    private int _fallbackCountdown;

    private readonly int _correlationStabilityToleranceSamples;

    private readonly double _minimumCorrelationConfidence;

    private int _correlationCandidateCount;

    private int _correlationCandidateAge;

    private double _correlationCandidateDelay;

    private double _correlationCandidateConfidence;

    public CompositeRenderDelayEstimator(
        IAcousticPathDelayEstimator primary,
        int maxDelayMs = 300,
        int stabilityToleranceMs = 8,
        double minimumCorrelationConfidence = 0.55)
    {
        _primary = primary ?? throw new ArgumentNullException(nameof(primary));
        if (maxDelayMs < 0 || maxDelayMs > 5000)
            throw new ArgumentOutOfRangeException(nameof(maxDelayMs));
        if (stabilityToleranceMs < 0 || stabilityToleranceMs > 5000)
            throw new ArgumentOutOfRangeException(nameof(stabilityToleranceMs));
        if (!double.IsFinite(minimumCorrelationConfidence) ||
            minimumCorrelationConfidence < 0 ||
            minimumCorrelationConfidence > 1)
            throw new ArgumentOutOfRangeException(nameof(minimumCorrelationConfidence));

        _maxDelaySamples = checked(maxDelayMs * 16);
        _correlationStabilityToleranceSamples = checked(stabilityToleranceMs * 16);
        _minimumCorrelationConfidence = minimumCorrelationConfidence;
        int capacity = checked(_maxDelaySamples + MaximumCorrelationAnalysisWindow + 320);
        _renderHistory = new float[Math.Max(2048, capacity)];
        _captureHistory = new float[_renderHistory.Length];
    }

    public RenderDelayEstimate Process(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone)
    {
        if (render.Length != microphone.Length)
            throw new ArgumentException("Render and capture hop lengths must match.", nameof(microphone));

        RenderDelayEstimate result = _primary.TryEstimate(
            render,
            microphone,
            out AcousticPathDelayEstimate primaryEstimate)
            ? new RenderDelayEstimate(
                HasEstimate: true,
                primaryEstimate.DelaySamples,
                primaryEstimate.IsRefined ? 0.95 : 0.65,
                primaryEstimate.IsRefined)
            : default;
        Append(render, microphone);
        if (result.IsRefined)
            return result;

        if (--_fallbackCountdown > 0)
            return result;
        _fallbackCountdown = 5;
        RenderDelayEstimate correlation = TryEstimateCorrelation();
        return TryPromoteCorrelation(in correlation, out RenderDelayEstimate promoted)
            ? promoted
            : result;
    }

    public void Reset()
    {
        _primary.Reset();
        Array.Clear(_renderHistory);
        Array.Clear(_captureHistory);
        _writeIndex = 0;
        _count = 0;
        _totalSamples = 0;
        _fallbackCountdown = 0;
        ResetCorrelationCandidate();
    }

    public void Dispose()
    {
        _primary.Dispose();
    }

    private void Append(ReadOnlySpan<float> render, ReadOnlySpan<float> microphone)
    {
        // Written in runs bounded by the ring wrap so the index bookkeeping happens once
        // per run instead of once per sample. Stored values and their order are unchanged.
        int capacity = _renderHistory.Length;
        int offset = 0;
        int remaining = render.Length;
        while (remaining > 0)
        {
            int run = Math.Min(remaining, capacity - _writeIndex);
            Span<float> renderRun = _renderHistory.AsSpan(_writeIndex, run);
            Span<float> captureRun = _captureHistory.AsSpan(_writeIndex, run);
            for (int i = 0; i < run; i++)
            {
                float renderSample = render[offset + i];
                float captureSample = microphone[offset + i];
                renderRun[i] = float.IsFinite(renderSample) ? renderSample : 0;
                captureRun[i] = float.IsFinite(captureSample) ? captureSample : 0;
            }

            _writeIndex += run;
            if (_writeIndex == capacity)
                _writeIndex = 0;
            _count = Math.Min(capacity, _count + run);
            _totalSamples += run;
            offset += run;
            remaining -= run;
        }
    }

    private RenderDelayEstimate TryEstimateCorrelation()
    {
        const int coarseStep = 16;
        // Half of the buffered history balances statistical reliability against
        // searchable delay range. Covering the full tracked range earlier with a
        // short window was measured on the evaluation corpus and produced more wrong
        // large-delay locks than it prevented.
        int analysisWindow = (int)Math.Clamp(
            _count / 2,
            MinimumCorrelationAnalysisWindow,
            MaximumCorrelationAnalysisWindow);
        if (_count < analysisWindow || _maxDelaySamples == 0)
            return default;

        long historyStart = _totalSamples - _count;
        long captureStart = _totalSamples - analysisWindow;
        if (captureStart < historyStart)
            return default;

        double captureEnergy = 0;
        if (TryResolveWindow(captureStart, analysisWindow, out int captureEnergyIndex))
        {
            float[] captureHistory = _captureHistory;
            int capacity = captureHistory.Length;
            for (int i = 0; i < analysisWindow; i++)
            {
                double sample = captureHistory[captureEnergyIndex];
                captureEnergy += sample * sample;
                if (++captureEnergyIndex == capacity)
                    captureEnergyIndex = 0;
            }
        }
        else
        {
            for (int i = 0; i < analysisWindow; i++)
            {
                double sample = Read(_captureHistory, captureStart + i);
                captureEnergy += sample * sample;
            }
        }
        if (captureEnergy < 1e-6)
            return default;

        int availableMaximumDelay = (int)Math.Min(
            _maxDelaySamples,
            Math.Max(0, captureStart - historyStart));
        double bestScore = 0;
        double secondScore = 0;
        int bestDelay = 0;
        for (int delay = 0; delay <= availableMaximumDelay; delay += coarseStep)
        {
            double score = NormalizedCorrelation(captureStart, delay, analysisWindow, captureEnergy);
            if (score > bestScore)
            {
                secondScore = bestScore;
                bestScore = score;
                bestDelay = delay;
            }
            else if (score > secondScore)
            {
                secondScore = score;
            }
        }

        if (bestScore < 0.12 ||
            bestScore - secondScore < Math.Max(0.015, bestScore * 0.08))
        {
            return default;
        }

        int refinedStart = Math.Max(0, bestDelay - coarseStep);
        int refinedEnd = Math.Min(availableMaximumDelay, bestDelay + coarseStep);
        for (int delay = refinedStart; delay <= refinedEnd; delay++)
        {
            double score = NormalizedCorrelation(captureStart, delay, analysisWindow, captureEnergy);
            if (score > bestScore)
            {
                bestScore = score;
                bestDelay = delay;
            }
        }

        double confidence = Math.Clamp(
            0.55 + (bestScore - 0.12) * 0.75 + (bestScore - secondScore) * 1.5,
            0.55,
            0.95);
        return new RenderDelayEstimate(
            HasEstimate: true,
            bestDelay,
            confidence,
            IsRefined: false,
            IsCorrelationFallback: true);
    }

    private bool TryPromoteCorrelation(
        in RenderDelayEstimate correlation,
        out RenderDelayEstimate promoted)
    {
        if (!correlation.HasEstimate ||
            correlation.Confidence < _minimumCorrelationConfidence)
        {
            if (++_correlationCandidateAge > 3)
                ResetCorrelationCandidate();
            promoted = default;
            return false;
        }

        _correlationCandidateAge = 0;
        int toleranceSamples = Math.Max(16, _correlationStabilityToleranceSamples);
        if (_correlationCandidateCount == 0 ||
            Math.Abs(correlation.DelaySamples - _correlationCandidateDelay) > toleranceSamples)
        {
            _correlationCandidateCount = 1;
            _correlationCandidateDelay = correlation.DelaySamples;
            _correlationCandidateConfidence = correlation.Confidence;
        }
        else
        {
            _correlationCandidateCount++;
            double weight = 1.0 / _correlationCandidateCount;
            _correlationCandidateDelay +=
                (correlation.DelaySamples - _correlationCandidateDelay) * weight;
            _correlationCandidateConfidence +=
                (correlation.Confidence - _correlationCandidateConfidence) * weight;
        }

        if (_correlationCandidateCount < 3)
        {
            promoted = default;
            return false;
        }

        promoted = new RenderDelayEstimate(
            HasEstimate: true,
            DelaySamples: _correlationCandidateDelay,
            Confidence: _correlationCandidateConfidence,
            IsCorrelationFallback: true);
        ResetCorrelationCandidate();
        return true;
    }

    private void ResetCorrelationCandidate()
    {
        _correlationCandidateCount = 0;
        _correlationCandidateAge = 0;
        _correlationCandidateDelay = 0;
        _correlationCandidateConfidence = 0;
    }

    private double NormalizedCorrelation(
        long captureStart,
        int delay,
        int length,
        double captureEnergy)
    {
        long renderStart = captureStart - delay;
        double renderEnergy = 0;
        double dot = 0;
        // Both windows are fully in range for every delay the caller scans, so the ring
        // offsets can be resolved once instead of re-deriving bounds per sample. The
        // accumulation order is still i = 0..length-1, so the sums are unchanged.
        if (TryResolveWindow(captureStart, length, out int captureIndex) &&
            TryResolveWindow(renderStart, length, out int renderIndex))
        {
            float[] captureHistory = _captureHistory;
            float[] renderHistory = _renderHistory;
            int capacity = captureHistory.Length;
            for (int i = 0; i < length; i++)
            {
                double capture = captureHistory[captureIndex];
                double render = renderHistory[renderIndex];
                dot += capture * render;
                renderEnergy += render * render;
                if (++captureIndex == capacity)
                    captureIndex = 0;
                if (++renderIndex == capacity)
                    renderIndex = 0;
            }
        }
        else
        {
            for (int i = 0; i < length; i++)
            {
                double capture = Read(_captureHistory, captureStart + i);
                double render = Read(_renderHistory, renderStart + i);
                dot += capture * render;
                renderEnergy += render * render;
            }
        }

        double denominator = Math.Sqrt(captureEnergy * renderEnergy);
        return denominator > 1e-9 ? Math.Abs(dot / denominator) : 0;
    }

    /// <summary>
    /// Resolves the physical start index for a fully-retained window. Returns false when
    /// any sample of the window falls outside the retained history, in which case the
    /// caller must use the bounds-checking <see cref="Read"/> path that substitutes zero.
    /// </summary>
    private bool TryResolveWindow(long absoluteStart, int length, out int startIndex)
    {
        startIndex = 0;
        long historyStart = _totalSamples - _count;
        long offset = absoluteStart - historyStart;
        if (length <= 0 || offset < 0 || offset + length > _count)
            return false;

        int oldestIndex = _writeIndex - _count;
        if (oldestIndex < 0)
            oldestIndex += _renderHistory.Length;
        int index = oldestIndex + checked((int)offset);
        if (index >= _renderHistory.Length)
            index -= _renderHistory.Length;
        startIndex = index;
        return true;
    }

    private float Read(float[] buffer, long absoluteSample)
    {
        long start = _totalSamples - _count;
        long offset = absoluteSample - start;
        if (offset < 0 || offset >= _count)
            return 0;
        int oldestIndex = _writeIndex - _count;
        if (oldestIndex < 0)
            oldestIndex += _renderHistory.Length;
        int index = oldestIndex + checked((int)offset);
        if (index >= _renderHistory.Length)
            index -= _renderHistory.Length;
        return buffer[index];
    }
}
