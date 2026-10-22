using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using Microsoft.Extensions.Options;
using VoiceBox.Assistant.Api.Configuration;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Contracts;
using VoiceBox.Assistant.Modules.BargeIn.RenderAware.Estimation;
using VoiceBox.Core.Services.AcousticProcessing;
using VoiceBox.DTD.RenderAware;

namespace VoiceBox.Assistant.Modules.BargeIn.RenderAware.Controllers;

internal sealed class RenderAwarePlaybackBargeInDetectorFactory : IPlaybackBargeInDetectorFactory
{
    private readonly VoiceAssistantBargeInOptions _options;

    private readonly IAcousticPathDelayEstimatorFactory _delayEstimatorFactory;

    private readonly ILogger _logger;

    private readonly TimeProvider _timeProvider;

    public RenderAwarePlaybackBargeInDetectorFactory(
        VoiceAssistantBargeInOptions options,
        IAcousticPathDelayEstimatorFactory delayEstimatorFactory,
        ILogger? logger = null,
        TimeProvider? timeProvider = null)
    {
        _options = options ?? throw new ArgumentNullException(nameof(options));
        _delayEstimatorFactory = delayEstimatorFactory ?? throw new ArgumentNullException(nameof(delayEstimatorFactory));
        _logger = logger ?? NullLogger.Instance;
        _timeProvider = timeProvider ?? TimeProvider.System;
    }

    public IPlaybackBargeInDetector Create()
    {
        IRenderAwareDtdService renderAwareDtdService = CreateDtd();
        IRenderDelayEstimator? renderDelayEstimator = null;
        try
        {
            renderDelayEstimator = CreateDelayEstimator();
            return new RenderAwarePlaybackBargeInDetector(_options.Dtd, _options.RenderAlignment, renderAwareDtdService, renderDelayEstimator, _logger, _timeProvider);
        }
        catch
        {
            try
            {
                renderDelayEstimator?.Dispose();
            }
            catch
            {
            }
            try
            {
                renderAwareDtdService.Dispose();
            }
            catch
            {
            }
            throw;
        }
    }

    private IRenderAwareDtdService CreateDtd()
        => new RenderAwareDtdService(Options.Create(_options.Dtd.Model));

    private IRenderDelayEstimator CreateDelayEstimator()
        => new CompositeRenderDelayEstimator(
            _delayEstimatorFactory.Create(sampleRate: 16000, channelCount: 1)
                ?? throw new InvalidOperationException(
                    "The acoustic-path delay estimator factory returned null."),
            _options.RenderAlignment.MaxTrackedDelayMs,
            minimumCorrelationConfidence:
                _options.RenderAlignment.MinimumAlignmentConfidence);
}
