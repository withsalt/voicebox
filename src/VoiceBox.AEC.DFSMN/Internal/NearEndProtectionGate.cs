using VoiceBox.AEC.DFSMN.Models;

namespace VoiceBox.AEC.DFSMN.Internal
{
    /// <summary>
    /// Turns the per-hop near-end VAD probability into a protection envelope in [0, 1].
    /// 0 = echo-only (legacy full suppression), 1 = near-end confirmed (suppression bounded).
    ///
    /// Attack is instantaneous: the probability is produced by the same inference as the mask it
    /// gates, so the very first confirmed hop is already protected. Release is a one-pole decay so
    /// short gaps between words do not let the mask slam back to full suppression mid-utterance.
    /// </summary>
    internal sealed class NearEndProtectionGate
    {
        // 包络高于该值时维持自适应冻结: 给冻结提供与保护释放一致的 hangover,
        // 概率在阈值附近逐 hop 抖动时 NLMS 不会在叠说中途反复恢复自适应。
        private const float FreezeEnvelopeFloor = 0.5f;

        private readonly bool _enabled;
        private readonly float _engage;
        private readonly float _range;
        private readonly float _release;
        private readonly float _maskFloorLinear;      // legacy floor (0 when unlimited)
        private readonly float _protectedFloorLinear; // floor at full protection
        private readonly bool _gateResidual;
        private readonly bool _freezeAdaptation;
        private readonly float _freezeProbability;

        private float _envelope;

        public NearEndProtectionGate(
            DfsmnAecNearEndProtectionOptions options,
            float? maskFloorDb,
            int hopLength,
            int sampleRate)
        {
            ArgumentNullException.ThrowIfNull(options);

            _enabled = options.Enabled;
            _engage = options.EngageProbability;
            _range = Math.Max(1e-6f, options.FullProbability - options.EngageProbability);
            float hopMs = 1000f * hopLength / sampleRate;
            _release = DspMath.OnePoleCoefficient(hopMs, options.ReleaseMs);
            _maskFloorLinear = maskFloorDb is { } db ? DspMath.AttenuationDbToGain(db) : 0f;
            _protectedFloorLinear = DspMath.AttenuationDbToGain(options.MaxSuppressionDb);
            _gateResidual = options.GateResidualSuppressor;
            _freezeAdaptation = options.FreezeAdaptation;
            _freezeProbability = options.FreezeProbability;
        }

        /// <summary>Current protection envelope, for diagnostics/tests.</summary>
        public float Envelope => _envelope;

        /// <summary>
        /// Advance the envelope with this hop's near-end probability. Returns the envelope to use
        /// for this same hop (instant attack, smoothed release).
        /// </summary>
        public float Observe(float vadProbability)
        {
            if (!_enabled)
                return 0f;

            float p = float.IsFinite(vadProbability) ? Math.Clamp(vadProbability, 0f, 1f) : 0f;
            float instant = Math.Clamp((p - _engage) / _range, 0f, 1f);
            _envelope = MathF.Max(instant, _envelope * _release);
            if (_envelope < 1e-4f)
                _envelope = 0f;
            return _envelope;
        }

        /// <summary>
        /// Effective linear mask floor for the current envelope: legacy floor on echo-only hops,
        /// the protected floor at full protection, linear blend in between. Monotonic in the
        /// envelope because the protected floor is always the larger (less suppressing) bound.
        /// </summary>
        public float MaskFloor(float envelope)
        {
            if (!_enabled || envelope <= 0f)
                return _maskFloorLinear;

            float floor = _maskFloorLinear + (_protectedFloorLinear - _maskFloorLinear) * envelope;
            return MathF.Max(floor, _maskFloorLinear);
        }

        /// <summary>
        /// Gain limit for the scalar residual suppressor: 0 = unrestricted (legacy), values toward
        /// 1 forbid extra attenuation while the near-end talker is confirmed.
        /// </summary>
        public float ResidualGainFloor(float envelope)
            => _enabled && _gateResidual ? Math.Clamp(envelope, 0f, 1f) : 0f;

        /// <summary>
        /// Whether linear-AEC adaptation should freeze for the NEXT hop. Complements the
        /// energy-based DTD, which misses soft talkers over loud playback (it needs mic energy
        /// to exceed the expected echo by ~6 dB). Freezes on the raw per-hop probability OR
        /// while the protection envelope is still engaged (release hangover): single-hop
        /// probability dips mid-utterance must not let NLMS adapt into the talker.
        /// Call after <see cref="Observe"/> so the envelope reflects the current hop.
        /// </summary>
        public bool ShouldFreezeAdaptation(float vadProbability)
            => _enabled && _freezeAdaptation &&
               ((float.IsFinite(vadProbability) && vadProbability >= _freezeProbability) ||
                _envelope >= FreezeEnvelopeFloor);

        public void Reset() => _envelope = 0f;
    }
}
