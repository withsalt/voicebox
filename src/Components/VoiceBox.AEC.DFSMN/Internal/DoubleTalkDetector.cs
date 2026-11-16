using VoiceBox.AEC.DFSMN.Models;

namespace VoiceBox.AEC.DFSMN.Internal
{
    /// <summary>
    /// Lightweight envelope-coupling double-talk detector (DTD).
    ///
    /// It learns the render→mic echo coupling gain on echo-only hops, then flags a hop as
    /// double-talk when the microphone level exceeds what that coupling alone would explain —
    /// i.e. near-end (user) speech has been added on top of the echo. The caller freezes the
    /// adaptive echo canceller's coefficient update during flagged hops, so a barging-in talker
    /// does not pull the filter (and the neural mask fed by its echo estimate) into suppressing
    /// the near-end voice.
    ///
    /// Decisions use raw hop energies only — no per-sample state — so it is cheap. Crucially, on
    /// echo-only signals it never flags, leaving steady-state adaptation byte-for-byte as if the
    /// detector were absent.
    /// </summary>
    internal sealed class DoubleTalkDetector
    {
        private readonly bool _enabled;
        private readonly float _renderFloor;     // RMS below which the far-end is treated as silent
        private readonly float _nearThreshold;   // mic / expected-echo ratio that declares near-end
        private readonly float _couplingSmooth;  // EMA factor for the coupling estimate
        private readonly int _hangoverHops;

        private float _coupling = -1f;           // < 0 until bootstrapped on the first far-active hop
        private int _hold;                       // remaining hangover hops

        public DoubleTalkDetector(DfsmnAecDoubleTalkOptions options, int hopLength, int sampleRate)
        {
            ArgumentNullException.ThrowIfNull(options);

            _enabled = options.Enabled;
            _renderFloor = options.RenderActivityThreshold;
            _nearThreshold = options.NearEndThreshold;
            _couplingSmooth = options.CouplingSmoothing;

            float hopMs = 1000f * hopLength / sampleRate;
            _hangoverHops = hopMs > 0f ? Math.Max(1, (int)MathF.Ceiling(options.HangoverMs / hopMs)) : 1;
        }

        /// <summary>
        /// Evaluate one hop. Returns true when it is double-talk and the linear AEC should freeze
        /// its coefficient update for this hop.
        /// </summary>
        /// <param name="capture">Raw microphone hop (near-end + echo + noise).</param>
        /// <param name="render">Echo-aligned far-end reference hop (delayed render).</param>
        /// <param name="externalNearEndHint">
        /// External near-end evidence for this hop (e.g. the DFSMN model's VAD probability from the
        /// previous hop). Treated as a confirmed near-end: freezes with hangover and keeps the
        /// coupling EMA from absorbing the talker's energy — soft speech below the energy
        /// threshold would otherwise slowly inflate the coupling and desensitize the detector.
        /// </param>
        public bool IsDoubleTalk(ReadOnlySpan<float> capture, ReadOnlySpan<float> render, bool externalNearEndHint = false)
        {
            if (!_enabled)
                return false;

            float renderRms = Rms(render);

            // No far-end energy → no echo to confuse with near-end. Leave the canceller free to
            // adapt (it self-protects against a ~zero reference via its NLMS epsilon), but keep
            // honoring an in-flight hangover from a double-talk burst that just ended.
            if (renderRms <= _renderFloor)
                return CountDownHold();

            float micRms = Rms(capture);

            // Bootstrap the coupling on the first far-active hop, assuming it is echo-only. One hop
            // of potential mis-estimate is negligible (the linear filter starts from zero anyway)
            // and the EMA below corrects it on subsequent echo-only hops. A hop with confirmed
            // near-end must not seed the coupling — and the confirmation carries the same hangover
            // as the normal path, so a single-hop probability dip mid-utterance cannot slip a
            // speech-contaminated ratio into the seed either.
            if (_coupling < 0f)
            {
                if (externalNearEndHint)
                {
                    _hold = _hangoverHops;
                    return true;
                }
                if (_hold > 0)
                {
                    _hold--;
                    return true;
                }
                _coupling = SafeRatio(micRms, renderRms);
                return false;
            }

            float expectedEchoRms = _coupling * renderRms;
            bool nearPresent = externalNearEndHint || micRms > expectedEchoRms * _nearThreshold;
            if (nearPresent)
                _hold = _hangoverHops;

            bool frozen = _hold > 0;
            if (frozen)
            {
                _hold--;
            }
            else
            {
                // Echo-only hop (no near-end, not in hangover): track the slowly-varying coupling.
                _coupling = _couplingSmooth * _coupling + (1f - _couplingSmooth) * SafeRatio(micRms, renderRms);
            }

            return frozen;
        }

        public void Reset()
        {
            _coupling = -1f;
            _hold = 0;
        }

        private bool CountDownHold()
        {
            if (_hold <= 0)
                return false;
            _hold--;
            return true;
        }

        private static float Rms(ReadOnlySpan<float> x)
        {
            if (x.IsEmpty)
                return 0f;
            double acc = 0;
            for (int i = 0; i < x.Length; i++)
                acc += (double)x[i] * x[i];
            return (float)Math.Sqrt(acc / x.Length);
        }

        private static float SafeRatio(float numerator, float denominator)
            => denominator > 1e-12f ? numerator / denominator : 0f;
    }
}
