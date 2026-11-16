// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System.Collections.Concurrent;

namespace VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures
{
    public class FbankComputer : IFeatureComputer
    {
        public FeatureOptions _opts { get; }
        private float _logEnergyFloor;
        private readonly Rfft _rfft;
        private readonly int _paddedWindowSize;

        // Cache of MelBanks2 instances shared across FbankComputer instances (and therefore
        // across ASR sessions). MelBanks2 is immutable after construction, so the shared
        // instance is safe to read concurrently.
        private static readonly ConcurrentDictionary<MelBanksCacheKey, MelBanks2> _sharedMelBanks = new();

        public FbankComputer(FeatureOptions opts)
        {
            _opts = opts;
            _paddedWindowSize = opts.FrameOpts.PaddedWindowSize();
            _rfft = new Rfft(_paddedWindowSize);

            if (opts.EnergyFloor > 0.0f)
            {
                _logEnergyFloor = (float)Math.Log(opts.EnergyFloor);
            }
            // Eagerly populate the cache for vtlnWarp = 1.0 (the only value FireRed uses).
            _ = GetMelBanks(1.0f);
        }

        private MelBanks2 GetMelBanks(float vtlnWarp)
        {
            var key = new MelBanksCacheKey(
                _opts.MelOpts.numBins,
                _opts.MelOpts.lowFreq,
                _opts.MelOpts.highFreq,
                _opts.MelOpts.vtlnLow,
                _opts.MelOpts.vtlnHigh,
                _opts.MelOpts.htkMode,
                _opts.MelOpts.isLibrosa,
                _opts.MelOpts.norm ?? string.Empty,
                _opts.MelOpts.useSlaneyMelScale,
                _opts.MelOpts.floorToIntBin,
                _opts.FrameOpts.SampFreq,
                _paddedWindowSize,
                vtlnWarp);
            return _sharedMelBanks.GetOrAdd(key, k => new MelBanks2(_opts.MelOpts, _opts.FrameOpts, vtlnWarp));
        }

        public int Dim()
        {
            return _opts.MelOpts.numBins + (_opts.UseEnergy ? 1 : 0);
        }

        public bool NeedRawLogEnergy()
        {
            return _opts.UseEnergy && _opts.RawEnergy;
        }

        public void Compute(float? signalRawLogEnergy, float vtlnWarp, Span<float> signalFrame, Span<float> feature)
        {
            if (signalRawLogEnergy == null) return;
            var melBanks = GetMelBanks(vtlnWarp);
            if (signalFrame.Length != _paddedWindowSize)
            {
                throw new ArgumentException("Invalid signal frame size.");
            }
            // Compute energy after window function (not the raw one).
            if (_opts.UseEnergy && !_opts.RawEnergy)
            {
                signalRawLogEnergy = (float)Math.Log(Math.Max(Utils.FeatureFunctions.InnerProduct(signalFrame, signalFrame, signalFrame.Length), float.Epsilon));
            }
            _rfft.Compute(signalFrame);  // signal_frame is modified in-place
            Utils.FeatureFunctions.ComputePowerSpectrum(signalFrame);
            // Use magnitude instead of power if requested.
            if (!_opts.UsePower)
            {
                int half = signalFrame.Length / 2 + 1;
                for (int i = 0; i < half; i++)
                {
                    signalFrame[i] = (float)Math.Sqrt(signalFrame[i]);
                }
            }
            // Write mel energies directly into the appropriate slice of `feature`. The legacy
            // code allocated a scratch + double-copied into/out of feature; for the htkCompat
            // case it produced a feature.Length-sized scratch where only the first numBins
            // were actually computed.
            int numBins = _opts.MelOpts.numBins;
            int melOffset = _opts.UseEnergy && !_opts.HtkCompat ? 1 : 0;
            Span<float> melOut = feature.Slice(melOffset, numBins);
            // Sum with mel filter banks over the power spectrum (uses only the first n/2+1 bins).
            melBanks.Compute(signalFrame, melOut);
            if (_opts.UseLogFbank)
            {
                for (int i = 0; i < numBins; i++)
                {
                    melOut[i] = (float)Math.Log(Math.Max(melOut[i], float.Epsilon));
                }
            }
            // Copy energy as first value (or the last, if htk_compat == true).
            if (_opts.UseEnergy)
            {
                if (_opts.EnergyFloor > 0.0 && signalRawLogEnergy < _logEnergyFloor)
                {
                    signalRawLogEnergy = _logEnergyFloor;
                }
                int energyIndex = _opts.HtkCompat ? numBins : 0;
                feature[energyIndex] = (float)signalRawLogEnergy;
            }
        }

        public FrameExtractionOptions GetFrameOptions()
        {
            return _opts.FrameOpts;
        }

        internal readonly record struct MelBanksCacheKey(
            int NumBins,
            float LowFreq,
            float HighFreq,
            float VtlnLow,
            float VtlnHigh,
            bool HtkMode,
            bool IsLibrosa,
            string Norm,
            bool UseSlaneyMelScale,
            bool FloorToIntBin,
            float SampFreq,
            int PaddedWindowSize,
            float VtlnWarpFactor);
    }
}
