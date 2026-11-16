// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes

namespace VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures
{
    public class OnlineFbank : IDisposable
    {
        internal bool _disposed = false;
        internal FeatureOptions _opts = new FeatureOptions();
        internal IFeature _feature;
        private float _sample_rate = 16000.0F;
        private int _num_bins = 80;
        // long-valued absolute frame counter — int would overflow at ~248 days of
        // continuous streaming (100 frames/sec @ 10 ms shift) and turn the n calculation
        // below into a large negative number.
        private long _last_frame_index = 0;
        // Reused per-frame scratch; FeatureShip.GetFeature writes into pData.data which is just a
        // reference to the underlying frame buffer — allocating per frame served no purpose.
        // Not readonly because FeatureShip.GetFeature takes its argument by ref.
        private FeatureData _featureDataScratch = new FeatureData();

        /// <summary>
        /// OnlineFbank
        /// </summary>
        /// <param name="dither"></param>
        /// <param name="snip_edges"></param>
        /// <param name="sample_rate"></param>
        /// <param name="num_bins"></param>
        /// <param name="frame_shift"></param>
        /// <param name="frame_length"></param>
        /// <param name="energy_floor"></param>
        /// <param name="debug_mel"></param>
        /// <param name="window_type">window_type (string): Type of window ('hamming'|'hanning'|'povey'|'rectangular'|'blackman')</param>
        /// <param name="feature_type">feature_type (string): Type of feature ('whisper'|'fbank')</param>
        public OnlineFbank(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming", string feature_type = "fbank")
        {
            _sample_rate = sample_rate;
            _num_bins = num_bins;
            this._opts = FeatureShip.GetFeatureOptions(
                 dither: dither,
                 snip_edges: snip_edges,
                 sample_rate: sample_rate,
                 num_bins: num_bins,
                 frame_shift: frame_shift,
                 frame_length: frame_length,
                 energy_floor: energy_floor,
                 debug_mel: debug_mel,
                 window_type: window_type
                 );
            this._feature = FeatureShip.GetOnlineFeature(this._opts, featureType: feature_type);
        }

        /// <summary>
        /// Get one frame at a time
        /// </summary>
        /// <param name="samples"></param>
        /// <returns></returns>
        public float[] GetFbank(float[] samples) => GetFbank(samples, samples.Length);

        /// <summary>
        /// Compute FBank features from the first <paramref name="sampleCount"/> samples in
        /// <paramref name="samples"/>. Lets callers pass a rented buffer that may be larger
        /// than the live region.
        /// </summary>
        public float[] GetFbank(float[] samples, int sampleCount)
        {
            FeatureShip.AcceptWaveform(_feature, _sample_rate, samples, sampleCount);
            long framesNum = FeatureShip.GetNumFramesReady(_feature);
            // n is the delta between calls; in steady state it equals the frames produced
            // by this call (~ms of audio / 10), so int is safe even with long counters.
            int n = checked((int)(framesNum - _last_frame_index));
            float[] fbanks = new float[n * _num_bins];
            for (long i = _last_frame_index; i < framesNum; i++)
            {
                FeatureShip.GetFeature(_feature, i, ref _featureDataScratch);
                float[] _featureData = _featureDataScratch.data;
                Array.Copy(_featureData, 0, fbanks, (int)(i - _last_frame_index) * _num_bins, _featureData.Length);
            }
            _last_frame_index += n;
            return fbanks;
        }

        /// <summary>
        /// Get all current frames each time
        /// Faster than GetFbank
        /// </summary>
        /// <param name="samples"></param>
        /// <returns></returns>
        public float[] GetFbankIndoor(float[] samples)
        {
            FeatureShip.AcceptWaveform(_feature, _sample_rate, samples, samples.Length);
            long framesNum = FeatureShip.GetNumFramesReady(_feature);
            FeatureDatas featureDatas = new FeatureDatas();
            FeatureShip.GetFeatures(_feature, _last_frame_index, ref featureDatas);
            float[] _fbankDatas = featureDatas.data;
            //Marshal.Copy(fbankDatas.data, _fbankDatas, 0, fbankDatas.data_length);
            _last_frame_index = framesNum - 1;
            return _fbankDatas;
        }
        public void InputFinished()
        {
            FeatureShip.InputFinished(_feature);
        }


        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            // No unmanaged resources to release; this class is fully managed. The Dispose
            // method is preserved so callers wrapping it in `using` get the expected
            // contract, and so the `_disposed` flag can be observed by future guards.
        }
    }
}