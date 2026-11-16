// See https://github.com/manyeyes for more information
// Copyright (c)  2024 by manyeyes
using System.Text;

namespace VoiceBox.ASR.FireRedASR2.Internal.SpeechFeatures
{
    // 模拟 C++ 结构体的类
    public class FeatureData
    {
        public float[] data = Array.Empty<float>();
        public int data_length;
    }

    public class FeatureDatas
    {
        public float[] data = Array.Empty<float>();
        public int data_length;
    }

    public struct FeatureOptions
    {
        public FrameExtractionOptions FrameOpts = new FrameExtractionOptions();
        // append an extra dimension with energy to the filter banks
        public MelBanksOptions MelOpts = new MelBanksOptions();
        public bool UseEnergy = false;
        public float EnergyFloor = 0.0f;  // active iff use_energy==true
        // If true, compute log_energy before preemphasis and windowing
        // If false, compute log_energy after preemphasis ans windowing
        public bool RawEnergy = true;  // active iff use_energy==true
        // If true, put energy last (if using energy)
        // If false, put energy first
        public bool HtkCompat = false;  // active iff use_energy==true

        // if true (default), produce log-filterbank, else linear
        public bool UseLogFbank = true;
        // if true (default), use power in filterbank
        // analysis, else magnitude.
        public bool UsePower = true;
        public FeatureOptions()
        {
            MelOpts.numBins = 23;
        }
        public override string ToString()
        {
            StringBuilder os = new StringBuilder();
            os.AppendLine("frame_opts: ");
            os.AppendLine(FrameOpts.ToString());
            os.AppendLine();

            os.AppendLine("mel_opts: ");
            os.AppendLine(MelOpts.ToString());

            os.AppendLine($"use_energy: {UseEnergy}");
            os.AppendLine($"energy_floor: {EnergyFloor}");
            os.AppendLine($"raw_energy: {RawEnergy}");
            os.AppendLine($"htk_compat: {HtkCompat}");
            os.AppendLine($"use_log_fbank: {UseLogFbank}");
            os.AppendLine($"use_power: {UsePower}");

            return os.ToString();
        }
    }

    public class FeatureShip
    {
        // No global lock: each IFeature instance is owned by exactly one session,
        // and per-session feature extraction is single-threaded (Channel SingleReader=true).
        // The legacy lock(mutex) here serialised every session through one global lock,
        // capping concurrency at 1.

        public static FeatureOptions GetFeatureOptions(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift = 10.0f, float frame_length = 25.0f, float energy_floor = 0.0f, bool debug_mel = false, string window_type = "hamming")//(float dither, bool snip_edges, float sample_rate, int num_bins, float frame_shift, float frame_length, float energy_floor, bool debug_mel, string window_type)
        {
            FeatureOptions opts = new FeatureOptions();
            opts.FrameOpts.Dither = dither;
            opts.FrameOpts.SnipEdges = snip_edges;
            opts.FrameOpts.SampFreq = sample_rate;
            opts.FrameOpts.WindowType = window_type;
            opts.FrameOpts.FrameShiftMs = frame_shift;
            opts.FrameOpts.FrameLengthMs = frame_length;
            opts.MelOpts.numBins = num_bins;
            opts.MelOpts.debugMel = debug_mel;
            opts.EnergyFloor = energy_floor;
            return opts;
        }

        public static IFeature GetOnlineFeature(FeatureOptions opts, string featureType = "fbank")
        {
            IFeature? feature = null;
            if (featureType == "fbank")
            {
                feature = new FbankFeature(new FbankComputer(opts));
            }
            else if (featureType == "whisper")
            {
                feature = new WhisperFeature(new WhisperComputer(opts));
            }
            else
            {
                throw new NotImplementedException();
            }
            return feature;
        }

        public static void AcceptWaveform(IFeature feature, float sample_rate, float[] samples, int samples_size)
        {
            feature.AcceptWaveform(sample_rate, samples, samples_size);
        }

        public static void InputFinished(IFeature feature)
        {
            feature.InputFinished();
        }

        public static long GetNumFramesReady(IFeature feature)
        {
            return feature.NumFramesReady();
        }

        public static void GetFeature(IFeature feature, long currFrameIndex, ref FeatureData pData)
        {
            long n = feature.NumFramesReady();
            if (n <= 0)
            {
                throw new Exception("Please first call AcceptWaveform()");
            }
            int discard_num = currFrameIndex == 0 ? 0 : 1;
            int feature_dim = feature.Dim();
            float[] f = feature.GetFrame(currFrameIndex);
            pData.data_length = feature_dim;
            pData.data = f;
            feature.Pop(discard_num);
        }

        public static void GetFeatures(IFeature feature, long lastFrameIndex, ref FeatureDatas pData)
        {
            long n = feature.NumFramesReady();
            if (n <= 0)
            {
                throw new Exception("Please first call AcceptWaveform()");
            }
            List<float> features = GetFrames(feature, lastFrameIndex);
            pData.data = features.ToArray();
            pData.data_length = features.Count;
        }

        private static List<float> GetFrames(IFeature feature, long lastFrameIndex)
        {
            long n = feature.NumFramesReady();
            if (n - lastFrameIndex < 0)
            {
                throw new Exception("Please first call AcceptWaveform()");
            }
            long framesNum = lastFrameIndex == 0 ? n : n - (lastFrameIndex + 1);
            int discard_num = checked((int)(lastFrameIndex == 0 ? n : n - (lastFrameIndex + 1)));
            int feature_dim = feature.Dim();
            List<float> features = new List<float>(checked((int)(framesNum * feature_dim)));
            long currFrameIndex = lastFrameIndex == 0 ? 0 : lastFrameIndex + 1;
            for (long i = currFrameIndex; i < n; i++)
            {
                float[] f = feature.GetFrame(i);
                features.AddRange(f);
            }
            feature.Pop(discard_num);
            return features;
        }

    }
}