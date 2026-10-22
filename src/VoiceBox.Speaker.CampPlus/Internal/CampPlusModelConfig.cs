using System.Globalization;

namespace VoiceBox.Speaker.CampPlus.Internal
{
    /// <summary>
    /// 模型目录 config.ini 的强类型视图 (扁平 key=value, 忽略 section, 与 DNR 系列约定一致)。
    /// 由导出脚本 scripts/campplus-onnx-export/01_export_onnx.py 生成。
    /// </summary>
    internal sealed class CampPlusModelConfig
    {
        public string ModelFile { get; private set; } = "campplus.onnx";
        public int SampleRate { get; private set; } = 16000;
        public int FeatureDim { get; private set; } = 80;
        public int EmbeddingSize { get; private set; } = 192;
        public float Threshold { get; private set; } = 0.31f;

        public int FrameLengthMs { get; private set; } = 25;
        public int FrameShiftMs { get; private set; } = 10;
        public int NumMelBins { get; private set; } = 80;
        public string WindowType { get; private set; } = "povey";
        public float Dither { get; private set; }
        public float Preemphasis { get; private set; } = 0.97f;
        public bool RemoveDcOffset { get; private set; } = true;
        public bool SnipEdges { get; private set; } = true;
        public float LowFreq { get; private set; } = 20f;
        public float HighFreq { get; private set; }     // 0 = Nyquist (kaldi 约定)
        public float WaveformScale { get; private set; } = 32768f;
        public bool ApplyCmn { get; private set; } = true;

        public static CampPlusModelConfig Load(string configPath)
        {
            if (!File.Exists(configPath))
                throw new FileNotFoundException($"CampPlus model config not found: {configPath}");
            var map = ParseIni(configPath);

            var cfg = new CampPlusModelConfig
            {
                ModelFile = map.TryGetValue("model_file", out var mf) && !string.IsNullOrWhiteSpace(mf) ? mf : "campplus.onnx",
                SampleRate = GetInt(map, "sample_rate", 16000, configPath),
                FeatureDim = GetInt(map, "feature_dim", 80, configPath),
                EmbeddingSize = GetInt(map, "embedding_size", 192, configPath),
                Threshold = GetFloat(map, "threshold", 0.31f, configPath),
                FrameLengthMs = GetInt(map, "frame_length_ms", 25, configPath),
                FrameShiftMs = GetInt(map, "frame_shift_ms", 10, configPath),
                NumMelBins = GetInt(map, "num_mel_bins", 80, configPath),
                WindowType = map.TryGetValue("window_type", out var w) ? w.ToLowerInvariant() : "povey",
                Dither = GetFloat(map, "dither", 0f, configPath),
                Preemphasis = GetFloat(map, "preemphasis", 0.97f, configPath),
                RemoveDcOffset = GetInt(map, "remove_dc_offset", 1, configPath) != 0,
                SnipEdges = GetInt(map, "snip_edges", 1, configPath) != 0,
                LowFreq = GetFloat(map, "low_freq", 20f, configPath),
                HighFreq = GetFloat(map, "high_freq", 0f, configPath),
                WaveformScale = GetFloat(map, "waveform_scale", 32768f, configPath),
                ApplyCmn = GetInt(map, "apply_cmn", 1, configPath) != 0,
            };
            cfg.Validate(configPath);
            return cfg;
        }

        /// <summary>校验配置在本实现支持的范围内 (fbank 前端只实现了导出脚本使用的参数组合)。</summary>
        private void Validate(string configPath)
        {
            if (SampleRate != 16000)
                throw new InvalidOperationException($"Unsupported sample_rate {SampleRate} (expect 16000). [{configPath}]");
            if (FeatureDim is < 8 or > 256)
                throw new InvalidOperationException($"Invalid feature_dim {FeatureDim}. [{configPath}]");
            if (EmbeddingSize is < 8 or > 4096)
                throw new InvalidOperationException($"Invalid embedding_size {EmbeddingSize}. [{configPath}]");
            if (FrameLengthMs != 25 || FrameShiftMs != 10)
                throw new InvalidOperationException(
                    $"Unsupported frame config {FrameLengthMs}/{FrameShiftMs}ms (expect 25/10). [{configPath}]");
            if (WindowType is not ("povey" or "hamming"))
                throw new InvalidOperationException($"Unsupported window_type '{WindowType}'. [{configPath}]");
            if (Dither != 0f)
                throw new InvalidOperationException($"Unsupported dither {Dither} (expect 0). [{configPath}]");
            if (!SnipEdges)
                throw new InvalidOperationException($"Unsupported snip_edges=0. [{configPath}]");
            // StreamingKaldiFbank 将 mel 频带写死为 20Hz~Nyquist; 配置声明其他值时必须响亮失败而非静默失配
            if (LowFreq != 20f || HighFreq != 0f)
                throw new InvalidOperationException(
                    $"Unsupported mel range low_freq={LowFreq}/high_freq={HighFreq} (expect 20/0). [{configPath}]");
            if (NumMelBins != FeatureDim)
                throw new InvalidOperationException(
                    $"num_mel_bins {NumMelBins} != feature_dim {FeatureDim}. [{configPath}]");
            if (!(Preemphasis >= 0f && Preemphasis <= 1f))
                throw new InvalidOperationException($"Invalid preemphasis {Preemphasis}. [{configPath}]");
            if (!(WaveformScale > 0f) || !float.IsFinite(WaveformScale))
                throw new InvalidOperationException($"Invalid waveform_scale {WaveformScale}. [{configPath}]");
            if (!(Threshold >= -1f && Threshold <= 1f))
                throw new InvalidOperationException($"Invalid threshold {Threshold}. [{configPath}]");
        }

        // ---- config.ini 解析 (扁平 key=value, 忽略 section; 与 Hush/DNR 系列一致) ----
        private static Dictionary<string, string> ParseIni(string path)
        {
            var map = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string raw in File.ReadLines(path))
            {
                string line = raw.Trim();
                if (line.Length == 0 || line[0] == '[' || line[0] == '#' || line[0] == ';') continue;
                int eq = line.IndexOf('=');
                if (eq <= 0) continue;
                map[line[..eq].Trim()] = line[(eq + 1)..].Trim();
            }
            return map;
        }

        private static int GetInt(
            Dictionary<string, string> cfg,
            string key,
            int fallback,
            string configPath)
        {
            if (!cfg.TryGetValue(key, out string? value)) return fallback;
            if (int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
                return parsed;
            throw new InvalidOperationException(
                $"Invalid integer value '{value}' for {key}. [{configPath}]");
        }

        private static float GetFloat(
            Dictionary<string, string> cfg,
            string key,
            float fallback,
            string configPath)
        {
            if (!cfg.TryGetValue(key, out string? value)) return fallback;
            if (float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
                return parsed;
            throw new InvalidOperationException(
                $"Invalid floating-point value '{value}' for {key}. [{configPath}]");
        }
    }
}
