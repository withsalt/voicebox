using VoiceBox.Core.Onnx;

namespace VoiceAssistantDemo
{
    internal class AudioDefaults
    {
        public const int SampleRate = 16000;

        public const int ChannelCount = 1;

        /// <summary>
        /// 采集主链降噪模式。通过环境变量 VOICEBOX_NOISE_SUPPRESSION 选择：
        /// none = 仅 AEC；webrtc = AEC + WebRTC NS；dnr = AEC + 独立 DPDFNet DNR。
        /// WebRTC NS 与独立 DNR 有意互斥。
        /// </summary>
        public static CaptureNoiseSuppressionMode NoiseSuppressionMode { get; } = ResolveNoiseSuppressionMode();

        public static bool UsesWebRtcNoiseSuppression=> AudioDefaults.NoiseSuppressionMode == CaptureNoiseSuppressionMode.WebRtc;

        public static bool UsesIndependentDnr=> AudioDefaults.NoiseSuppressionMode == CaptureNoiseSuppressionMode.IndependentDnr;

        public const string BASE_MODEL_DIR = "D:\\ML\\Audio\\voicebox_models\\v1";

        public static string WakeupKwsPath = Path.Combine(AppContext.BaseDirectory, "runtimes", "models", "kws", "kws-xiaoyun-onnx");

        public static string WakeupWord = "小云小云";

        /// <summary>
        /// 模型名称
        /// </summary>
        /// <remarks>
        /// "doubao-seed-2-1-turbo-260628"; "deepseek-v4-1-flash-260910"; "doubao-seed-2-1-lite-260915";
        /// </remarks>
        public static string LlmModelName = "doubao-seed-2-1-turbo-260628";

        /// <summary>
        /// 火山引擎
        /// </summary>
        public static Uri LlmEndpoint = new Uri("https://ark.cn-beijing.volces.com/api/v3");

        public static string AssistantName = ConvertWakeupWordToAssistantName(WakeupWord);

        public static string LlmOnnxModelPath = Path.Combine(BASE_MODEL_DIR, "llm", "genai", "Qwen3-0.6B-ONNX-INT4-CPU");

        public static string LlmGgufModelPath = Path.Combine(BASE_MODEL_DIR, "llm", "llama", "Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf");

        public static string Qwen3AsrPath = Path.Combine(BASE_MODEL_DIR, "asr", "qwen3", "qwen3-asr-1.7B-onnx-fp16");

        public static string Confucius4AsrPath = Path.Combine(BASE_MODEL_DIR, "asr", "Confucius4-R2T2", "fp16");

        public static string Qwen3TtsPath = Path.Combine(BASE_MODEL_DIR, "tts", "qwen3", "qwen3-tts-0.6b-onnx-fp32");

        public static string FireRedAsr2Path = Path.Combine(BASE_MODEL_DIR, "asr", "fireredasr2-aed", "fireredasr2-aed-large-zh-en-int8-onnx-offline-20260212");

        public static string MossNanoTtsPath = Path.Combine(BASE_MODEL_DIR, "tts", "open-moss", "MOSS-TTS-Nano-100M-ONNX");

        public static string VoxCPMTtsPath = Path.Combine(BASE_MODEL_DIR, "tts", "voxcpm", "voxcpm-0.5b-onnx-int8");

        public static string KokoroTtsPath = Path.Combine(BASE_MODEL_DIR, "tts", "kokoro", "kokoro-v1.1");

        /// <summary>
        /// Qwen3-TTS (speakerId): serena / vivian / uncle_fu / ryan / aiden / ono_anna / sohee / eric(四川话) / dylan(北京话)。当前启用 Qwen3-TTS。
        /// MossNano: Junhao / Zhiming / Weiguo / Xiaoyu / Yuewen / Lingyu / Trump / Ava / Bella / Adam 等。同一常量会同时下发到 MossNano 选项
        /// KokoroTTS: af_maple,af_sol,bf_vale,zf_001,zf_002,zf_003,zf_004,zf_005,zf_006,zf_007,zf_008,zf_017,zf_018,zf_019,zf_021,zf_022,zf_023,zf_024,zf_026,zf_027,zf_028,zf_032,zf_036,zf_038,zf_039,zf_040,zf_042,zf_043,zf_044,zf_046,zf_047,zf_048,zf_049,zf_051,zf_059,zf_060,zf_067,zf_070,zf_071,zf_072,zf_073,zf_074,zf_075,zf_076,zf_077,zf_078,zf_079,zf_083,zf_084,zf_085,zf_086,zf_087,zf_088,zf_090,zf_092,zf_093,zf_094,zf_099,zm_009,zm_010,zm_011,zm_012,zm_013,zm_014,zm_015,zm_016,zm_020,zm_025,zm_029,zm_030,zm_031,zm_033,zm_034,zm_035,zm_037,zm_041,zm_045,zm_050,zm_052,zm_053,zm_054,zm_055,zm_056,zm_057,zm_058,zm_061,zm_062,zm_063,zm_064,zm_065,zm_066,zm_068,zm_069,zm_080,zm_081,zm_082,zm_089,zm_091,zm_095,zm_096,zm_097,zm_098,zm_100
        /// zf -> 中国女声；zm -> 中国男声；
        /// </summary>
        public static Dictionary<string, string> TtsVoiceNameMap = new Dictionary<string, string>
        {
            { "Qwen3TTSService", "serena" },
            { "MossNanoTTSService", "Xiaoyu" },
            { "KokoroTTSService", "zf_001" },
            { "VoxCPMTTSService", "xiaoxiao_female" },
            { "DoubaoOnlineTTSService", "zh_female_vv_uranus_bigtts" },
        };

        public static float TtsSpeed = 1.0f;

        public static bool IsEnableThinking = false;

        public static bool IsContinuousConversationEnabled = true;

        public static string DefaultAssistantPrompt =
            $"你叫‘{AssistantName}’，你是一个自然、真实、简洁的日常聊天伙伴。和我对话时，不要使用机器化、模板化的表达，也不要频繁提到\"AI\"\"模型\"\"程序\"等身份信息。\n" +
            "你的说话方式要像一个普通人：\n" +
            "1. 语气自然，像朋友聊天。\n" +
            "2. 回答简洁，不说废话。\n" +
            "3. 不要过度解释，除非我追问。\n" +
            "4. 不要用\"作为AI\"这类开场。\n" +
            "5. 不要刻意客套，不要假装很懂。\n" +
            "6. 可以有正常的语气词，比如\"嗯\"\"确实\"\"我懂\"\"可以\"。\n" +
            "7. 如果不确定，就直接说\"不太确定\"。\n" +
            "8. 根据我的语气调整回应，轻松、自然、有分寸。\n" +
            "9. 你的回复会被朗读出来。严格不要输出 emoji、表情符号、颜文字 (例如 :) ^_^)、markdown 格式 (* ` # _ ~ 等)，也不要在括号里写舞台指示 (例如 (笑) (停顿))。只输出可以直接朗读的文字。\n" +
            "10. 我的发言是通过语音识别 (ASR) 转成文字的，可能存在错别字、同音字混淆、漏字、断句错误或夹杂噪声。处理方式：\n" +
            "    - 如果名称错了，不要纠正，就当在正常喊你就行 \n" +
            "    - 如果句子大体通顺、能猜到我的意思 (例如同音字)，直接按你的理解回应，不需要先复述确认。\n" +
            "    - 如果出现明显不通顺、词语错乱、语义跳跃过大，先把它当作可能的识别错误：自然地问一句\"没太听清，再说一遍？\"或\"你是想说 xx 吗？\"，不要凭空编造意思，也不要硬接一个完全无关的话题。\n" +
            "    - 单字或孤立语气词 (嗯、啊、对、行、好) 按正常对话回应即可，不要当成错误。\n" +
            "11. 不要在回复里复述\"你刚才说的是 xxx\"这种确认句, 除非确实需要澄清。日常对话不需要复读。\n" +
            "你不需要强调自己的身份，只需要像一个真实的聊天对象一样回应我。";

        private static CaptureNoiseSuppressionMode ResolveNoiseSuppressionMode()
        {
            string? value = Environment.GetEnvironmentVariable("VOICEBOX_NOISE_SUPPRESSION");
            return value?.Trim().ToLowerInvariant() switch
            {
                null or "" or "none" or "disabled" => CaptureNoiseSuppressionMode.Disabled,
                "webrtc" or "ns" => CaptureNoiseSuppressionMode.WebRtc,
                "dnr" or "independent" => CaptureNoiseSuppressionMode.IndependentDnr,
                _ => throw new InvalidOperationException(
                    "VOICEBOX_NOISE_SUPPRESSION must be one of: none, webrtc, dnr."),
            };
        }

        public static string ConvertWakeupWordToAssistantName(string wakeupWord)
        {
            ArgumentNullException.ThrowIfNull(wakeupWord);

            for (int unitLength = 1; unitLength <= wakeupWord.Length / 2; unitLength++)
            {
                if (wakeupWord.Length % unitLength != 0)
                {
                    continue;
                }

                ReadOnlySpan<char> unit = wakeupWord.AsSpan(0, unitLength);
                bool isRepeated = true;
                for (int offset = unitLength; offset < wakeupWord.Length; offset += unitLength)
                {
                    if (!unit.SequenceEqual(wakeupWord.AsSpan(offset, unitLength)))
                    {
                        isRepeated = false;
                        break;
                    }
                }

                if (isRepeated)
                {
                    int assistantNameLength = Math.Max(unitLength, 2);
                    return wakeupWord[..assistantNameLength];
                }
            }

            return wakeupWord;
        }
    }

    internal enum CaptureNoiseSuppressionMode
    {
        Disabled,
        WebRtc,
        IndependentDnr,
    }
}
