using VoiceBox.Core.Onnx;
using VoiceBox.Core.Services.AudioService.Models;

namespace VoiceBox.Wakeup.SherpaOnnxKws.Models
{
    /// <summary>
    /// SherpaOnnxKws (基于 sherpa-onnx KWS Zipformer-Transducer 的流式命令词唤醒服务) 的配置。
    /// 模型来源: https://k2-fsa.github.io/sherpa/onnx/kws/pretrained_models/index.html
    /// </summary>
    public class SherpaOnnxKwsOptions : OnnxExecutionOptions
    {
        /// <summary>音频采样率 (Hz)。模型固定要求 16000Hz, 其它采样率请自行重采样。</summary>
        public uint SampleRate { get; set; } = 16000;

        /// <summary>声道数。仅支持 1 (单声道)。</summary>
        public uint ChannelCount { get; set; } = 1;

        /// <summary>采样格式。仅支持 Float32 ([-1, 1] 归一化振幅)。</summary>
        public SampleFormat SampleFormat { get; set; } = SampleFormat.Float32;

        /// <summary>
        /// Transducer Encoder ONNX 模型路径 (例如 encoder-epoch-13-avg-2-chunk-16-left-64.int8.onnx)。
        /// </summary>
        public string EncoderModelPath { get; set; } = string.Empty;

        /// <summary>Transducer Decoder ONNX 模型路径 (decoder-*.onnx)。</summary>
        public string DecoderModelPath { get; set; } = string.Empty;

        /// <summary>Transducer Joiner ONNX 模型路径 (joiner-*.onnx)。</summary>
        public string JoinerModelPath { get; set; } = string.Empty;

        /// <summary>符号表文件 (tokens.txt) 路径。</summary>
        public string TokensFilePath { get; set; } = string.Empty;

        /// <summary>
        /// 可选: CMU 风格英文发音词典路径 (每行 "WORD  PH1 PH2 ...", 例如模型自带的 en.phone)。
        /// 用于通过 <see cref="AddKeyword(string)"/> 等简化重载, 把英文关键词自动转成 token 序列。
        /// 留空则在 <see cref="TokensFilePath"/> 同目录寻找 "en.phone"。
        /// </summary>
        public string CmuPhoneDictionaryPath { get; set; } = string.Empty;

        /// <summary>
        /// 可选: 关键词文件路径 (每行一个关键词, token 之间空格分隔, 可选 :score, #threshold, @phrase 后缀)。
        /// 启动时会与 <see cref="Keywords"/> 合并; 若两者都为空则会抛异常。
        /// </summary>
        public string KeywordsFile { get; set; } = string.Empty;

        /// <summary>用代码方式动态添加的关键词列表。</summary>
        public IList<KwsKeywordDefinition> Keywords { get; } = new List<KwsKeywordDefinition>();

        /// <summary>波束搜索的活跃路径数 (modified beam-search)。默认 4。</summary>
        public int MaxActivePaths { get; set; } = 4;

        /// <summary>关键词命中后需保持的尾部空白帧数, 用于稳定触发。默认 1。</summary>
        public int NumTrailingBlanks { get; set; } = 1;

        /// <summary>
        /// 命中即触发: 一旦 beam 中任意一条假设走到关键词结尾节点, 且平均声学概率 ≥ 阈值,
        /// 立即触发, 不再等待 <see cref="NumTrailingBlanks"/> 个尾部空白帧。
        /// <para>
        /// 用途: 解决连续/快速说话场景。原算法只看最佳假设, 并要求关键词后还要有静音帧;
        /// 当用户说 "小爱同学打开灯" 时, 关键词路径在 "学" 刚出现后立即被新词路径挤出最佳位,
        /// trailing blanks 永远凑不齐 → 永不触发。开启此项后:
        /// </para>
        /// <list type="bullet">
        ///   <item>扫描 beam 中所有假设找命中, 而不仅是 best (避免被后续语音挤掉)。</item>
        ///   <item>满足阈值即触发, 跳过 trailing blanks 检查。</item>
        /// </list>
        /// 默认 true；如需更保守（减少误唤醒）可置 false。
        /// </summary>
        public bool EmitImmediately { get; set; } = true;

        /// <summary>全局 boost score (上下文偏置加分), 单个关键词若未单独设置则使用该值。默认 1.5。</summary>
        public float KeywordsScore { get; set; } = 1.5f;

        /// <summary>全局声学触发阈值 (0~1], 单个关键词若未单独设置则使用该值。默认 0.25。</summary>
        public float KeywordsThreshold { get; set; } = 0.2f;

        /// <summary>
        /// 全局防抖时间 (ms)。命中后该时间内不再重复触发同一关键词。单个关键词可独立设置 (DebounceMs 优先)。默认 800ms。
        /// </summary>
        public int DebounceMs { get; set; } = 800;

        /// <summary>
        /// 长时间静音后, 自动重置流式解码器 (秒)。
        /// 模型解码到 N 个连续空白帧后会自动重置, 默认 1.5 秒。
        /// </summary>
        public float ResetAfterSilenceSeconds { get; set; } = 1.5f;

        /// <summary>
        /// 自动分词时, 单字回落 (即未命中短语词典) 的多音字是否展开成多条候选读音。
        /// 关闭时只取每字第一读音; 开启时把所有读音做笛卡尔积 (受 <see cref="MaxAutoVariantsPerKeyword"/> 上限保护)。
        /// 短语词典命中的字段不展开 (短语已限定上下文读音)。默认 true。
        /// </summary>
        public bool ExpandPolyphoneVariants { get; set; } = true;

        /// <summary>
        /// 单个关键词经 AutoTokenizer 自动展开 (3+3 上声变调 + 多音字笛卡尔积) 后允许产出的最大变体数,
        /// 含规范读音本身。≤0 表示不限。默认 16, 足够覆盖 4 字以内多音字短语 + 变调, 同时避免长词爆炸。
        /// </summary>
        public int MaxAutoVariantsPerKeyword { get; set; } = 16;

        /// <summary>
        /// 自动分词时叠加的口音 / 方言混淆变体规则 (例如 zh↔z, n↔l, in↔ing, …)。
        /// 默认 <see cref="KwsConfusionVariants.None"/> 不展开;
        /// 开启会扩张上下文图、提高误唤醒概率, 应根据目标用户口音场景主动选择。
        /// 总变体数仍受 <see cref="MaxAutoVariantsPerKeyword"/> 上限保护。
        /// </summary>
        public KwsConfusionVariants ConfusionVariants { get; set; } = KwsConfusionVariants.None;

        /// <summary>
        /// 链式添加关键词。
        /// </summary>
        /// <param name="keyword">关键词原文 (用于回调事件)。</param>
        /// <param name="tokens">空格分隔的 token 序列, 必须能在 tokens.txt 中找到。</param>
        /// <param name="boostScore">该关键词的 boost score, ≤0 表示沿用全局。</param>
        /// <param name="threshold">该关键词的触发阈值, ≤0 表示沿用全局。</param>
        /// <param name="debounceMs">该关键词的防抖时间 (ms), ≤0 表示沿用全局。</param>
        public SherpaOnnxKwsOptions AddKeyword(
            string keyword,
            string tokens,
            float boostScore = 0f,
            float threshold = 0f,
            int debounceMs = -1)
        {
            if (string.IsNullOrWhiteSpace(keyword))
                throw new ArgumentException("Keyword must not be empty.", nameof(keyword));
            if (string.IsNullOrWhiteSpace(tokens))
                throw new ArgumentException("Tokens must not be empty.", nameof(tokens));

            Keywords.Add(new KwsKeywordDefinition
            {
                Keyword = keyword,
                Tokens = tokens,
                BoostScore = boostScore,
                Threshold = threshold,
                DebounceMs = debounceMs > 0 ? debounceMs : DebounceMs,
            });
            return this;
        }

        /// <summary>
        /// 链式添加关键词 (仅原文, tokens 在启动时由 <see cref="Internal.AutoTokenizer"/> 自动生成)。
        /// 中文走拼音字典 → 声母+带调韵母; 英文走 CMU 风格发音字典 (默认查 tokens.txt 同目录的 en.phone)。
        /// 字典查不到的字 / 词会在初始化时抛出, 此时请改用 <see cref="AddKeyword(string, string, float, float, int)"/>
        /// 显式指定 Tokens。
        /// </summary>
        public SherpaOnnxKwsOptions AddKeyword(
            string keyword,
            float boostScore = 0f,
            float threshold = 0f,
            int debounceMs = -1)
        {
            if (string.IsNullOrWhiteSpace(keyword))
                throw new ArgumentException("Keyword must not be empty.", nameof(keyword));

            Keywords.Add(new KwsKeywordDefinition
            {
                Keyword = keyword,
                // Tokens 留空, 触发 KeywordEncoder 内部的自动生成
                Tokens = string.Empty,
                BoostScore = boostScore,
                Threshold = threshold,
                DebounceMs = debounceMs > 0 ? debounceMs : DebounceMs,
            });
            return this;
        }

        /// <summary>
        /// 链式添加关键词, 同时注册多条「发音变体」, 所有变体命中时都上报同一 <paramref name="keyword"/>。
        /// <para>
        /// 用于弥补 ASR 模型对小语料关键词置信度低的问题: 把模型常误识的结果当作合法
        /// token 序列一起注册。例如 "小爱同学" 常被解成 "小爱通学" / "小爱同鞋", 把它们
        /// 作为 variants 注册后，OnWakeup 事件里的 keyword 字段仍是 "小爱同学"。
        /// </para>
        /// 每条变体都会经 AutoTokenizer 解析, 因此必须由中文/英文/中英混排构成,
        /// 不能直接传入 token 字符串。
        /// </summary>
        /// <param name="keyword">主关键词原文; 也是回调事件里展示的名字。</param>
        /// <param name="variants">发音变体原文集合 (例如 "小爱通学", "小爱同鞋", ...)。空集合等价于 AddKeyword。</param>
        /// <param name="boostScore">作用于全部变体的 boost score。≤0 表示沿用全局。</param>
        /// <param name="threshold">作用于全部变体的触发阈值。≤0 表示沿用全局。</param>
        /// <param name="debounceMs">主关键词的防抖时间 (ms)。≤0 表示沿用全局。</param>
        public SherpaOnnxKwsOptions AddKeywordWithVariants(
            string keyword,
            IEnumerable<string> variants,
            float boostScore = 0f,
            float threshold = 0f,
            int debounceMs = -1)
        {
            if (string.IsNullOrWhiteSpace(keyword))
                throw new ArgumentException("Keyword must not be empty.", nameof(keyword));

            var def = new KwsKeywordDefinition
            {
                Keyword = keyword,
                Tokens = string.Empty,
                BoostScore = boostScore,
                Threshold = threshold,
                DebounceMs = debounceMs > 0 ? debounceMs : DebounceMs,
            };
            if (variants != null)
            {
                foreach (var v in variants)
                {
                    if (!string.IsNullOrWhiteSpace(v)) def.Variants.Add(v);
                }
            }
            Keywords.Add(def);
            return this;
        }
    }
}
