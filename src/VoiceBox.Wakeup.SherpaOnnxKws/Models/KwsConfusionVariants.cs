namespace VoiceBox.Wakeup.SherpaOnnxKws.Models
{
    /// <summary>
    /// AutoTokenizer 在生成中文关键词的发音变体时, 可叠加的「混淆音」规则。
    /// 对每条 base 变体 (规范读音 / 上声变调 / 多音字组合) 的每个 token 做单点替换,
    /// 替换后必须能在 tokens.txt 中找到, 否则自然丢弃。
    /// <para>
    /// 默认 <see cref="None"/>。开启后上下文图会扩张, 误唤醒率上升, 应根据目标用户口音场景
    /// 主动选择。常见取值见 <see cref="Common"/> 与 <see cref="All"/>。
    /// </para>
    /// </summary>
    [Flags]
    public enum KwsConfusionVariants
    {
        /// <summary>不生成任何混淆变体。</summary>
        None = 0,

        /// <summary>平翘舌互换: zh↔z, ch↔c, sh↔s (西南官话 / 江淮官话 / 部分南方口音常见)。</summary>
        RetroflexFlat = 1 << 0,

        /// <summary>声母 n↔l 互换 (湖南/湖北/西南部分地区常见)。</summary>
        NasalLateral = 1 << 1,

        /// <summary>韵尾 n↔ng 互换 (in/ing, en/eng, an/ang, ian/iang, uan/uang; 东南方言常见)。</summary>
        NasalFinal = 1 << 2,

        /// <summary>声母 f↔h 互换 (闽南/客家/部分湘语常见)。</summary>
        FrontVelar = 1 << 3,

        /// <summary>声母 r↔l 互换 (部分粤语/西南口音常见)。</summary>
        Rhotic = 1 << 4,

        /// <summary>常用三类: 平翘舌 + 前后鼻 + 鼻边音, 覆盖最大常见口音群体。</summary>
        Common = RetroflexFlat | NasalFinal | NasalLateral,

        /// <summary>启用全部混淆规则。</summary>
        All = RetroflexFlat | NasalFinal | NasalLateral | FrontVelar | Rhotic,
    }
}
