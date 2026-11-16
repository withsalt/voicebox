namespace VoiceBox.LLM.OpenAI
{
    /// <summary>
    /// OpenAI-compatible Chat API 的深度思考开关, 映射到火山方舟等服务的 <c>thinking</c> 字段。
    /// </summary>
    public enum ThinkingMode
    {
        /// <summary>
        /// 强制开启深度思考。
        /// </summary>
        Enabled = 0,

        /// <summary>
        /// 强制关闭深度思考。
        /// </summary>
        Disabled,

        /// <summary>
        /// 不下发 thinking 字段, 由服务端或模型默认策略决定是否进行深度思考。
        /// </summary>
        Auto,
    }
}
