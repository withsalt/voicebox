using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace VoiceBox.LLM.LLamaSharp.Internal
{
    /// <summary>标记类别: 命中后对解码状态的影响。</summary>
    internal enum LlamaMarkerKind
    {
        /// <summary>进入 thinking 块。</summary>
        ThinkingStart,

        /// <summary>退出 thinking 块。</summary>
        ThinkingEnd,

        /// <summary>仅从输出剔除, 不改变状态 (泄漏的 special token 文本)。</summary>
        Strip,

        /// <summary>文本级停止词: 终止生成, 停止词与其后内容全部丢弃。</summary>
        Stop,
    }

    /// <summary>
    /// 由 <see cref="ILlamaModelProfile"/> 预编译出的标记表: 解码循环逐 token 高频使用,
    /// 首字符集合与最大标记长度在构造期算好。
    /// </summary>
    internal sealed class LlamaStreamDecoderConfig
    {
        public LlamaStreamDecoderConfig(ILlamaModelProfile profile)
        {
            ArgumentNullException.ThrowIfNull(profile);

            var markers = new List<(string Text, LlamaMarkerKind Kind)>();
            void AddAll(IReadOnlyList<string> texts, LlamaMarkerKind kind)
            {
                foreach (string text in texts)
                {
                    if (!string.IsNullOrEmpty(text))
                        markers.Add((text, kind));
                }
            }

            AddAll(profile.ThinkingStartMarkers, LlamaMarkerKind.ThinkingStart);
            AddAll(profile.ThinkingEndMarkers, LlamaMarkerKind.ThinkingEnd);
            AddAll(profile.StripMarkers, LlamaMarkerKind.Strip);
            AddAll(profile.StopTexts, LlamaMarkerKind.Stop);

            Markers = markers.ToArray();
            MaxMarkerLength = Markers.Length == 0 ? 0 : Markers.Max(m => m.Text.Length);
            FirstChars = Markers.Select(m => m.Text[0]).Distinct().ToArray();
        }

        public (string Text, LlamaMarkerKind Kind)[] Markers { get; }

        /// <summary>最长标记的字符数, 也是安全输出时需要保留的 lookahead 窗口。</summary>
        public int MaxMarkerLength { get; }

        /// <summary>全部标记的首字符集合 (快速排除无标记的正文段落)。</summary>
        public char[] FirstChars { get; }
    }

    /// <summary>
    /// 处理来自 <c>StreamingTokenDecoder</c> 的逐 token 文本流 (设计对齐
    /// <c>VoiceBox.LLM.OnnxGenAI</c> 的 GenAIStreamDecoder), 完成三件事:
    /// <list type="number">
    ///   <item>把 thinking 块内容路由到 thinking 输出, 不进入正文;</item>
    ///   <item>剔除泄漏为文本的 special token;</item>
    ///   <item>命中文本级停止词时置 <see cref="Stopped"/>, 由生成循环终止解码。</item>
    /// </list>
    ///
    /// 标记可能被 token 边界劈开, 因此保留一个 <see cref="LlamaStreamDecoderConfig.MaxMarkerLength"/>
    /// 大小的尾部缓冲, 直到能判定标记是否完整才放行。标记表来自模型 profile — 不同模型
    /// 的差异全部收敛在 <see cref="ILlamaModelProfile"/>, 本状态机保持通用。
    /// </summary>
    internal sealed class LlamaStreamDecoder
    {
        private readonly LlamaStreamDecoderConfig _config;

        private readonly StringBuilder _buffer = new();

        // Feed 每次复用同两个输出缓冲, 避免逐 token new StringBuilder。
        private readonly StringBuilder _visibleOut = new();
        private readonly StringBuilder _thinkingOut = new();

        private bool _insideThinking;

        public LlamaStreamDecoder(LlamaStreamDecoderConfig config, bool startsInsideThinking = false)
        {
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _insideThinking = startsInsideThinking;
        }

        /// <summary>命中停止词后为 true; 生成循环应立即终止, 后续 Feed 输入全部丢弃。</summary>
        public bool Stopped { get; private set; }

        /// <summary>
        /// 投递新一段来自 tokenizer 的解码结果。
        /// </summary>
        /// <param name="text">当前 token 解码出的字符串 (可能是空串、半个标记、或多 token 合并)。</param>
        /// <param name="visible">本次能立即对外输出的可见文本 (可能为空)。</param>
        /// <param name="thinking">本次能立即对外输出的思考文本 (可能为空)。</param>
        public void Feed(string text, out string visible, out string thinking)
        {
            visible = string.Empty;
            thinking = string.Empty;
            if (string.IsNullOrEmpty(text) || Stopped) return;

            _buffer.Append(text);

            _visibleOut.Clear();
            _thinkingOut.Clear();
            // 可能跨越多个状态切换, 反复扫描直到 buffer 不能再前进。
            while (TryAdvance()) { }

            if (_visibleOut.Length > 0) visible = _visibleOut.ToString();
            if (_thinkingOut.Length > 0) thinking = _thinkingOut.ToString();
        }

        /// <summary>
        /// 在生成结束时调用, 把缓冲区残留的内容按当前状态刷出 (含无法配对的半截标记)。
        /// </summary>
        public void Flush(out string visible, out string thinking)
        {
            visible = string.Empty;
            thinking = string.Empty;
            if (_buffer.Length == 0 || Stopped)
            {
                _buffer.Clear();
                return;
            }

            string remaining = _buffer.ToString();
            _buffer.Clear();
            if (_insideThinking)
                thinking = remaining;
            else
                visible = remaining;
        }

        private bool TryAdvance()
        {
            if (_buffer.Length == 0) return false;

            if (!ContainsMarkerFirstChar())
            {
                EmitSafePrefix(_buffer.ToString());
                return false;
            }

            // 每次前进只把 buffer 物化成字符串一次, 所有标记查找都在这份快照上做。
            string buf = _buffer.ToString();

            int firstIdx = -1;
            string? matchedText = null;
            LlamaMarkerKind matchedKind = default;
            foreach (var (text, kind) in _config.Markers)
            {
                int idx = buf.IndexOf(text, StringComparison.Ordinal);
                if (idx < 0) continue;
                // 更早者优先; 同位置更长者优先 (如 "</think>" 优先于假想的 "</t")。
                if (firstIdx < 0 || idx < firstIdx || (idx == firstIdx && text.Length > matchedText!.Length))
                {
                    firstIdx = idx;
                    matchedText = text;
                    matchedKind = kind;
                }
            }

            if (firstIdx < 0)
            {
                // 没有完整标记。保留尾部 MaxMarkerLength 个字符以防标记被 token 边界切开,
                // 其余部分按当前状态刷出。
                EmitSafePrefix(buf);
                return false;
            }

            // 先把标记之前的部分按当前状态刷出 (须在切换状态前), 再连同标记一起从 buffer 移除。
            if (firstIdx > 0)
                EmitByState(buf.Substring(0, firstIdx));

            _buffer.Remove(0, firstIdx + matchedText!.Length);

            switch (matchedKind)
            {
                case LlamaMarkerKind.ThinkingStart:
                    _insideThinking = true;
                    break;
                case LlamaMarkerKind.ThinkingEnd:
                    _insideThinking = false;
                    break;
                case LlamaMarkerKind.Strip:
                    break;
                case LlamaMarkerKind.Stop:
                    Stopped = true;
                    _buffer.Clear();
                    return false;
            }
            return true;
        }

        /// <summary>刷出 buffer 中除 lookahead 尾窗之外的部分。</summary>
        private void EmitSafePrefix(string buf)
        {
            int safeLen = buf.Length - _config.MaxMarkerLength;
            if (safeLen <= 0) return;

            EmitByState(buf.Substring(0, safeLen));
            _buffer.Remove(0, safeLen);
        }

        private bool ContainsMarkerFirstChar()
        {
            char[] firstChars = _config.FirstChars;
            if (firstChars.Length == 0) return false;

            for (int i = 0; i < _buffer.Length; i++)
            {
                char c = _buffer[i];
                for (int j = 0; j < firstChars.Length; j++)
                {
                    if (c == firstChars[j]) return true;
                }
            }
            return false;
        }

        private void EmitByState(string segment)
        {
            if (segment.Length == 0) return;

            // 所有标记 (含 Strip) 都参与完整匹配并整体移除, 到这里的段落必然不含完整标记,
            // 无需再做二次剔除。
            if (_insideThinking)
                _thinkingOut.Append(segment);
            else
                _visibleOut.Append(segment);
        }
    }
}
