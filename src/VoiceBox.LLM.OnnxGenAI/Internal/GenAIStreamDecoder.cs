using System;
using System.Text;

namespace VoiceBox.LLM.OnnxGenAI.Internal
{
    /// <summary>
    /// 处理来自 <c>TokenizerStream.Decode</c> 的逐 token 文本流, 完成两件事:
    /// <list type="number">
    ///   <item>把 thinking 块 (<c>&lt;|channel&gt; ... &lt;channel|&gt;</c> 或 <c>&lt;think&gt; ... &lt;/think&gt;</c>) 里的内容路由到 thinking 输出, 不进入正文。</item>
    ///   <item>过滤掉 Gemma 4 的 special token 与 tool_call 元数据, 返回干净的可见文本。</item>
    /// </list>
    ///
    /// 状态机有跨 token 边界的窗口逻辑: thinking/tool_call 起止标记可能被 token 边界劈开,
    /// 所以保留一个小缓冲, 直到能判定标记是否完整。
    /// </summary>
    internal sealed class GenAIStreamDecoder
    {
        // 与 D:/Temp/gemma/src/streaming.ts 的 SPECIAL_TOKENS 集合保持一致, 见 streaming.ts:3-12。
        private static readonly string[] SpecialTokens =
        {
            "<eos>", "<bos>", "<end_of_turn>", "<start_of_turn>",
            "<|turn>", "<turn|>",
            "<|tool>", "<tool|>",
            "<|tool_call>", "<tool_call|>",
            "<|tool_response>", "<tool_response|>",
            "<|channel>", "<channel|>",
            "<think>", "</think>",
            "<|think|>",
            "<|image|>",
            "<|audio|>",
            "<|\"|>",
        };

        // 保留足够的尾部字符来覆盖最长 special token 被 token 边界切开的情况。
        private const int MaxLookaheadChars = 16;

        // Gemma 4 在 thinking 段开头产出的元信息行, 与思考内容无关, 必须剔除。
        private const string ThoughtNewline = "thought\n";

        private readonly StringBuilder _raw = new();
        private readonly StringBuilder _buffer = new();

        // Feed 每次复用同两个输出缓冲, 避免逐 token new StringBuilder。
        private readonly StringBuilder _visibleOut = new();
        private readonly StringBuilder _thinkingOut = new();

        // thinking 起始后累计的前缀缓冲。thought\n 只有 8 字符, 但 TryAdvance 的 16 字符安全窗口
        // 会把它切成碎片 (如先发 "thou" 再发 "ght\n"), 逐段 StartsWith 无法命中 -> 元信息泄漏。
        // 这里累计到 >= ThoughtNewline.Length 再一次性判定, 覆盖流式分片与 Flush 短段两种情况。
        private readonly StringBuilder _thinkingStartBuffer = new();

        private bool _insideThinking;
        private bool _insideToolCall;
        private bool _atThinkingStart;

        public string Raw => _raw.ToString();

        /// <summary>
        /// 投递新一段来自 tokenizer 的解码结果。
        /// </summary>
        /// <param name="text">tokenizer 当前 token 解码出的字符串 (可能是空字符串、半个 token、或多 token 合并)。</param>
        /// <param name="visible">本次能立即对外输出的可见文本 (可能为空)。</param>
        /// <param name="thinking">本次能立即对外输出的思考文本 (可能为空)。</param>
        public void Feed(string text, out string visible, out string thinking)
        {
            visible = string.Empty;
            thinking = string.Empty;
            if (string.IsNullOrEmpty(text)) return;

            _raw.Append(text);
            _buffer.Append(text);

            // 可能跨越多个状态切换, 反复扫描直到 buffer 不能再前进。复用输出缓冲, 逐 token 零额外分配。
            _visibleOut.Clear();
            _thinkingOut.Clear();
            while (TryAdvance(_visibleOut, _thinkingOut)) { }

            if (_visibleOut.Length > 0) visible = _visibleOut.ToString();
            if (_thinkingOut.Length > 0) thinking = _thinkingOut.ToString();
        }

        /// <summary>
        /// 在生成结束时调用, 把缓冲区残留的内容按当前状态刷出。
        /// </summary>
        public void Flush(out string visible, out string thinking)
        {
            visible = string.Empty;
            thinking = string.Empty;
            _visibleOut.Clear();
            _thinkingOut.Clear();

            if (_buffer.Length > 0)
            {
                string remaining = _buffer.ToString();
                _buffer.Clear();
                string clean = StripSpecialTokens(remaining);
                if (_insideThinking)
                {
                    // 末段也走 thinking-start 累计: 短段 (<16 字符) 从未触发过 TryAdvance 安全输出,
                    // 整段滞留到 Flush, 须补 thought\n 判定 (否则整段带元信息输出)。
                    if (_atThinkingStart)
                    {
                        _thinkingStartBuffer.Append(clean);
                        FlushThinkingStart(_thinkingOut);
                    }
                    else
                    {
                        _thinkingOut.Append(clean);
                    }
                }
                else if (!_insideToolCall)
                {
                    _visibleOut.Append(clean);
                }
            }
            else if (_atThinkingStart)
            {
                // _buffer 已空但 thinking-start 前缀仍被 hold (短段未达 8 字符即结束): 释放, 不丢。
                FlushThinkingStart(_thinkingOut);
            }

            if (_visibleOut.Length > 0) visible = _visibleOut.ToString();
            if (_thinkingOut.Length > 0) thinking = _thinkingOut.ToString();
        }

        private bool TryAdvance(StringBuilder visibleSb, StringBuilder thinkingSb)
        {
            if (_buffer.Length == 0) return false;

            if (!ContainsMarkerStart(_buffer))
            {
                int safeLen = _buffer.Length - MaxLookaheadChars;
                if (safeLen <= 0) return false;

                EmitByState(_buffer.ToString(0, safeLen), visibleSb, thinkingSb);
                _buffer.Remove(0, safeLen);
                return false;
            }

            // 每次前进只把 buffer 物化成字符串一次, 所有标记查找都在这份快照上做。
            // 所有 special token 都参与匹配: 非状态 token 也必须整体吞掉, 否则一旦被安全窗口切开,
            // StripSpecialTokens 就再也看不到完整 token, 会把碎片泄漏到用户文本里。
            string buf = _buffer.ToString();

            int firstMarker = -1;
            string? matched = null;
            foreach (string marker in SpecialTokens)
            {
                int idx = buf.IndexOf(marker, StringComparison.Ordinal);
                if (idx < 0) continue;
                // 同一位置理论上只会命中一个标记 (首字符各不相同); 保留"更长者优先"仅作防御。
                if (firstMarker < 0 || idx < firstMarker || (idx == firstMarker && marker.Length > matched!.Length))
                {
                    firstMarker = idx;
                    matched = marker;
                }
            }

            if (firstMarker < 0)
            {
                // 没有完整标记。安全输出: 保留尾部 MaxLookaheadChars 个字符以防标记被切开,
                // 其余部分按当前状态刷出。
                int safeLen = buf.Length - MaxLookaheadChars;
                if (safeLen <= 0) return false;

                EmitByState(buf.Substring(0, safeLen), visibleSb, thinkingSb);
                _buffer.Remove(0, safeLen);
                return false;
            }

            // 先把 marker 之前的部分按当前状态刷出 (须在切换状态前), 再连同 marker 一起从 buffer 移除。
            if (firstMarker > 0)
                EmitByState(buf.Substring(0, firstMarker), visibleSb, thinkingSb);

            _buffer.Remove(0, firstMarker + matched!.Length);
            ApplyMarker(matched, thinkingSb);
            return true;
        }

        private static bool ContainsMarkerStart(StringBuilder sb)
        {
            for (int i = 0; i < sb.Length; i++)
            {
                if (sb[i] == '<') return true;
            }
            return false;
        }

        private void ApplyMarker(string marker, StringBuilder thinkingSb)
        {
            bool wasInsideThinking = _insideThinking;
            switch (marker)
            {
                case "<|channel>": _insideThinking = true; _atThinkingStart = true; break;
                case "<channel|>": _insideThinking = false; _atThinkingStart = false; break;
                case "<think>": _insideThinking = true; _atThinkingStart = true; break;
                case "</think>": _insideThinking = false; _atThinkingStart = false; break;
                case "<|tool_call>": _insideToolCall = true; break;
                case "<tool_call|>": _insideToolCall = false; break;
                case "<tool_response|>": _insideToolCall = false; break;
            }
            // 若该 marker 结束了 thinking, 释放被 hold 的 thinking-start 前缀 (含 thought\n 判定),
            // 否则短前缀滞留在 _thinkingStartBuffer 直到下次 thinking 才被错误释放/丢失。
            // 用状态转移判定 (wasInside && !_insideThinking), 不依赖具体 marker 文本。
            if (wasInsideThinking && !_insideThinking)
                FlushThinkingStart(thinkingSb);
        }

        private void EmitByState(string segment, StringBuilder visibleSb, StringBuilder thinkingSb)
        {
            if (string.IsNullOrEmpty(segment)) return;

            string clean = StripSpecialTokens(segment);
            if (clean.Length == 0) return;

            if (_insideThinking)
            {
                if (_atThinkingStart)
                {
                    // 累计 thinking 起始前缀, 凑满 thought\n 长度再一次性判定剔除:
                    // 16 字符安全窗口会把 8 字符的 "thought\n" 切碎, 逐段 StartsWith 会漏判致泄漏。
                    _thinkingStartBuffer.Append(clean);
                    if (_thinkingStartBuffer.Length >= ThoughtNewline.Length)
                        FlushThinkingStart(thinkingSb);
                }
                else
                {
                    thinkingSb.Append(clean);
                }
            }
            else if (!_insideToolCall)
            {
                visibleSb.Append(clean);
            }
            // tool_call 块默认丢弃 (本项目当前不暴露 tool 调用; 后续 Agent 层接管时再分流)。
        }

        /// <summary>
        /// 释放累计的 thinking-start 前缀: 若以 "thought\n" 开头则剔除该 8 字符元信息行,
        /// 其余原样输出。不依赖 _atThinkingStart 标志 (ApplyMarker 结束 thinking 时该标志已被置 false)。
        /// </summary>
        private void FlushThinkingStart(StringBuilder thinkingSb)
        {
            if (_thinkingStartBuffer.Length == 0)
            {
                _atThinkingStart = false;
                return;
            }
            string prefix = _thinkingStartBuffer.ToString();
            _thinkingStartBuffer.Clear();
            if (prefix.StartsWith(ThoughtNewline, StringComparison.Ordinal))
                prefix = prefix[ThoughtNewline.Length..];
            if (prefix.Length > 0)
                thinkingSb.Append(prefix);
            _atThinkingStart = false;
        }

        private static string StripSpecialTokens(string text)
        {
            if (string.IsNullOrEmpty(text)) return string.Empty;
            // 快速路径: 所有 special token 都以 '<' 开头, 段内没有 '<' 就不可能命中,
            // 直接返回, 省掉对全部 special token 的逐一扫描 (绝大多数正文段落如此)。
            if (text.IndexOf('<') < 0) return text;

            string result = text;
            for (int i = 0; i < SpecialTokens.Length; i++)
            {
                string tok = SpecialTokens[i];
                if (result.IndexOf(tok, StringComparison.Ordinal) >= 0)
                    result = result.Replace(tok, string.Empty);
            }
            return result;
        }
    }
}
