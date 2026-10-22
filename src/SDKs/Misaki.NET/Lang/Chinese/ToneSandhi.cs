namespace Misaki.NET.Lang.Chinese;

/// <summary>
/// Implements Mandarin tone sandhi rules for Chinese TTS.
/// Adapted from PaddleSpeech tone sandhi module.
/// </summary>
public class ToneSandhi
{
    #region Constants

    private static readonly HashSet<string> XEng = new() { "x", "eng" };

    #endregion

    #region Must Neural Tone Words

    /// <summary>
    /// Words where the second syllable becomes neutral tone (tone 5).
    /// </summary>
    private static readonly HashSet<string> MustNeuralToneWords = new()
    {
        "麻烦", "麻利", "鸳鸯", "高粱", "骨头", "骆驼", "马虎", "首饰",
        "馒头", "馄饨", "风筝", "难为", "队伍", "阔气", "闺女", "门道",
        "锄头", "铺盖", "铃铛", "铁匠", "钥匙", "里脊", "里头", "部分",
        "那么", "道士", "造化", "迷糊", "连累", "这么", "这个", "运气",
        "过去", "软和", "转悠", "踏实", "跳蚤", "跟头", "趔趄", "财主",
        "豆腐", "讲究", "记性", "记号", "认识", "规矩", "见识", "裁缝",
        "补丁", "衣裳", "衣服", "衙门", "街坊", "行李", "行当", "蛤蟆",
        "蘑菇", "薄荷", "葫芦", "葡萄", "萝卜", "荸荠", "苗条", "苗头",
        "苍蝇", "芝麻", "舒服", "舒坦", "舌头", "自在", "膏药", "脾气",
        "脑袋", "脊梁", "能耐", "胳膊", "胭脂", "胡萝", "胡琴", "胡同",
        "聪明", "耽误", "耽搁", "耷拉", "耳朵", "老爷", "老实", "老婆",
        "戏弄", "将军", "翻腾", "罗嗦", "罐头", "编辑", "结实", "红火",
        "累赘", "糨糊", "糊涂", "精神", "粮食", "簸箕", "篱笆", "算计",
        "算盘", "答应", "笤帚", "笑语", "笑话", "窟窿", "窝囊", "窗户",
        "稳当", "稀罕", "称呼", "秧歌", "秀气", "秀才", "福气", "祖宗",
        "砚台", "码头", "石榴", "石头", "石匠", "知识", "眼睛", "眯缝",
        "眨巴", "眉毛", "相声", "盘算", "白净", "痢疾", "痛快", "疟疾",
        "疙瘩", "疏忽", "畜生", "生意", "甘蔗", "琵琶", "琢磨", "琉璃",
        "玻璃", "玫瑰", "玄乎", "狐狸", "状元", "特务", "牲口", "牙碜",
        "牌楼", "爽快", "爱人", "热闹", "烧饼", "烟筒", "烂糊", "点心",
        "炊帚", "灯笼", "火候", "漂亮", "滑溜", "溜达", "温和", "清楚",
        "消息", "浪头", "活泼", "比方", "正经", "欺负", "模糊", "槟榔",
        "棺材", "棒槌", "棉花", "核桃", "栅栏", "柴火", "架势", "枕头",
        "枇杷", "机灵", "本事", "木头", "木匠", "朋友", "月饼", "月亮",
        "暖和", "明白", "时候", "新鲜", "故事", "收拾", "收成", "提防",
        "挖苦", "挑剔", "指甲", "指头", "拾掇", "拳头", "拨弄", "招牌",
        "招呼", "抬举", "护士", "折腾", "扫帚", "打量", "打算", "打扮",
        "打听", "打发", "扎实", "扁担", "戒指", "懒得", "意识", "意思",
        "悟性", "怪物", "思量", "怎么", "念头", "念叨", "别人", "快活",
        "忙活", "志气", "心思", "得罪", "张罗", "弟兄", "开通", "应酬",
        "庄稼", "干事", "帮手", "帐篷", "希罕", "师父", "师傅", "巴结",
        "巴掌", "差事", "工夫", "岁数", "屁股", "尾巴", "少爷", "小气",
        "小伙", "将就", "对头", "对付", "寡妇", "家伙", "客气", "实在",
        "官司", "学问", "字号", "嫁妆", "媳妇", "媒人", "婆家", "娘家",
        "委屈", "姑娘", "姐夫", "妯娌", "妥当", "妖精", "奴才", "女婿",
        "头发", "太阳", "大爷", "大方", "大意", "大夫", "多少", "多么",
        "外甥", "壮实", "地道", "地方", "在乎", "困难", "嘴巴", "嘱咐",
        "嘟囔", "嘀咕", "喜欢", "喇嘛", "喇叭", "商量", "唾沫", "哑巴",
        "哈欠", "哆嗦", "咳嗽", "和尚", "告诉", "告示", "含糊", "吓唬",
        "后头", "名字", "名堂", "合同", "吆喝", "叫唤", "口袋", "厚道",
        "厉害", "千斤", "包袱", "包涵", "匀称", "勤快", "动静", "动弹",
        "功夫", "力气", "前头", "刺猬", "刺激", "别扭", "利落", "利索",
        "利害", "分析", "出息", "凑合", "凉快", "冷战", "冤枉", "冒失",
        "养活", "关系", "先生", "兄弟", "便宜", "使唤", "佩服", "作坊",
        "体面", "位置", "似的", "伙计", "休息", "什么", "人家", "亲戚",
        "亲家", "交情", "云彩", "事情", "买卖", "主意", "丫头", "丧气",
        "两口", "东西", "东家", "世故", "不由", "下水", "下巴", "上头",
        "上司", "丈夫", "丈人", "一辈", "那个", "菩萨", "父亲", "母亲",
        "咕噜", "邋遢", "费用", "冤家", "甜头", "介绍", "荒唐", "大人",
        "泥鳅", "幸福", "熟悉", "计划", "扑腾", "蜡烛", "姥爷", "照顾",
        "喉咙", "吉他", "弄堂", "蚂蚱", "凤凰", "拖沓", "寒碜", "糟蹋",
        "倒腾", "报复", "逻辑", "盘缠", "喽啰", "牢骚", "咖喱", "扫把",
        "惦记"
    };

    #endregion

    #region Must Not Neural Tone Words

    /// <summary>
    /// Words that should NOT get neutral tone applied.
    /// </summary>
    private static readonly HashSet<string> MustNotNeuralToneWords = new()
    {
        "男子", "女子", "分子", "原子", "量子", "莲子", "石子", "瓜子",
        "电子", "人人", "虎虎", "幺幺", "干嘛", "学子", "哈哈", "数数",
        "袅袅", "局地", "以下", "娃哈哈", "花花草草", "留得", "耕地",
        "想想", "熙熙", "攘攘", "卵子", "死死", "冉冉", "恳恳", "佼佼",
        "吵吵", "打打", "考考", "整整", "莘莘", "落地", "算子", "家家户户",
        "青青"
    };

    #endregion

    #region Pinyin Utilities

    /// <summary>
    /// Gets the tone number from a pinyin final string (last character).
    /// </summary>
    private static int GetTone(string final)
    {
        if (string.IsNullOrEmpty(final))
            return 0;
        char last = final[^1];
        if (char.IsDigit(last))
            return last - '0';
        return 0;
    }

    /// <summary>
    /// Replaces the tone number in a pinyin final string.
    /// </summary>
    private static string SetTone(string final, int tone)
    {
        if (string.IsNullOrEmpty(final))
            return final;
        char last = final[^1];
        if (char.IsDigit(last))
            return final[..^1] + tone.ToString();
        return final + tone.ToString();
    }

    /// <summary>
    /// Checks if all characters in a word have tone 3.
    /// </summary>
    private static bool AllToneThree(List<string> finals)
    {
        foreach (var f in finals)
        {
            if (GetTone(f) != 3)
                return false;
        }
        return true;
    }

    #endregion

    #region Pre-Merge Methods

    /// <summary>
    /// Merges segments for correct sandhi application.
    /// </summary>
    public static List<(string word, string pos)> PreMergeForModify(List<(string word, string pos)> segments)
    {
        // A per-call cache for GetWordFinals keyed by the literal word.
        // Both MergeContinuousThreeTones and MergeContinuousThreeTones2 call GetWordFinals
        // for nearly the same set of words; sharing the cache halves the lexicon hits
        // for any segment whose word survives the first merge.
        var finalsCache = new Dictionary<string, List<string>?>();

        var result = segments;
        result = MergeBu(result);
        result = MergeYi(result);
        result = MergeReduplication(result);
        result = MergeContinuousThreeTones(result, finalsCache);
        result = MergeContinuousThreeTones2(result, finalsCache);
        result = MergeEr(result);
        return result;
    }

    private static List<string>? GetWordFinalsCached(string word, Dictionary<string, List<string>?> cache)
    {
        if (cache.TryGetValue(word, out var cached))
            return cached;
        var fresh = GetWordFinals(word);
        cache[word] = fresh;
        return fresh;
    }

    /// <summary>
    /// Merge 不 with the following word.
    /// </summary>
    private static List<(string word, string pos)> MergeBu(List<(string word, string pos)> segments)
    {
        var result = new List<(string word, string pos)>();
        for (int i = 0; i < segments.Count; i++)
        {
            if (segments[i].word == "不" && i + 1 < segments.Count && !XEng.Contains(segments[i + 1].pos))
            {
                result.Add((segments[i].word + segments[i + 1].word, segments[i + 1].pos));
                i++; // skip next
            }
            else
            {
                result.Add(segments[i]);
            }
        }
        return result;
    }

    /// <summary>
    /// Merge 一 with surrounding words.
    /// Handles reduplication pattern like 看一看 (V一V) — requires POS == "v" to match Python.
    /// </summary>
    private static List<(string word, string pos)> MergeYi(List<(string word, string pos)> segments)
    {
        var result = new List<(string word, string pos)>();
        int i = 0;
        while (i < segments.Count)
        {
            if (segments[i].word == "一" && i > 0 && i + 1 < segments.Count)
            {
                // Check for V一V reduplication pattern: previous POS must be verb
                var prev = result.Count > 0 ? result[^1] : (word: "", pos: "");
                var next = segments[i + 1];
                if (prev.word.Length == 1 && next.word.Length == 1 &&
                    prev.word == next.word &&
                    prev.pos == "v" &&
                    !XEng.Contains(next.pos))
                {
                    // Merge into V一V pattern
                    result[^1] = (prev.word + "一" + next.word, prev.pos);
                    i += 2;
                    continue;
                }
            }

            if (segments[i].word == "一" && i + 1 < segments.Count && !XEng.Contains(segments[i + 1].pos))
            {
                result.Add(("一" + segments[i + 1].word, segments[i + 1].pos));
                i += 2;
            }
            else
            {
                result.Add(segments[i]);
                i++;
            }
        }
        return result;
    }

    /// <summary>
    /// Merge consecutive identical words (e.g. AA, ABAB → AAAB doubled forms).
    /// Mirrors Python tone_sandhi._merge_reduplication: compares full words of any length.
    /// </summary>
    private static List<(string word, string pos)> MergeReduplication(List<(string word, string pos)> segments)
    {
        var result = new List<(string word, string pos)>();
        for (int i = 0; i < segments.Count; i++)
        {
            var (word, pos) = segments[i];
            if (result.Count > 0 &&
                word == result[^1].word &&
                !XEng.Contains(pos))
            {
                var last = result[^1];
                result[^1] = (last.word + word, last.pos);
            }
            else
            {
                result.Add((word, pos));
            }
        }
        return result;
    }

    /// <summary>
    /// Merge adjacent all-tone-3 segments (max 3 chars total).
    /// Mirrors Python tone_sandhi._merge_continuous_three_tones:
    /// - Iterates segment-by-segment and merges into the previous result
    /// - Uses a merge_last array to prevent cascade (a segment that was just merged
    ///   in cannot itself trigger another merge)
    /// - Skips merging if the previous segment is itself a 2-char reduplication
    /// </summary>
    private static List<(string word, string pos)> MergeContinuousThreeTones(
        List<(string word, string pos)> segments,
        Dictionary<string, List<string>?> finalsCache)
    {
        var result = new List<(string word, string pos)>();
        if (segments.Count == 0)
            return result;

        // Pre-compute finals for each segment (mirrors Python sub_finals_list)
        var subFinals = new List<List<string>?>(segments.Count);
        foreach (var (word, pos) in segments)
        {
            if (XEng.Contains(pos))
            {
                subFinals.Add(null); // Python uses ['0'] sentinel; null means "treat as not all-tone-3"
            }
            else
            {
                subFinals.Add(GetWordFinalsCached(word, finalsCache));
            }
        }

        var mergeLast = new bool[segments.Count];

        for (int i = 0; i < segments.Count; i++)
        {
            var (word, pos) = segments[i];
            bool merged = false;

            if (i - 1 >= 0 && !XEng.Contains(pos))
            {
                var prevFinals = subFinals[i - 1];
                var curFinals = subFinals[i];
                if (prevFinals != null && curFinals != null &&
                    AllToneThree(prevFinals) && AllToneThree(curFinals) &&
                    !mergeLast[i - 1])
                {
                    var prevWord = segments[i - 1].word;
                    if (!IsReduplication(prevWord) && prevWord.Length + word.Length <= 3)
                    {
                        // Merge into the last result item (which is segments[i-1] possibly already merged)
                        var last = result[^1];
                        result[^1] = (last.word + word, last.pos);
                        mergeLast[i] = true;
                        merged = true;
                    }
                }
            }

            if (!merged)
            {
                result.Add((word, pos));
            }
        }

        return result;
    }

    /// <summary>
    /// Mirrors Python tone_sandhi._is_reduplication: word of length 2 with identical chars.
    /// </summary>
    private static bool IsReduplication(string word)
    {
        return word.Length == 2 && word[0] == word[1];
    }

    /// <summary>
    /// Merge when last char of prev segment and first char of current segment are both tone 3.
    /// Mirrors Python tone_sandhi._merge_continuous_three_tones_2 with merge_last cascade prevention.
    /// </summary>
    private static List<(string word, string pos)> MergeContinuousThreeTones2(
        List<(string word, string pos)> segments,
        Dictionary<string, List<string>?> finalsCache)
    {
        var result = new List<(string word, string pos)>();
        if (segments.Count == 0)
            return result;

        var subFinals = new List<List<string>?>(segments.Count);
        foreach (var (word, pos) in segments)
        {
            subFinals.Add(XEng.Contains(pos) ? null : GetWordFinalsCached(word, finalsCache));
        }

        var mergeLast = new bool[segments.Count];

        for (int i = 0; i < segments.Count; i++)
        {
            var (word, pos) = segments[i];
            bool merged = false;

            if (i - 1 >= 0 && !XEng.Contains(pos))
            {
                var prevFinals = subFinals[i - 1];
                var curFinals = subFinals[i];
                if (prevFinals != null && curFinals != null &&
                    prevFinals.Count > 0 && curFinals.Count > 0 &&
                    GetTone(prevFinals[^1]) == 3 &&
                    GetTone(curFinals[0]) == 3 &&
                    !mergeLast[i - 1])
                {
                    var prevWord = segments[i - 1].word;
                    if (!IsReduplication(prevWord) && prevWord.Length + word.Length <= 3)
                    {
                        var last = result[^1];
                        result[^1] = (last.word + word, last.pos);
                        mergeLast[i] = true;
                        merged = true;
                    }
                }
            }

            if (!merged)
            {
                result.Add((word, pos));
            }
        }

        return result;
    }

    /// <summary>
    /// Merge 儿 with the preceding word.
    /// </summary>
    private static List<(string word, string pos)> MergeEr(List<(string word, string pos)> segments)
    {
        var result = new List<(string word, string pos)>();
        for (int i = 0; i < segments.Count; i++)
        {
            if (segments[i].word == "儿" && result.Count > 0 && !XEng.Contains(result[^1].pos))
            {
                var last = result[^1];
                result[^1] = (last.word + "儿", last.pos);
            }
            else
            {
                result.Add(segments[i]);
            }
        }
        return result;
    }

    #endregion

    #region Tone Modification

    /// <summary>
    /// Applies tone sandhi rules to a word's finals.
    /// Rules are applied in order: Bu, Yi, Neural, Three.
    /// </summary>
    /// <param name="word">The Chinese word.</param>
    /// <param name="pos">Part-of-speech tag from jieba.</param>
    /// <param name="finals">List of pinyin finals with tone numbers (e.g., "a1", "eng2").</param>
    /// <returns>Modified list of finals with sandhi applied.</returns>
    public static List<string> ModifiedTone(string word, string pos, List<string> finals)
    {
        var result = new List<string>(finals);

        // Apply sandhi rules in order (in-place on the same list)
        BuSandhi(word, result);
        YiSandhi(word, result);
        NeuralSandhi(word, pos, result);
        ThreeSandhi(word, result);

        return result;
    }

    /// <summary>
    /// 不 tone sandhi:
    /// - Before tone 4: becomes tone 2 (bu2)
    /// - In V不V pattern: becomes neutral tone (bu5)
    /// </summary>
    private static void BuSandhi(string word, List<string> result)
    {
        if (word.Length < 2)
            return;

        for (int i = 0; i < word.Length; i++)
        {
            if (word[i] == '不' && i < result.Count)
            {
                // V不V pattern: 不 becomes neutral tone
                if (i > 0 && i + 1 < word.Length && word[i - 1] == word[i + 1])
                {
                    result[i] = SetTone(result[i], 5);
                }
                // Before tone 4: becomes tone 2
                else if (i + 1 < result.Count && GetTone(result[i + 1]) == 4)
                {
                    result[i] = SetTone(result[i], 2);
                }
            }
        }
    }

    /// <summary>
    /// 一 tone sandhi (mirrors Python tone_sandhi._yi_sandhi):
    /// - In pure digit sequences (一零零, 二一零): unchanged
    /// - In V一V reduplication (length-3 word with word[0]==word[2]): yi5
    /// - When word starts with "第一" (e.g. 第一, 第一次, 第一个): yi1 stays
    /// - Before tone 4 OR tone 5: yi2
    /// - Before other (non-zero) tones: yi4
    /// </summary>
    private static void YiSandhi(string word, List<string> result)
    {
        if (word.Length < 2)
            return;

        // Pure digit sequences: 一零, 一二三, 二一零 — 一 stays unchanged
        if (IsPureDigitSequence(word))
            return;

        // V一V reduplication: e.g. 看一看 (length-3, word[0] == word[-1])
        if (word.Length == 3 && word[1] == '一' && word[0] == word[^1] && result.Count >= 2)
        {
            result[1] = SetTone(result[1], 5);
            return;
        }

        // 第一X: 一 stays tone 1 (covers 第一, 第一次, 第一个, etc.)
        if (word.StartsWith("第一", StringComparison.Ordinal))
        {
            if (result.Count >= 2)
                result[1] = SetTone(result[1], 1);
            return;
        }

        for (int i = 0; i < word.Length; i++)
        {
            if (word[i] == '一' && i + 1 < word.Length && i < result.Count && i + 1 < result.Count)
            {
                int nextTone = GetTone(result[i + 1]);
                // Before tone 4 or tone 5 (neutral): becomes tone 2
                if (nextTone == 4 || nextTone == 5)
                {
                    result[i] = SetTone(result[i], 2);
                }
                // Before other tones (1, 2, 3): becomes tone 4
                else if (nextTone != 0)
                {
                    result[i] = SetTone(result[i], 4);
                }
            }
        }
    }

    /// <summary>
    /// Returns true if the word contains 一 and all non-一 characters are Chinese digits.
    /// Mirrors Python: all(item.isnumeric() for item in word if item != YI).
    /// </summary>
    private static bool IsPureDigitSequence(string word)
    {
        if (word.IndexOf('一') < 0)
            return false;
        foreach (var c in word)
        {
            if (c == '一')
                continue;
            if (!IsCjkNumeric(c))
                return false;
        }
        return true;
    }

    /// <summary>
    /// Neural (neutral) tone sandhi — mirrors Python tone_sandhi._neural_sandhi.
    /// Applies tone 5 to particles, suffixes, reduplication, directional complements,
    /// and words in the must_neural list (including sub-word matches via jieba cut_for_search).
    /// </summary>
    private static void NeuralSandhi(string word, string pos, List<string> result)
    {
        if (string.IsNullOrEmpty(word) || result.Count == 0)
            return;

        // Must-not list: early return
        if (MustNotNeuralToneWords.Contains(word))
            return;

        // Reduplication: for each j, if word[j] == word[j-1] and pos starts with n/v/a,
        // set finals[j] to tone 5.
        if (!string.IsNullOrEmpty(pos))
        {
            char posFirst = pos[0];
            if (posFirst == 'n' || posFirst == 'v' || posFirst == 'a')
            {
                for (int j = 1; j < word.Length && j < result.Count; j++)
                {
                    if (word[j] == word[j - 1])
                    {
                        result[j] = SetTone(result[j], 5);
                    }
                }
            }
        }

        int geIdx = word.IndexOf('个');
        char lastChar = word[^1];
        int lastIdx = result.Count - 1;

        bool handled = false;

        // Sentence-final modal particles: 吧呢啊呐噻嘛吖嗨呐哦哒滴哩哟喽啰耶喔诶
        if (ModalParticles.Contains(lastChar))
        {
            result[lastIdx] = SetTone(result[lastIdx], 5);
            handled = true;
        }
        // Structural particles at word end: 的地得
        else if (lastChar == '的' || lastChar == '地' || lastChar == '得')
        {
            result[lastIdx] = SetTone(result[lastIdx], 5);
            handled = true;
        }
        // Single-char aspect particles: 了着过 with aspect POS
        else if (word.Length == 1 && (lastChar == '了' || lastChar == '着' || lastChar == '过') &&
                 (pos == "ul" || pos == "uz" || pos == "ug"))
        {
            result[lastIdx] = SetTone(result[lastIdx], 5);
            handled = true;
        }
        // Plural/diminutive suffix: 们/子 on pronouns/nouns
        else if (word.Length > 1 && (lastChar == '们' || lastChar == '子') &&
                 (pos == "r" || pos == "n"))
        {
            result[lastIdx] = SetTone(result[lastIdx], 5);
            handled = true;
        }
        // Locative suffix: 上/下 on space/location/direction POS
        else if (word.Length > 1 && (lastChar == '上' || lastChar == '下') &&
                 (pos == "s" || pos == "l" || pos == "f"))
        {
            result[lastIdx] = SetTone(result[lastIdx], 5);
            handled = true;
        }
        // Directional complement: 来/去 after 上下进出回过起开
        else if (word.Length > 1 && (lastChar == '来' || lastChar == '去') &&
                 DirectionalStems.Contains(word[^2]))
        {
            result[lastIdx] = SetTone(result[lastIdx], 5);
            handled = true;
        }
        // 个 patterns: preceded by a numeric or 几有两半多各整每做是; or standalone 个
        else if ((geIdx >= 1 && geIdx < result.Count &&
                  (IsCjkNumeric(word[geIdx - 1]) || GePrecedents.Contains(word[geIdx - 1])))
                 || word == "个")
        {
            if (geIdx >= 0 && geIdx < result.Count)
            {
                result[geIdx] = SetTone(result[geIdx], 5);
            }
            else if (word == "个")
            {
                result[lastIdx] = SetTone(result[lastIdx], 5);
            }
            handled = true;
        }

        // Fallback: check full word or last 2 chars in must_neural
        if (!handled)
        {
            string lastTwo = word.Length >= 2 ? word[^2..] : word;
            if (MustNeuralToneWords.Contains(word) || MustNeuralToneWords.Contains(lastTwo))
            {
                result[lastIdx] = SetTone(result[lastIdx], 5);
            }
        }

        // Sub-word check via jieba cut_for_search (mirrors Python lines 131-138).
        // Split into two sub-words, for each check must_neural membership.
        // SplitWord always returns 2 elements (second may be empty).
        if (word.Length >= 1)
        {
            var wordList = JiebaBridge.SplitWord(word);
            int sub1Len = wordList[0].Length;
            // Sub-word 1: finals[0..sub1Len-1], last final at index sub1Len-1
            if (sub1Len > 0 && sub1Len <= result.Count)
            {
                string sub1 = wordList[0];
                string sub1LastTwo = sub1.Length >= 2 ? sub1[^2..] : sub1;
                if (MustNeuralToneWords.Contains(sub1) || MustNeuralToneWords.Contains(sub1LastTwo))
                {
                    result[sub1Len - 1] = SetTone(result[sub1Len - 1], 5);
                }
            }
            // Sub-word 2: finals[sub1Len..], last final at end of list
            string sub2 = wordList[1];
            if (!string.IsNullOrEmpty(sub2) && sub1Len < result.Count)
            {
                string sub2LastTwo = sub2.Length >= 2 ? sub2[^2..] : sub2;
                if (MustNeuralToneWords.Contains(sub2) || MustNeuralToneWords.Contains(sub2LastTwo))
                {
                    result[lastIdx] = SetTone(result[lastIdx], 5);
                }
            }
        }
    }

    private static readonly HashSet<char> ModalParticles = new(
        "吧呢啊呐噻嘛吖嗨哦哒滴哩哟喽啰耶喔诶");

    private static readonly HashSet<char> DirectionalStems = new("上下进出回过起开");

    private static readonly HashSet<char> GePrecedents = new("几有两半多各整每做是");

    private static readonly HashSet<char> CjkNumerics = new(
        "零一二三四五六七八九十百千万亿兆〇○两壹贰叁肆伍陆柒捌玖拾佰仟萬億");

    /// <summary>
    /// Matches Python str.isnumeric() for CJK numeric characters.
    /// .NET's CharUnicodeInfo.GetNumericValue does not recognize most CJK ideographic numerics,
    /// so we use an explicit set covering common Chinese number characters.
    /// </summary>
    private static bool IsCjkNumeric(char c)
    {
        return char.IsAsciiDigit(c) || CjkNumerics.Contains(c);
    }

    /// <summary>
    /// Third tone sandhi — mirrors Python tone_sandhi._three_sandhi.
    /// - len 2: if all tone 3, first becomes tone 2
    /// - len 3: use jieba sub-word split (2+1 vs 1+2 prosody); handle all-tone-3 and mixed
    /// - len 4: split 2+2; apply all-tone-3 rule to each pair independently
    /// - other lengths (1, 5+): no changes (Python behavior)
    /// </summary>
    private static void ThreeSandhi(string word, List<string> result)
    {
        if (result.Count < 2)
            return;

        // len 2
        if (word.Length == 2 && result.Count == 2)
        {
            if (AllToneThree(result))
            {
                result[0] = SetTone(result[0], 2);
            }
            return;
        }

        // len 3 — requires jieba sub-word split for 2+1 vs 1+2 prosody
        if (word.Length == 3 && result.Count == 3)
        {
            var wordList = JiebaBridge.SplitWord(word);
            int sub1Len = wordList[0].Length;

            if (AllToneThree(result))
            {
                // 2+1 prosody (e.g. 蒙古/包): finals[0]=2, finals[1]=2
                if (sub1Len == 2)
                {
                    result[0] = SetTone(result[0], 2);
                    result[1] = SetTone(result[1], 2);
                }
                // 1+2 prosody (e.g. 纸/老虎): finals[1]=2
                else if (sub1Len == 1)
                {
                    result[1] = SetTone(result[1], 2);
                }
            }
            else
            {
                // Mixed tones — split finals and apply per sub-unit rules
                var sub1Finals = result.GetRange(0, sub1Len);
                var sub2Finals = result.GetRange(sub1Len, result.Count - sub1Len);
                var finalsList = new[] { sub1Finals, sub2Finals };

                for (int i = 0; i < finalsList.Length; i++)
                {
                    var sub = finalsList[i];
                    // e.g. 所有/人 — sub all-tone-3 with len 2: first → 2
                    if (sub.Count == 2 && AllToneThree(sub))
                    {
                        sub[0] = SetTone(sub[0], 2);
                    }
                    // e.g. 好/喜欢 — second sub not all-tone-3 but starts with tone 3,
                    // and first sub's last char is tone 3: first sub's last → 2
                    else if (i == 1 && !AllToneThree(sub) && sub.Count > 0 &&
                             GetTone(sub[0]) == 3 &&
                             finalsList[0].Count > 0 && GetTone(finalsList[0][^1]) == 3)
                    {
                        finalsList[0][^1] = SetTone(finalsList[0][^1], 2);
                    }
                }

                // Reassemble into result list
                int idx = 0;
                foreach (var sub in finalsList)
                {
                    foreach (var f in sub)
                    {
                        result[idx++] = f;
                    }
                }
            }
            return;
        }

        // len 4 — split 2+2, apply all-tone-3 rule to each pair independently
        if (word.Length == 4 && result.Count == 4)
        {
            var first = result.GetRange(0, 2);
            var second = result.GetRange(2, 2);
            if (AllToneThree(first))
            {
                result[0] = SetTone(result[0], 2);
            }
            if (AllToneThree(second))
            {
                result[2] = SetTone(result[2], 2);
            }
            return;
        }

        // Other lengths (1, 5+): Python performs no tone-3 sandhi changes at word level
    }

    #endregion

    #region Helper Methods

    /// <summary>
    /// Gets pinyin finals for each character in a word.
    /// Returns null if lookup is not available.
    /// This is a placeholder that should be connected to PinyinLookup when available.
    /// </summary>
    private static List<string>? GetWordFinals(string word)
    {
        // Delegate to PinyinLookup if available
        return PinyinLookupBridge.GetFinals(word);
    }

    #endregion
}

/// <summary>
/// Bridge to PinyinLookup for tone sandhi merge operations.
/// This allows ToneSandhi to work with or without a full pinyin dictionary.
/// </summary>
public static class PinyinLookupBridge
{
    private static volatile Func<string, List<string>?>? s_lookupFunc;

    /// <summary>
    /// Delegate for looking up pinyin finals of a word.
    /// Each string in the returned list is a final with tone number (e.g., "ao3").
    /// </summary>
    public static Func<string, List<string>?>? LookupFunc
    {
        get => s_lookupFunc;
        set => s_lookupFunc = value;
    }

    /// <summary>
    /// Gets the pinyin finals for a word using the registered lookup function.
    /// Returns null if no lookup function is registered.
    /// </summary>
    public static List<string>? GetFinals(string word)
    {
        return s_lookupFunc?.Invoke(word);
    }

    /// <summary>
    /// Initializes the bridge with a PinyinLookup instance.
    /// </summary>
    public static void Initialize(PinyinLookup lookup)
    {
        s_lookupFunc = word => lookup.GetFinals(word);
    }
}

/// <summary>
/// Bridge to jieba's CutForSearch for sub-word splitting in NeuralSandhi.
/// Mirrors the behavior of Python's jieba.cut_for_search used by tone_sandhi._split_word.
/// </summary>
public static class JiebaBridge
{
    private static volatile Func<string, IEnumerable<string>>? s_cutForSearchFunc;

    /// <summary>
    /// Delegate for jieba's cut_for_search (search-mode segmentation).
    /// </summary>
    public static Func<string, IEnumerable<string>>? CutForSearchFunc
    {
        get => s_cutForSearchFunc;
        set => s_cutForSearchFunc = value;
    }

    /// <summary>
    /// Splits a word using jieba's search-mode segmentation.
    /// Mirrors Python tone_sandhi._split_word exactly: always returns 2 elements
    /// (the second may be empty string when the word cannot be sub-segmented).
    /// </summary>
    public static List<string> SplitWord(string word)
    {
        var func = s_cutForSearchFunc;
        if (func == null || string.IsNullOrEmpty(word))
            return new List<string> { word, string.Empty };

        var subWords = func(word).Where(w => !string.IsNullOrEmpty(w)).ToList();
        if (subWords.Count == 0)
            return new List<string> { word, string.Empty };

        // Python: sorted(word_list, key=len, reverse=False)[0]
        var firstSubword = subWords.OrderBy(w => w.Length).First();
        var firstBeginIdx = word.IndexOf(firstSubword, StringComparison.Ordinal);

        if (firstBeginIdx == 0)
        {
            var second = firstSubword.Length >= word.Length
                ? string.Empty
                : word[firstSubword.Length..];
            return new List<string> { firstSubword, second };
        }
        else
        {
            var second = firstSubword.Length >= word.Length
                ? string.Empty
                : word[..^firstSubword.Length];
            return new List<string> { second, firstSubword };
        }
    }
}
