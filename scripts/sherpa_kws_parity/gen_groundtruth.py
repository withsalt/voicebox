"""Generate ground-truth token sequences for AutoTokenizer parity tests.

Uses sherpa_onnx.text2token (tokens_type='phone+ppinyin') to expand random
zh / en / mixed keywords into the same token format that the C# AutoTokenizer
should produce.

Output: JSONL (one record per line) at the path given on the CLI.
Run inside the openWakeWord conda env:
  conda activate openWakeWord
  python scripts/sherpa_kws_parity/gen_groundtruth.py <output.jsonl>
"""
import json
import random
import sys
from pathlib import Path

from sherpa_onnx import text2token

MODEL_DIR = Path(r"E:/ML/Audio/models/kws/sherpa-onnx-kws-zipformer-zh-en-3M-2025-12-20")
TOKENS = MODEL_DIR / "tokens.txt"
LEXICON = MODEL_DIR / "en.phone"

# A modest but representative pool. Each phrase listed must be encodable —
# Chinese ones must consist of CJK chars covered by Misaki's pinyin dict;
# English phrases must be in CMU en.phone (uppercase). Items here are kept
# short on purpose to keep diffs readable when a parity case fails.
ZH_WORDS = [
    "你好", "小爱", "同学", "天气", "中国", "上海", "北京", "苹果", "电脑", "手机",
    "音乐", "播放", "暂停", "下一首", "上一首", "增大音量", "减小音量",
    "周望军", "朱丽楠", "蒋友伯", "文森特", "落实", "见面会", "女儿", "法国",
    "你好世界", "小爱同学", "我爱你", "我饿了", "我们走吧", "天天向上",
    "好好学习", "可爱的", "彩虹", "太阳花", "西红柿", "草莓", "葡萄", "鱼香肉丝",
    "重要", "银行", "长大", "重新开始", "行行行", "请稍等",
    # 含上声字的, 用于触发 3+3 sandhi 测试
    "你好", "好好", "美好", "很好", "美女", "勇敢", "理想", "晚上好", "你早",
]

EN_WORDS = [
    "HELLO", "WORLD", "LIGHT", "UP", "STOP", "PLAY", "PAUSE", "NEXT", "PREVIOUS",
    "VOLUME", "WEATHER", "FRIDAY", "HAPPY", "FRIEND", "COMPUTER", "MUSIC", "RADIO",
    "GOOD", "MORNING", "EVENING", "NIGHT", "OPEN", "CLOSE", "DOOR", "WINDOW",
    "TURN", "ON", "OFF", "DIM", "BRIGHT",
]

MIXED = [
    "HELLO 小爱",
    "PLAY 音乐",
    "STOP 播放",
    "OPEN 门",
    "TURN ON 灯",
    "GOOD 早上",
    "小爱 PLAY",
    "你好 WORLD",
    "LIGHT UP 房间",
]


def random_combo(rng: random.Random) -> str:
    """Build a random keyword by sampling 1..3 fragments from the pools."""
    kind = rng.choice(["zh", "en", "mixed", "zh-multi", "en-multi"])
    if kind == "zh":
        return rng.choice(ZH_WORDS)
    if kind == "en":
        return rng.choice(EN_WORDS)
    if kind == "mixed":
        return rng.choice(MIXED)
    if kind == "zh-multi":
        return "".join(rng.sample(ZH_WORDS, k=2))  # 直接拼接, 形成长一点的中文短语
    # en-multi: 用空格隔开
    return " ".join(rng.sample(EN_WORDS, k=rng.randint(2, 3)))


def main(out_path: str, n_random: int = 60, seed: int = 20260529) -> None:
    rng = random.Random(seed)
    pool: list[str] = []
    pool.extend(ZH_WORDS)
    pool.extend(EN_WORDS)
    pool.extend(MIXED)
    for _ in range(n_random):
        pool.append(random_combo(rng))

    # 去重 (保持插入顺序)
    seen: set[str] = set()
    texts: list[str] = []
    for t in pool:
        if t not in seen:
            seen.add(t)
            texts.append(t)

    print(f"[gen] generating {len(texts)} expected token sequences via sherpa_onnx.text2token", file=sys.stderr)
    encoded = text2token(
        texts=texts,
        tokens=str(TOKENS),
        tokens_type="phone+ppinyin",
        lexicon=str(LEXICON),
    )

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for text, toks in zip(texts, encoded):
            rec = {"text": text, "tokens": " ".join(toks)}
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[gen] wrote {out_path}", file=sys.stderr)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(2)
    main(sys.argv[1])
