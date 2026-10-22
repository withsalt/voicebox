"""Reference streaming policy, shared by Python comparison and benchmark tools."""


def normalize(raw):
    raw = raw.split('|', 1)[0].replace('\ufffd', '')
    western, chinese = ',.!?;:()', '，。！？；：（）'
    result, previous = [], ''
    for char in raw:
        if '\u4e00' <= previous <= '\u9fff' and char in western:
            char = chinese[western.index(char)]
        elif previous and previous.isascii() and (previous.isalnum() or previous in '\'"') and char in chinese:
            char = western[chinese.index(char)]
        result.append(char)
        if not char.isspace():
            previous = char
    return ''.join(result)


def rollback(tokenizer, text, count):
    if count == 0:
        return text
    ids = tokenizer.encode(text)
    for keep in range(max(0, len(ids) - count), 0, -1):
        prefix = tokenizer.decode(ids[:keep])
        if '\ufffd' not in prefix:
            return prefix
    return ''


def stream(audio, tokenizer, infer, chunk_ms=160, lookahead_ms=160, rollback_tokens=1):
    prefix = ''
    snapshots = []
    for samples in range((chunk_ms + lookahead_ms) * 16, len(audio) + 1, chunk_ms * 16):
        raw = normalize(infer(audio[:samples], prefix, 4))
        proposed = rollback(tokenizer, raw, rollback_tokens)
        if proposed.startswith(prefix):
            prefix = proposed
        snapshots.append({'samples': samples, 'text': raw.strip(), 'committed': prefix.strip()})
    raw = normalize(infer(audio, prefix, 256))
    snapshots.append({'samples': len(audio), 'text': raw.strip(), 'committed': raw.strip(), 'final': True})
    return {'text': raw.strip(), 'snapshots': snapshots}
