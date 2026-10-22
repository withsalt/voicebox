"""Materialize existing evaluation speech as mono 16k float PCM for both runtimes."""
import argparse
import json
from pathlib import Path
from confucius_onnx.common import read_audio, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    original = json.loads(args.manifest.read_text('utf-8'))
    records = []
    for item in original['speechAssets']:
        if Path(item['id']).name != item['id'] or any(c in item['id'] for c in '/\\:'):
            raise ValueError('Manifest IDs must be simple filenames')
        audio = read_audio(args.manifest.parent / item['audioPath'])
        path = (args.out_dir / (item['id'] + '.f32')).resolve()
        audio.astype('<f4').tofile(path)
        records.append({'id': item['id'], 'audio': str(path), 'text': item['text'],
                        'language': 'Chinese', 'corpus': item['corpus'], 'samples': len(audio)})
    write_json(args.out_dir / 'manifest.json', records)
    print(f'{len(records)} cases, {sum(x["samples"] for x in records)/16000:.1f}s; '
          f'max {max(x["samples"] for x in records)/16000:.1f}s', flush=True)


if __name__ == '__main__':
    main()
