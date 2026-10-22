"""Download the small LibriSpeech test fixture for reproducible English evaluation.

Requires pyarrow (evaluation only). Audio is kept locally, outside version control.
The source dataset card contains no license field; original LibriSpeech is CC BY 4.0.
"""
import argparse
import io
import json
from pathlib import Path
import urllib.request
import pyarrow.parquet as pq
import soundfile as sf

URL = ('https://huggingface.co/datasets/hf-internal-testing/librispeech_asr_dummy/'
       'resolve/main/clean/validation-00000-of-00001.parquet')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out-dir', type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(URL, timeout=60) as response:
        raw = response.read()
    records = []
    for row in pq.read_table(io.BytesIO(raw)).to_pylist():
        audio, rate = sf.read(io.BytesIO(row['audio']['bytes']), dtype='float32')
        if rate != 16000 or audio.ndim != 1:
            raise ValueError('Expected mono 16 kHz LibriSpeech audio')
        if len(audio) > 480000:
            continue
        identifier = str(row['id'])
        if not identifier or any(c not in '0123456789-' for c in identifier):
            raise ValueError(f'Unexpected dataset identifier: {identifier}')
        path = (args.out_dir / (identifier + '.f32')).resolve()
        audio.astype('<f4').tofile(path)
        records.append({'id': identifier, 'audio': str(path), 'text': row['text'],
                        'language': 'English', 'samples': len(audio), 'corpus': 'librispeech-test-clean-dummy'})
    (args.out_dir / 'manifest.json').write_text(json.dumps(records, indent=2), encoding='utf-8')
    (args.out_dir / 'source.json').write_text(json.dumps({'url': URL, 'samples': len(records),
        'license': 'Dataset card has no license field; original LibriSpeech is CC BY 4.0. Evaluation copy only.'}, indent=2), encoding='utf-8')
    print(f'Prepared {len(records)} English samples')


if __name__ == '__main__':
    main()
