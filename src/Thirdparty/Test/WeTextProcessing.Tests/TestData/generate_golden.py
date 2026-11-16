# -*- coding: utf-8 -*-
"""Exports the FST models and regenerates the golden corpus (*.tsv).

The golden files pin WeTextProcessing.NET output to the Python WeTextProcessing
package: every line is "<input>\t<python_output>" where python_output comes from
the official Python implementation running the same FSTs exported to
runtimes/models/WeTextProcessing. Inputs are collected from the official
per-rule test data of the Python package.

Usage (requires pynini and importlib_resources):
    python generate_golden.py <path-to-WeTextProcessing-source>
    python generate_golden.py E:\\Temp\\wetextprocessing-1.3.0
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path, help="Extracted WeTextProcessing 1.3.0 source distribution")
args = parser.parse_args()
PKG = str(args.source.resolve())
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = Path(__file__).resolve().parents[3] / "WeTextProcessing.NET/runtimes/models/WeTextProcessing"
if "\nVersion: 1.3.0\n" not in (args.source / "PKG-INFO").read_text(encoding="utf-8"):
    sys.exit("Expected the official WeTextProcessing 1.3.0 source distribution")
sys.path.insert(0, PKG)

import pynini  # noqa: E402
from tn.chinese.normalizer import Normalizer as ZhNormalizer  # noqa: E402
from tn.english.normalizer import Normalizer as EnNormalizer  # noqa: E402
from itn.chinese.inverse_normalizer import InverseNormalizer as ZhInverseNormalizer  # noqa: E402
from itn.english.inverse_normalizer import InverseNormalizer as EnInverseNormalizer  # noqa: E402


def collect_inputs(pattern):
    inputs = []
    for path in sorted(glob.glob(pattern)):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if " => " not in line:
                    continue
                text = line.split(" => ")[0].strip()
                if text and "\t" not in text:
                    inputs.append(text)
    return list(dict.fromkeys(inputs))


def generate(name, normalizer, pattern):
    inputs = collect_inputs(pattern)
    if not inputs:
        raise ValueError(f"No upstream test inputs found: {pattern}")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    for role in ("tagger", "verbalizer"):
        getattr(normalizer, role).write(str(MODEL_DIR / f"{name}_{role}.fst"))
    out_path = os.path.join(OUT_DIR, f"{name}.tsv")
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for text in inputs:
            f.write(f"{text}\t{normalizer.normalize(text)}\n")
    print(f"{name}: {len(inputs)} entries written -> {out_path}", flush=True)


generate(
    "zh_tn",
    ZhNormalizer(cache_dir=False),
    os.path.join(PKG, "tn", "chinese", "test", "data", "*.txt"),
)
generate(
    "zh_itn",
    ZhInverseNormalizer(cache_dir=False),
    os.path.join(PKG, "itn", "chinese", "test", "data", "*.txt"),
)
generate(
    "en_tn",
    EnNormalizer(cache_dir=False),
    os.path.join(PKG, "tn", "english", "test", "data", "*.txt"),
)
generate(
    "en_itn",
    EnInverseNormalizer(cache_dir=False),
    os.path.join(PKG, "itn", "english", "test", "data", "*.txt"),
)

manifest = {
    "wetextprocessing_version": "1.3.0",
    "source_archive": "wetextprocessing-1.3.0.tar.gz",
    "source_sha256": "217a939201fc1e3324c67bc9983c4f4e56dc9dc37b850845b3d2e0a1daf6fb5e",
    "pynini_version": pynini.__version__,
    "options": "Default constructor options; cache_dir=False",
    "sha256": {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(MODEL_DIR.glob("*.fst"))
    },
}
(MODEL_DIR / "manifest.json").write_text(
    json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
)
