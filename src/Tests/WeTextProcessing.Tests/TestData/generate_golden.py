# -*- coding: utf-8 -*-
"""Regenerates the golden corpus files (*.tsv) in this directory.

The golden files pin WeTextProcessing.NET output to the Python WeTextProcessing
package: every line is "<input>\t<python_output>" where python_output comes from
the official Python implementation running the same precompiled FSTs that ship
in runtimes/models/WeTextProcessing. Inputs are collected from the official
per-rule test data of the Python package.

Usage (requires pynini, e.g. the "voicebox" conda env):
    python generate_golden.py <path-to-WeTextProcessing-source>
    python generate_golden.py E:\\Temp\\wetextprocessing-1.2.0
"""
import glob
import os
import sys

if len(sys.argv) != 2:
    sys.exit(__doc__)
PKG = sys.argv[1]
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PKG)

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
    out_path = os.path.join(OUT_DIR, f"{name}.tsv")
    skipped = 0
    with open(out_path, "w", encoding="utf-8", newline="\n") as f:
        for text in inputs:
            try:
                f.write(f"{text}\t{normalizer.normalize(text)}\n")
            except Exception as e:
                skipped += 1
                print(f"  SKIP ({e!r}): {text}")
    print(f"{name}: {len(inputs) - skipped} entries written, {skipped} skipped -> {out_path}")


generate(
    "zh_tn",
    ZhNormalizer(cache_dir=os.path.join(PKG, "tn")),
    os.path.join(PKG, "tn", "chinese", "test", "data", "*.txt"),
)
generate(
    "zh_itn",
    ZhInverseNormalizer(cache_dir=os.path.join(PKG, "itn")),
    os.path.join(PKG, "itn", "chinese", "test", "data", "*.txt"),
)
generate(
    "en_tn",
    EnNormalizer(cache_dir=os.path.join(PKG, "tn")),
    os.path.join(PKG, "tn", "english", "test", "data", "*.txt"),
)
generate(
    "en_itn",
    EnInverseNormalizer(cache_dir=os.path.join(PKG, "itn")),
    os.path.join(PKG, "itn", "english", "test", "data", "*.txt"),
)
