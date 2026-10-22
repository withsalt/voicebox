# -*- coding: utf-8 -*-
"""Baseline: run FunASR FsmnKWS on the bundled example wav to verify env + model."""
import os
import sys

MODEL_DIR = os.path.join(os.path.dirname(__file__), "model", "speech_charctc_kws_phone-xiaoyun")
WAV = os.path.join(MODEL_DIR, "example", "kws_xiaoyunxiaoyun.wav")

from funasr import AutoModel

model = AutoModel(
    model=MODEL_DIR,
    keywords="小云小云",
    device="cpu",
    disable_update=True,
)

res = model.generate(input=WAV, cache={}, output_dir=os.path.join(os.path.dirname(__file__), "outputs", "baseline"))
print("RESULT:", res)
