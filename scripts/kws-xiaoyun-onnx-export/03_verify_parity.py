# -*- coding: utf-8 -*-
"""End-to-end verification of the streaming ONNX KWS on the real example wav.

1. fbank80(dither=0) -> LFR(5,3) -> CMVN  (deterministic frontend)
2. full-utterance torch forward vs chunked ONNX streaming parity
3. streaming CTC-prefix-beam-search keyword decode (prototype of the C# decoder)
4. dump test vectors for C# parity tests
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
import soundfile as sf
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model", "speech_charctc_kws_phone-xiaoyun")
ONNX_DIR = os.path.join(SCRIPT_DIR, "outputs", "onnx")
VEC_DIR = os.path.join(SCRIPT_DIR, "outputs", "vectors")
WAV = os.path.join(MODEL_DIR, "example", "kws_xiaoyunxiaoyun.wav")

NUM_LAYERS, PROJ_DIM, CACHE_LEN, DELAY, WARMUP = 4, 128, 11, 8, 35
CHUNK = 6  # LFR frames per ONNX call (= 180ms audio)


def make_frontend():
    from funasr.frontends.wav_frontend import WavFrontend
    return WavFrontend(
        fs=16000, window="hamming", n_mels=80, frame_length=25, frame_shift=10,
        lfr_m=5, lfr_n=3, dither=0.0,
        cmvn_file=os.path.join(MODEL_DIR, "funasr", "am.mvn.dim80_l2r2"),
    )


class StreamingKwsDecoder:
    """Frame-synchronous CTC prefix beam search keyword spotter.

    Mirrors funasr KwsCtcPrefixDecoder but adapted for endless streams:
    renormalization, stale-hypothesis pruning, max keyword span.
    """

    def __init__(self, keyword_token_ids, score_beam=3, path_beam=20,
                 prob_floor=0.05, max_span_frames=83, threshold=0.5):
        self.keyword = list(keyword_token_ids)
        self.tokenset = {0} | set(self.keyword)
        self.score_beam = score_beam
        self.path_beam = path_beam
        self.prob_floor = prob_floor
        self.max_span = max_span_frames  # 2.5s / 30ms
        self.threshold = threshold
        self.reset()

    def reset(self):
        self.t = 0
        self.hyps = [(tuple(), (1.0, 0.0, []))]

    def _is_sublist(self, main, check):
        n, m = len(main), len(check)
        if n < m:
            return -1
        for i in range(n - m + 1):
            if list(main[i:i + m]) == list(check):
                return i
        return -1

    def step(self, probs):
        """probs: np array [vocab]; returns (detected, score, end_frame) or None"""
        t = self.t
        self.t += 1

        top_idx = np.argpartition(probs, -self.score_beam)[-self.score_beam:]
        filtered = [(int(i), float(probs[i])) for i in top_idx
                    if probs[i] > self.prob_floor and int(i) in self.tokenset]

        if filtered:
            next_hyps = defaultdict(lambda: (0.0, 0.0, []))
            for s, ps in filtered:
                for prefix, (pb, pnb, nodes) in self.hyps:
                    last = prefix[-1] if prefix else None
                    if s == 0:
                        n_pb, n_pnb, _ = next_hyps[prefix]
                        next_hyps[prefix] = (n_pb + (pb + pnb) * ps, n_pnb, list(nodes))
                    elif s == last:
                        if not math.isclose(pnb, 0.0, abs_tol=1e-6):
                            n_pb, n_pnb, n_nodes = next_hyps[prefix]
                            nn = list(nodes)
                            if ps > nn[-1][2]:
                                nn[-1] = (s, t, ps)
                            next_hyps[prefix] = (n_pb, n_pnb + pnb * ps, nn)
                        if not math.isclose(pb, 0.0, abs_tol=1e-6):
                            n_prefix = prefix + (s,)
                            n_pb, n_pnb, n_nodes = next_hyps[n_prefix]
                            nn = list(nodes) + [(s, t, ps)]
                            next_hyps[n_prefix] = (n_pb, n_pnb + pb * ps, nn)
                    else:
                        n_prefix = prefix + (s,)
                        n_pb, n_pnb, n_nodes = next_hyps[n_prefix]
                        if n_nodes:
                            nn = n_nodes
                            if ps > nn[-1][2]:
                                nn[-1] = (s, t, ps)
                        else:
                            nn = list(nodes) + [(s, t, ps)]
                        next_hyps[n_prefix] = (n_pb + 0.0, n_pnb + (pb + pnb) * ps, nn)
            hyps = sorted(next_hyps.items(), key=lambda x: x[1][0] + x[1][1], reverse=True)
            self.hyps = hyps[:self.path_beam]

        # prune stale hyps + renormalize
        kept = []
        best = 0.0
        for prefix, (pb, pnb, nodes) in self.hyps:
            if nodes and t - nodes[-1][1] > self.max_span:
                continue
            kept.append((prefix, (pb, pnb, nodes)))
            best = max(best, pb + pnb)
        if not any(len(p) == 0 for p, _ in kept):
            kept.append((tuple(), (best if best > 0 else 1.0, 0.0, [])))
        if best > 0 and (best < 1e-10 or best > 1e10):
            kept = [(p, (pb / best, pnb / best, nodes)) for p, (pb, pnb, nodes) in kept]
        self.hyps = kept

        # keyword match over all hypotheses
        for prefix, (pb, pnb, nodes) in self.hyps:
            off = self._is_sublist(prefix, self.keyword)
            if off == -1:
                continue
            span_nodes = nodes[off:off + len(self.keyword)]
            if span_nodes[-1][1] - span_nodes[0][1] > self.max_span:
                continue
            score = 1.0
            for _, _, p in span_nodes:
                score *= p
            score = math.sqrt(score)
            if score >= self.threshold:
                self.reset()
                return True, score, t
        return None


def main():
    os.makedirs(VEC_DIR, exist_ok=True)

    samples, sr = sf.read(WAV, dtype="float32")
    assert sr == 16000
    print(f"wav: {len(samples)} samples ({len(samples)/sr:.2f}s)")

    frontend = make_frontend()
    speech = torch.from_numpy(samples).unsqueeze(0)
    lens = torch.tensor([len(samples)], dtype=torch.int32)
    feats, feats_len = frontend(speech, lens)
    feats_np = feats.numpy()
    T = feats_np.shape[1]
    print(f"feats: {feats_np.shape}")

    # full-utterance reference (torch, via funasr encoder)
    from funasr import AutoModel
    am = AutoModel(model=MODEL_DIR, keywords="小云小云", device="cpu", disable_update=True)
    encoder = am.model.encoder.eval()
    with torch.no_grad():
        full_probs = torch.softmax(encoder(feats), dim=-1).numpy()[0]

    # chunked ONNX streaming
    import onnxruntime as ort
    sess = ort.InferenceSession(os.path.join(ONNX_DIR, "kws_streaming.onnx"),
                                providers=["CPUExecutionProvider"])
    caches = [np.zeros((1, PROJ_DIM, CACHE_LEN, 1), dtype=np.float32) for _ in range(NUM_LAYERS)]
    outs = []
    for s in range(0, T, CHUNK):
        res = sess.run(None, {"feats": feats_np[:, s:s + CHUNK, :],
                              **{f"cache{i}": caches[i] for i in range(NUM_LAYERS)}})
        outs.append(res[0])
        caches = res[1:]
    stream_probs = np.concatenate(outs, axis=1)[0]  # [T, 2599]

    diff = np.abs(stream_probs[WARMUP:T] - full_probs[WARMUP - DELAY:T - DELAY]).max()
    print(f"real-wav steady-state parity (onnx stream vs torch full): {diff:.3e}")
    assert diff < 1e-4

    # keyword tokens
    with open(os.path.join(ONNX_DIR, "tokens.txt"), encoding="utf-8") as f:
        tokens = [line.rstrip("\n") for line in f]
    tok2id = {tok: i for i, tok in enumerate(tokens)}
    kw_ids = [tok2id[c] for c in "小云小云"]
    print(f"keyword token ids: {kw_ids}")

    # streaming decode
    decoder = StreamingKwsDecoder(kw_ids)
    detections = []
    for t in range(stream_probs.shape[0]):
        hit = decoder.step(stream_probs[t])
        if hit:
            detections.append(hit)
            print(f"DETECTED at output frame {hit[2]} (~{hit[2]*0.03:.2f}s) score={hit[1]:.4f}")
    assert len(detections) == 1, f"expected exactly 1 detection, got {len(detections)}"

    # negative check: pure silence must not trigger
    sil = np.zeros(16000 * 5, dtype=np.float32)
    sf_feats, _ = frontend(torch.from_numpy(sil).unsqueeze(0), torch.tensor([len(sil)], dtype=torch.int32))
    sf_np = sf_feats.numpy()
    caches = [np.zeros((1, PROJ_DIM, CACHE_LEN, 1), dtype=np.float32) for _ in range(NUM_LAYERS)]
    dec2 = StreamingKwsDecoder(kw_ids)
    hits = 0
    for s in range(0, sf_np.shape[1], CHUNK):
        res = sess.run(None, {"feats": sf_np[:, s:s + CHUNK, :],
                              **{f"cache{i}": caches[i] for i in range(NUM_LAYERS)}})
        caches = res[1:]
        for t in range(res[0].shape[1]):
            if dec2.step(res[0][0, t]):
                hits += 1
    print(f"silence false alarms: {hits}")
    assert hits == 0

    # dump C# test vectors
    samples.tofile(os.path.join(VEC_DIR, "input_samples.bin"))
    feats_np[0].astype(np.float32).tofile(os.path.join(VEC_DIR, "feats.bin"))
    stream_probs.astype(np.float32).tofile(os.path.join(VEC_DIR, "probs_stream.bin"))
    with open(os.path.join(VEC_DIR, "vectors_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "wav": "input_samples.bin float32 mono 16k",
            "num_samples": int(len(samples)),
            "feats_shape": [int(T), 400],
            "probs_shape": [int(stream_probs.shape[0]), 2599],
            "chunk_lfr_frames": CHUNK,
            "keyword_token_ids": kw_ids,
            "detection_frames": [d[2] for d in detections],
            "detection_scores": [d[1] for d in detections],
        }, f, ensure_ascii=False, indent=2)
    print(f"test vectors written to {VEC_DIR}")
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
