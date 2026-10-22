"""
Export a DeepPhonemizer ForwardTransformer checkpoint (e.g.
en_us_cmudict_ipa_forward.pt) to ONNX, and dump the tokenizer config the
C# runtime needs to encode/decode tensors.

Usage:
    pip install deep-phonemizer torch onnx onnxruntime
    python export_deepphonemizer_onnx.py \
        --checkpoint en_us_cmudict_ipa_forward.pt \
        --out-dir ./out

Outputs:
    out/deep_phonemizer.onnx
    out/deep_phonemizer_tokenizer.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

# PyTorch 2.6 flipped torch.load's weights_only default to True. The DP
# checkpoint pickles Preprocessor / SequenceTokenizer objects, so the safe
# loader rejects it. We trust the official checkpoint, so force the legacy
# behavior by patching torch.load before DP touches it.
_orig_torch_load = torch.load


def _torch_load_unsafe(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


torch.load = _torch_load_unsafe  # type: ignore[assignment]

from dp.phonemizer import Phonemizer  # noqa: E402  (must import after torch.load patch above)
from dp.preprocessing.text import SequenceTokenizer  # noqa: E402

DEFAULT_CHECKPOINT = Path(__file__).parent / "weights" / "en_us_cmudict_ipa_forward.pt"
DEFAULT_OUT_DIR = Path(__file__).parent / "out"


def _dump_tokenizer(phonemizer: Phonemizer, path: Path) -> None:
    """Persist the text + phoneme tokenizers as JSON for the C# side.

    SequenceTokenizer doesn't expose a `symbols` attribute — the real source
    of truth is `token_to_idx` (str -> int) and `idx_to_token` (int -> str),
    plus per-language start tokens like '<en_us>'. We dump all of it so the
    C# encoder can build inputs and decode outputs without re-deriving anything.
    """
    pre: SequenceTokenizer = phonemizer.predictor.text_tokenizer
    post: SequenceTokenizer = phonemizer.predictor.phoneme_tokenizer

    def serialize(tok: SequenceTokenizer) -> dict:
        # Per-language start token indices: SequenceTokenizer builds them as
        # f"<{lang}>" during __init__, so reproduce that mapping here.
        lang_to_start_index = {
            lang: tok.token_to_idx[f"<{lang}>"] for lang in tok.languages
        }
        return {
            "languages": list(tok.languages),
            "char_repeats": tok.char_repeats,
            "lowercase": tok.lowercase,
            "append_start_end": tok.append_start_end,
            "pad_index": tok.pad_index,
            "end_index": tok.end_index,
            "vocab_size": tok.vocab_size,
            "lang_to_start_index": lang_to_start_index,
            "token_to_idx": dict(tok.token_to_idx),
            # idx_to_token keys are ints; JSON forces them to strings — the
            # C# side will parse them back to ints.
            "idx_to_token": {str(i): t for i, t in tok.idx_to_token.items()},
            "special_tokens": sorted(tok.special_tokens),
        }

    payload = {
        "text_tokenizer": serialize(pre),
        "phoneme_tokenizer": serialize(post),
        "model_type": "forward_transformer",
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


class ForwardWrapper(torch.nn.Module):
    """ForwardTransformer expects a dict batch; ONNX wants positional tensors."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        # ForwardTransformer.generate returns argmax indices; for ONNX we want
        # the raw logits so the C# side can decide on argmax / beam / temperature.
        out = self.model({"text": text})
        # Newer DP versions return a dict {"logits": ...}; older return a tensor.
        if isinstance(out, dict):
            return out["logits"]
        return out


def export(checkpoint: Path, out_dir: Path, opset: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading checkpoint: {checkpoint}")
    phonemizer = Phonemizer.from_checkpoint(str(checkpoint))
    model = phonemizer.predictor.model
    model.eval()

    tokenizer_path = out_dir / "deep_phonemizer_tokenizer.json"
    print(f"[2/4] Dumping tokenizer config: {tokenizer_path}")
    _dump_tokenizer(phonemizer, tokenizer_path)

    # Dummy input: build a real tokenized word so _make_len_mask doesn't mask
    # every position (token id 0 == pad → fully-masked encoder → NaNs and a
    # useless sanity check). Use the first available language and a 5-letter
    # word to exercise char_repeats. Sequence length is a dynamic axis at
    # export, so any length works at runtime.
    text_tok = phonemizer.predictor.text_tokenizer
    sample_lang = text_tok.languages[0]
    dummy_ids = text_tok("hello", sample_lang)
    dummy = torch.tensor([dummy_ids], dtype=torch.long)
    wrapped = ForwardWrapper(model)
    wrapped.eval()  # propagate eval mode to the wrapper; the new dynamo
                    # exporter inspects the outer module's `training` flag.

    # Sanity-check the wrapper before tracing — if the dict-vs-tensor branch
    # is wrong, fail loudly here instead of with a cryptic ONNX trace error.
    with torch.no_grad():
        ref_out = wrapped(dummy)
    print(f"      reference output: shape={tuple(ref_out.shape)} dtype={ref_out.dtype}")

    onnx_path = out_dir / "deep_phonemizer.onnx"
    print(f"[3/4] Exporting ONNX (opset={opset}): {onnx_path}")
    torch.onnx.export(
        wrapped,
        (dummy,),
        str(onnx_path),
        input_names=["text"],
        output_names=["logits"],
        dynamic_axes={
            "text": {0: "batch", 1: "seq_len"},
            "logits": {0: "batch", 1: "seq_len"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )

    print("[4/4] Verifying ONNX output against PyTorch")
    try:
        import numpy as np
        import onnxruntime as ort

        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        onnx_out = sess.run(None, {"text": dummy.numpy()})[0]
        diff = float(np.abs(onnx_out - ref_out.numpy()).max())
        print(f"      max abs diff vs torch: {diff:.6g}")
        if diff > 1e-3:
            print("      WARNING: large divergence — inspect the wrapper / opset")
    except ImportError:
        print("      (skipped: install onnxruntime + numpy to verify)")

    print(f"\nDone. ONNX: {onnx_path}\n     Tokenizer: {tokenizer_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, type=Path)
    p.add_argument("--out-dir", default=DEFAULT_OUT_DIR, type=Path)
    p.add_argument(
        "--opset",
        default=19,
        type=int,
        help="ONNX opset (19 is safe for ONNX Runtime >= 1.16)",
    )
    args = p.parse_args()
    export(args.checkpoint, args.out_dir, args.opset)


if __name__ == "__main__":
    main()
