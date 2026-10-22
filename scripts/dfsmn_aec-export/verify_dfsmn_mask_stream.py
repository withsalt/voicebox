import argparse
import shutil
import site
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from export_dfsmn_mask_stream import StreamingDfsmnMaskVadCore


ROOT = Path(__file__).resolve().parent


def load_dfsmn_model(model_dir: Path):
    patched = ROOT / "modeling_modified" / "uni_deep_fsmn.py"
    dst = Path(site.getsitepackages()[-1]) / "modelscope" / "models" / "audio" / "aec" / "layers" / "uni_deep_fsmn.py"
    shutil.copyfile(patched, dst)

    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    pipe = pipeline(
        Tasks.acoustic_echo_cancellation,
        model=str(model_dir),
        device="cpu",
        trust_remote_code=True,
    )
    return pipe.model.cpu().eval().float()


def main():
    parser = argparse.ArgumentParser(description="Verify streaming ONNX mask+VAD core against PyTorch.")
    parser.add_argument("--model-dir", default=str(ROOT / "dependents" / "speech_dfsmn_aec_psm_16k"))
    parser.add_argument("--onnx", default=str(ROOT / "export" / "dfsmn_aec_mask_vad_stream.onnx"))
    parser.add_argument("--frames", type=int, default=17)
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    feat = rng.normal(0, 1, size=(1, args.frames, 240)).astype(np.float32)

    session = ort.InferenceSession(str(Path(args.onnx).resolve()), providers=["CPUExecutionProvider"])
    output_names = {o.name for o in session.get_outputs()}
    missing = {"mask", "vad_prob", "cache_out"} - output_names
    if missing:
        raise SystemExit(f"ONNX model is missing required output(s): {', '.join(sorted(missing))}")

    model = load_dfsmn_model(Path(args.model_dir).resolve())
    pt_model = StreamingDfsmnMaskVadCore(model).eval()
    pt_cache = torch.zeros(pt_model.num_layers, 1, pt_model.hidden_size, pt_model.cache_frames)
    pt_masks = []
    pt_vads = []
    with torch.inference_mode():
        for i in range(args.frames):
            mask, vad_prob, pt_cache = pt_model(torch.from_numpy(feat[:, i : i + 1]), pt_cache)
            pt_masks.append(mask.numpy())
            pt_vads.append(vad_prob.numpy())
    pt_mask_out = np.concatenate(pt_masks, axis=1)
    pt_vad_out = np.concatenate(pt_vads, axis=1)

    cache = np.zeros((pt_model.num_layers, 1, pt_model.hidden_size, pt_model.cache_frames), dtype=np.float32)
    ort_masks = []
    ort_vads = []
    requested_outputs = ["mask", "vad_prob", "cache_out"]
    for i in range(args.frames):
        results = session.run(requested_outputs, {"feat": feat[:, i : i + 1], "cache_in": cache})
        mask, vad_prob, cache = results
        ort_masks.append(mask)
        ort_vads.append(vad_prob)
    ort_mask_out = np.concatenate(ort_masks, axis=1)
    ort_vad_out = np.concatenate(ort_vads, axis=1)

    mask_diff = np.abs(pt_mask_out - ort_mask_out)
    print(f"max_abs={mask_diff.max():.8f}")
    print(f"mean_abs={mask_diff.mean():.8f}")
    if mask_diff.max() > 1e-4:
        raise SystemExit(1)
    vad_diff = np.abs(pt_vad_out - ort_vad_out)
    print(f"vad_max_abs={vad_diff.max():.8f}")
    print(f"vad_mean_abs={vad_diff.mean():.8f}")
    if vad_diff.max() > 1e-4:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
