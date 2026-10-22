import argparse
import json
import shutil
import site
from pathlib import Path

import numpy as np
import onnxruntime as ort
import scipy.io.wavfile as wav
import scipy.signal
import torch
import yaml

from export_dfsmn_mask_stream import StreamingDfsmnMaskVadCore, load_dfsmn_model


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_DIR = ROOT / "dependents" / "speech_dfsmn_aec_psm_16k"
DEFAULT_ONNX = ROOT / ".." / ".." / "src" / "VoiceBox.AEC.DFSMN" / "runtimes" / "models" / "aec" / "dfsmn" / "dfsmn_aec_mask_vad_stream.onnx"


def patch_modelscope_fsmn():
    patched = ROOT / "modeling_modified" / "uni_deep_fsmn.py"
    dst = Path(site.getsitepackages()[-1]) / "modelscope" / "models" / "audio" / "aec" / "layers" / "uni_deep_fsmn.py"
    shutil.copyfile(patched, dst)


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = np.abs(a - b)
    return {
        "max_abs": float(diff.max()),
        "mean_abs": float(diff.mean()),
        "p999_abs": float(np.percentile(diff, 99.9)),
    }


def synthesize(base: torch.Tensor, mask: np.ndarray) -> np.ndarray:
    window = torch.hamming_window(640, periodic=False)
    spectrum = torch.view_as_real(
        torch.stft(base, 640, 320, 640, center=False, window=window, return_complex=True))
    torch_mask = torch.from_numpy(mask).permute(2, 1, 0).float()
    masked = spectrum * torch_mask
    audio = torch.istft(
        torch.view_as_complex(masked),
        640,
        320,
        640,
        center=False,
        window=window,
        length=len(base))
    return to_numpy(audio)


def audio_diff_stats(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "max_abs": float(abs_diff.max()),
        "mean_abs": float(abs_diff.mean()),
        "rms_diff": float(np.sqrt(np.mean(np.square(diff)))),
    }


def encode_float32_base64(array: np.ndarray) -> str:
    return __import__("base64").b64encode(np.ascontiguousarray(array, dtype=np.float32).tobytes()).decode("ascii")


def run_pair(
        name,
        near_name,
        far_name,
        model,
        core,
        session,
        preprocessor,
        examples_dir: Path,
        golden_frames: int = 0):
    processed = preprocessor({
        "nearend_mic": str(examples_dir / near_name),
        "farend_speech": str(examples_dir / far_name),
    })
    feat = processed["feature"].detach().cpu().numpy().astype(np.float32)
    base = processed["base"].detach().cpu().float()
    frames = feat.shape[0]

    with torch.inference_mode():
        pt_full_mask, pt_full_vad = model(torch.from_numpy(feat).unsqueeze(0))
    pt_full_mask = to_numpy(pt_full_mask)
    pt_full_vad = to_numpy(pt_full_vad)

    pt_cache = torch.zeros(core.num_layers, 1, core.hidden_size, core.cache_frames)
    pt_stream_masks = []
    pt_stream_vads = []
    with torch.inference_mode():
        for i in range(frames):
            mask, vad, pt_cache = core(torch.from_numpy(feat[i:i + 1]).unsqueeze(0), pt_cache)
            pt_stream_masks.append(to_numpy(mask))
            pt_stream_vads.append(to_numpy(vad))
    pt_stream_mask = np.concatenate(pt_stream_masks, axis=1)
    pt_stream_vad = np.concatenate(pt_stream_vads, axis=1)

    cache = np.zeros((core.num_layers, 1, core.hidden_size, core.cache_frames), dtype=np.float32)
    onnx_masks = []
    onnx_vads = []
    for i in range(frames):
        mask, vad, cache = session.run(
            ["mask", "vad_prob", "cache_out"],
            {"feat": feat[i:i + 1][None, :, :], "cache_in": cache})
        onnx_masks.append(mask)
        onnx_vads.append(vad)
    onnx_mask = np.concatenate(onnx_masks, axis=1)
    onnx_vad = np.concatenate(onnx_vads, axis=1)

    audio_pt_full = synthesize(base, pt_full_mask)
    audio_pt_stream = synthesize(base, pt_stream_mask)
    audio_onnx = synthesize(base, onnx_mask)

    result = {
        "name": name,
        "nearend_mic": near_name,
        "farend_speech": far_name,
        "samples": int(len(base)),
        "frames": int(frames),
        "feature_shape": list(feat.shape),
        "pt_stream_vs_onnx": {
            "mask": diff_stats(pt_stream_mask, onnx_mask),
            "vad_prob": diff_stats(pt_stream_vad, onnx_vad),
            "audio": audio_diff_stats(audio_pt_stream, audio_onnx),
        },
        "pt_full_vs_pt_stream": {
            "mask": diff_stats(pt_full_mask, pt_stream_mask),
            "vad_prob": diff_stats(pt_full_vad, pt_stream_vad),
        },
        "pt_full_vs_onnx": {
            "mask": diff_stats(pt_full_mask, onnx_mask),
            "vad_prob": diff_stats(pt_full_vad, onnx_vad),
            "audio": audio_diff_stats(audio_pt_full, audio_onnx),
        },
        "vad_range": {
            "pt_min": float(pt_full_vad.min()),
            "pt_max": float(pt_full_vad.max()),
            "onnx_min": float(onnx_vad.min()),
            "onnx_max": float(onnx_vad.max()),
        },
    }

    if golden_frames > 0:
        take = min(golden_frames, frames)
        result["golden"] = {
            "frames": int(take),
            "feature_shape": [int(take), 240],
            "mask_shape": [1, int(take), 321],
            "vad_shape": [1, int(take), 1],
            "feature_b64": encode_float32_base64(feat[:take]),
            "mask_b64": encode_float32_base64(pt_full_mask[:, :take, :]),
            "vad_b64": encode_float32_base64(pt_full_vad[:, :take, :]),
        }

    return result


def load_local_wav(path):
    sample_rate, data = wav.read(path)
    if len(data.shape) > 1:
        raise ValueError(f"Expected mono wav: {path}")
    if sample_rate != 16000:
        gcd = np.gcd(sample_rate, 16000)
        data = scipy.signal.resample_poly(data.astype(np.float32), 16000 // gcd, sample_rate // gcd)
    return data.astype(np.float32), 16000


def main():
    parser = argparse.ArgumentParser(description="Verify exported DFSMN ONNX on the bundled example wav files.")
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--onnx", default=str(DEFAULT_ONNX))
    parser.add_argument("--tolerance", type=float, default=1e-4)
    parser.add_argument("--golden-frames", type=int, default=0)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    onnx_path = Path(args.onnx).resolve()
    examples_dir = model_dir / "examples"

    patch_modelscope_fsmn()

    config = yaml.full_load((model_dir / "dey_mini.yaml").read_text(encoding="utf-8"))
    config["io"]["mvn"] = str(model_dir / "feature.DEY.mvn.txt")

    # Import after patching site-packages so the preprocessor and model use the same FSMN implementation.
    from modelscope.preprocessors import LinearAECAndFbank

    # The bundled example wav files are local mono files. ModelScope's default loader imports
    # librosa unconditionally for optional resampling, so use scipy here to keep this verifier
    # runnable in the project's lean voicebox environment.
    LinearAECAndFbank.load_wav = staticmethod(load_local_wav)

    model = load_dfsmn_model(model_dir).eval().float()
    core = StreamingDfsmnMaskVadCore(model).eval()
    preprocessor = LinearAECAndFbank(config["io"])
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

    pairs = [
        ("example2", "nearend_mic.wav", "farend_speech.wav"),
        ("example1", "nearend_mic1.wav", "farend_speech1.wav"),
    ]
    results = [
        run_pair(name, near, far, model, core, session, preprocessor, examples_dir, args.golden_frames)
        for name, near, far in pairs
    ]

    print(json.dumps({
        "model_dir": str(model_dir),
        "onnx": str(onnx_path),
        "results": results,
    }, indent=2, ensure_ascii=False))

    max_mask = max(item["pt_stream_vs_onnx"]["mask"]["max_abs"] for item in results)
    max_vad = max(item["pt_stream_vs_onnx"]["vad_prob"]["max_abs"] for item in results)
    if max(max_mask, max_vad) > args.tolerance:
        raise SystemExit(
            f"ONNX streaming parity failed: mask_max_abs={max_mask:.8g}, vad_max_abs={max_vad:.8g}")


if __name__ == "__main__":
    main()
