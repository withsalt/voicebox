"""Generate reproducible WeSpeaker parity vectors for the C# test suite.

The reference path intentionally mirrors wespeaker/bin/infer_onnx.py:
PCM float -> x32768 -> torchaudio Kaldi Hamming fbank -> utterance CMN ->
cnceleb_resnet34.onnx. The output is used to test each C# pipeline stage.
"""

import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import numpy as np
import onnxruntime as ort
import soundfile as sf
import torch
import torchaudio.compliance.kaldi as kaldi


REPO_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "VoiceBox.Speaker.WeSpeaker"
    / "runtimes"
    / "models"
    / "speaker"
    / "wespeaker"
    / "cnceleb_resnet34.onnx"
)
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "vectors"
SPEECHBRAIN_REVISION = "e5cb1f65b940634215650aa1171e0440d0808123"
SAMPLES = {
    "spk1_snt1": (
        "tests/samples/ASR/spk1_snt1.wav",
        "2f7315ccf543b6368528b098cd895098c154f222cd55ff7b5f66b50feb383c56",
    ),
    "spk1_snt2": (
        "tests/samples/ASR/spk1_snt2.wav",
        "e1f125a318f4026991bf78f5c41481be0fbcf9b3b2fd72284f9824cb851c3034",
    ),
    "spk2_snt1": (
        "tests/samples/ASR/spk2_snt1.wav",
        "5f3f534ba99f159358c6ee2e4fb5e42b3b9399f37f4e290651405c76095f26f0",
    ),
}
MODEL_SHA256 = "78817ca21a9707ad886d50745162231027a09c997fbf2ecf741c5d8ff4db1bf8"
WINDOW_FRAMES = 198
HOP_FRAMES = 25


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_samples(download_dir: Path) -> None:
    download_dir.mkdir(parents=True, exist_ok=True)
    for stem, (relative_path, expected_hash) in SAMPLES.items():
        destination = download_dir / f"{stem}.wav"
        url = (
            "https://raw.githubusercontent.com/speechbrain/speechbrain/"
            f"{SPEECHBRAIN_REVISION}/{relative_path}"
        )
        if not destination.exists() or sha256(destination) != expected_hash:
            with urllib.request.urlopen(url) as response, destination.open("wb") as output:
                shutil.copyfileobj(response, output)
        actual_hash = sha256(destination)
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {destination}: {actual_hash} != {expected_hash}"
            )


def compute_fbank(samples: np.ndarray) -> np.ndarray:
    waveform = torch.from_numpy(samples).unsqueeze(0) * (1 << 15)
    features = kaldi.fbank(
        waveform,
        num_mel_bins=80,
        frame_length=25,
        frame_shift=10,
        dither=0.0,
        sample_frequency=16000,
        window_type="hamming",
        use_energy=False,
    )
    return features.numpy()


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(normalize(left), normalize(right)))


def save_float32(path: Path, values: np.ndarray) -> None:
    np.ascontiguousarray(values, dtype=np.float32).tofile(path)


def main() -> None:
    if sha256(MODEL_PATH) != MODEL_SHA256:
        raise RuntimeError(f"Missing or invalid model: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    download_samples(OUTPUT_DIR)
    session_options = ort.SessionOptions()
    session_options.inter_op_num_threads = 1
    session_options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(MODEL_PATH),
        sess_options=session_options,
        providers=["CPUExecutionProvider"],
    )

    def embed(features: np.ndarray) -> np.ndarray:
        return session.run(["embs"], {"feats": features[None].astype(np.float32)})[0][0]

    embeddings = {}
    metadata = {
        "reference": "wespeaker/bin/infer_onnx.py",
        "model_sha256": MODEL_SHA256,
        "sample_rate": 16000,
        "waveform_scale": 32768.0,
        "feature_dim": 80,
        "embedding_size": 256,
        "window_frames": WINDOW_FRAMES,
        "hop_frames": HOP_FRAMES,
        "threshold": 0.55,
        "files": {},
        "scores": {},
    }

    for stem in SAMPLES:
        wav_path = OUTPUT_DIR / f"{stem}.wav"
        samples, sample_rate = sf.read(wav_path, dtype="float32", always_2d=True)
        if sample_rate != 16000 or samples.shape[1] != 1:
            raise RuntimeError(f"Expected 16kHz mono audio: {wav_path}")
        mono = samples[:, 0]
        fbank = compute_fbank(mono)
        cmn = fbank - fbank.mean(axis=0, keepdims=True)
        embedding = embed(cmn)
        embeddings[stem] = embedding

        windows = []
        for start in range(0, fbank.shape[0] - WINDOW_FRAMES + 1, HOP_FRAMES):
            window = fbank[start : start + WINDOW_FRAMES]
            windows.append(embed(window - window.mean(axis=0, keepdims=True)))
        window_array = (
            np.stack(windows) if windows else np.zeros((0, 256), dtype=np.float32)
        )

        save_float32(OUTPUT_DIR / f"{stem}.fbank.bin", fbank)
        save_float32(OUTPUT_DIR / f"{stem}.cmn.bin", cmn)
        save_float32(OUTPUT_DIR / f"{stem}.embedding.bin", embedding)
        save_float32(OUTPUT_DIR / f"{stem}.windows.bin", window_array)
        metadata["files"][stem] = {
            "wav": wav_path.name,
            "num_samples": int(mono.shape[0]),
            "num_frames": int(fbank.shape[0]),
            "num_windows": int(window_array.shape[0]),
        }

    stems = list(SAMPLES)
    for left_index, left in enumerate(stems):
        for right in stems[left_index + 1 :]:
            metadata["scores"][f"{left}|{right}"] = round(
                cosine(embeddings[left], embeddings[right]), 8
            )

    with (OUTPUT_DIR / "meta.json").open("w", encoding="utf-8") as output:
        json.dump(metadata, output, indent=2)
        output.write("\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
