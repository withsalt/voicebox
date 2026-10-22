from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from dtd_model import DtdModelConfig, RenderAwareDtd, StreamingDtdWrapper


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export and verify the render-aware streaming DTD model."
    )
    parser.add_argument("checkpoint")
    parser.add_argument("--output")
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--parity-frames", type=int, default=20)
    return parser.parse_args()


def add_metadata(path: Path, metadata: dict[str, str]) -> None:
    graph = onnx.load(str(path))
    del graph.metadata_props[:]
    for key, value in metadata.items():
        item = graph.metadata_props.add()
        item.key = key
        item.value = value
    onnx.checker.check_model(graph)
    onnx.save(graph, str(path))


def verify_streaming_parity(
    wrapper: StreamingDtdWrapper,
    onnx_path: Path,
    frames: int,
) -> dict[str, float]:
    rng = np.random.default_rng(20260729)
    microphone = rng.normal(0.0, 0.05, (frames, 640)).astype(np.float32)
    render = rng.normal(0.0, 0.05, (frames, 640)).astype(np.float32)
    torch_state = wrapper.model.initial_state(1)
    torch_probability: list[np.ndarray] = []
    with torch.inference_mode():
        for index in range(frames):
            probability, torch_state = wrapper(
                torch.from_numpy(microphone[index : index + 1]),
                torch.from_numpy(render[index : index + 1]),
                torch_state,
            )
            torch_probability.append(probability.numpy())

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )
    ort_state = np.zeros(
        (
            wrapper.model.config.gru_layers,
            1,
            wrapper.model.config.hidden_size,
        ),
        dtype=np.float32,
    )
    ort_probability: list[np.ndarray] = []
    for index in range(frames):
        probability, ort_state = session.run(
            None,
            {
                "microphone": microphone[index : index + 1],
                "render": render[index : index + 1],
                "state_in": ort_state,
            },
        )
        ort_probability.append(probability)

    torch_values = np.concatenate(torch_probability, axis=1)
    ort_values = np.concatenate(ort_probability, axis=1)
    difference = np.abs(torch_values - ort_values)
    state_difference = np.abs(torch_state.numpy() - ort_state)
    return {
        "probability_max_abs": float(difference.max(initial=0.0)),
        "probability_mean_abs": float(difference.mean()),
        "state_max_abs": float(state_difference.max(initial=0.0)),
        "state_mean_abs": float(state_difference.mean()),
    }


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint).resolve()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = DtdModelConfig(**checkpoint["model_config"])
    model = RenderAwareDtd(config)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    wrapper = StreamingDtdWrapper(model).eval()

    output = Path(args.output).resolve() if args.output else checkpoint_path.with_suffix(".onnx")
    output.parent.mkdir(parents=True, exist_ok=True)
    microphone = torch.zeros(1, config.frame_length, dtype=torch.float32)
    render = torch.zeros(1, config.frame_length, dtype=torch.float32)
    state = model.initial_state(1)
    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (microphone, render, state),
            str(output),
            input_names=["microphone", "render", "state_in"],
            output_names=["nearend_probability", "state_out"],
            dynamic_axes=None,
            do_constant_folding=True,
            opset_version=args.opset,
            dynamo=False,
        )

    training_args = checkpoint.get("training_args", {})
    training_source = (
        "Enriched multi-source: AEC Challenge narrow/fullband/real plus "
        "external speech, noise, and RIR online synthesis"
        if training_args.get("enriched_root")
        else "Microsoft AEC Challenge synthetic train split"
    )
    metadata = {
        "model_type": "render-aware-nearend-dtd",
        "sample_rate": str(config.sample_rate),
        "frame_length": str(config.frame_length),
        "hop_length": str(config.hop_length),
        "input_names": "microphone,render,state_in",
        "output_names": "nearend_probability,state_out",
        "audio_input_shape": f"1,{config.frame_length}",
        "state_shape": f"{config.gru_layers},1,{config.hidden_size}",
        "frequency_bands": str(config.frequency_bands),
        "probability_shape": "1,1,1",
        "decision_threshold": str(checkpoint["decision_threshold"]),
        "label_definition": (
            "propagated near-end at microphone; 20ms RMS >= "
            "max(0.0005,3% utterance peak); gaps <=100ms filled"
        ),
        "training_source": training_source,
        "trainable_parameters": str(model.complexity().trainable_parameters),
        "streaming_mmacs_per_second": str(
            model.complexity().total_macs_per_second / 1_000_000
        ),
        "runtime_resampling": (
            "source files read at per-file header rate and resampled to 16kHz"
        ),
    }
    add_metadata(output, metadata)
    contract_path = output.with_suffix(".contract.json")
    contract_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    parity = verify_streaming_parity(wrapper, output, args.parity_frames)
    parity_path = output.with_suffix(".parity.json")
    parity_path.write_text(
        json.dumps(parity, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if parity["probability_max_abs"] > 1e-4 or parity["state_max_abs"] > 1e-4:
        raise RuntimeError(f"ONNX parity check failed: {parity}")
    print(
        json.dumps(
            {
                "onnx": str(output),
                "contract": str(contract_path),
                "parity": parity,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
