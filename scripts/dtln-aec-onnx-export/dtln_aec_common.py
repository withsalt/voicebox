from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx


SCRIPT_DIR = Path(__file__).resolve().parent
UPSTREAM_DIR = SCRIPT_DIR / "DTLN-aec"
DEFAULT_TFLITE_DIR = UPSTREAM_DIR / "pretrained_models"
DEFAULT_ONNX_DIR = SCRIPT_DIR / "onnx_out_named"
DEFAULT_SAMPLE_MIC = (
    UPSTREAM_DIR
    / "audio_samples"
    / "DMTgmZwtgUilp4omPK7-OQ_doubletalk_mic.wav"
)

SUPPORTED_UNITS = (128, 256, 512)
BLOCK_LEN = 512
BLOCK_SHIFT = 128
SAMPLE_RATE = 16_000

PART_IO = {
    1: {
        "inputs": ("mic_mag", "state1_in", "lpb_mag"),
        "outputs": ("mask", "state1_out"),
    },
    2: {
        "inputs": ("estimated_block", "state2_in", "lpb_block"),
        "outputs": ("enhanced_block", "state2_out"),
    },
}


def parse_units(value: str) -> list[int]:
    value = value.strip().lower()
    if value == "all":
        return list(SUPPORTED_UNITS)

    units: list[int] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            unit = int(token)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(f"Invalid unit count: {token}") from exc
        if unit not in SUPPORTED_UNITS:
            allowed = ", ".join(str(x) for x in SUPPORTED_UNITS)
            raise argparse.ArgumentTypeError(
                f"Unsupported unit count {unit}; expected one of {allowed}"
            )
        units.append(unit)

    if not units:
        raise argparse.ArgumentTypeError("No unit counts were provided.")
    return units


def iter_model_parts(units: Iterable[int]) -> Iterable[tuple[int, int]]:
    for unit in units:
        for part in (1, 2):
            yield unit, part


def tflite_path(tflite_dir: Path, units: int, part: int) -> Path:
    return tflite_dir / f"dtln_aec_{units}_{part}.tflite"


def onnx_path(onnx_dir: Path, units: int, part: int) -> Path:
    return onnx_dir / f"dtln_aec_{units}_{part}.onnx"


def rename_onnx_io(model: onnx.ModelProto, input_names: tuple[str, ...], output_names: tuple[str, ...]) -> None:
    old_inputs = [value.name for value in model.graph.input]
    old_outputs = [value.name for value in model.graph.output]
    if len(old_inputs) != len(input_names):
        raise ValueError(f"Expected {len(input_names)} inputs, found {len(old_inputs)}.")
    if len(old_outputs) != len(output_names):
        raise ValueError(f"Expected {len(output_names)} outputs, found {len(old_outputs)}.")

    for old, new in zip(old_inputs, input_names):
        rename_value(model, old, new)
    for old, new in zip(old_outputs, output_names):
        rename_value(model, old, new)


def rename_value(model: onnx.ModelProto, old: str, new: str) -> None:
    for value in list(model.graph.input) + list(model.graph.output) + list(model.graph.value_info):
        if value.name == old:
            value.name = new

    for node in model.graph.node:
        for idx, name in enumerate(node.input):
            if name == old:
                node.input[idx] = new
        for idx, name in enumerate(node.output):
            if name == old:
                node.output[idx] = new


def add_metadata(model: onnx.ModelProto, units: int, part: int, opset: int) -> None:
    metadata = {
        "dtln_aec.units": str(units),
        "dtln_aec.part": str(part),
        "dtln_aec.sample_rate": str(SAMPLE_RATE),
        "dtln_aec.block_len": str(BLOCK_LEN),
        "dtln_aec.block_shift": str(BLOCK_SHIFT),
        "dtln_aec.opset": str(opset),
        "dtln_aec.source": "breizhn/DTLN-aec TFLite pretrained model",
    }

    del model.metadata_props[:]
    for key, value in metadata.items():
        prop = model.metadata_props.add()
        prop.key = key
        prop.value = value


def tensor_shape(value_info: onnx.ValueInfoProto) -> list[int | str]:
    shape = value_info.type.tensor_type.shape
    dims: list[int | str] = []
    for dim in shape.dim:
        if dim.dim_value:
            dims.append(dim.dim_value)
        elif dim.dim_param:
            dims.append(dim.dim_param)
        else:
            dims.append("?")
    return dims


def load_checked_model(path: Path) -> onnx.ModelProto:
    model = onnx.load(path)
    onnx.checker.check_model(model)
    return model


def zeros_for_shape(shape: Iterable[int | str | None]) -> np.ndarray:
    dims: list[int] = []
    for dim in shape:
        if isinstance(dim, int):
            dims.append(dim)
        elif isinstance(dim, str) and dim.isdigit():
            dims.append(int(dim))
        else:
            raise ValueError(f"Cannot allocate zero tensor for dynamic shape: {list(shape)}")
    return np.zeros(dims, dtype=np.float32)
