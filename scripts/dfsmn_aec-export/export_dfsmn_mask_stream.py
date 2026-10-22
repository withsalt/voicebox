import argparse
import json
import shutil
import site
from pathlib import Path

import onnx
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parent


class StreamingUniDeepFsmn(torch.nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.linear = layer.linear
        self.norm = layer.norm
        self.act = layer.act
        self.project = layer.project
        self.conv_weight = torch.nn.Parameter(layer.conv1.weight.detach().clone(), requires_grad=False)
        self.skip_connect = bool(layer.skip_connect)
        self.lorder = int(layer.lorder)
        self.hidden_size = int(layer.output_dim)
        self.cache_frames = self.lorder - 1

    def forward(self, x, cache):
        # x: [B,T,H], cache: [B,H,lorder-1]
        h = self.project(self.norm(self.act(self.linear(x))))
        h_t = h.transpose(1, 2)
        context = torch.cat([cache, h_t], dim=2)
        weight = self.conv_weight.squeeze(-1)
        y = F.conv1d(context, weight, groups=self.hidden_size).transpose(1, 2)
        if self.skip_connect:
            y = y + h
        out = x + y
        next_cache = context[:, :, -self.cache_frames:]
        return out, next_cache


class StreamingDfsmnMaskCore(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.linear1 = model.linear1
        self.relu = model.relu
        self.layers = torch.nn.ModuleList([StreamingUniDeepFsmn(layer) for layer in model.deepfsmn])
        self.linear2 = model.linear2
        self.sig = model.sig
        self.num_layers = len(self.layers)
        self.hidden_size = self.layers[0].hidden_size
        self.cache_frames = self.layers[0].cache_frames

    def encode(self, feat, cache_in):
        # feat: [1,T,240], cache_in: [L,1,512,19]
        x = self.relu(self.linear1(feat))
        next_cache = []
        for i, layer in enumerate(self.layers):
            x, cache = layer(x, cache_in[i])
            next_cache.append(cache)
        return x, torch.stack(next_cache, dim=0)

    def forward(self, feat, cache_in):
        x, next_cache = self.encode(feat, cache_in)
        mask = self.sig(self.linear2(x))
        return mask, next_cache


class StreamingDfsmnMaskVadCore(StreamingDfsmnMaskCore):
    def __init__(self, model):
        super().__init__(model)
        if not hasattr(model, "linear3"):
            raise AttributeError("DFSMN model does not expose the VAD head `linear3`.")
        self.linear3 = model.linear3

    def forward(self, feat, cache_in):
        x, next_cache = self.encode(feat, cache_in)
        mask = self.sig(self.linear2(x))
        vad_prob = torch.sigmoid(self.linear3(x))
        return mask, vad_prob, next_cache


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


def add_metadata(path: Path, metadata: dict[str, str]):
    model = onnx.load(str(path))
    del model.metadata_props[:]
    for key, value in metadata.items():
        item = model.metadata_props.add()
        item.key = key
        item.value = str(value)
    onnx.save(model, str(path))


def main():
    parser = argparse.ArgumentParser(description="Export streaming DFSMN AEC mask core ONNX.")
    parser.add_argument("--model-dir", default=str(ROOT / "dependents" / "speech_dfsmn_aec_psm_16k"))
    parser.add_argument("--output")
    parser.add_argument("--chunk-frames", type=int, default=1)
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    model_dir = Path(args.model_dir).resolve()
    default_name = "dfsmn_aec_mask_vad_stream.onnx"
    output = Path(args.output or (ROOT / "export" / default_name)).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    model = load_dfsmn_model(model_dir)
    core = StreamingDfsmnMaskVadCore(model).eval()

    feat = torch.zeros(1, args.chunk_frames, 240, dtype=torch.float32)
    cache = torch.zeros(core.num_layers, 1, core.hidden_size, core.cache_frames, dtype=torch.float32)
    output_names = ["mask", "vad_prob", "cache_out"]

    with torch.inference_mode():
        torch.onnx.export(
            core,
            (feat, cache),
            str(output),
            input_names=["feat", "cache_in"],
            output_names=output_names,
            dynamic_axes=None,
            do_constant_folding=True,
            opset_version=args.opset,
            dynamo=False,
        )

    metadata = {
        "sample_rate": "16000",
        "feature_dim": "240",
        "feature_bins": "80",
        "mask_bins": "321",
        "frame_length": "640",
        "hop_length": "320",
        "num_layers": str(core.num_layers),
        "hidden_size": str(core.hidden_size),
        "cache_frames": str(core.cache_frames),
        "chunk_frames": str(args.chunk_frames),
        "outputs": ",".join(output_names),
        "vad_dim": "1",
    }
    add_metadata(output, metadata)

    contract_path = output.with_suffix(".contract.json")
    contract_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    mvn_src = model_dir / "feature.DEY.mvn.txt"
    if mvn_src.exists():
        shutil.copyfile(mvn_src, output.parent / "feature.DEY.mvn.txt")

    print(f"Exported {output}")
    print(f"Wrote {contract_path}")


if __name__ == "__main__":
    main()
