"""
构建 VoiceBox.DNR.DeepFilterNet 所需的有状态流式 ONNX（combined_streaming.onnx）。

DeepFilterNet 官方只发布「离线」ONNX（enc.onnx / erb_dec.onnx / df_dec.onnx），其
GRU 初始状态被固化为 0、不对外暴露 I/O，无法在 ONNX Runtime 下逐帧流式。本脚本把它们
转换为「有状态流式」模型（三个 GRU 隐状态外置为输入/输出），整个过程仅依赖 onnx + numpy
(+ 可选 onnxsim)，无需 PyTorch。

流水线：
  1. 若工作目录缺少离线 ONNX，则从官方仓库下载 DeepFilterNet3_ll_onnx.tar.gz 并解压
     (enc.onnx / erb_dec.onnx / df_dec.onnx / config.ini)。
  2. split_encoder_and_patch_decoders → 拆分编码器 + 给解码器打补丁，产出
     enc_conv_streaming / enc_gru_streaming / erb_dec_streaming / df_dec_streaming。
  3. merge_split_models → 合并为单个 combined_streaming.onnx（保留全部 GRU 状态 I/O）。
  4. 把 combined_streaming.onnx 与 config.ini 安装到 C# 工程的 runtimes 目录。

用法：
  python build_streaming_onnx.py                 # 下载并构建，安装到默认 C# runtimes 目录
  python build_streaming_onnx.py --force          # 重新下载并重建
  python build_streaming_onnx.py --out <dir>      # 指定安装目录
  python build_streaming_onnx.py --work <dir>     # 指定工作/缓存目录
  python build_streaming_onnx.py --url <url>      # 覆盖下载地址
"""
import argparse
import os
import shutil
import sys
import tarfile
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # 便于 import 同目录脚本

DEFAULT_URL = (
    "https://github.com/Rikorose/DeepFilterNet/raw/refs/heads/main/"
    "models/DeepFilterNet3_ll_onnx.tar.gz"
)
OFFLINE_FILES = ["enc.onnx", "erb_dec.onnx", "df_dec.onnx", "config.ini"]
STREAMING_FILES = [
    "enc_conv_streaming.onnx", "enc_gru_streaming.onnx",
    "erb_dec_streaming.onnx", "df_dec_streaming.onnx",
    "combined_streaming.onnx",
]


def default_out_dir() -> Path:
    # scripts/DeepFilterNet-onnx-export -> repo 根 -> C# runtimes/.../dfn3_ll
    return (HERE.parent.parent / "src" / "VoiceBox.DNR.DeepFilterNet"
            / "runtimes" / "models" / "dnr" / "deepfilternet" / "dfn3_ll")


def have_offline(work: Path) -> bool:
    return all((work / f).exists() for f in OFFLINE_FILES)


def download_and_extract(url: str, work: Path) -> None:
    work.mkdir(parents=True, exist_ok=True)
    tar_path = work / "DeepFilterNet3_ll_onnx.tar.gz"
    if not tar_path.exists():
        print(f"Downloading offline model:\n  {url}")
        try:
            urllib.request.urlretrieve(url, tar_path)
        except Exception as e:
            print(f"ERROR: download failed: {e}")
            print("提示：可手动下载该 .tar.gz 放到工作目录后重试，或用 --url 指定镜像。")
            sys.exit(1)
        print(f"  saved {tar_path} ({tar_path.stat().st_size / 1e6:.1f} MB)")
    else:
        print(f"Archive already downloaded: {tar_path}")

    print("Extracting offline ONNX (enc / erb_dec / df_dec + config.ini)...")
    with tarfile.open(tar_path, "r:gz") as tf:
        for m in tf.getmembers():
            base = os.path.basename(m.name)
            if base in OFFLINE_FILES:
                fobj = tf.extractfile(m)
                if fobj is None:
                    continue
                (work / base).write_bytes(fobj.read())
                print(f"  {base}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Build stateful streaming DeepFilterNet3-LL ONNX.")
    ap.add_argument("--work", default=str(HERE / "models" / "dfn3_ll"),
                    help="工作/缓存目录（存放离线 ONNX 与中间产物）。")
    ap.add_argument("--out", default=None,
                    help="安装目录（默认 C# 工程 runtimes/.../dfn3_ll）。")
    ap.add_argument("--url", default=DEFAULT_URL, help="离线模型下载地址。")
    ap.add_argument("--force", action="store_true",
                    help="强制重新下载并重建全部中间/最终产物。")
    args = ap.parse_args()

    work = Path(args.work)
    out = Path(args.out) if args.out else default_out_dir()

    # 1) 离线 ONNX：缺失或 --force 时下载解压。
    if args.force or not have_offline(work):
        download_and_extract(args.url, work)
    else:
        print(f"Offline ONNX already present in {work}, skipping download.")
    if not have_offline(work):
        print("ERROR: offline ONNX files missing after download/extract.")
        sys.exit(1)

    # --force：清理旧的流式中间/最终产物，确保从头重建。
    if args.force:
        for f in STREAMING_FILES:
            p = work / f
            if p.exists():
                p.unlink()

    import onnx  # 延迟到此处，给出更友好的缺包提示
    try:
        from split_encoder_and_patch_decoders import split_encoder, patch_decoder
        from merge_split_models import merge_split_models
    except ImportError as e:
        print(f"ERROR: 无法导入构建脚本：{e}（确保与本脚本同目录）。")
        sys.exit(1)

    # 2) 拆分编码器 + 给解码器打补丁。
    print("\n[1/2] Split encoder + patch decoders (externalize GRU states)...")
    if not (work / "enc_conv_streaming.onnx").exists():
        split_encoder(work)
    patch_decoder(work, "erb_dec.onnx", "erb_h0", "erb_h1", "erb")
    patch_decoder(work, "df_dec.onnx", "df_h0", "df_h1", "df")

    # 3) 合并为单个 combined_streaming.onnx。
    print("\n[2/2] Merge split models into combined_streaming.onnx...")
    combined = merge_split_models(work)
    try:
        from onnxsim import simplify
        print("Running onnxsim...")
        combined, ok = simplify(combined)
        if not ok:
            print("WARNING: onnxsim 校验未通过，使用未简化模型。")
    except ImportError:
        print("onnxsim 未安装，跳过简化（pip install onnxsim 可得到更小/更快的图）。")
    combined_path = work / "combined_streaming.onnx"
    onnx.save(combined, str(combined_path))
    print(f"Saved {combined_path} ({combined_path.stat().st_size / 1e6:.1f} MB)")

    # 4) 安装到 C# runtimes。
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(combined_path, out / "combined_streaming.onnx")
    shutil.copy2(work / "config.ini", out / "config.ini")
    print(f"\nInstalled -> {out}")
    print("  combined_streaming.onnx")
    print("  config.ini")
    print("\n下一步：python dump_reference.py  生成对拍向量，再跑 C# 测试验证。")


if __name__ == "__main__":
    main()
