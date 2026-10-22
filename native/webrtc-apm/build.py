#!/usr/bin/env python3
"""Build the VoiceBox WebRTC APM shim from an official Stable WebRTC checkout."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, NoReturn


API_URL = "https://chromiumdash.appspot.com/fetch_releases"
WEBRTC_URL = "https://webrtc.googlesource.com/src.git"
DEPOT_TOOLS_URL = "https://chromium.googlesource.com/chromium/tools/depot_tools.git"
VERSION_RE = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+)$")
SHA_RE = re.compile(r"^[0-9a-f]{40}$")
SUPPORTED_RIDS = ("win-x64", "linux-x64", "linux-arm64")
EXPECTED_EXPORTS = {
    "apm_abi_version",
    "apm_create",
    "apm_create_with_neural_residual_echo_model",
    "apm_delay_create",
    "apm_delay_destroy",
    "apm_delay_process",
    "apm_delay_reset",
    "apm_destroy",
    "apm_get_stats",
    "apm_process_capture",
    "apm_process_reverse",
    "apm_reset",
    "apm_set_stream_delay_ms",
    "apm_webrtc_version",
}
WINDOWS_SYSTEM_DEPENDENCIES = {
    "advapi32.dll",
    "avrt.dll",
    "bcrypt.dll",
    "dbghelp.dll",
    "gdi32.dll",
    "iphlpapi.dll",
    "kernel32.dll",
    "ntdll.dll",
    "ole32.dll",
    "oleaut32.dll",
    "secur32.dll",
    "shell32.dll",
    "user32.dll",
    "winmm.dll",
    "ws2_32.dll",
}
LINUX_SYSTEM_DEPENDENCIES = {
    "libc.so.6",
    "libdl.so.2",
    "libgcc_s.so.1",
    "libm.so.6",
    "libpthread.so.0",
}
MAX_GLIBC_VERSION = (2, 31)

NATIVE_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = NATIVE_DIR.parents[1]
LOCK_PATH = NATIVE_DIR / "webrtc.lock.json"
MODEL_PATH = (
    REPOSITORY_ROOT
    / "src/VoiceBox.APM.WebRtc/runtimes/models/aec/webrtc/v20260212/model.tflite"
)


class BuildError(RuntimeError):
    """An actionable build, metadata, or environment failure."""


def fail(message: str) -> NoReturn:
    raise BuildError(message)


def run(
    command: list[str | Path],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    capture: bool = False,
) -> str:
    args = [str(value) for value in command]
    print("+", subprocess.list2cmdline(args), flush=True)
    try:
        result = subprocess.run(
            args,
            cwd=cwd,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        )
    except FileNotFoundError as exc:
        fail(f"Required command was not found: {args[0]}")
    except subprocess.CalledProcessError as exc:
        fail(f"Command failed with exit code {exc.returncode}: {subprocess.list2cmdline(args)}")
    return result.stdout.strip() if capture else ""


def get_json(url: str) -> Any:
    request = urllib.request.Request(url, headers={"User-Agent": "voicebox-webrtc-build/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.load(response)
    except Exception as exc:
        fail(f"Could not read ChromiumDash API: {exc}")


def parse_version(version: str) -> tuple[int, int, int, int]:
    match = VERSION_RE.fullmatch(version)
    if not match:
        fail(f"Invalid Chrome version '{version}', expected MAJOR.MINOR.BUILD.PATCH.")
    return tuple(int(part) for part in match.groups())  # type: ignore[return-value]


def resolve_stable(version: str | None, milestone: int | None, num: int) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {"channel": "Stable", "platform": "Windows", "num": str(num)}
    )
    payload = get_json(f"{API_URL}?{query}")
    if not isinstance(payload, list):
        fail("ChromiumDash returned an unexpected response.")
    releases = [item for item in payload if isinstance(item, dict)]

    if version is not None:
        parse_version(version)
        matches = [item for item in releases if item.get("version") == version]
    elif milestone is not None:
        matches = [item for item in releases if item.get("milestone") == milestone]
    else:
        matches = releases
    if not matches:
        fail("No matching Windows Stable release was returned by ChromiumDash.")

    matches.sort(
        key=lambda item: parse_version(str(item.get("version", "0.0.0.0"))),
        reverse=True,
    )
    release = matches[0]
    chrome_version = str(release.get("version", ""))
    _, _, branch, _ = parse_version(chrome_version)
    hashes = release.get("hashes")
    commit = str(hashes.get("webrtc", "")) if isinstance(hashes, dict) else ""
    if not SHA_RE.fullmatch(commit):
        fail(f"Stable release {chrome_version} has no valid hashes.webrtc revision.")
    return {
        "chrome_version": chrome_version,
        "milestone": int(release["milestone"]),
        "chromium_branch": branch,
        "webrtc_commit": commit,
    }


def load_lock() -> dict[str, Any]:
    try:
        lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        fail(f"Could not read WebRTC lock file '{LOCK_PATH}': {exc}")
    for key in (
        "chrome_version",
        "milestone",
        "chromium_branch",
        "webrtc_commit",
        "depot_tools_commit",
    ):
        if key not in lock:
            fail(f"WebRTC lock file is missing '{key}'.")
    parse_version(str(lock["chrome_version"]))
    if not SHA_RE.fullmatch(str(lock["webrtc_commit"])):
        fail("WebRTC lock file contains an invalid webrtc_commit.")
    if not SHA_RE.fullmatch(str(lock["depot_tools_commit"])):
        fail("WebRTC lock file contains an invalid depot_tools_commit.")
    return lock


def save_lock(lock: dict[str, Any]) -> None:
    LOCK_PATH.write_text(json.dumps(lock, indent=2) + "\n", encoding="utf-8", newline="\n")


def default_rid() -> str:
    machine = platform.machine().lower()
    if os.name == "nt" and machine in ("amd64", "x86_64"):
        return "win-x64"
    if sys.platform.startswith("linux"):
        if machine in ("amd64", "x86_64"):
            return "linux-x64"
        if machine in ("arm64", "aarch64"):
            return "linux-arm64"
    fail(f"No default RID for host {platform.system()} {platform.machine()}.")


def default_cache_root() -> Path:
    if os.name == "nt":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if not local_app_data:
            fail("LOCALAPPDATA is not set.")
        return Path(local_app_data) / "VoiceBox" / "webrtc-apm"
    return Path.home() / ".cache" / "voicebox-webrtc-apm"


def validate_host(rid: str) -> None:
    if rid.startswith("win-") and os.name != "nt":
        fail(f"{rid} must be built on Windows.")
    if rid.startswith("linux-") and not sys.platform.startswith("linux"):
        fail(f"{rid} must be built on Linux or WSL.")
    if rid == "linux-arm64" and platform.machine().lower() not in (
        "amd64",
        "x86_64",
        "arm64",
        "aarch64",
    ):
        fail(f"linux-arm64 cross build is unsupported on {platform.machine()}.")


def build_environment(proxy: str | None, depot_tools: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PATH"] = str(depot_tools) + os.pathsep + env.get("PATH", "")
    env["DEPOT_TOOLS_METRICS"] = "0"
    env["DEPOT_TOOLS_UPDATE"] = "0"
    env["GCLIENT_SUPPRESS_GIT_VERSION_WARNING"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    if os.name == "nt":
        env["DEPOT_TOOLS_WIN_TOOLCHAIN"] = "0"
        env["GYP_MSVS_VERSION"] = "2022"
        env["GIT_CONFIG_COUNT"] = "1"
        env["GIT_CONFIG_KEY_0"] = "core.longpaths"
        env["GIT_CONFIG_VALUE_0"] = "true"
    if proxy:
        for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            env[name] = proxy
    return env


def find_visual_studio() -> Path:
    program_files_x86 = os.environ.get("ProgramFiles(x86)")
    if not program_files_x86:
        fail("ProgramFiles(x86) is not set; Visual Studio cannot be located.")
    vswhere = (
        Path(program_files_x86)
        / "Microsoft Visual Studio/Installer/vswhere.exe"
    )
    if not vswhere.is_file():
        fail("vswhere.exe was not found. Install Visual Studio C++ build tools.")
    installation = run(
        [
            vswhere,
            "-latest",
            "-products",
            "*",
            "-requires",
            "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
            "-property",
            "installationPath",
        ],
        capture=True,
    )
    if not installation:
        fail("Visual Studio with the C++ x64 tools was not found.")
    return Path(installation)


def import_visual_studio_environment(env: dict[str, str]) -> dict[str, str]:
    if os.name != "nt":
        return env
    vcvars = find_visual_studio() / "VC/Auxiliary/Build/vcvars64.bat"
    if not vcvars.is_file():
        fail(f"Visual Studio environment script was not found: {vcvars}")

    command = f'cmd.exe /d /s /c ""{vcvars}" >nul && set"'
    print("+", command, flush=True)
    try:
        result = subprocess.run(
            command,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
        )
    except subprocess.CalledProcessError as exc:
        fail(f"Visual Studio environment initialization failed with exit code {exc.returncode}.")

    imported = env.copy()
    preserved_ps_module_path = imported.get("PSModulePath")
    for line in result.stdout.splitlines():
        name, separator, value = line.partition("=")
        if separator and name and name.casefold() != "psmodulepath":
            for existing in tuple(imported):
                if existing.casefold() == name.casefold():
                    del imported[existing]
            imported[name] = value
    if preserved_ps_module_path is not None:
        imported["PSModulePath"] = preserved_ps_module_path
    return imported


def ensure_depot_tools(path: Path, commit: str, env: dict[str, str]) -> None:
    if not (path / ".git").is_dir():
        path.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", DEPOT_TOOLS_URL, path], env=env)
    run(["git", "fetch", "origin", commit], cwd=path, env=env)
    run(["git", "checkout", "--force", commit], cwd=path, env=env)


def write_gclient(checkout: Path) -> None:
    content = """solutions = [
  {
    "name": "src",
    "url": "https://webrtc.googlesource.com/src.git",
    "deps_file": "DEPS",
    "custom_deps": {},
  },
]
target_os = []
"""
    path = checkout / ".gclient"
    if not path.exists() or path.read_text(encoding="utf-8") != content:
        checkout.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")


def ensure_checkout(
    checkout: Path,
    branch: int,
    commit: str,
    depot_tools: Path,
    env: dict[str, str],
    force_sync: bool,
) -> Path:
    source = checkout / "src"
    marker = checkout / ".voicebox-synced-revision"
    write_gclient(checkout)
    if not (source / ".git").is_dir():
        run(
            ["git", "clone", "--filter=blob:none", "--no-checkout", WEBRTC_URL, source],
            env=env,
        )

    current = run(["git", "rev-parse", "HEAD"], cwd=source, env=env, capture=True)
    synced = marker.read_text(encoding="utf-8").strip() if marker.exists() else ""
    if force_sync or current != commit or synced != commit:
        remote_ref = f"refs/branch-heads/{branch}"
        local_ref = f"refs/remotes/origin/branch-heads/{branch}"
        run(["git", "fetch", "origin", f"{remote_ref}:{local_ref}"], cwd=source, env=env)
        try:
            run(["git", "checkout", "--detach", commit], cwd=source, env=env)
        except BuildError:
            run(["git", "fetch", "origin", commit], cwd=source, env=env)
            run(["git", "checkout", "--detach", commit], cwd=source, env=env)
        gclient = find_program("gclient", env)
        run([gclient, "sync", "-D", "--no-history"], cwd=checkout, env=env)
        marker.write_text(commit + "\n", encoding="utf-8", newline="\n")
    return source


def install_overlay(source: Path, version: str) -> Path:
    destination = source / "voicebox_apm"
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(NATIVE_DIR / "gn", destination)
    shutil.copytree(NATIVE_DIR / "include", destination / "include")
    shutil.copytree(NATIVE_DIR / "src", destination / "src")
    shutil.copytree(NATIVE_DIR / "test", destination / "test")

    shim = destination / "src/apm_shim.cc"
    text = shim.read_text(encoding="utf-8")
    text = re.sub(
        r'return "m\d+\.\d+\.\d+\.\d+";',
        f'return "m{version}";',
        text,
        count=1,
    )
    shim.write_text(text, encoding="utf-8", newline="\n")
    return destination


def ensure_target_sysroot(source: Path, rid: str, env: dict[str, str]) -> None:
    if not rid.startswith("linux-"):
        return
    arch = "amd64" if rid == "linux-x64" else "arm64"
    sysroot = source / f"build/linux/debian_bullseye_{arch}-sysroot"
    if sysroot.is_dir():
        return
    run(
        [
            sys.executable,
            source / "build/linux/sysroot_scripts/install-sysroot.py",
            f"--arch={arch}",
        ],
        cwd=source,
        env=env,
    )


def gn_args(rid: str) -> list[str]:
    target_os, target_cpu = rid.split("-", 1)
    args = [
        f'target_os="{target_os}"',
        f'target_cpu="{target_cpu}"',
        "is_debug=false",
        "is_component_build=false",
        "symbol_level=0",
        "treat_warnings_as_errors=false",
        "rtc_enable_protobuf=true",
        "rtc_build_examples=false",
        "rtc_build_tools=false",
        "rtc_include_tests=false",
        "rtc_use_h264=false",
        "rtc_use_perfetto=false",
        "rtc_use_pipewire=false",
        "rtc_use_x11=false",
        "libyuv_include_tests=false",
        "use_rtti=true",
    ]
    if target_os == "win":
        args.append("use_custom_libcxx=false")
    return args


def find_program(name: str, env: dict[str, str]) -> str:
    value = shutil.which(name, path=env.get("PATH"))
    if value is None:
        fail(f"Required command was not found: {name}")
    return value


def build_target(source: Path, output: Path, rid: str, env: dict[str, str]) -> Path:
    gn = find_program("gn", env)
    ninja = find_program("ninja", env)
    output.mkdir(parents=True, exist_ok=True)
    generate_command = [
        gn,
        "gen",
        output,
        "--root-target=//voicebox_apm:abi_probe",
        "--args=" + " ".join(gn_args(rid)),
    ]
    attempts = 3 if os.name == "nt" else 1
    for attempt in range(1, attempts + 1):
        try:
            run(generate_command, cwd=source, env=env)
            break
        except BuildError:
            if attempt == attempts:
                raise
            print(f"==> GN generation failed; retrying ({attempt}/{attempts})...", flush=True)
            time.sleep(1)
    run([ninja, "-C", output, "voicebox_apm:abi_probe"], cwd=source, env=env)
    library_name = "webrtc-apm.dll" if rid.startswith("win-") else "libwebrtc-apm.so"
    library = output / library_name
    if not library.is_file():
        fail(f"GN completed but the native library is missing: {library}")
    return library


def validate_pe(library: Path, env: dict[str, str]) -> None:
    with library.open("rb") as stream:
        if stream.read(2) != b"MZ":
            fail(f"Windows output is not a PE image: {library}")
        stream.seek(0x3C)
        pe_offset = struct.unpack("<I", stream.read(4))[0]
        stream.seek(pe_offset)
        if stream.read(4) != b"PE\0\0":
            fail(f"Windows output has no valid PE signature: {library}")
        machine = struct.unpack("<H", stream.read(2))[0]
    if machine != 0x8664:
        fail(f"win-x64 output has unexpected PE machine 0x{machine:04x}.")

    dumpbin = find_program("dumpbin", env)
    exports_output = run([dumpbin, "/EXPORTS", library], env=env, capture=True)
    exports = set(re.findall(r"^\s+\d+\s+[0-9A-F]+\s+[0-9A-F]+\s+(\S+)", exports_output, re.MULTILINE))
    if exports != EXPECTED_EXPORTS:
        fail(
            "Windows export audit failed: "
            f"missing={sorted(EXPECTED_EXPORTS - exports)}, "
            f"unexpected={sorted(exports - EXPECTED_EXPORTS)}"
        )

    dependencies_output = run([dumpbin, "/DEPENDENTS", library], env=env, capture=True)
    dependencies = {
        value.casefold()
        for value in re.findall(r"^\s+([A-Za-z0-9_.+-]+\.dll)\s*$", dependencies_output, re.MULTILINE)
    }
    unexpected_dependencies = dependencies - WINDOWS_SYSTEM_DEPENDENCIES
    if unexpected_dependencies or "kernel32.dll" not in dependencies:
        fail(
            "Windows dependency audit failed: "
            f"unexpected={sorted(unexpected_dependencies)}, "
            f"dependencies={sorted(dependencies)}"
        )


def validate_elf(library: Path, rid: str, env: dict[str, str]) -> None:
    expected_machine = 0x3E if rid == "linux-x64" else 0xB7
    with library.open("rb") as stream:
        header = stream.read(20)
    if len(header) != 20 or header[:6] != b"\x7fELF\x02\x01":
        fail(f"Linux output is not a 64-bit little-endian ELF image: {library}")
    machine = struct.unpack_from("<H", header, 18)[0]
    if machine != expected_machine:
        fail(f"{rid} output has unexpected ELF machine 0x{machine:04x}.")

    nm = find_program("nm", env)
    symbols_output = run(
        [nm, "-D", "--defined-only", "--extern-only", library],
        env=env,
        capture=True,
    )
    exports = {
        match.group(1)
        for line in symbols_output.splitlines()
        if (match := re.search(r"\b(apm_[A-Za-z0-9_]+)$", line))
    }
    if exports != EXPECTED_EXPORTS:
        fail(
            "Linux export audit failed: "
            f"missing={sorted(EXPECTED_EXPORTS - exports)}, "
            f"unexpected={sorted(exports - EXPECTED_EXPORTS)}"
        )

    readelf = find_program("readelf", env)
    dynamic_output = run([readelf, "-d", library], env=env, capture=True)
    dependencies = set(re.findall(r"Shared library: \[([^]]+)]", dynamic_output))
    dynamic_loader = (
        "ld-linux-x86-64.so.2" if rid == "linux-x64" else "ld-linux-aarch64.so.1"
    )
    allowed_dependencies = LINUX_SYSTEM_DEPENDENCIES | {dynamic_loader}
    unexpected_dependencies = dependencies - allowed_dependencies
    if unexpected_dependencies or "libc.so.6" not in dependencies:
        fail(
            "Linux dependency audit failed: "
            f"unexpected={sorted(unexpected_dependencies)}, "
            f"dependencies={sorted(dependencies)}"
        )

    versions_output = run([readelf, "--version-info", library], env=env, capture=True)
    glibc_versions = {
        tuple(int(part) for part in value.split("."))
        for value in re.findall(r"GLIBC_([0-9]+(?:\.[0-9]+)+)", versions_output)
    }
    if not glibc_versions:
        fail("Linux output contains no GLIBC version requirements.")
    maximum = max(glibc_versions)
    if maximum > MAX_GLIBC_VERSION:
        fail(
            f"Linux output requires GLIBC {'.'.join(map(str, maximum))}; "
            f"maximum supported is {'.'.join(map(str, MAX_GLIBC_VERSION))}."
        )


def verify_and_deploy(
    output: Path,
    library: Path,
    rid: str,
    env: dict[str, str],
) -> None:
    if rid.startswith("win-"):
        validate_pe(library, env)
    else:
        validate_elf(library, rid, env)

    probe_name = "abi_probe.exe" if rid.startswith("win-") else "abi_probe"
    probe = output / probe_name
    host_rid = default_rid()
    if rid == host_rid:
        probe_env = os.environ.copy()
        if rid.startswith("linux-"):
            existing = probe_env.get("LD_LIBRARY_PATH")
            probe_env["LD_LIBRARY_PATH"] = (
                str(output) if not existing else str(output) + os.pathsep + existing
            )
        run([probe, MODEL_PATH], cwd=output, env=probe_env)
    else:
        print(f"==> Skipping execution of cross-compiled {rid} probe.")

    destination = (
        REPOSITORY_ROOT / "src/VoiceBox.APM.WebRtc/runtimes" / rid / "native"
    )
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copy2(library, destination / library.name)
    if rid.startswith("win-"):
        pdb = output / "webrtc-apm.dll.pdb"
        if pdb.is_file():
            shutil.copy2(pdb, destination / "webrtc-apm.pdb")
    print(f"==> Installed {library.name} -> {destination}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rid", choices=SUPPORTED_RIDS, default=default_rid())
    parser.add_argument("--cache-root", type=Path, default=default_cache_root())
    parser.add_argument("--proxy")
    parser.add_argument("--sync", action="store_true", help="Force gclient dependency synchronization.")
    parser.add_argument("--resolve-stable", action="store_true", help="Print the selected Stable metadata and exit.")
    parser.add_argument("--update-stable", action="store_true", help="Update the lock file to the selected Stable release.")
    selector = parser.add_mutually_exclusive_group()
    selector.add_argument("--chrome-version")
    selector.add_argument("--milestone", type=int)
    parser.add_argument("--num", type=int, default=100)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    lock = load_lock()
    if args.resolve_stable or args.update_stable:
        stable = resolve_stable(args.chrome_version, args.milestone, args.num)
        candidate = {**stable, "depot_tools_commit": lock["depot_tools_commit"]}
        print(json.dumps(candidate, indent=2))
        if args.update_stable:
            save_lock(candidate)
            print(f"Updated {LOCK_PATH}")
        return 0

    validate_host(args.rid)
    if not MODEL_PATH.is_file():
        fail(f"Neural residual echo model is missing: {MODEL_PATH}")

    cache_root = args.cache_root.expanduser().resolve()
    depot_tools = cache_root / "depot_tools"
    checkout = cache_root / "checkout"
    output = cache_root / "out" / str(lock["webrtc_commit"])[:12] / args.rid
    bootstrap_env = build_environment(args.proxy, depot_tools)
    ensure_depot_tools(depot_tools, str(lock["depot_tools_commit"]), bootstrap_env)
    env = import_visual_studio_environment(
        build_environment(args.proxy, depot_tools)
    )
    source = ensure_checkout(
        checkout,
        int(lock["chromium_branch"]),
        str(lock["webrtc_commit"]),
        depot_tools,
        env,
        args.sync,
    )
    ensure_target_sysroot(source, args.rid, env)
    install_overlay(source, str(lock["chrome_version"]))
    library = build_target(source, output, args.rid, env)
    verify_and_deploy(output, library, args.rid, env)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1)
