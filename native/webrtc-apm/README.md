# webrtc-apm 原生构建

`webrtc-apm` 是 WebRTC Audio Processing Module 的 C ABI 适配层，提供 AEC3、
噪声抑制、AGC2、独立 AEC3 延迟估计器和 Neural Residual Echo Estimator。

仓库只有一个构建入口：`build.py`。脚本直接使用官方 WebRTC 源码，自动完成
版本检出、依赖同步、GN 配置、Ninja 编译、模型探针、产物审计和部署。不需要
预编译的 `libwebrtc`，也不需要手工应用补丁。

## 支持平台

| RID | 构建主机 | 产物 |
| --- | --- | --- |
| `win-x64` | Windows x64 | `webrtc-apm.dll`、`webrtc-apm.pdb` |
| `linux-x64` | Linux x64 / WSL x64 | `libwebrtc-apm.so` |
| `linux-arm64` | Linux arm64，或 Linux / WSL x64 交叉编译 | `libwebrtc-apm.so` |

Linux 产物使用 Chromium 的 Debian Bullseye sysroot 和静态 Chromium libc++，
不需要随包部署额外的 C++ 运行库。脚本要求最高 GLIBC 版本不超过 2.31。

## 环境要求

所有平台都需要 Python 3.10+、Git，以及能够访问 ChromiumDash、
Google Source、CIPD 和 Google Storage 的网络。

Windows 需要安装带有“使用 C++ 的桌面开发”工作负载的 Visual Studio。
脚本会通过 `vswhere.exe` 自动定位 Visual Studio，并导入 x64 编译环境。

Ubuntu 22.04 / WSL 可安装以下基础依赖：

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential ca-certificates git python3 binutils
```

WebRTC 使用的 `depot_tools`、Clang、GN、Ninja 和目标 sysroot 由脚本固定版本并
下载，无需使用系统版本。

## 构建

在仓库根目录执行：

```powershell
# Windows x64
python native/webrtc-apm/build.py --rid win-x64
```

```bash
# Linux x64
python3 native/webrtc-apm/build.py --rid linux-x64

# 在 x64 Linux / WSL 上交叉编译 Linux arm64
python3 native/webrtc-apm/build.py --rid linux-arm64
```

未指定 `--rid` 时，脚本根据当前主机选择 `win-x64`、`linux-x64` 或
`linux-arm64`。首次构建需要同步完整 WebRTC 依赖，后续构建复用缓存并增量
编译。默认缓存位置：

- Windows：`%LOCALAPPDATA%\VoiceBox\webrtc-apm`
- Linux：`~/.cache/voicebox-webrtc-apm`

可使用 `--cache-root <path>` 更改缓存目录，或使用 `--proxy <url>` 为源码和
依赖下载设置 HTTP/HTTPS 代理。`--sync` 会强制重新执行 `gclient sync`。

构建成功后，脚本自动部署到：

- `src/VoiceBox.APM.WebRtc/runtimes/win-x64/native/`
- `src/VoiceBox.APM.WebRtc/runtimes/linux-x64/native/`
- `src/VoiceBox.APM.WebRtc/runtimes/linux-arm64/native/`

模型保持为独立 runtime 资源：

```text
src/VoiceBox.APM.WebRtc/runtimes/models/aec/webrtc/v20260212/model.tflite
```

本机目标会运行 `abi_probe`，同时验证普通 APM 和真实 Neural REE 模型创建。
交叉编译目标不能在宿主机执行，但仍会在部署前验证 ELF 架构、完整导出表、
动态依赖和 GLIBC 版本。Windows 同样会验证 PE x64、完整导出表和系统 DLL
依赖。任何检查失败都不会覆盖 runtime 产物。

## 版本锁与 Stable 更新

`webrtc.lock.json` 固定 Chrome Stable 版本、WebRTC 精确提交和
`depot_tools` 精确提交。普通构建只读取锁文件，不访问 Stable API 来改变版本，
因此同一个仓库提交在不同日期仍使用相同源码。

查看 ChromiumDash 当前 Windows Stable 对应的 WebRTC 提交：

```powershell
python native/webrtc-apm/build.py --resolve-stable
python native/webrtc-apm/build.py --resolve-stable --milestone 151
python native/webrtc-apm/build.py --resolve-stable --chrome-version 151.0.7922.138
```

显式更新锁文件后，再分别重建三个 RID：

```powershell
python native/webrtc-apm/build.py --update-stable
```

也可以将 `--milestone` 或 `--chrome-version` 与 `--update-stable` 组合使用。
更新版本时必须提交新的锁文件、三个原生 runtime 产物，并运行托管集成测试和
AEC Challenge 对照评测。

## 目录结构

```text
native/webrtc-apm/
  build.py             唯一构建入口
  webrtc.lock.json     WebRTC 与 depot_tools 版本锁
  gn/BUILD.gn          最终共享库和 ABI probe 的 GN target
  gn/apm.map           Linux 公共符号白名单
  include/apm_shim.h   跨平台 C ABI
  src/                 C ABI 实现
  test/abi_probe.cpp   原生 ABI 与模型探针
```
