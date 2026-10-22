# 获取 Chrome WebRTC Neural Residual Echo Estimator 模型

本文说明如何让 Google Chrome 下载 Optimization Target 71：
`WEBRTC_NEURAL_RESIDUAL_ECHO_ESTIMATOR`。该模型用于 WebRTC AEC 的神经网络残余回声估计。

## 重要说明

目前没有已确认可用的公开模型直链或第三方镜像。Hugging Face 仓库
[`dejanseo/chrome_models`](https://huggingface.co/dejanseo/chrome_models) 的 README 将目标 71
标记为 `Downloaded`，但当前仓库及其主分支历史中都没有 `71/` 目录。该状态来自一次只修改
README、没有上传模型文件的提交：
[`bc0c738`](https://huggingface.co/dejanseo/chrome_models/commit/bc0c738e55278aa131eabdc397d058e76f7470ec)。

目标 71 由 Chrome 的 Optimization Guide 服务按需下发，不是一个固定的公开下载地址。
Chromium 源码显示：

- 功能开关 `WebRtcAudioNeuralResidualEchoEstimation` 默认关闭。
- 启用功能后，Chrome 会注册 Optimization Target 71。
- Chrome 观察到麦克风输入流创建后，才会请求模型，以避免不必要的下载。

## 前置条件

- Windows 10 或 Windows 11。
- 较新版本的正式版 Google Chrome。普通 Chromium 构建可能无法访问 Google 的模型下发服务。
- 可以正常访问 Chrome Optimization Guide 服务。
- 可以在测试页面上授予麦克风权限。

## 下载步骤

### 1. 启动独立的 Chrome 配置

在 PowerShell 中执行：

```powershell
$chromeExe = "C:\Program Files\Google\Chrome\Application\chrome.exe"
$modelProfile = Join-Path $env:LOCALAPPDATA "ChromeModel71"

if (-not (Test-Path -LiteralPath $chromeExe)) {
    throw "未找到 Google Chrome: $chromeExe"
}

Start-Process -FilePath $chromeExe -ArgumentList @(
    "--user-data-dir=$modelProfile",
    "--enable-features=WebRtcAudioNeuralResidualEchoEstimation",
    "https://webrtc.github.io/samples/src/content/getusermedia/audio/"
)
```

这里使用独立的 `ChromeModel71` 用户数据目录，因此不需要关闭日常使用的 Chrome，也不会修改
现有 Chrome 配置。

### 2. 触发模型请求

在新打开的 Chrome 窗口中：

1. 在 WebRTC 示例页面允许麦克风权限。
2. 启动音频采集，确保浏览器实际创建麦克风输入流。
3. 保持浏览器运行几分钟，等待后台模型请求和下载。

只打开浏览器而不创建麦克风输入流，通常不会触发目标 71 的模型请求。

### 3. 检查 Chrome 内部状态

在同一 Chrome 窗口打开：

```text
chrome://optimization-guide-internals/#modelss
```

在 Models 页面查找 `WebRTCNeuralResidualEchoEstimator` 或 Optimization Target 71，检查模型是否已
注册、下载或报告错误。

也可以打开 `chrome://version`，确认 Command Line 中包含：

```text
--enable-features=WebRtcAudioNeuralResidualEchoEstimation
```

### 4. 检查本地模型目录

在 PowerShell 中执行：

```powershell
$modelDir = Join-Path $env:LOCALAPPDATA `
    "ChromeModel71\optimization_guide_model_store\71"

if (Test-Path -LiteralPath $modelDir) {
    Get-ChildItem -LiteralPath $modelDir -Recurse -Force |
        Select-Object FullName, Length, LastWriteTime
} else {
    Write-Host "目标 71 尚未下载：$modelDir"
}
```

下载成功后，应保留整个 `71` 目录及其子目录。不要只复制其中一个文件，因为模型元数据和版本
信息也可能是加载过程的一部分。

## 下载没有发生时

依次检查：

1. `chrome://version` 中是否包含正确的 `--enable-features` 参数。
2. 是否在独立 Chrome 窗口中真正启动过麦克风采集。
3. `chrome://optimization-guide-internals` 是否显示目标 71，以及是否有网络或模型服务错误。
4. Chrome 是否为较新的官方 Google Chrome，而不是没有 Google API 配置的 Chromium 构建。
5. 等待几分钟后重启独立 Chrome，再次创建麦克风输入流。

即使以上条件都满足，Google 后端仍可能没有向当前 Chrome 版本、平台或实验组提供模型。此时无法
通过构造一个固定 URL 来下载，只能等待官方下发或寻找由其他用户合法发布的模型副本。

## 清理独立配置

不再需要该测试配置时，先关闭使用 `ChromeModel71` 的所有 Chrome 窗口，再删除下面的目录：

```text
%LOCALAPPDATA%\ChromeModel71
```

删除该目录也会删除已经下载的目标 71 模型。需要保留模型时，请先备份完整的 `71` 目录。

## 参考资料

- [Chromium：目标 71 的注册逻辑](https://source.chromium.org/chromium/chromium/src/+/main:chrome/browser/media/audio_process_ml_model_forwarder.cc;l=169)
- [Chromium：创建音频输入流后才注册模型观察器](https://source.chromium.org/chromium/chromium/src/+/main:chrome/browser/media/audio_process_ml_model_forwarder.cc;l=219)
- [Chromium：功能默认关闭](https://source.chromium.org/chromium/chromium/src/+/main:media/base/media_switches.cc;l=1080)
- [Hugging Face 模型集合](https://huggingface.co/dejanseo/chrome_models)

模型文件的许可证和再分发权限并未在上述 Hugging Face 仓库中明确说明。将提取的模型用于发布、
再分发或商业用途前，应单独确认授权条件。
