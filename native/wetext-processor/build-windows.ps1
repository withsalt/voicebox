<#
.SYNOPSIS
  Builds wetext_processor_c.dll and copies it into the managed project's
  runtimes/<rid>/native/ folder.

.DESCRIPTION
  Configures and builds with CMake using the MSVC toolchain. Works from any
  PowerShell: if cl.exe is not on PATH the script imports vcvars automatically,
  and if cmake/ninja are not on PATH the copies bundled with Visual Studio are
  used. Ninja is preferred (single-config); when it is unavailable the script
  falls back to the Visual Studio multi-config generator.

  OpenFst/glog/gflags are built from third_party/ local caches (or fetched from
  GitHub by CMake FetchContent on first use) into a single self-contained
  wetext_processor_c.dll. There is no separate prebuilt drop to download.

.PARAMETER Rid
  Target RID: win-x64 (default) or win-x86.

.PARAMETER Config
  CMake build configuration: Release (default) or Debug.

.EXAMPLE
  ./build-windows.ps1
  ./build-windows.ps1 -Rid win-x64 -Config Release
#>
[CmdletBinding()]
param(
  [ValidateSet('win-x64', 'win-x86')]
  [string]$Rid = 'win-x64',
  [ValidateSet('Release', 'Debug')]
  [string]$Config = 'Release'
)

$ErrorActionPreference = 'Stop'
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$buildDir = Join-Path $scriptDir "build/$Rid"
$outDir = Join-Path $scriptDir "../../src/SDKs/WeTextProcessing.NET/runtimes/$Rid/native"

# ---- Locate Visual Studio (vswhere) -----------------------------------------
function Get-VsPath {
  $vswhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
  if (-not (Test-Path $vswhere)) { return $null }
  & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
}

# ---- Locate cmake / ninja (PATH first, then the VS-bundled copies) ----------
$vsPath = $null
$cmake = (Get-Command cmake -ErrorAction SilentlyContinue).Source
if (-not $cmake) {
  $vsPath = Get-VsPath
  if (-not $vsPath) { throw 'Neither cmake on PATH nor a Visual Studio installation was found.' }
  $cmake = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
  if (-not (Test-Path $cmake)) { throw "CMake not found in Visual Studio at: $cmake" }
}
$ninja = (Get-Command ninja -ErrorAction SilentlyContinue).Source
if (-not $ninja) {
  if (-not $vsPath) { $vsPath = Get-VsPath }
  if ($vsPath) {
    $candidate = Join-Path $vsPath 'Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe'
    if (Test-Path $candidate) { $ninja = $candidate }
  }
}

# ---- Ensure the MSVC environment is present (cl on PATH) ---------------------
if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) {
  if (-not $vsPath) { $vsPath = Get-VsPath }
  if (-not $vsPath) { throw 'MSVC not found: install Visual Studio with the C++ workload.' }
  $arch = if ($Rid -eq 'win-x86') { '32' } else { '64' }
  $vcvars = Join-Path $vsPath "VC\Auxiliary\Build\vcvars$arch.bat"
  Write-Host "==> Importing MSVC env from $vcvars"
  cmd /c "`"$vcvars`" && set" | ForEach-Object {
    # Never overwrite PSModulePath: vcvars reports the *machine* value, and
    # replacing the process value breaks PowerShell module auto-loading
    # (Get-FileHash etc.).
    if ($_ -match '^(.*?)=(.*)$' -and $matches[1] -ne 'PSModulePath') {
      Set-Item -Path "env:$($matches[1])" -Value $matches[2]
    }
  }
  if (-not (Get-Command cl.exe -ErrorAction SilentlyContinue)) { throw 'cl.exe still not found after importing vcvars.' }
}

# ---- Configure, build, deploy ------------------------------------------------
Write-Host "==> Configuring ($Rid, $Config)"
$dllDir = Join-Path $buildDir 'processor'
if ($ninja) {
  & $cmake -S $scriptDir -B $buildDir -G Ninja `
    "-DCMAKE_MAKE_PROGRAM=$ninja" `
    "-DCMAKE_BUILD_TYPE=$Config"
} else {
  # Fall back to the Visual Studio generator (multi-config).
  $platform = if ($Rid -eq 'win-x86') { 'Win32' } else { 'x64' }
  & $cmake -S $scriptDir -B $buildDir -A $platform
  $dllDir = Join-Path $dllDir $Config
}
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed.' }

Write-Host '==> Building'
& $cmake --build $buildDir --config $Config --target wetext_processor_c --parallel
if ($LASTEXITCODE -ne 0) { throw 'CMake build failed.' }

New-Item -ItemType Directory -Force -Path $outDir | Out-Null
Copy-Item (Join-Path $dllDir 'wetext_processor_c.dll') $outDir -Force
$pdb = Join-Path $dllDir 'wetext_processor_c.pdb'
if (Test-Path $pdb) { Copy-Item $pdb $outDir -Force }
Write-Host "==> Copied wetext_processor_c.dll -> $(Resolve-Path $outDir)"
