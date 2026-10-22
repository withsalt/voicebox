[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $OutputDir,

    [string] $Resume = "",

    [string] $Python = "E:\.packages\conda\envs\voicebox\python.exe",

    [switch] $DryRun
)

$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Python executable is missing: $Python"
}

$launcher = Join-Path $PSScriptRoot "training_preset.py"
$preset = Join-Path $PSScriptRoot "configs\candidate-12.json"
$arguments = @(
    $launcher,
    "--preset", $preset,
    "--output-dir", $OutputDir
)
if ($Resume) {
    $arguments += @("--resume", $Resume)
}
if ($DryRun) {
    $arguments += "--dry-run"
}

& $Python @arguments
exit $LASTEXITCODE
