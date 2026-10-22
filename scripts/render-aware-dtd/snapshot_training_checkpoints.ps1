param(
    [Parameter(Mandatory = $true)]
    [int] $TargetProcessId,

    [Parameter(Mandatory = $true)]
    [string] $CandidateDirectory,

    [int] $IntervalSeconds = 2
)

$candidate = [System.IO.Path]::GetFullPath($CandidateDirectory)
$checkpointDirectory = Join-Path $candidate "checkpoints"
[System.IO.Directory]::CreateDirectory($checkpointDirectory) | Out-Null

function Save-LatestCheckpoint {
    $historyPath = Join-Path $candidate "history.json"
    $lastPath = Join-Path $candidate "last.pt"
    if (!(Test-Path -LiteralPath $historyPath) -or !(Test-Path -LiteralPath $lastPath)) {
        return
    }
    try {
        $history = Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
    }
    catch {
        return
    }
    if (!$history -or $history.Count -eq 0) {
        return
    }
    $epoch = [int] $history[-1].epoch
    $destination = Join-Path $checkpointDirectory ("epoch-{0:D3}.pt" -f $epoch)
    if (!(Test-Path -LiteralPath $destination)) {
        Copy-Item -LiteralPath $lastPath -Destination $destination
    }
}

while (Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue) {
    Save-LatestCheckpoint
    Start-Sleep -Seconds $IntervalSeconds
}
Save-LatestCheckpoint
