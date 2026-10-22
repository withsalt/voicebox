param(
    [Parameter(Mandatory = $true)]
    [int] $TargetProcessId,

    [Parameter(Mandatory = $true)]
    [string] $OutputPath,

    [int] $IntervalSeconds = 2
)

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$header = @(
    "timestamp"
    "committed_bytes"
    "commit_limit_bytes"
    "percent_committed"
    "available_bytes"
    "process_tree_working_set_bytes"
    "gpu_used_mib"
) -join ","
[System.IO.File]::WriteAllText($resolvedOutput, $header + [Environment]::NewLine)

while (Get-Process -Id $TargetProcessId -ErrorAction SilentlyContinue) {
    $memory = Get-CimInstance Win32_PerfFormattedData_PerfOS_Memory
    $processes = @(Get-CimInstance Win32_Process)
    $targetIds = [System.Collections.Generic.HashSet[int]]::new()
    [void] $targetIds.Add($TargetProcessId)
    do {
        $added = $false
        foreach ($process in $processes) {
            if (
                $targetIds.Contains([int] $process.ParentProcessId) -and
                $targetIds.Add([int] $process.ProcessId)
            ) {
                $added = $true
            }
        }
    } while ($added)
    $workingSet = 0L
    foreach ($processId in $targetIds) {
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        if ($process) {
            $workingSet += [int64] $process.WorkingSet64
        }
    }
    $gpuUsed = (
        & nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits |
            Select-Object -First 1
    )
    $row = @(
        [DateTimeOffset]::Now.ToString("o")
        [int64] $memory.CommittedBytes
        [int64] $memory.CommitLimit
        [int] $memory.PercentCommittedBytesInUse
        [int64] $memory.AvailableBytes
        $workingSet
        [int] $gpuUsed
    ) -join ","
    [System.IO.File]::AppendAllText(
        $resolvedOutput,
        $row + [Environment]::NewLine
    )
    Start-Sleep -Seconds $IntervalSeconds
}
