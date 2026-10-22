param(
    [int] $Candidate7ProcessId = 0
)

$ErrorActionPreference = "Stop"

$python = "E:\.packages\conda\envs\voicebox\python.exe"
$trainerScript = "E:\Projects\voicebox\scripts\render-aware-dtd\train_dtd.py"
$monitorScript = "E:\Projects\voicebox\scripts\render-aware-dtd\monitor_training_resources.ps1"
$datasetRoot = "E:\Projects\render-aware-dtd\datasets\AEC-Challenge"
$enrichedRoot = "E:\Projects\render-aware-dtd\datasets\render-aware-dtd-v4"
$artifactRoot = "E:\Projects\render-aware-dtd\artifacts-v4"
$frozenInputs = [ordered]@{
    $trainerScript = "9E02478B21780BF75F8FD80F0A8C989C7B1FB39046D119013DA347F6EEF65AEC"
    "E:\Projects\voicebox\scripts\render-aware-dtd\dtd_model.py" = "54D42E9A7BBB3C7015B77942557C136FF9742AC0587A65BC56BE52C51C4B2362"
    "E:\Projects\voicebox\scripts\render-aware-dtd\dtd_metrics.py" = "CBD8EFB16B78518C070366790B8A679EECB2510BE6252E5A7FAC5D13BAFD2966"
    "E:\Projects\voicebox\scripts\render-aware-dtd\enriched_dataset.py" = "524040B9D9538D4E0EBDE57157AA0AAE1508FBCC507E52B201291B63DFF5AC78"
    "E:\Projects\voicebox\scripts\render-aware-dtd\export_dtd_onnx.py" = "9C8E9BB72914951DC5776B7754E032611E9B19FB4C2CB96ECC83A43855A3DB05"
    (Join-Path $enrichedRoot "dataset_summary.json") = "347ED8D7935FEDF49708FCB3B828BCB94A6270B2825282A4F7D1DBF6D87613DB"
}

$specifications = @(
    [ordered]@{
        Candidate = 7
        ChangedGroup = "256x3 / 150-epoch anchor"
        LearningRate = "8e-4"
        WarmupEpochs = "5"
        WeightDecay = "1e-4"
        PositiveWeight = "1.5"
        Dropout = "0.1"
        AecNarrowWeight = "0.40"
        WenetWeight = "0.20"
    },
    [ordered]@{
        Candidate = 8
        ChangedGroup = "LR schedule: peak 6e-4, warmup 8"
        LearningRate = "6e-4"
        WarmupEpochs = "8"
        WeightDecay = "1e-4"
        PositiveWeight = "1.5"
        Dropout = "0.1"
        AecNarrowWeight = "0.40"
        WenetWeight = "0.20"
    },
    [ordered]@{
        Candidate = 9
        ChangedGroup = "Source mixture: narrow .35, Wenet .25"
        LearningRate = "8e-4"
        WarmupEpochs = "5"
        WeightDecay = "1e-4"
        PositiveWeight = "1.5"
        Dropout = "0.1"
        AecNarrowWeight = "0.35"
        WenetWeight = "0.25"
    },
    [ordered]@{
        Candidate = 10
        ChangedGroup = "Loss weighting: positive weight 1.75"
        LearningRate = "8e-4"
        WarmupEpochs = "5"
        WeightDecay = "1e-4"
        PositiveWeight = "1.75"
        Dropout = "0.1"
        AecNarrowWeight = "0.40"
        WenetWeight = "0.20"
    },
    [ordered]@{
        Candidate = 11
        ChangedGroup = "Regularization: dropout .15"
        LearningRate = "8e-4"
        WarmupEpochs = "5"
        WeightDecay = "1e-4"
        PositiveWeight = "1.5"
        Dropout = "0.15"
        AecNarrowWeight = "0.40"
        WenetWeight = "0.20"
    },
    [ordered]@{
        Candidate = 12
        ChangedGroup = "AdamW regularization: weight decay 2e-4"
        LearningRate = "8e-4"
        WarmupEpochs = "5"
        WeightDecay = "2e-4"
        PositiveWeight = "1.5"
        Dropout = "0.1"
        AecNarrowWeight = "0.40"
        WenetWeight = "0.20"
    }
)

function Get-CandidateDirectory {
    param([int] $Candidate)
    return Join-Path $artifactRoot "candidate-$Candidate"
}

function Assert-FrozenInputs {
    foreach ($entry in $frozenInputs.GetEnumerator()) {
        if (-not (Test-Path -LiteralPath $entry.Key -PathType Leaf)) {
            throw "frozen input is missing: $($entry.Key)"
        }
        $actual = (
            Get-FileHash -Algorithm SHA256 -LiteralPath $entry.Key
        ).Hash
        if ($actual -ne $entry.Value) {
            throw (
                "frozen input hash mismatch: path={0}, expected={1}, actual={2}" -f
                $entry.Key,
                $entry.Value,
                $actual
            )
        }
    }
}

function Get-TrainingArguments {
    param(
        [System.Collections.IDictionary] $Specification,
        [int] $Workers,
        [string] $ResumePath = ""
    )

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    $arguments = @(
        $trainerScript,
        "--dataset-root", $datasetRoot,
        "--enriched-root", $enrichedRoot,
        "--output-dir", $candidateDirectory,
        "--epochs", "150",
        "--batch-size", "128",
        "--sequence-seconds", "8",
        "--learning-rate", $Specification.LearningRate,
        "--lr-schedule", "cosine",
        "--warmup-epochs", $Specification.WarmupEpochs,
        "--min-learning-rate", "1e-5",
        "--weight-decay", $Specification.WeightDecay,
        "--positive-weight", $Specification.PositiveWeight,
        "--model-size", "256",
        "--gru-layers", "3",
        "--dropout", $Specification.Dropout,
        "--aec-narrow-weight", $Specification.AecNarrowWeight,
        "--aec-fullband-weight", "0.20",
        "--wenet-weight", $Specification.WenetWeight,
        "--aec-real-weight", "0.20",
        "--wenet-far-only-probability", "0.40",
        "--examples-per-epoch", "12000",
        "--validation-examples", "1500",
        "--validation-modulo", "20",
        "--train-limit", "0",
        "--validation-limit", "0",
        "--num-workers", [string]$Workers,
        "--seed", "43",
        "--device", "cuda",
        "--early-stopping-patience", "0",
        "--early-stopping-min-delta", "0.0002"
    )
    if ($ResumePath) {
        $arguments += @("--resume", $ResumePath)
    }
    return $arguments
}

function Test-CompletedCandidate {
    param([System.Collections.IDictionary] $Specification)

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    $summaryPath = Join-Path $candidateDirectory "training_summary.json"
    $historyPath = Join-Path $candidateDirectory "history.json"
    $checkpointDirectory = Join-Path $candidateDirectory "checkpoints"
    if (
        -not (Test-Path -LiteralPath $summaryPath) -or
        -not (Test-Path -LiteralPath $historyPath) -or
        -not (Test-Path -LiteralPath $checkpointDirectory)
    ) {
        return $false
    }

    try {
        $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
        $historyRaw = (
            Get-Content -LiteralPath $historyPath -Raw | ConvertFrom-Json
        )
        $history = @($historyRaw)
    }
    catch {
        return $false
    }
    $checkpointNames = @(
        Get-ChildItem -LiteralPath $checkpointDirectory -Filter "epoch-*.pt" |
            ForEach-Object { $_.Name } |
            Sort-Object
    )
    $expectedCheckpointNames = @(
        0..149 | ForEach-Object { "epoch-{0:D3}.pt" -f $_ }
    )
    $historyEpochs = @(
        $history | ForEach-Object { [int]$_.epoch }
    )
    $expectedHistoryEpochs = @(0..149)
    $checkpointMismatch = @(
        Compare-Object $expectedCheckpointNames $checkpointNames
    ).Count -ne 0
    $historyMismatch = @(
        Compare-Object $expectedHistoryEpochs $historyEpochs -SyncWindow 0
    ).Count -ne 0

    $trainingArgs = $summary.training_args
    return (
        [int]$summary.epochs_completed -eq 150 -and
        [int]$summary.model_config.projection_size -eq 256 -and
        [int]$summary.model_config.hidden_size -eq 256 -and
        [int]$summary.model_config.gru_layers -eq 3 -and
        [double]$summary.model_config.dropout -eq [double]$Specification.Dropout -and
        [string]$trainingArgs.dataset_root -eq $datasetRoot -and
        [string]$trainingArgs.enriched_root -eq $enrichedRoot -and
        [int]$trainingArgs.epochs -eq 150 -and
        [int]$trainingArgs.batch_size -eq 128 -and
        [double]$trainingArgs.sequence_seconds -eq 8.0 -and
        [double]$trainingArgs.learning_rate -eq [double]$Specification.LearningRate -and
        [string]$trainingArgs.lr_schedule -eq "cosine" -and
        [int]$trainingArgs.warmup_epochs -eq [int]$Specification.WarmupEpochs -and
        [double]$trainingArgs.min_learning_rate -eq 1e-5 -and
        [double]$trainingArgs.weight_decay -eq [double]$Specification.WeightDecay -and
        [double]$trainingArgs.positive_weight -eq [double]$Specification.PositiveWeight -and
        [int]$trainingArgs.model_size -eq 256 -and
        [int]$trainingArgs.gru_layers -eq 3 -and
        [double]$trainingArgs.dropout -eq [double]$Specification.Dropout -and
        [double]$trainingArgs.aec_narrow_weight -eq [double]$Specification.AecNarrowWeight -and
        [double]$trainingArgs.aec_fullband_weight -eq 0.20 -and
        [double]$trainingArgs.wenet_weight -eq [double]$Specification.WenetWeight -and
        [double]$trainingArgs.aec_real_weight -eq 0.20 -and
        [double]$trainingArgs.wenet_far_only_probability -eq 0.40 -and
        [int]$trainingArgs.examples_per_epoch -eq 12000 -and
        [int]$trainingArgs.validation_examples -eq 1500 -and
        [int]$trainingArgs.validation_modulo -eq 20 -and
        [int]$trainingArgs.train_limit -eq 0 -and
        [int]$trainingArgs.validation_limit -eq 0 -and
        [int]$trainingArgs.seed -eq 43 -and
        [string]$trainingArgs.device -eq "cuda" -and
        [int]$trainingArgs.early_stopping_patience -eq 0 -and
        [double]$trainingArgs.early_stopping_min_delta -eq 0.0002 -and
        $history.Count -eq 150 -and
        -not $historyMismatch -and
        -not $checkpointMismatch
    )
}

function Test-HasTrainingEvidence {
    param([System.Collections.IDictionary] $Specification)

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    foreach ($name in @("history.json", "last.pt", "training.stdout.log")) {
        if (Test-Path -LiteralPath (Join-Path $candidateDirectory $name)) {
            return $true
        }
    }
    $checkpointDirectory = Join-Path $candidateDirectory "checkpoints"
    if (Test-Path -LiteralPath $checkpointDirectory -PathType Container) {
        $checkpointCount = @(
            Get-ChildItem `
                -LiteralPath $checkpointDirectory `
                -Filter "epoch-*.pt" `
                -ErrorAction SilentlyContinue
        ).Count
        if ($checkpointCount -gt 0) {
            return $true
        }
    }
    return $false
}

function Test-HasRecoverableCheckpoint {
    param([System.Collections.IDictionary] $Specification)

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    if (Test-Path -LiteralPath (Join-Path $candidateDirectory "last.pt")) {
        return $true
    }
    $checkpointDirectory = Join-Path $candidateDirectory "checkpoints"
    if (-not (Test-Path -LiteralPath $checkpointDirectory -PathType Container)) {
        return $false
    }
    return @(
        Get-ChildItem `
            -LiteralPath $checkpointDirectory `
            -Filter "epoch-*.pt" `
            -ErrorAction SilentlyContinue
    ).Count -gt 0
}

function Get-NextResumeAttempt {
    param([System.Collections.IDictionary] $Specification)

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    $maximum = 0
    foreach (
        $file in Get-ChildItem `
            -LiteralPath $candidateDirectory `
            -Filter "training.resume-*.stdout.log" `
            -ErrorAction SilentlyContinue
    ) {
        if ($file.Name -match "^training\.resume-(\d{3})\.stdout\.log$") {
            $maximum = [Math]::Max($maximum, [int]$Matches[1])
        }
    }
    return $maximum + 1
}

function Write-ExperimentMetadata {
    param([System.Collections.IDictionary] $Specification)

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    [System.IO.Directory]::CreateDirectory($candidateDirectory) | Out-Null
    $path = Join-Path $candidateDirectory "EXPERIMENT.json"
    if (Test-Path -LiteralPath $path) {
        throw "refusing to overwrite experiment metadata: $path"
    }
    $metadata = [ordered]@{
        schema_version = 1
        candidate = "candidate-$($Specification.Candidate)"
        anchor = "candidate-7"
        changed_parameter_group = $Specification.ChangedGroup
        formal_training = $true
        test_data_opened = $false
        recipe = [ordered]@{
            model_size = 256
            gru_layers = 3
            epochs = 150
            batch_size = 128
            sequence_seconds = 8.0
            seed = 43
            learning_rate = $Specification.LearningRate
            lr_schedule = "cosine"
            warmup_epochs = [int]$Specification.WarmupEpochs
            min_learning_rate = "1e-5"
            weight_decay = $Specification.WeightDecay
            positive_weight = $Specification.PositiveWeight
            dropout = $Specification.Dropout
            aec_narrow_weight = $Specification.AecNarrowWeight
            aec_fullband_weight = "0.20"
            wenet_weight = $Specification.WenetWeight
            aec_real_weight = "0.20"
            wenet_far_only_probability = "0.40"
            examples_per_epoch = 12000
            validation_examples = 1500
            initial_workers = 8
            recovery_workers = 4
            early_stopping_patience = 0
        }
    }
    [System.IO.File]::WriteAllText(
        $path,
        ($metadata | ConvertTo-Json -Depth 8) + [Environment]::NewLine
    )
}

function Invoke-Training {
    param(
        [System.Collections.IDictionary] $Specification,
        [int] $Workers,
        [int] $Attempt
    )

    Assert-FrozenInputs
    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    $suffix = if ($Attempt -eq 0) { "" } else { ".resume-{0:D3}" -f $Attempt }
    $stdout = Join-Path $candidateDirectory "training$suffix.stdout.log"
    $stderr = Join-Path $candidateDirectory "training$suffix.stderr.log"
    $resourceLog = Join-Path $candidateDirectory "resource-monitor$suffix.csv"
    $monitorStdout = Join-Path $candidateDirectory "resource-monitor$suffix.stdout.log"
    $monitorStderr = Join-Path $candidateDirectory "resource-monitor$suffix.stderr.log"
    foreach ($path in @($stdout, $stderr, $resourceLog, $monitorStdout, $monitorStderr)) {
        if (Test-Path -LiteralPath $path) {
            throw "refusing to overwrite orchestration output: $path"
        }
    }

    $resumePath = if ($Attempt -eq 0) {
        ""
    } else {
        Join-Path $candidateDirectory "last.pt"
    }
    $arguments = Get-TrainingArguments $Specification $Workers $resumePath
    $trainer = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -PassThru `
        -WindowStyle Hidden
    Write-Host (
        "{0:o} candidate-{1} attempt={2} workers={3} trainer_pid={4}" -f
        [DateTimeOffset]::Now,
        $Specification.Candidate,
        $Attempt,
        $Workers,
        $trainer.Id
    )

    $monitorArguments = @(
        "-NoProfile",
        "-ExecutionPolicy", "Bypass",
        "-File", $monitorScript,
        "-TargetProcessId", [string]$trainer.Id,
        "-OutputPath", $resourceLog,
        "-IntervalSeconds", "2"
    )
    $monitor = Start-Process `
        -FilePath "powershell.exe" `
        -ArgumentList $monitorArguments `
        -RedirectStandardOutput $monitorStdout `
        -RedirectStandardError $monitorStderr `
        -PassThru `
        -WindowStyle Hidden

    $trainer.WaitForExit()
    $trainer.Refresh()
    $exitCode = $trainer.ExitCode
    if ($null -eq $exitCode) {
        $exitCode = if (Test-CompletedCandidate $Specification) { 0 } else { -1 }
    }
    [void]$monitor.WaitForExit(15000)
    Write-Host (
        "{0:o} candidate-{1} attempt={2} exit_code={3}" -f
        [DateTimeOffset]::Now,
        $Specification.Candidate,
        $Attempt,
        $exitCode
    )
    return $exitCode
}

function Complete-OrRecoverCandidate {
    param(
        [System.Collections.IDictionary] $Specification,
        [bool] $InitialAttemptAlreadyRan
    )

    if (Test-CompletedCandidate $Specification) {
        Write-Output "candidate-$($Specification.Candidate) already complete"
        return
    }

    $initialAttemptRan = (
        $InitialAttemptAlreadyRan -or
        (Test-HasTrainingEvidence $Specification)
    )
    if (-not $initialAttemptRan) {
        $exitCode = Invoke-Training $Specification 8 0
        if (Test-CompletedCandidate $Specification) {
            return
        }
    }

    $candidateDirectory = Get-CandidateDirectory $Specification.Candidate
    for ($recovery = 1; $recovery -le 3; $recovery++) {
        $lastPath = Join-Path $candidateDirectory "last.pt"
        if (-not (Test-HasRecoverableCheckpoint $Specification)) {
            throw "candidate-$($Specification.Candidate) failed without a recoverable checkpoint"
        }
        $attempt = Get-NextResumeAttempt $Specification
        $resumeExitCode = Invoke-Training $Specification 4 $attempt
        if (Test-CompletedCandidate $Specification) {
            return
        }
    }
    throw "candidate-$($Specification.Candidate) did not complete after three recoveries"
}

Write-Output ("{0:o} 256x3 round orchestrator started" -f [DateTimeOffset]::Now)

$candidate7 = $specifications[0]
if (-not (Test-CompletedCandidate $candidate7)) {
    if ($Candidate7ProcessId -le 0) {
        throw "candidate-7 is incomplete and no live process id was supplied"
    }
    $candidate7Process = Get-Process -Id $Candidate7ProcessId -ErrorAction SilentlyContinue
    if ($candidate7Process) {
        Write-Output "waiting for existing candidate-7 pid=$Candidate7ProcessId"
        $candidate7Process.WaitForExit()
    }
    Complete-OrRecoverCandidate $candidate7 $true
}

foreach ($specification in $specifications | Select-Object -Skip 1) {
    $candidateDirectory = Get-CandidateDirectory $specification.Candidate
    if (-not (Test-Path -LiteralPath $candidateDirectory)) {
        Write-ExperimentMetadata $specification
    }
    Complete-OrRecoverCandidate $specification $false
}

Write-Output ("{0:o} all six candidates complete" -f [DateTimeOffset]::Now)
