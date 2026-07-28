param(
    [int]$CurrentBenchmarkProcessId = 22576,
    [string]$PythonExe = "",
    [string]$Device = "cuda:0",
    [int]$Epochs = 500,
    [int]$PollSeconds = 15
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = Split-Path $repoRoot -Parent
$currentRunDir = Join-Path $repoRoot "logs\bench_2dmd_high_all_models"
$sparseRunDir = Join-Path $repoRoot "logs\megnet_sparse_reproduction"
$historyPath = Join-Path $currentRunDir "alignn\2dmd_high\full_x\seed123_history.csv"
$checkpointPath = Join-Path $currentRunDir "alignn\2dmd_high\full_x\seed123_best_checkpoint.pth"
$statusPath = Join-Path $repoRoot "logs\megnet_sparse_interleave_status.log"
$resumeOutputPath = Join-Path $currentRunDir "resume_after_megnet_sparse.log"
$sparseScript = Join-Path $PSScriptRoot "run_megnet_sparse_benchmark.ps1"

if (-not $PythonExe) {
    if ($env:CONDA_PREFIX) {
        $condaPython = Join-Path $env:CONDA_PREFIX "python.exe"
        if (Test-Path -LiteralPath $condaPython) {
            $PythonExe = $condaPython
        }
    }
    if (-not $PythonExe) {
        $PythonExe = (Get-Command python).Source
    }
}

"$(Get-Date -Format o) waiting for ALIGNN 2dmd_high/full_x seed123" |
    Set-Content -LiteralPath $statusPath

while ($true) {
    $benchmarkProcess = Get-Process -Id $CurrentBenchmarkProcessId -ErrorAction SilentlyContinue
    if (-not $benchmarkProcess) {
        throw "Benchmark PID $CurrentBenchmarkProcessId exited before full_x completed."
    }

    $testComplete = $false
    if (Test-Path -LiteralPath $historyPath) {
        try {
            $testComplete = Select-String `
                -LiteralPath $historyPath `
                -Pattern '^TEST,' `
                -Quiet
        }
        catch {
            $testComplete = $false
        }
    }
    if ($testComplete -and (Test-Path -LiteralPath $checkpointPath)) {
        break
    }
    Start-Sleep -Seconds $PollSeconds
}

"$(Get-Date -Format o) full_x complete; stopping PID $CurrentBenchmarkProcessId" |
    Add-Content -LiteralPath $statusPath
Stop-Process -Id $CurrentBenchmarkProcessId -Force
Wait-Process -Id $CurrentBenchmarkProcessId -ErrorAction SilentlyContinue

"$(Get-Date -Format o) starting MEGNET_SPARSE seed123" |
    Add-Content -LiteralPath $statusPath
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $sparseScript `
    -PythonExe $PythonExe `
    -Device $Device `
    -Epochs $Epochs `
    -RunDir $sparseRunDir
$sparseExitCode = $LASTEXITCODE
if ($sparseExitCode -ne 0) {
    "$(Get-Date -Format o) sparse failed with exit code $sparseExitCode; resuming 2dmd anyway" |
        Add-Content -LiteralPath $statusPath
}

"$(Get-Date -Format o) sparse phase ended; resuming 2dmd benchmark" |
    Add-Content -LiteralPath $statusPath
Push-Location $workspaceRoot
try {
    & $PythonExe -m HERA.main `
        --model all `
        --dataset 2dmd_high `
        --mode all `
        --r 0 `
        --epochs $Epochs `
        --device $Device `
        --seed 123 `
        --atom-init (Join-Path $repoRoot "atom_init.json") `
        --run-dir $currentRunDir `
        --resume *>> $resumeOutputPath
    $resumeExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

"$(Get-Date -Format o) resumed 2dmd benchmark finished with exit code $resumeExitCode" |
    Add-Content -LiteralPath $statusPath
if ($sparseExitCode -ne 0) {
    exit $sparseExitCode
}
exit $resumeExitCode
