param(
    [string]$PythonExe = "",
    [string]$Device = "cuda:0",
    [int]$Epochs = 500,
    [string]$RunDir = "",
    [int]$WaitForProcessId = 0
)

$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$workspaceRoot = Split-Path $repoRoot -Parent

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

if (-not $RunDir) {
    $RunDir = Join-Path $repoRoot "logs\megnet_sparse_reproduction"
}
$RunDir = [System.IO.Path]::GetFullPath($RunDir)
New-Item -ItemType Directory -Path $RunDir -Force | Out-Null

$statusPath = Join-Path $RunDir "benchmark_status.log"
$outputPath = Join-Path $RunDir "benchmark_output.log"
$exitCodePath = Join-Path $RunDir "benchmark_exit_code.txt"

if ($WaitForProcessId -gt 0) {
    $existingProcess = Get-Process -Id $WaitForProcessId -ErrorAction SilentlyContinue
    if ($existingProcess) {
        "$(Get-Date -Format o) queued behind PID $WaitForProcessId" |
            Set-Content -LiteralPath $statusPath
        Wait-Process -Id $WaitForProcessId
    }
}

"$(Get-Date -Format o) starting MEGNET_SPARSE benchmark" |
    Add-Content -LiteralPath $statusPath

Push-Location $workspaceRoot
try {
    & $PythonExe -m HERA.main `
        --model megnet `
        --dataset 2dmd_high vacancy `
        --mode sparse `
        --epochs $Epochs `
        --device $Device `
        --seed 123 `
        --atom-init (Join-Path $repoRoot "atom_init.json") `
        --run-dir $RunDir `
        --resume *>> $outputPath
    $benchmarkExitCode = $LASTEXITCODE
}
finally {
    Pop-Location
}

$benchmarkExitCode | Set-Content -LiteralPath $exitCodePath
"$(Get-Date -Format o) finished with exit code $benchmarkExitCode" |
    Add-Content -LiteralPath $statusPath
exit $benchmarkExitCode
