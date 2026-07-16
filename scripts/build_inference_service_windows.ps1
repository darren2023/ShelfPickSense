param(
    [string]$Name = "shelf-pick-inference-service",
    [string]$DistPath = "dist",
    [string]$WorkPath = "build\nuitka",
    [switch]$IncludeXGBoost,
    [switch]$IncludeLightGBM
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $Root

$OutputFilename = "$Name.exe"
$NuitkaArgs = @(
    "run", "--with", "nuitka", "--with", "zstandard", "python", "-m", "nuitka",
    "--standalone",
    "--onefile",
    "--assume-yes-for-downloads",
    "--output-dir=$DistPath",
    "--output-filename=$OutputFilename",
    "--remove-output",
    "--include-package=analysis",
    "--include-package=sklearn",
    "--include-package=joblib",
    "--nofollow-import-to=cv2",
    "--nofollow-import-to=xgboost.testing",
    "--nofollow-import-to=hypothesis",
    "--nofollow-import-to=pytest",
    "scripts\inference_service_entry.py"
)

if ($IncludeXGBoost) {
    $NuitkaArgs = $NuitkaArgs[0..($NuitkaArgs.Length - 2)] + @("--include-package=xgboost") + $NuitkaArgs[-1]
}
if ($IncludeLightGBM) {
    $NuitkaArgs = $NuitkaArgs[0..($NuitkaArgs.Length - 2)] + @("--include-package=lightgbm") + $NuitkaArgs[-1]
}

New-Item -ItemType Directory -Force -Path $WorkPath | Out-Null
& uv @NuitkaArgs

Write-Host "Built: $(Join-Path $DistPath $OutputFilename)"
