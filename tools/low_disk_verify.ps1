param(
    [switch]$Full,
    [switch]$KeepTemp,
    [switch]$KeepProjectVenv
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$WorkRoot = Join-Path $env:TEMP "edgefit-work"
$CargoTarget = Join-Path $WorkRoot "cargo-target"
$TmpRoot = Join-Path $WorkRoot "tmp"
$UvCache = Join-Path $WorkRoot "uv-cache"
$PyCachePrefix = Join-Path $WorkRoot "pycache"
$ReportDir = Join-Path $WorkRoot "reports"

$env:CARGO_TARGET_DIR = $CargoTarget
$env:TMP = $TmpRoot
$env:TEMP = $TmpRoot
$env:UV_CACHE_DIR = $UvCache
$env:PYTHONPYCACHEPREFIX = $PyCachePrefix
$env:PYTHONDONTWRITEBYTECODE = "1"

function Invoke-Step {
    param(
        [string]$Name,
        [scriptblock]$Command,
        [int[]]$AllowedExitCodes = @(0)
    )

    Write-Host "==> $Name"
    & $Command
    $code = if ($null -eq $global:LASTEXITCODE) { 0 } else { $global:LASTEXITCODE }
    if ($AllowedExitCodes -notcontains $code) {
        throw "$Name failed with exit code $code"
    }
    $global:LASTEXITCODE = 0
}

function Remove-GeneratedPath {
    param([string]$Path)

    if (-not (Test-Path -LiteralPath $Path)) {
        return
    }

    $resolved = (Resolve-Path -LiteralPath $Path).Path
    $item = Get-Item -LiteralPath $resolved -Force
    if ($item.PSIsContainer) {
        [System.IO.Directory]::Delete($resolved, $true)
    } else {
        [System.IO.File]::Delete($resolved)
    }

    if (Test-Path -LiteralPath $resolved) {
        throw "Failed to remove generated path: $resolved"
    }
}

function Remove-ProjectGeneratedFiles {
    $paths = @(
        (Join-Path $RepoRoot "target"),
        (Join-Path $RepoRoot ".uv-cache"),
        (Join-Path $RepoRoot "tmp"),
        (Join-Path $RepoRoot "reports"),
        (Join-Path $RepoRoot ".pytest_cache")
    )

    if (-not $KeepProjectVenv) {
        $paths += (Join-Path $RepoRoot ".venv")
    }

    foreach ($path in $paths) {
        Remove-GeneratedPath $path
    }

    Get-ChildItem -LiteralPath $RepoRoot -Recurse -Directory -Filter "__pycache__" -ErrorAction SilentlyContinue |
        ForEach-Object { Remove-GeneratedPath $_.FullName }
}

New-Item -ItemType Directory -Force -Path $CargoTarget, $TmpRoot, $UvCache, $PyCachePrefix, $ReportDir | Out-Null

$locationPushed = $false
try {
    Push-Location $RepoRoot
    $locationPushed = $true

    Invoke-Step "cargo test" { cargo test --workspace }

    $edgefit = Join-Path $CargoTarget "debug\edgefit.exe"
    Invoke-Step "validate esp32s3" { & $edgefit target validate targets\esp32s3.yaml }
    Invoke-Step "validate ort mobile" { & $edgefit target validate targets\ort-mobile-cpu.yaml }
    Invoke-Step "validate tflm micro" { & $edgefit target validate targets\tflm-micro.yaml }
    Invoke-Step "check good tiny" { & $edgefit check examples\models\good_tiny.edgefit.json --target targets\esp32s3.yaml }
    Invoke-Step "check bad detector expected fail" { & $edgefit check examples\models\bad_detector.edgefit.json --target targets\esp32s3.yaml --format sarif --out (Join-Path $ReportDir "edgefit.sarif") --summary (Join-Path $ReportDir "edgefit-summary.md") } @(1)
    Invoke-Step "check suppressed dynamic rank expected fail" { & $edgefit check examples\models\rank_dynamic.edgefit.json --target targets\esp32s3.yaml --format json --suppress EF0101,EF0102 } @(1)

    if ($Full) {
        Invoke-Step "python public trial tests" { python -m unittest tools\test_public_pr_trial_gate.py }
        $internalAuditTest = Join-Path $RepoRoot "tools\test_implementation_audit.py"
        if (Test-Path -LiteralPath $internalAuditTest) {
            Invoke-Step "python internal audit tests" { python -m unittest tools\test_implementation_audit.py }
        }
        Invoke-Step "python onnx tool tests" { python -m unittest discover -s tools\onnx-normalize -p "test_*.py" }
    }
} finally {
    if ($locationPushed) {
        Pop-Location
    }

    Remove-ProjectGeneratedFiles

    if (-not $KeepTemp) {
        Remove-GeneratedPath $WorkRoot
    }
}

Write-Host "EdgeFit low-disk verification passed."
if ($KeepTemp) {
    Write-Host "Temporary work root kept at: $WorkRoot"
} else {
    Write-Host "Temporary work root removed: $WorkRoot"
}
