<#
    One-time setup for the silero-tts plugin: creates .venv with PyTorch (CPU)
    in the plugin folder and verifies the worker can import torch.
    Run from anywhere; it targets the folder this script lives in.
    Requires Python 3.9+ (or uv).
#>
$ErrorActionPreference = "Stop"
$Root = $PSScriptRoot
if (-not $Root) { $Root = Get-Location }

$Venv = Join-Path $Root ".venv"
$VenvPy = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $VenvPy)) {
    Write-Host "Creating .venv ..."
    if (Get-Command python -ErrorAction SilentlyContinue) {
        python -m venv $Venv
    } elseif (Get-Command uv -ErrorAction SilentlyContinue) {
        uv venv --seed $Venv
    } else {
        throw "neither python nor uv found"
    }
}

if (-not (Test-Path $VenvPy)) {
    throw "venv python not found: $VenvPy"
}

Write-Host "Installing PyTorch (CPU) + numpy ..."
& $VenvPy -m pip install --upgrade pip numpy torch
if ($LASTEXITCODE -ne 0) {
    # A uv-created venv may have no pip bundled; retry through uv itself.
    Write-Host "pip unavailable inside the venv, retrying with uv ..."
    uv pip install --python $VenvPy numpy torch
    if ($LASTEXITCODE -ne 0) { throw "package install failed" }
}

Write-Host "Checking the worker runtime ..."
& $VenvPy -c "import sys, torch, numpy; print('python', sys.version.split()[0]); print('torch', torch.__version__); print('numpy', numpy.__version__)"
if ($LASTEXITCODE -ne 0) { throw "worker runtime check failed" }

Write-Host ""
Write-Host "Done. First synthesis will download the model (~140 MB) once."