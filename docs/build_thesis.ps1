# Build the thesis PDF with MiKTeX (no Perl/latexmk needed).
# Builds docs/thesis_working_distilling_overleaf/main.tex (the live thesis).
# Usage:  powershell -ExecutionPolicy Bypass -File docs/build_thesis.ps1
# biblatex backend is bibtex (NOT biber).
param([string]$Name = "main")

$ErrorActionPreference = "Stop"
$bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $bin) { $env:PATH = "$bin;$env:PATH" }
Set-Location (Join-Path $PSScriptRoot "thesis_working_distilling_overleaf")

Write-Host "pdflatex pass 1..." -ForegroundColor Cyan
& pdflatex -interaction=nonstopmode "$Name.tex" | Out-Null
Write-Host "bibtex..."          -ForegroundColor Cyan
& bibtex $Name                  | Out-Null
Write-Host "pdflatex pass 2..." -ForegroundColor Cyan
& pdflatex -interaction=nonstopmode "$Name.tex" | Out-Null
Write-Host "pdflatex pass 3..." -ForegroundColor Cyan
& pdflatex -interaction=nonstopmode "$Name.tex" | Out-Null

$pdf = "$Name.pdf"
if (Test-Path $pdf) {
    $kb = [math]::Round((Get-Item $pdf).Length / 1KB)
    Write-Host "OK -> $pdf ($kb KB)" -ForegroundColor Green
} else {
    Write-Host "BUILD FAILED - see $Name.log" -ForegroundColor Red
    exit 1
}
