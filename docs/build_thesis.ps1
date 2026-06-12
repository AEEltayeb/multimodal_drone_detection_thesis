# Build thesis_working.pdf with MiKTeX (no Perl/latexmk needed).
# Usage:  cd docs ;  .\build_thesis.ps1            (builds thesis_working.tex)
#         .\build_thesis.ps1 thesis_chapters       (builds a different file, no .tex)
param([string]$Name = "thesis_working")

$ErrorActionPreference = "Stop"
$bin = "$env:LOCALAPPDATA\Programs\MiKTeX\miktex\bin\x64"
if (Test-Path $bin) { $env:PATH = "$bin;$env:PATH" }
Set-Location $PSScriptRoot

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
    Invoke-Item $pdf
} else {
    Write-Host "BUILD FAILED - see $Name.log" -ForegroundColor Red
    exit 1
}
