$ErrorActionPreference = "Stop"

if (-not (Get-Command xelatex -ErrorAction SilentlyContinue)) {
    throw "xelatex was not found. Install TeX Live or MiKTeX first."
}

$packageRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$reports = foreach ($prefix in @("1_*", "2_*")) {
    $directory = Get-ChildItem -LiteralPath $packageRoot -Directory |
        Where-Object { $_.Name -like $prefix } |
        Select-Object -First 1
    if (-not $directory) {
        throw "Report directory matching '$prefix' was not found."
    }

    $texFile = Get-ChildItem -LiteralPath $directory.FullName -File -Filter "*.tex" |
        Select-Object -First 1
    if (-not $texFile) {
        throw "No TeX report was found in '$($directory.FullName)'."
    }

    @{
        Directory = $directory.FullName
        File = $texFile.Name
    }
}

foreach ($report in $reports) {
    Push-Location $report.Directory
    try {
        Write-Host "Compiling $($report.File) (3 passes)..."
        for ($round = 1; $round -le 3; $round++) {
            Write-Host "  Pass $round"
            & xelatex -interaction=nonstopmode -halt-on-error $report.File
            if ($LASTEXITCODE -ne 0) {
                throw "$($report.File) failed on pass $round."
            }
        }

        $baseName = [System.IO.Path]::GetFileNameWithoutExtension($report.File)
        foreach ($suffix in @(".aux", ".log", ".out", ".toc", ".synctex.gz")) {
            Remove-Item -LiteralPath (Join-Path $report.Directory ($baseName + $suffix)) `
                -Force -ErrorAction SilentlyContinue
        }
    }
    finally {
        Pop-Location
    }
}

Write-Host "Both reports compiled; contents and cross-references are current."
