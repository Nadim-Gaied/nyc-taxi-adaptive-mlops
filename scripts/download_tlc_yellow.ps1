[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-(0[1-9]|1[0-2])$')]
    [string]$StartMonth,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-(0[1-9]|1[0-2])$')]
    [string]$EndMonth,

    [ValidateRange(1, 8)]
    [int]$Parallelism = 1
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$rawRoot = Join-Path $projectRoot 'data\raw\tlc\yellow'
$curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
$aria2 = Get-Command 'aria2c.exe' -ErrorAction SilentlyContinue
$current = [datetime]::ParseExact("$StartMonth-01", 'yyyy-MM-dd', $null)
$last = [datetime]::ParseExact("$EndMonth-01", 'yyyy-MM-dd', $null)

if ($last -lt $current) {
    throw 'EndMonth must not be earlier than StartMonth.'
}

$downloads = @()
while ($current -le $last) {
    $month = $current.ToString('yyyy-MM')
    $year = $current.ToString('yyyy')
    $filename = "yellow_tripdata_$month.parquet"
    $url = "https://d37ci6vzurychx.cloudfront.net/trip-data/$filename"
    $targetDirectory = Join-Path $rawRoot $year
    $target = Join-Path $targetDirectory $filename
    $partial = "$target.partial"

    New-Item -ItemType Directory -Path $targetDirectory -Force | Out-Null
    if (Test-Path -LiteralPath $target -PathType Leaf) {
        Write-Host "Already present: $filename"
    }
    else {
        $downloads += [pscustomobject]@{
            Filename = $filename
            Url = $url
            Target = $target
            Partial = $partial
        }
    }
    $current = $current.AddMonths(1)
}

if ($downloads.Count -eq 0) {
    Write-Host 'All requested files are already present.'
    exit 0
}

if ($Parallelism -eq 1 -or ($null -eq $curl -and $null -eq $aria2)) {
    foreach ($download in $downloads) {
        Write-Host "Downloading: $($download.Filename)"
        if ($null -ne $curl) {
            & $curl.Source `
                --fail `
                --location `
                --retry 3 `
                --retry-delay 2 `
                --continue-at - `
                --output $download.Partial `
                $download.Url
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed for $($download.Filename) with exit code $LASTEXITCODE"
            }
        }
        else {
            Invoke-WebRequest -Uri $download.Url -OutFile $download.Partial
        }
        Move-Item -LiteralPath $download.Partial -Destination $download.Target
    }
    exit 0
}

if ($null -ne $aria2) {
    Write-Host "Downloading $($downloads.Count) files with $Parallelism parallel segmented transfers."
    $inputLines = @()
    foreach ($download in $downloads) {
        $inputLines += $download.Url
        $inputLines += "  dir=$(Split-Path -Parent $download.Partial)"
        $inputLines += "  out=$(Split-Path -Leaf $download.Partial)"
    }
    $inputLines -join "`n" | & $aria2.Source `
        --input-file=- `
        --max-concurrent-downloads=$Parallelism `
        --split=4 `
        --max-connection-per-server=4 `
        --min-split-size=1M `
        --continue=true `
        --auto-file-renaming=false `
        --allow-overwrite=true `
        --summary-interval=10 `
        --console-log-level=notice
    $aria2ExitCode = $LASTEXITCODE
    $incomplete = @(
        $downloads | Where-Object {
            -not (Test-Path -LiteralPath $_.Partial -PathType Leaf) -or
            (Test-Path -LiteralPath "$($_.Partial).aria2" -PathType Leaf)
        }
    )
    if ($incomplete.Count -gt 0) {
        throw "aria2 left $($incomplete.Count) incomplete transfer(s) with exit code $aria2ExitCode; rerun to resume"
    }
    if ($aria2ExitCode -ne 0) {
        Write-Warning "aria2 returned exit code $aria2ExitCode, but every requested transfer completed"
    }
    foreach ($download in $downloads) {
        if (-not (Test-Path -LiteralPath $download.Partial -PathType Leaf)) {
            throw "download completed without expected partial file: $($download.Filename)"
        }
        Move-Item -LiteralPath $download.Partial -Destination $download.Target
        Write-Host "Completed: $($download.Filename)"
    }
    exit 0
}

Write-Host "Downloading $($downloads.Count) files with $Parallelism parallel transfers."
$curlArguments = @(
    '--fail',
    '--location',
    '--retry', '3',
    '--retry-delay', '2',
    '--continue-at', '-',
    '--parallel',
    '--parallel-immediate',
    '--parallel-max', "$Parallelism",
    '--silent',
    '--show-error'
)
foreach ($download in $downloads) {
    $curlArguments += @('--output', $download.Partial, '--url', $download.Url)
}
& $curl.Source @curlArguments
if ($LASTEXITCODE -ne 0) {
    throw "parallel curl failed with exit code $LASTEXITCODE; rerun to resume partial files"
}
foreach ($download in $downloads) {
    if (-not (Test-Path -LiteralPath $download.Partial -PathType Leaf)) {
        throw "download completed without expected partial file: $($download.Filename)"
    }
    Move-Item -LiteralPath $download.Partial -Destination $download.Target
    Write-Host "Completed: $($download.Filename)"
}
