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

function Test-ParquetFooter([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        if ($stream.Length -lt 8) { return $false }
        $magic = [byte[]]::new(4)
        [void]$stream.Read($magic, 0, 4)
        if ([System.Text.Encoding]::ASCII.GetString($magic) -ne 'PAR1') { return $false }
        [void]$stream.Seek(-4, [System.IO.SeekOrigin]::End)
        [void]$stream.Read($magic, 0, 4)
        return [System.Text.Encoding]::ASCII.GetString($magic) -eq 'PAR1'
    }
    finally { $stream.Dispose() }
}

function Test-Aria2Control([string]$Partial) {
    return (Test-Path -LiteralPath "$Partial.aria2" -PathType Leaf) -or
        (Test-Path -LiteralPath "$Partial.aria2__temp" -PathType Leaf)
}

function Invoke-ResumableCurl([string]$Url, [string]$Output, [string]$Filename) {
    for ($attempt = 1; $attempt -le 10; $attempt++) {
        & $curl.Source --fail --location --continue-at - --silent --show-error --output $Output $Url
        if ($LASTEXITCODE -eq 0) { return }
        Write-Warning "curl attempt $attempt failed for $Filename (exit $LASTEXITCODE); resuming"
        if ($attempt -lt 10) { Start-Sleep -Seconds 2 }
    }
    throw "curl could not finish $Filename after 10 attempts"
}

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
    elseif ((Test-ParquetFooter $partial) -and -not (Test-Aria2Control $partial)) {
        Move-Item -LiteralPath $partial -Destination $target
        Write-Host "Recovered completed transfer: $filename"
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
        if (-not (Test-ParquetFooter $download.Partial)) {
            throw "downloaded file is not a complete Parquet file: $($download.Filename)"
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
    $aria2Log = Join-Path $rawRoot 'aria2-download.log'
    $inputLines -join "`n" | & $aria2.Source `
        --input-file=- `
        --max-concurrent-downloads=$Parallelism `
        --split=4 `
        --max-connection-per-server=4 `
        --min-split-size=1M `
        --continue=true `
        --auto-file-renaming=false `
        --allow-overwrite=true `
        --max-tries=5 `
        --retry-wait=2 `
        --log=$aria2Log `
        --summary-interval=10 `
        --console-log-level=notice
    $aria2ExitCode = $LASTEXITCODE
    $incomplete = @()
    foreach ($download in $downloads) {
        if ((Test-ParquetFooter $download.Partial) -and -not (Test-Aria2Control $download.Partial)) {
            Move-Item -LiteralPath $download.Partial -Destination $download.Target
            Write-Host "Completed: $($download.Filename)"
        }
        else {
            $incomplete += $download
        }
    }
    if ($incomplete.Count -gt 0) {
        Write-Warning "aria2 left $($incomplete.Count) incomplete file(s); retrying fresh copies with curl. Details: $aria2Log"
        foreach ($download in $incomplete) {
            $repair = "$($download.Partial).repair"
            Write-Host "Repairing: $($download.Filename)"
            if ($null -ne $curl) {
                Invoke-ResumableCurl $download.Url $repair $download.Filename
            }
            else {
                Invoke-WebRequest -Uri $download.Url -OutFile $repair
            }
            if (-not (Test-ParquetFooter $repair)) {
                throw "repair did not produce a complete Parquet file: $($download.Filename); aria2 log: $aria2Log"
            }
            Move-Item -LiteralPath $repair -Destination $download.Target
            foreach ($stale in @($download.Partial, "$($download.Partial).aria2", "$($download.Partial).aria2__temp")) {
                if (Test-Path -LiteralPath $stale -PathType Leaf) {
                    Remove-Item -LiteralPath $stale
                }
            }
            Write-Host "Completed after repair: $($download.Filename)"
        }
    }
    elseif ($aria2ExitCode -ne 0) {
        Write-Warning "aria2 returned exit code $aria2ExitCode, but all files passed validation"
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
    if (-not (Test-ParquetFooter $download.Partial)) {
        throw "downloaded file is not a complete Parquet file: $($download.Filename)"
    }
    Move-Item -LiteralPath $download.Partial -Destination $download.Target
    Write-Host "Completed: $($download.Filename)"
}
