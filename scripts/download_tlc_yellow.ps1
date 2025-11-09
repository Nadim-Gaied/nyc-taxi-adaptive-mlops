[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-(0[1-9]|1[0-2])$')]
    [string]$StartMonth,

    [Parameter(Mandatory = $true)]
    [ValidatePattern('^\d{4}-(0[1-9]|1[0-2])$')]
    [string]$EndMonth
)

$ErrorActionPreference = 'Stop'
$projectRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$rawRoot = Join-Path $projectRoot 'data\raw\tlc\yellow'
$curl = Get-Command 'curl.exe' -ErrorAction SilentlyContinue
$current = [datetime]::ParseExact("$StartMonth-01", 'yyyy-MM-dd', $null)
$last = [datetime]::ParseExact("$EndMonth-01", 'yyyy-MM-dd', $null)

if ($last -lt $current) {
    throw 'EndMonth must not be earlier than StartMonth.'
}

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
        Write-Host "Downloading: $filename"
        if ($null -ne $curl) {
            & $curl.Source `
                --fail `
                --location `
                --retry 3 `
                --retry-delay 2 `
                --continue-at - `
                --output $partial `
                $url
            if ($LASTEXITCODE -ne 0) {
                throw "curl failed for $filename with exit code $LASTEXITCODE"
            }
        }
        else {
            Invoke-WebRequest -Uri $url -OutFile $partial
        }
        Move-Item -LiteralPath $partial -Destination $target
    }
    $current = $current.AddMonths(1)
}
