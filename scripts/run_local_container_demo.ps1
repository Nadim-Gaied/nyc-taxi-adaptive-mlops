param(
    [ValidatePattern('^[a-z0-9][a-z0-9_-]*$')]
    [string]$ProjectName = 'nyc-taxi-demo',
    [switch]$Reset,
    [switch]$Down
)

$ErrorActionPreference = 'Stop'
$compose = @('compose', '--project-name', $ProjectName, '--profile', 'demo')

if ($Reset -or $Down) {
    & docker @compose down --volumes --remove-orphans
    if ($LASTEXITCODE -ne 0) {
        throw 'Docker Compose cleanup failed.'
    }
}

if ($Down) {
    return
}

$required = @(
    (
        'models/history/' +
            'fixed_periodic_28d__periodic__recent_365d__20251224T050000Z.txt'
    ),
    'data/processed/features_monthly/2025/features_2025-12.parquet'
)
foreach ($path in $required) {
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) {
        throw "Required demonstration input is missing: $path"
    }
}

& docker @compose build
if ($LASTEXITCODE -ne 0) {
    throw 'Docker image build failed.'
}

& docker @compose up --detach --wait mlflow
if ($LASTEXITCODE -ne 0) {
    throw 'MLflow did not become healthy.'
}

& docker @compose run --rm bootstrap
if ($LASTEXITCODE -ne 0) {
    throw 'Production bootstrap failed.'
}

& docker @compose run --rm --no-deps demo
if ($LASTEXITCODE -ne 0) {
    throw 'Containerized production cycles failed.'
}

Write-Output 'Container demonstration passed.'
Write-Output 'MLflow UI: http://localhost:5000'
Write-Output "Remove the isolated state with: .\scripts\run_local_container_demo.ps1 -ProjectName $ProjectName -Down"
