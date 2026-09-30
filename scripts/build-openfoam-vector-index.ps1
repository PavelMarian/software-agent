param(
    [string] $EnvironmentFile = ".env",
    [string] $MemoryDatabase = "data\openfoam-memory\openfoam-v10.sqlite",
    [string] $VectorDatabase = "data\openfoam-memory\openfoam-v10.lance",
    [string] $SoftwareId = "software_44748dff87126de54560b16c"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = (Join-Path $projectRoot "src")
if (-not (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf)) {
    throw "Missing $EnvironmentFile. Copy .env.example to .env and set SOFTWARE_MEMORY_EMBEDDING_API_KEY."
}
Get-Content -LiteralPath $EnvironmentFile | ForEach-Object {
    $line = $_.Trim()
    if ($line -and -not $line.StartsWith("#")) {
        $pair = $line.Split("=", 2)
        if ($pair.Count -eq 2 -and $pair[0]) { Set-Item -Path "Env:$($pair[0].Trim())" -Value $pair[1].Trim() }
    }
}
if (-not $env:SOFTWARE_MEMORY_EMBEDDING_API_KEY) { throw "SOFTWARE_MEMORY_EMBEDDING_API_KEY is empty." }
if (-not (Test-Path -LiteralPath $MemoryDatabase -PathType Leaf)) { throw "Missing memory database: $MemoryDatabase" }
python -m software_multiagent.software_memory.embeddings $MemoryDatabase $VectorDatabase $SoftwareId --model $env:SOFTWARE_MEMORY_EMBEDDING_MODEL --base-url $env:SOFTWARE_MEMORY_EMBEDDING_BASE_URL --provider $env:SOFTWARE_MEMORY_EMBEDDING_PROVIDER
