param(
    [ValidateSet("openai", "openrouter")] [string] $Provider,
    [string] $Model,
    [Parameter(Mandatory = $true)] [string] $BasicJson,
    [Parameter(Mandatory = $true)] [string] $AdvancedJson,
    [string] $UpstreamRoot = "third_party\cfdllmbench",
    [string] $CorpusRoot = "data\corpus",
    [string] $Image = "software-multiagent-foambench:openfoam10",
    [string] $EnvironmentFile = ".env"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot

if (Test-Path -LiteralPath $EnvironmentFile -PathType Leaf) {
    Get-Content -LiteralPath $EnvironmentFile | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#")) {
            $pair = $line.Split("=", 2)
            if ($pair.Count -eq 2 -and $pair[0] -and -not (Test-Path "Env:$($pair[0].Trim())")) {
                Set-Item -Path "Env:$($pair[0].Trim())" -Value $pair[1].Trim()
            }
        }
    }
}
$Provider = if ($Provider) { $Provider } else { $env:SOFTWARE_MULTIAGENT_PROVIDER }
$Model = if ($Model) { $Model } else { $env:SOFTWARE_MULTIAGENT_MODEL }
if ($Provider -notin @("openai", "openrouter")) { throw "Set SOFTWARE_MULTIAGENT_PROVIDER to openai or openrouter." }
if (-not $Model) { throw "Set SOFTWARE_MULTIAGENT_MODEL in $EnvironmentFile or pass -Model." }

$keyName = if ($Provider -eq "openrouter") { "OPENROUTER_API_KEY" } else { "OPENAI_API_KEY" }
if (-not [Environment]::GetEnvironmentVariable($keyName)) {
    throw "Set $keyName in $EnvironmentFile or in the current environment before running setup."
}
foreach ($file in @($BasicJson, $AdvancedJson)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Dataset JSON is missing: $file"
    }
}
if (-not (Get-Command git -ErrorAction SilentlyContinue)) { throw "git is required." }
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) { throw "Docker is required." }

$upstream = Join-Path $projectRoot $UpstreamRoot
if (-not (Test-Path -LiteralPath $upstream)) {
    git clone https://github.com/NLR-Theseus/cfdllmbench.git $upstream
}
$foamBench = Join-Path $upstream "FoamBench"
if (-not (Test-Path -LiteralPath (Join-Path $foamBench "score_calculation.py") -PathType Leaf)) {
    throw "The checkout at $upstream does not contain FoamBench evaluation scripts."
}

python -m pip install -e ".[foambench-evaluator,providers,embeddings,dev]"
python -m foambench.cli import-foambench --dataset $BasicJson --corpus $CorpusRoot --split basic
python -m foambench.cli import-foambench --dataset $AdvancedJson --corpus $CorpusRoot --split advanced
docker build -f docker\Dockerfile.foambench -t $Image .

$configuration = @{
    provider = $Provider
    model = $Model
    corpus = (Resolve-Path $CorpusRoot).Path
    upstream_foambench = (Resolve-Path $foamBench).Path
    image = $Image
    created_at = (Get-Date).ToUniversalTime().ToString("o")
}
$configuration | ConvertTo-Json | Set-Content -LiteralPath "experiment.local.json" -Encoding utf8
Write-Output "Setup complete. Local configuration: $projectRoot\experiment.local.json"
