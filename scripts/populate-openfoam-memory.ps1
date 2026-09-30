param(
    [Parameter(Mandatory = $true)] [string] $UserGuideCatalog,
    [Parameter(Mandatory = $true)] [string] $SourceCatalog,
    [string] $MemoryDatabase = "data\openfoam-memory\openfoam-v10.sqlite",
    [string] $ReportDirectory = "data\openfoam-memory\reports",
    [string] $SoftwareId = "software_44748dff87126de54560b16c"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
Set-Location $projectRoot
$env:PYTHONPATH = Join-Path $projectRoot "src"

foreach ($catalog in @($UserGuideCatalog, $SourceCatalog)) {
    if (-not (Test-Path -LiteralPath (Join-Path $catalog "manifest.json") -PathType Leaf)) {
        throw "Catalog does not contain manifest.json: $catalog"
    }
}

$databaseParent = Split-Path -Parent $MemoryDatabase
if ($databaseParent) { New-Item -ItemType Directory -Path $databaseParent -Force | Out-Null }
New-Item -ItemType Directory -Path $ReportDirectory -Force | Out-Null

function Invoke-CheckedPython {
    param([Parameter(ValueFromRemainingArguments = $true)] [string[]] $PythonArguments)
    & python @PythonArguments
    if ($LASTEXITCODE -ne 0) { throw "Python command failed with exit code $LASTEXITCODE" }
}

$identity = @(
    "--product", "OpenFOAM",
    "--vendor", "OpenFOAM Foundation",
    "--edition", "Foundation",
    "--distribution", "Foundation",
    "--platform", "Linux",
    "--evidence-only"
)

$userGuideArguments = @(
    "-m", "software_multiagent.software_memory.adapters.api_doc_knowledge",
    $UserGuideCatalog, $MemoryDatabase
) + $identity + @("--report", (Join-Path $ReportDirectory "user-guide-import.json"))
Invoke-CheckedPython -PythonArguments $userGuideArguments

$sourceArguments = @(
    "-m", "software_multiagent.software_memory.adapters.api_doc_knowledge",
    $SourceCatalog, $MemoryDatabase
) + $identity + @("--report", (Join-Path $ReportDirectory "source-import.json"))
Invoke-CheckedPython -PythonArguments $sourceArguments

$compileArguments = @(
    "-m", "software_multiagent.software_memory.extraction",
    $MemoryDatabase, $SoftwareId, $ReportDirectory
)
Invoke-CheckedPython -PythonArguments $compileArguments

Write-Output "OpenFOAM memory populated: $MemoryDatabase"
Write-Output "Compilation and review reports: $ReportDirectory"
