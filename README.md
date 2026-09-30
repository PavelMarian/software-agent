# Experimental FoamBench run

This guide configures an experimental multi-agent run from a clean checkout.
The project is standalone and uses only its domain-neutral core plus the
sibling `foambench` adapter package.

## 1. Prerequisites

- Python 3.10 or newer;
- Docker Engine for comparable Linux/OpenFOAM runs;
- an OpenFOAM 10 image with the solvers and MPI required by the selected tasks;
- a local copy of the original FoamBench Basic and/or Advanced JSON dataset
  (download the two files from [FoamBench on Kaggle](https://www.kaggle.com/datasets/nithinsekhar/foambench));
- credentials for the selected model provider.

Clone the official repository separately for its authoritative evaluator scripts:

```powershell
git clone https://github.com/NLR-Theseus/cfdllmbench.git third_party\cfdllmbench
```

Do not place the corpus, provider credentials, `GT_Files`, or run outputs in
Git. `.gitignore` already excludes the normal locations.

## 2. Install the project

From the repository root:

```powershell
cd software-multiagent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[foambench-evaluator,providers,embeddings,dev]"
```

Verify the standalone boundary and tests:

```powershell
python -m pytest -q
python -c "import software_multiagent; import foambench"
```

Both imports must complete successfully.

## One-command setup

Download `FoamBench_basic.json` and `FoamBench_advanced.json` from the
[upstream Kaggle dataset](https://www.kaggle.com/datasets/nithinsekhar/foambench),
then set `$DatasetDir` to the directory where you saved them and create the
ignored local configuration once:

```powershell
Copy-Item .env.example .env
# Edit .env: select SOFTWARE_MULTIAGENT_PROVIDER/SOFTWARE_MULTIAGENT_MODEL
# and fill OPENROUTER_API_KEY or OPENAI_API_KEY.
$DatasetDir = "<path-to-dataset-directory>"
```

The following one command then performs the local setup. Provider and model are
read from `.env`; `-Provider` and `-Model` remain optional overrides:

```powershell
.\scripts\setup-foambench.ps1 `
  -BasicJson "$DatasetDir\FoamBench_basic.json" `
  -AdvancedJson "$DatasetDir\FoamBench_advanced.json"
```

It clones the official repository into `third_party/`, installs the project
extras, imports both splits, builds the OpenFOAM/MPI image, and writes the
ignored `experiment.local.json`. It does not put secrets in that file.

## 3. Import a corpus

Obtain the upstream JSON from [FoamBench on Kaggle](https://www.kaggle.com/datasets/nithinsekhar/foambench)
outside this repository, then import it locally. Reuse `$DatasetDir` from the
previous section (or set it here):

```powershell
foambench import-foambench `
  --dataset "$DatasetDir\FoamBench_basic.json" `
  --corpus data\corpus `
  --split basic

foambench import-foambench `
  --dataset "$DatasetDir\FoamBench_advanced.json" `
  --corpus data\corpus `
  --split advanced
```

Inspect one public task before using a provider:

```powershell
foambench show-task --corpus data\corpus --task foambench__basic__LidDrivenCavity-1
```

The corpus keeps its agent-visible `seed/` directory separate from
`private/reference/`. A model driver receives neither private paths nor their
contents.

## 4. Configure the built-in multi-agent driver

All local secrets and model selection live in the ignored `.env` file:

```dotenv
SOFTWARE_MULTIAGENT_PROVIDER=openrouter
SOFTWARE_MULTIAGENT_MODEL=<provider-model-id>
OPENROUTER_API_KEY=...
OPENAI_API_KEY=

SOFTWARE_MEMORY_EMBEDDING_PROVIDER=openrouter
SOFTWARE_MEMORY_EMBEDDING_MODEL=<embedding-model-id>
SOFTWARE_MEMORY_EMBEDDING_BASE_URL=https://openrouter.ai/api/v1
SOFTWARE_MEMORY_EMBEDDING_API_KEY=...

# Enable canonical shared memory after populating it. Leave DATABASE empty for
# a memory-free run.
SOFTWARE_MEMORY_DATABASE=data/openfoam-memory/openfoam-v10.sqlite
SOFTWARE_MEMORY_VECTOR_DATABASE=data/openfoam-memory/openfoam-v10.lance
SOFTWARE_MEMORY_SOFTWARE_ID=software_44748dff87126de54560b16c
SOFTWARE_MEMORY_SOFTWARE_VERSION=10
```

For an OpenAI run, set `SOFTWARE_MULTIAGENT_PROVIDER=openai`, select a model
available from that provider, and fill `OPENAI_API_KEY`. The embedding provider
and embedding model can be selected independently. The included provider
boundary creates LangChain's `ChatOpenAI` integration for either endpoint.

The CLI loads `.env` automatically. Explicit `--provider` and `--model` flags
override its defaults. `run-task` and `run-split` use the built-in four-role driver by default:
Planner → Researcher → Executor → Evaluator. LangGraph owns the state and role
transitions; LangChain owns messages, model binding, and structured tool calls.
Each role uses a bounded ReAct loop with one native action per observation. The
run manifest records short decision/observation summaries in `reasoning_trace`,
not private chain-of-thought.
The graph exposes only neutral tools: `read_file`, `list_files`, `write_file`,
`run_program`, and `verify_workspace`. Failed deterministic verification follows
the bounded LangGraph repair edge back to Executor.

The default `foambench.drivers:run` module is only an adapter. It converts the
public benchmark task and workspace into a neutral `SoftwareRunRequest` and
delegates to `SoftwareMultiAgent`, which is the single place where the model,
four-role team, and optional shared memory are constructed.

`--driver package.module:function` remains optional for experimental custom
workflows. Provider credentials remain environment variables and are never
written to `run.json` or `experiment.local.json`.

## OpenFOAM memory: collect, populate, and index

The repository contains the product-neutral `acquisition.corpus` collector.
URLs, product labels, versions, repository refs, roots, and optional file
filters are inputs rather than hard-coded OpenFOAM settings. The following
profile collects the OpenFOAM v10 documentation and every textual/source file
in the official `OpenFOAM/OpenFOAM-10` repository. These steps require
network access and may take a long time.

First collect the two stable JSON/JSONL catalogs:

```powershell
$env:PYTHONPATH = (Resolve-Path src)
python -m software_multiagent.software_memory.acquisition.corpus `
  collect-web data\openfoam-catalog\user-guide `
  --url https://doc.cfd.direct/openfoam/user-guide-v10/contents `
  --software "OpenFOAM" `
  --version 10

python -m software_multiagent.software_memory.acquisition.corpus `
  collect-github data\openfoam-catalog\source `
  --url https://github.com/OpenFOAM/OpenFOAM-10 `
  --software "OpenFOAM" `
  --version 10
```

Each generated catalog contains `manifest.json`, `repository_snapshot.json`,
`sources.jsonl`, `sections.jsonl`, and `operations.jsonl`. Keep the manifest:
it records the selected version/commit and whether collection was truncated.
The neutral collector uses all repository roots and accepts all files that pass
its UTF-8 text check, including extensionless files. Its normal safety ceilings
are 10,000 web pages, 100,000 repository files, a 500 MB archive, and 1 GB of
uncompressed text. Reaching a ceiling, failing a documentation URL, or producing
`truncated=true` makes the command fail: a partial catalog is never accepted as
a successful full collection. Raise the relevant neutral CLI limit and rerun if
the source legitimately exceeds one of these ceilings.

Import both catalogs into canonical SQLite memory and compile the extracted
entities and operation contracts:

```powershell
.\scripts\populate-openfoam-memory.ps1 `
  -UserGuideCatalog data\openfoam-catalog\user-guide\OpenFOAM\10 `
  -SourceCatalog data\openfoam-catalog\source\OpenFOAM\10 `
  -MemoryDatabase data\openfoam-memory\openfoam-v10.sqlite `
  -ReportDirectory data\openfoam-memory\reports
```

The importer is idempotent for unchanged catalogs. Do not delete or overwrite a
previous database when updating sources; use a new database path if you need an
independent experiment. Inspect `compilation-report.json` and
`human-review-queue.json` before indexing. The review queue contains ambiguous
or conflicting extractions that should not silently become trusted knowledge.

Finally set the embedding provider, model, and credentials in `.env`, then
build the derived semantic index:

```powershell
.\scripts\build-openfoam-vector-index.ps1 `
  -MemoryDatabase data\openfoam-memory\openfoam-v10.sqlite `
  -VectorDatabase data\openfoam-memory\openfoam-v10.lance
```

The command prints indexed and skipped item counts. Put that same directory in
`SOFTWARE_MEMORY_VECTOR_DATABASE`. Re-index into a new empty `.lance` path after
changing the embedding provider, model, dimensions, or the canonical SQLite
corpus. The vector index is disposable derived data; SQLite is the source of
truth, and neither artifact contains the API key.

At runtime retrieval is deliberately SQLite-first. Exact, full-text, and graph
results from SQLite are ranked and packed first. Vector search may only append
canonical SQLite records that those methods did not already find, using the
remaining item and token budget. A vector duplicate neither replaces an SQLite
result nor changes its score. If the vector path is empty, retrieval remains
SQLite-only; if it is set but the matching index is absent, the run fails
instead of silently dropping semantic retrieval.

After population, copy the SQLite path, vector directory, printed software ID,
and version into the four `SOFTWARE_MEMORY_*` variables shown above. Every role
then reads a bounded role-specific view from the same store. Writes are active
as well:
Planner records compact plan/gap handoffs, Researcher stores attributed raw
evidence as an unpromoted candidate, Executor stores progress and produced
artifacts, and independently verified operations can be recorded as validator
observations. A configured but missing database or software ID is a hard error;
the runner never silently substitutes empty memory.

Before a benchmark preflight, validate the complete model path with disposable
synthetic tasks:

```powershell
python scripts\run-synthetic-smoke.py --output runs\smoke-hello
python scripts\run-memory-repair-smoke.py --output runs\smoke-mechanisms
```

The first command checks the production composition root, LangGraph role flow,
LangChain tool binding, file mutation, and deterministic verification. The
second checks role-conditioned reads, Planner/Researcher/Executor writes,
SQLite evidence persistence, a forced verifier repair cycle, and recovery after
an observed tool error. Both use the real provider/model selected in `.env` and
return a nonzero exit code if an asserted mechanism is absent.

## 5. Run a preflight task

Use a disposable output directory. Start with one Basic task and inspect its
public artifacts before any large run:

```powershell
foambench run-task `
  --corpus data\corpus `
  --task foambench__basic__LidDrivenCavity-1 `
  --provider openrouter `
  --model $env:SOFTWARE_MULTIAGENT_MODEL `
  --image software-multiagent-foambench:openfoam10 `
  --seed 17 `
  --output runs\preflight\LidDrivenCavity-1
```

Review `run.json`, `prediction.json`, and `workspace/output/`. A completed
driver run is not a scientific pass: `public_verification.passed` must also be
true before evaluator scoring.

## 6. Run a split and resume it

```powershell
foambench run-split `
  --corpus data\corpus `
  --split basic `
  --provider openrouter `
  --model $env:SOFTWARE_MULTIAGENT_MODEL `
  --image software-multiagent-foambench:openfoam10 `
  --seed 17 `
  --output runs\basic-17

foambench run-split `
  --corpus data\corpus `
  --split basic `
  --provider openrouter `
  --model $env:SOFTWARE_MULTIAGENT_MODEL `
  --image software-multiagent-foambench:openfoam10 `
  --seed 17 `
  --output runs\basic-17 `
  --resume
```

`summary.json` contains a stable per-task status list. Keep model ID, seed,
image digest, corpus hash, driver revision, and resource limits with each
experiment before comparing results.

## 7. Evaluation boundary

Run an upstream-compatible evaluator only after agents have stopped. It must
receive a reconstructed submission from `prediction.json` and private reference
assets in a fresh evaluation container. Do not mount `private/reference/` into
an agent container or workspace. Compare its scores against upstream on fixed
golden submissions before reporting any result.

The project can invoke the four official scripts in their required order against
that checkout after submission files have been staged into its `Dataset/` layout:

```powershell
foambench evaluate-upstream --foambench-root third_party\cfdllmbench\FoamBench
```

First stage each completed run into the official layout (the folder name is the
algorithm identity consumed by the upstream reports):

```powershell
foambench stage-upstream --run runs\preflight\LidDrivenCavity-1 --foambench-root third_party\cfdllmbench\FoamBench
foambench evaluate-upstream --foambench-root third_party\cfdllmbench\FoamBench
```

Build the published-compatible execution image:

```powershell
docker build -f docker\Dockerfile.foambench -t software-multiagent-foambench:openfoam10 .
docker run --rm --network none software-multiagent-foambench:openfoam10 bash -lc "source /opt/openfoam10/etc/bashrc && mpirun --version && blockMesh -help"
```

## Current experimental limits

The runner, built-in provider driver, corpus boundary, per-command Docker
execution, upstream staging, and official report invocation are implemented.
Before claiming a full official FoamBench result, pin the Docker image digest
and validate the official reports on fixed golden submissions, especially for
the Advanced split.
