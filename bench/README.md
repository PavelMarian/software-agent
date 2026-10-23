# Software Benchmark

Software Benchmark is a framework-agnostic harness for evaluating how LLM
agents use complex software to complete executable tasks. It supports solo and
multi-agent runs under shared budgets, reproducible execution environments,
application-specific tools, and evaluation against hidden native-format
oracles.

## Overview

The benchmark owns the experimental protocol: task ingestion, agent topology,
roles and permissions, aggregate budgets, tool access, traces, submission
collection, and scoring. Model providers and agent frameworks connect through
narrow adapters, so changing either does not redefine the task or evaluator.

```text
Source data
    ↓
TaskBundle
    ↓
Benchmark run (model + mode + shared budget)
    ↓
Software environment and MCP tools
    ↓
Submission (patch, artifacts, or live state)
    ↓
Independent evaluation
    ↓
Scores, evidence, and traces
```

The primary comparison is result quality under the same aggregate limits for
tokens, wall time, and tool calls. Agent count and coordination metrics are
reported as experimental descriptors; they do not add to task score.

The main package boundaries are:

```text
software_bench/  Import, orchestration, execution, evaluation, and MCP runtime
data/tasks/       Integrated TaskBundles, grouped by profile and instance
configs/          Modes, roles, permissions, and budgets
environments/     Reproducible application environments
criteria/         Task-selection and structural-review contracts
schemas/          JSON contracts for tasks, runs, traces, and results
docs/             Detailed contracts and architecture decisions
tests/            Unit and integration fixtures
```

Install the benchmark for development and validate a bundle before running it:

```bash
python -m pip install -e "..[dev]"
python -m pip install -e ".[dev]"

software-bench validate \
  --bundle data/tasks/PROFILE/INSTANCE \
  --mode configs/modes/full_mas.toml
```

A run produces a normalized prediction; evaluation applies that prediction in
a fresh environment and emits structured evidence and a score:

```bash
software-bench run \
  --bundle data/tasks/PROFILE/INSTANCE \
  --mode configs/modes/compute_matched_solo.toml \
  --model-adapter openrouter \
  --model MODEL \
  --output runs/example \
  --run-id example

software-bench evaluate \
  --bundle data/tasks/PROFILE/INSTANCE \
  --prediction runs/example/prediction.json \
  --output runs/example/result.json
```

See [the architecture notes](docs/architecture.md) for runtime boundaries and
[the code-structure guide](docs/code-structure.md) for package ownership.

## Data and TaskBundles

`TaskBundle` is the benchmark's dataset boundary. Each instance lives at
`data/tasks/<profile>/<instance_id>/` and contains four contracts:

```text
task.json         Public objective and target software
evaluation.json   Submission type, hidden checks, parsers, and oracles
environment.json  Workspace, backend, image, network, and runtime requirements
provenance.json   Private source identity and reference provenance
```

Bundle-backed tasks may also contain `workspace/`, which is copied into the
agent's disposable environment, and `evaluator/`, which is withheld until
evaluation and mounted as `.benchmark/`. Agents never receive private
provenance or hidden evaluator assets.

The same contract covers three submission forms:

- Git patches for software-evolution tasks;
- recursive artifact bundles, conventionally collected from `output/`;
- selected environment state for live services and mutable systems.

Importers translate upstream records into this common layout. Dataset storage
formats therefore do not leak into the agent runtime or scoring interface.
Static bundle validation confirms the contract; scored admission additionally
requires a runnable environment and an oracle that checks correctness rather
than mere completion. Task selection and structural complexity remain separate
from domain difficulty and runtime burden.

See [the TaskBundle contract](docs/task-bundle.md), [the executable-task import
flow](docs/universal-executable-adapter.md), and [dataset importer
documentation](docs/application-datasets.md).

## Software Environments and MCP

Each TaskBundle selects a local, Docker, or installed execution backend. The
environment defines the workspace, image, working directory, network policy,
timeouts, allowed credentials, and any managed external service. Run and
evaluation use separate disposable environments so hidden checks do not alter
the agent's workspace or depend on its process state.

Application access is exposed through compact MCP profiles. A profile provides
workflow-level operations such as inspecting state, running a solver, editing a
document, applying a database change, rendering an artifact, or validating a
result. The harness selects the profile from public TaskBundle metadata and
runs it inside the same environment and permission boundary as the task.
Mutating tools must be enabled explicitly; role policies determine which agents
can use environment tools.

Before spending model budget, check that the selected environment contains the
executables required by its MCP profile:

```bash
software-bench preflight \
  --bundle data/tasks/PROFILE/INSTANCE \
  --mcp-profile auto \
  --output runs/preflight.json
```

New applications can start as external declarative MCP specifications and move
into the built-in catalog after adding a pinned environment, native validator,
dataset ingestion path, and integration tests. This keeps software-specific
commands and validation at the benchmark boundary without adding domain
branches to the agent loop.

See [the MCP guide](docs/mcp-applications.md), [environment and validator
documentation](docs/application-environments.md), and [the MCP application
schema](schemas/mcp-application.schema.json).
