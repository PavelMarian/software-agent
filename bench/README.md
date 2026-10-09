# Software Benchmark

Software Benchmark is a framework-agnostic prototype for measuring how teams of
LLM agents coordinate on tasks whose successful completion requires executing
and using complex target software.

The current slice combines:

- universal executable-task ingestion across scientific, engineering, data,
  desktop, and live service applications;
- automatic application MCP selection with reproducible environments and
  native-format validators;
- patch, artifact, workspace-file, and live-state submissions with native validators;
- benchmark-owned roles, permissions, communication, budgets, and traces;
- solo/MAS comparison modes and planner/verifier ablations;
- a narrow model-adapter boundary, so agent frameworks do not redefine the
  experiment.

Docker and local execution backends are available. OpenRouter and other
OpenAI-compatible model endpoints can drive the native agents. Independent-
ensemble selection and MCP variants remain outside this prototype slice.

## TaskBundle

Every instance is a directory with four contracts:

```text
task.json         public objective, target software, and workstreams
evaluation.json   submission type, hidden checks, parsers, and oracles
environment.json  reproducible workspace and execution requirements
provenance.json   private source metadata and reference provenance
```

The contract supports software-evolution tasks that submit a Git patch,
artifact-producing tasks that submit the recursive contents of `output/`, and
live tasks whose environment state is evaluated. Scientific solvers, CAD/GIS
tools, SRE environments, databases, and desktop applications can therefore use
domain-specific hidden evaluators. See
[`docs/task-bundle.md`](docs/task-bundle.md).

## Repository layout

```text
software_bench/  Importers, harness, evaluation, MCP, and runtime bridge
configs/         Role policies, modes, and aggregate budgets
docs/            Contracts and architecture decisions
environments/    Reproducible application environments
schemas/         JSON contracts for tasks, runs, traces, and results
tests/           Unit and integration tests plus fixtures
```

## Development

Requires Python 3.10 or newer and the parent `software-multiagent` package.

```bash
python -m pip install -e "..[dev]"
python -m pip install -e ".[dev]"
pytest
```

Validate the fixture bundle and a mode:

```bash
software-bench validate \
  --bundle tests/fixtures/task_bundle \
  --mas-ready \
  --mode configs/modes/full_mas.toml
```

Run the benchmark-owned protocol, then evaluate its normalized prediction:

```bash
software-bench run \
  --bundle tests/fixtures/task_bundle \
  --mode configs/modes/full_mas.toml \
  --model-adapter mock \
  --workspace tests/fixtures/task_bundle/workspace \
  --output runs/smoke \
  --run-id smoke

software-bench evaluate \
  --bundle tests/fixtures/task_bundle \
  --prediction runs/smoke/prediction.json \
  --workspace tests/fixtures/task_bundle/workspace \
  --output runs/smoke/result.json
```

The mock model and local evaluation backend exist only for contract and fixture
tests. They are not suitable for reported benchmark results.

## Comparison rule

The primary comparison is task quality under the same aggregate team budget:
tokens, wall-clock time, and tool calls. Agent count is reported as an execution
descriptor and never contributes to the score. Coordination metrics are also
reported separately; they explain how a team used its budget, but cannot
compensate for a worse task result.

The common import, run, submission, and evaluation flow for executable task
sources is documented in
[`docs/universal-executable-adapter.md`](docs/universal-executable-adapter.md).

Official scientific regression suites can be imported through a source-aware
recipe that explicitly separates public models from hidden numerical oracles:

```text
software-bench import-dataset \
  --source openmc-tests \
  --upstream /datasets/openmc \
  --recipe recipes/openmc.json \
  --output data/tasks/openmc
```

The supported source catalog covers the Quantum ESPRESSO test-suite, OpenMC
tests, SU2 TestCases, EnergyPlus testfiles, and MODFLOW 6 autotests. See
[`docs/scientific-imports.md`](docs/scientific-imports.md).

The remaining application profiles have source-aware importers for FoamBench,
QGIS processing test data, ParaView regression tests, FreeCAD Examples, GROMACS
regressiontests, LAMMPS examples, and the Blender Benchmark bundle:

```text
software-bench import-dataset \
  --source lammps-examples \
  --upstream /datasets/lammps \
  --recipe recipes/lammps.json \
  --output data/tasks/lammps
```

See [`docs/application-datasets.md`](docs/application-datasets.md) for the
catalog and recipe contract.
Reproducible images and hidden native-format validators for every built-in
application profile are described in
[`docs/application-environments.md`](docs/application-environments.md).

Before spending model budget, verify that every executable required by a task's
application profile exists in its selected environment:

```text
software-bench preflight --bundle TASK --mcp-profile auto --output preflight.json
```

Add a new application by writing its MCP specification directly. This path has
no dependency on the parent agent package and does not invoke automatic tool
generation. Validate the contract statically, then check its executables in the
actual application workspace:

```text
python -m software_bench.mcp validate-spec \
  --spec configs/my-application.mcp.json

python -m software_bench.mcp validate-spec \
  --spec configs/my-application.mcp.json \
  --workspace /path/to/application-workspace \
  --output runs/my-application-preflight.json
```

Start from [`templates/application-profile/application.mcp.json`](templates/application-profile/application.mcp.json)
and see [`docs/mcp-applications.md`](docs/mcp-applications.md) for the manual
external-profile and built-in-profile workflows.

## Native agent prototype

Each mode explicitly selects an `agent_topology`:

- `single_agent` runs one generalist role responsible for planning,
  implementation, and verification;
- `multi_agent` runs configured role phases over a shared workspace and message
  bus; the full mode includes a final executor repair phase after verification;
- `independent_ensemble` is declared but is not executable yet.

The single-agent topology uses the parent `software_multiagent` package through
a thin bridge in `software_bench/harness/agents/software_multiagent.py`. The benchmark still
owns its model adapter, tools, trace format, environment, and aggregate budget
ledger. The current sequential MAS path is the next topology to migrate to this
runtime before final compute-matched comparisons. Role instructions and
permissions live in `configs/roles`, while phase order and topology live in
`configs/modes`.

The internal package boundaries and the maintainability audit are documented in
[`docs/code-structure.md`](docs/code-structure.md).

## OpenRouter model adapter

Install the optional SDK and provide the API key through the environment:

```bash
python -m pip install -e ".[openrouter]"
export OPENROUTER_API_KEY="..."
```

Then select any OpenRouter model that supports tool calling:

```bash
software-bench run \
  --bundle tests/fixtures/task_bundle \
  --mode configs/modes/compute_matched_solo.toml \
  --model-adapter openrouter \
  --model anthropic/claude-sonnet-4 \
  --workspace /path/to/disposable/workspace \
  --output runs/openrouter-solo \
  --run-id openrouter-solo
```

For another OpenAI-compatible endpoint, use `--model-adapter
openai-compatible` together with `--base-url` and, when needed,
`--api-key-env`. Keys are never accepted as CLI arguments or written to run
artifacts.

Token usage is taken from each provider response. If usage is absent, or a
request succeeds only after a retry, the run is conservatively marked with
`measurement_complete=false`, because billed usage across all attempts cannot
be proven exact.

## Environment backends

Use `--no-docker` to override a TaskBundle that prefers Docker and execute both
the agent run and evaluation directly on the host:

```bash
software-bench run --no-docker ...
software-bench evaluate --no-docker ...
```

For bundle-backed tasks, the run receives a temporary copy of the public seed
workspace automatically. Git-backed tasks require an explicit disposable
workspace via `--workspace`. `--environment-backend local` and `--backend local`
are equivalent explicit forms. Host execution is not isolated: the host must
provide the software declared by the task, such as Python packages, LibreOffice,
database clients, or OpenFOAM.

Docker execution uses the image and workdir declared by the TaskBundle. Git
tasks reset an image repository to a base commit; bundle-backed tasks copy only
their public seed workspace. Docker mode does not require a host workspace:

```bash
software-bench run --environment-backend docker ...
software-bench evaluate --backend docker ...
```

The Docker CLI and a running daemon are required only for the Docker backend or
when an upstream runtime intrinsically depends on Docker (notably SREGym/Kind).
In Docker mode, the agent run and evaluation use separate containers, and
managed containers are removed after each operation. Tool allowlists and direct
file writes are enforced by the harness, but run-capable roles currently share
one mutable task container. Per-role read-only OS mounts are future work.

Git task images must provide Git. All task images must provide `/bin/sh` and
the POSIX `timeout` command; software-specific images may also declare a shell
initialization command such as the OpenFOAM environment setup.
