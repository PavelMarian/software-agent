# `compchem-bench__qe-ecutwfc-convergence`

This directory is a complete Software Benchmark `TaskBundle` adapted from the
CompChemBench `qe-ecutwfc-convergence` task.

The agent sees only `task.json` and the seeded `workspace/`. The harness captures
the five raw Quantum ESPRESSO logs and `results.json` as environment-state
evidence. During evaluation it injects `evaluator/` as `.benchmark/`; neither the
reference energies nor the oracle implementation is exposed during execution.

The task intentionally uses the calibrated Quantum ESPRESSO 7.4 environment,
not the benchmark's generic Quantum ESPRESSO 7.5 image. Build it once from the
included reproducible context before running the bundle:

```text
docker build -t software-bench/quantum-espresso-7.4:compchem-bench-20260720 image
```

Then validate and run from the `bench` directory:

```text
software-bench validate --bundle data/tasks/quantum_espresso/compchem-bench__qe-ecutwfc-convergence --mas-ready
software-bench run --bundle data/tasks/quantum_espresso/compchem-bench__qe-ecutwfc-convergence --mode configs/modes/unrestricted_solo.toml --model-adapter MODEL_ADAPTER --output RUN --run-id qe-ecutwfc-1
software-bench evaluate --bundle data/tasks/quantum_espresso/compchem-bench__qe-ecutwfc-convergence --prediction RUN/prediction.json --output RUN/score.json
```

`image/` is build context only and is not copied into the agent workspace.
