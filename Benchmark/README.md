# Path-Planning Benchmark — Generated Scenarios

This benchmark is intended to compare **A\***, **RRT**, and **A\*-guided DWA** on the **synthetic scenarios produced by the scenario generator**, not just the six manually defined example scenarios.

> **Version requirement:** The original `Benchmark/benchmark.py` accepts only scenarios 1–6 and **does not support `--scenario-dir`**. The commands below require the generated-scenario-enabled benchmark (`benchmark_updated.py`) installed as `Benchmark/benchmark.py`, **and** a compatible `common.py` whose `scenario(...)` function accepts `scenario_dir`. Merely changing this README does not enable generated scenarios. Check `python Benchmark/benchmark.py --help` before running.

## Folder layout

```text
project-root/
├── Benchmark/
│   ├── benchmark.py
│   ├── metrics.py
│   └── README.md
├── AStar/
│   └── astar.py
├── RRT/
│   └── rrt.py
├── DWA/
│   └── dwa.py
├── common.py
└── generated/
    ├── scenarios/
    │   ├── generated_000000.json
    │   └── ...
    └── experiments/
        ├── generated_000000_compact.json
        └── ...
```

Planner filenames in this layout correspond to the generated-scenario-enabled benchmark's `MODULE_PATHS`; change those entries if your project uses different filenames.

## 1. Verify the benchmark supports generated scenarios

Run from the **project root**:

```bash
python Benchmark/benchmark.py --help
```

Verify that `--scenario-dir` appears. If it does not, you are running the old benchmark and must update the script first.

## 2. Run all three algorithms on all generated experiments (recommended)

```bash
python Benchmark/benchmark.py \
  --scenario-dir generated/experiments \
  --trials 10 \
  --repeat-deterministic \
  --output Benchmark/results_generated
```

The benchmark automatically discovers every `*.json` file directly in `generated/experiments/`, and runs **A\***, **RRT**, and **DWA** on each experiment. This is the recommended input when each experiment specifies a particular scenario and vehicle configuration.

- `--trials 10`: 10 RRT trials per experiment with different seeds.
- `--repeat-deterministic`: also run A* and DWA 10 times per experiment (otherwise they run once each).
- `--output`: where CSV summaries and graphs are saved.

**Example run count:** 1,000 road scenarios × 4 vehicle configurations × 3 algorithms × 10 trials = **120,000 planner runs**, assuming the generator produced 4,000 experiment JSON files and the benchmark loads each vehicle configuration correctly.

## 3. Run all three algorithms on generated road scenarios only

```bash
python Benchmark/benchmark.py \
  --scenario-dir generated/scenarios \
  --trials 10 \
  --repeat-deterministic \
  --output Benchmark/results_roads
```

Use this option for road-scenario JSON files. To benchmark all vehicle variants, use `generated/experiments` instead.

## 4. Quick test before a large benchmark

Test one generated experiment:

```bash
python Benchmark/benchmark.py \
  --scenario-dir generated/experiments \
  --scenarios generated_000000_compact \
  --trials 2 \
  --repeat-deterministic \
  --output Benchmark/results_smoke
```

Test a selected set:

```bash
python Benchmark/benchmark.py \
  --scenario-dir generated/experiments \
  --scenarios generated_000000_compact generated_000001_compact \
  --trials 5
```

`--scenarios` accepts **filename stems**, not the old numeric IDs. Omit it to run all JSON files in the chosen directory.

## 5. Optional planning parameters

```bash
python Benchmark/benchmark.py \
  --scenario-dir generated/experiments \
  --trials 10 \
  --rrt-iterations 20000 \
  --dwa-steps 800 \
  --resolution 0.25 \
  --field-resolution 0.10 \
  --seed 42
```

- `--rrt-iterations`: maximum RRT sampling iterations.
- `--dwa-steps`: maximum DWA control steps.
- `--resolution`: planning-grid resolution (meters).
- `--field-resolution`: DWA field resolution (meters).
- `--seed`: base random seed for reproducibility.

## 6. Outputs and interpretation

The generated-scenario-enabled benchmark writes:

```text
Benchmark/results_generated/
├── raw_results.csv
├── summary_results.csv
└── graphs/
    └── ... comparison plots ...
```

`raw_results.csv` contains one row per planner run, including scenario/experiment, algorithm, trial, success, search and validation times, and path metrics. `summary_results.csv` aggregates performance. Graphs compare the algorithms; large generated sets are better interpreted using aggregate plots than one chart per scenario.

**Primary comparison metrics:** success rate, search time, time to valid path, path length, minimum clearance, and turning. The DWA implementation is A*-guided, so its timing can include generating an A* reference path. Compare consistent timing definitions across planners.

## Troubleshooting

- **`unrecognized arguments: --scenario-dir`** — You are running the original six-scenario benchmark. Replace/update `Benchmark/benchmark.py` with the generated-scenario-enabled version.
- **`scenario() got an unexpected keyword argument 'scenario_dir'`** — Your `common.py` does not support loading scenario JSON files from a directory. Update the loader before running.
- **`No scenario JSON files found`** — Verify the directory exists and contains JSON files directly inside it.
- **Planner file not found** — Check `MODULE_PATHS` in `benchmark.py` against actual planner filenames.
- **All experiments fail** — Confirm the scenario loader understands the generated experiment JSON schema, including vehicle configuration. Inspect the `error` and `failure_reason` fields in `raw_results.csv`.

**Recommended command (after installing compatible benchmark and loader):**

```bash
python Benchmark/benchmark.py --scenario-dir generated/experiments --trials 10 --repeat-deterministic --output Benchmark/results_generated
```
