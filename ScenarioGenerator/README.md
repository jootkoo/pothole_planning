# Synthetic Scenario Generator

Generates **planner-compatible JSON scenarios** for the existing `common_optimized.py` loader, using five controlled dimensions:

1. **Road geometry:** straight, gentle curve, sharp curve, S-curve, variable width.
2. **Pothole geometry:** circular, elliptical, irregular convex, concave, elongated, clustered (polygon footprints).
3. **Obstacle placement:** random, lane-centered, alternating, clustered, narrow passage, blockage.
4. **Traffic conditions:** none, low, medium, high (stationary, locally road-aligned vehicles).
5. **Vehicle configuration:** compact, sedan, SUV, van; same environment paired with each vehicle.

## Install and run

Copy the `ScenarioGenerator/` folder to your existing project root, next to `common_optimized.py` and `scenarios/`. No external Python dependencies are needed for generation.

From the project root:

```bash
python -m ScenarioGenerator.generator --count 25 --seed 42 --output generated
python -m ScenarioGenerator.generator --count 1500 --seed 42 --output generated
```

Adjust ranges in `ScenarioGenerator/config.json`. The generator records a deterministic seed per scenario. Reusing the same count, seed and configuration reproduces identical outputs. The output directory is overwritten **for matching filenames**; other files are not deleted. Use a new output directory for each dataset version.

## Outputs

- `generated/scenarios/generated_000000.json`: road environment with a default sedan.
- `generated/experiments/generated_000000_compact.json`, etc.: full scenario JSON for each vehicle; **these are directly compatible with the existing `scenario()` loader**.
- `generated/metadata.csv`: road/pothole/placement/traffic family and generation seed.
- `generated/features.csv`: initial numeric ML features, one row per vehicle experiment.
- `generated/experiments.csv`: experiment-to-JSON mapping.
- `generated/manifest.json`: dataset seed, counts and settings.

## Using generated scenarios with the existing planners

Your current `common_optimized.py` loads scenarios only from the project root's `scenarios/` directory. To benchmark an experiment, copy its JSON file into `scenarios/` and reference its filename stem:

```bash
cp generated/experiments/generated_000000_sedan.json scenarios/
python Benchmark/benchmark.py --scenarios generated_000000_sedan --trials 5
```

Use `copy` instead of `cp` on Windows. For large batches, extend the benchmark to read `generated/experiments.csv` or set its scenario directory explicitly; this package does **not** modify your existing planner or benchmark code.

## Validation and modeling caveats

- Polygon validity and start/goal **point positions** inside the road are checked. This is **not** a vehicle-footprint feasibility check; a generated experiment may be blocked or invalid for its selected vehicle. Such failures should be retained in the benchmark.
- The validator checks polygon self-intersections and pothole vertices inside the road. It does not guarantee obstacle non-overlap or that a drivable path exists.
- Some intended placement patterns (including `blockage` and `narrow_passage`) are *sampling strategies*, not guarantees of a particular clearance or complete obstruction. Use post-generation geometric metrics if exact difficulty quotas are needed.
- Road boundaries are centerline offsets. Extreme parameter ranges can produce invalid boundaries; generation rejects these rather than repairing them silently.
- Your shared `Road` currently uses an **axis-aligned ego footprint**, even on curved roads. These scenarios are suitable for testing that approximation, not for claiming heading-aware vehicle safety.
- `features.csv` contains initial easy-to-compute features, not exact minimum passage width or curvature. It is **unlabeled**: run planners and assign winner labels only after benchmarking.
- Vehicle-specific experiments should stay grouped by base `scenario_id` when splitting train/validation/test data.
- The category combinations cycle deterministically; they are not a perfectly balanced full-factorial design. Check category coverage before model training.

## Quick smoke test

```bash
python -m ScenarioGenerator.generator --count 8 --seed 42 --output generated
python -m unittest ScenarioGenerator.tests -v
```
