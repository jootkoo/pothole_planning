# A* + Dynamic Window Approach (DWA) Pothole Avoidance Simulator

A two-stage pothole-avoidance planner combining **global A*** with **local Dynamic Window Approach (DWA)** trajectory selection. Global A* produces a reference route on a default **0.25 m occupancy grid**. DWA then chooses speed and steering commands using a default **0.10 m obstacle-distance field**, a kinematic bicycle-model rollout, and a weighted trajectory score.

The implementation is a simulated local planner; it does not execute commands on a physical vehicle.

## 1. Features

- Shared road, obstacle, and vehicle-footprint geometry.
- Global A* reference route plus local DWA motion control.
- Speed and steering candidates constrained by acceleration and steering-rate limits.
- Two-second local trajectory rollouts at 0.25-second intervals.
- Distance-field clearance scoring and geometry checks near obstacles.
- Validation of the global reference, each executed motion segment, and the completed trajectory.
- Separate grid-build, search, and geometry-validation timing fields.
- JSON scenarios, optional plots, and headless runs.

## Project layout

```text
project_root/
├── common_optimized.py
├── planning_grid.py
├── geometry_validation.py
├── scenarios/
├── AStar/
├── RRT/rrt_pothole_simulator.py
├── DWA/dwa_pothole_simulator.py
└── Benchmark/
```

Run commands from the project root. Scenario identifiers are `1`, `2`, `3`, `4`, `5`, `6`, and `curved_polygon` in the supplied project archive. Both planners use the shared road geometry and vehicle-footprint collision model. The footprint is axis-aligned, not rotated with heading.

## 2. Running the planner

```bash
# All scenarios
python DWA/dwa_pothole_simulator.py

# One scenario without plots
python DWA/dwa_pothole_simulator.py --scenario 4 --no-show

# Allow more control steps
python DWA/dwa_pothole_simulator.py --scenario 6 --steps 800 --no-show

# Set global and local grid resolutions
python DWA/dwa_pothole_simulator.py --scenario curved_polygon --resolution 0.25 --field-resolution 0.10 --no-show
```

| Argument | Default | Purpose |
|---|---:|---|
| `--scenario` | all | Scenario ID |
| `--steps` | `550` | Maximum executed control steps |
| `--resolution` | `0.25` | Global A* occupancy-grid resolution (m) |
| `--field-resolution` | `0.10` | Local distance-field resolution (m) |
| `--no-show` | off | Disable displayed figures |

## 3. How A* + DWA works

1. **Build two grids.** `PlanningGrid` creates a global traversability grid and a finer local grid with obstacle-distance estimates.
2. **Plan globally.** `grid_astar()` computes an A* reference route from start to goal, using its default traffic weighting.
3. **Validate the reference.** Reject the reference if `validate_path()` finds a geometric violation.
4. **Initialize vehicle state.** Start at the scenario position with heading, speed, and steering angle set to zero.
5. **Select a lookahead target.** Find the closest reference waypoint at or beyond the previous progress index, then look ahead approximately 3.5 m along the reference.
6. **Sample reachable controls.** At each control step, evaluate **5 speeds × 11 steering angles** constrained by the current speed/steer and their rate limits.
7. **Roll out each control.** Simulate each command over the **2.0 s horizon**, checking estimated clearance along the way and consulting original road geometry when near occupied regions.
8. **Choose the minimum-score safe command.** Apply the first 0.25 s of that command, validate the executed segment against original geometry, and repeat.
9. **Finish or fail.** Success requires reaching within **0.7 m** of the goal and passing final executed-path validation. Failure can result from no global path, no safe local command, invalid executed motion, or exhausting the control-step limit.

## 4. DWA trajectory cost function

For each safe candidate command, the implementation minimizes:

\[
J = 2.5\,d_{target} + 0.65\,e_{heading} + 0.65\,(v_{max}-v) + rac{0.12}{c_{min}+0.15}
\]

| Term | Meaning | Effect |
|---|---|---|
| `d_target` | Euclidean distance from simulated rollout endpoint to local lookahead target | Favors progress toward reference path |
| `e_heading` | Absolute wrapped angle between simulated final heading and direction to target | Favors alignment with target |
| `v_max - v` | Difference between maximum allowed speed and candidate speed | Favors faster feasible motion |
| `c_min` | Minimum estimated clearance sampled along rollout | Penalizes proximity to obstacles |

The score is **minimized**, not maximized. The clearance denominator includes `0.15` to avoid division by zero. Candidates judged unsafe are **discarded before scoring**, rather than merely assigned a large penalty.

### Example (illustrative, not measured)

Suppose a candidate rollout ends 1.0 m from its target, with 0.2 rad heading error, speed 2.0 m/s, and minimum estimated clearance 0.5 m. With `v_max = 2.5 m/s`:

\[
J = 2.5(1.0)+0.65(0.2)+0.65(2.5-2.0)+rac{0.12}{0.5+0.15}pprox 3.14
\]

A competing safe command with a lower score is preferred. These are **implementation-specific weights**, not universally optimal DWA coefficients.

### Global A* cost function

The global reference planner uses the shared A* objective:

\[
f(n)=g(n)+h(n)
\]

In `planning_grid.py`, movement cost accumulates distance weighted by traffic proximity, and the heuristic estimates remaining Euclidean distance. The default `grid_astar()` traffic weight is **5.0**. This global A* cost is distinct from the **local DWA trajectory score** above.

## 5. Motion model and dynamic window

The vehicle state is `(x, y, heading, speed, steering_angle)`. The rollout uses a kinematic bicycle approximation:

\[
\dot{x}=v\cos	heta,\qquad \dot{y}=v\sin	heta,\qquad
\dot{	heta}=rac{v}{L}	an\delta
\]

Default model and sampling parameters:

| Parameter | Value |
|---|---:|
| Control interval `DT` | `0.25 s` |
| Rollout horizon | `2.0 s` |
| Wheelbase | `2.7 m` |
| Maximum speed | `2.5 m/s` |
| Maximum acceleration | `1.5 m/s²` |
| Maximum steering angle | `35°` |
| Maximum steering-rate magnitude | `90°/s` |
| Speed candidates | `5` |
| Steering candidates | `11` |
| Goal tolerance | `0.7 m` |
| Maximum control steps | `550` |

Reachable speeds are sampled from `[max(0, v−a_max DT), min(v_max, v+a_max DT)]`; steering candidates are similarly bounded by steering-rate and steering-angle limits. Commands are held constant during each simulated rollout.

## 6. Obstacle clearance and original-geometry validation

The local grid's distance field estimates distance to occupied grid cells; it is **not exact Euclidean clearance**. A rollout is rejected if its sampled distance is negative, or if it is near obstacles (below 0.6 m estimated clearance) and the original road's `free()` test fails. The executed 0.25 s segment is also passed to `validate_path()`, and the complete executed route is validated at goal arrival.

The global A* reference is separately validated before local execution begins. Collision validation is sampled, and the shared vehicle footprint is axis-aligned even though DWA simulates vehicle heading.

## 7. Performance and output metrics

| Metric | Meaning |
|---|---|
| `grid_build_ms` | Construction time for both global and local grids |
| `search_ms` | Global A* + local control search time, excluding measured geometry validation |
| `validation_ms` | Original-geometry validation time |
| `validation_checks` | Checks reported by the validator |
| `validated_candidates` | Complete reference/final-path candidates submitted for validation |
| Control steps | Number of executed control iterations |
| Path length / clearance / turning | Calculated by the shared benchmark for successful executed paths |

**Cold-start time** = grid build + search + validation. **Warm-start time** = search + validation, assuming both grids already exist. The implementation computes `time_to_valid_ms` for failed attempts too, so only call it *time to a valid path* if a validated executed path was returned.

## 8. Benchmarking recommendations

Compare **validated executed paths**, not just the A* reference. DWA's planning time includes local trajectory evaluation over multiple control steps, whereas A* and RRT output geometric global paths. This is a substantive difference in the task performed; present it explicitly when comparing times. Report success rate, failed-attempt durations, path length, clearance, and turning, and keep shared scenarios and validation rules consistent.

## 9. Limitations

- The local score weights are fixed and not automatically tuned.
- The distance field and collision checks are approximations.
- The vehicle model is simplified; collision footprint does not rotate with heading.
- A local minimum or blocked lookahead can prevent reaching the goal even when a global reference exists.
- DWA does not globally replan when local execution becomes stuck.
- The environment is static during a planning attempt.
- Global and local computations make timing comparisons with pure global planners non-equivalent unless explained.

## 10. Research context

This is the **hybrid global/local** baseline in the pothole-avoidance study. It tests whether a dynamically constrained local controller following an A* route offers useful trade-offs in route feasibility, clearance, smoothness, and computation relative to A* and RRT.
