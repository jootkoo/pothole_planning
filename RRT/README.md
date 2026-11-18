# RRT Pothole Avoidance Simulator

A two-dimensional **Rapidly-exploring Random Tree (RRT)** planner for finding routes around potholes and stationary traffic on a three-lane road. The implementation samples continuous coordinates, checks edges against a shared **0.25 m occupancy grid** by default, and validates candidate complete paths against the original road geometry.

This is **standard RRT**, not RRT*: it seeks a feasible route, not a provably shortest or minimum-cost route.

## 1. Features

- Continuous random sampling with configurable goal bias and extension length.
- Precomputed `PlanningGrid` for fast collision checks along tree edges.
- Original-geometry validation before accepting a completed path.
- Continues exploring after a candidate fails geometry validation.
- Configurable random seed for reproducible runs.
- Separate grid-build, search, and validation timing fields.
- Supports JSON scenarios, including irregular polygon potholes and curved road boundaries.

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
python RRT/rrt_pothole_simulator.py

# One scenario, without displaying plots
python RRT/rrt_pothole_simulator.py --scenario 4 --no-show

# Reproducible sampling and a larger search budget
python RRT/rrt_pothole_simulator.py --scenario 6 --seed 17 --iterations 12000 --no-show

# Change occupancy-grid resolution
python RRT/rrt_pothole_simulator.py --scenario curved_polygon --resolution 0.25 --no-show
```

| Argument | Default | Purpose |
|---|---:|---|
| `--scenario` | all | Scenario ID |
| `--seed` | `42` | Random generator seed |
| `--iterations` | `12000` | Maximum tree-growth iterations |
| `--resolution` | `0.25` | Occupancy-grid cell width (m) |
| `--no-show` | off | Disable displayed figures |

## 3. How RRT works

1. **Build the grid.** `PlanningGrid` precomputes traversability using the full axis-aligned vehicle footprint and safety margin.
2. **Validate endpoints.** The start and goal must be free in both the grid and original road geometry.
3. **Choose a sample.** With probability `goal_bias` the target is the goal; otherwise it is uniformly sampled from the road bounding box.
4. **Find the nearest node.** The current implementation performs a linear Euclidean-distance scan through existing tree nodes.
5. **Extend the tree.** Move from the nearest node toward the target by at most `step_size` meters.
6. **Check the edge.** `grid_segment_free()` checks the cells crossed by the proposed segment; blocked edges are discarded.
7. **Try the goal.** If the new node is within one step of the goal and the connecting edge is grid-free, reconstruct a complete candidate route.
8. **Validate the original geometry.** `validate_path(road, path)` checks the complete candidate. If invalid, increment the rejection counter and keep searching. Return the first validated route or fail after the iteration budget.

## 4. RRT sampling and selection rules (the algorithm's objective)

Unlike A* or DWA, this implementation **does not calculate a weighted path cost** to select among alternative routes. Its rules are geometric:

**Goal-biased sampling:**

```text
sample = goal                         with probability p_goal
sample = uniform(road bounding box)  otherwise
```

**Nearest-node selection:**

$$
q_{\mathrm{nearest}} = \operatorname*{arg\,min}_{q \in \mathrm{Tree}} \left\|q - q_{\mathrm{sample}}\right\|_2
$$

**Tree extension:**

$$
q_{\mathrm{new}} = q_{\mathrm{nearest}} + \min\left(1, \frac{\Delta}{\left\|q_{\mathrm{sample}} - q_{\mathrm{nearest}}\right\|_2}\right)\left(q_{\mathrm{sample}} - q_{\mathrm{nearest}}\right)
$$

Here `Δ = step_size`, default **0.8 m**, and `p_goal = goal_bias`, default **0.15**. A goal connection is attempted when the new point is within `step_size` of the goal. The first geometrically validated complete path wins.

**Important:** Traffic proximity is not a separate penalty in the RRT objective. Traffic obstacles affect feasibility through occupancy and geometry checks. Path length and clearance are **evaluation metrics**, not optimization terms in this RRT implementation.

## 5. Grid checks and original-geometry validation

The occupancy grid makes tree expansion inexpensive, but grid discretization can admit a segment that fails original-geometry validation. Every candidate complete path is checked with the shared `validate_path()` function. Rejected candidates are counted in `rejected_exact`; the search continues until a valid route is found or `max_iterations` is exhausted.

Validation samples the original geometry along path segments. It is not an exact continuous swept-volume proof, and the vehicle footprint does not rotate with its heading.

## 6. Performance and output metrics

| Metric | Meaning |
|---|---|
| `grid_build_ms` | Time to construct the occupancy grid |
| `search_ms` | Search duration excluding separately measured validation time |
| `validation_ms` | Time spent validating candidate complete paths |
| `validation_checks` | Checks reported by the geometry validator |
| `validated_candidates` | Number of candidate complete paths submitted for validation |
| `rejected_exact` | Candidates rejected by original-geometry validation |
| Iterations | Tree-growth iterations used |
| Path length / clearance / turning | Derived by the shared benchmark for successful paths |

**Cold-start time** = grid build + search + validation. **Warm-start time** = search + validation (assuming the grid is already available). The planner's `time_to_valid_ms` field is computed even for unsuccessful attempts, so only interpret it as **time to a valid path when a validated path was actually returned**; otherwise it is attempt duration.

## 7. Benchmarking recommendations

Run multiple random seeds per scenario and report success rate, median successful planning time, failed-attempt duration, path length, clearance, and turning. Use the same scenario and original-geometry validator as A* and A* + DWA. RRT's random sampling means a single run is not representative.

## 8. Limitations

- Standard RRT does not optimize route length or smoothness.
- Linear nearest-neighbor search can become expensive as the tree grows.
- Goal bias and extension length strongly affect results.
- A 0.25 m occupancy grid is an approximation.
- Geometry validation is sampled and uses an axis-aligned vehicle footprint.
- No steering, acceleration, or vehicle-dynamics constraints are imposed on the resulting RRT route.
- Obstacles are static within each planning run.

## 9. Research context

This planner is the randomized global-planning baseline in the pothole-avoidance comparison. Its role is to measure when sampling-based exploration finds safe routes efficiently relative to A* and A* + DWA, **without assuming RRT is optimal**.
