# A* Pothole Avoidance Simulator

A two-dimensional **A\*** path planner for pothole avoidance using the project's shared road model, **0.25 m occupancy grid** (configurable), and **original-geometry path validation**. The planner finds a grid-based route from a start to a goal, optionally displays it, and reports preprocessing, search, and validation timings separately.

This is a path-planning baseline, **not** a vehicle-dynamics simulator or a complete reproduction of the traffic-aware planning system discussed by Sistla et al.

## Features

- Uses shared road and scenario definitions from `common.py`, including the geometry represented by each scenario.
- Builds an occupancy grid with `PlanningGrid` from `planning_grid.py`.
- Runs eight-connected grid-based A* through `grid_astar`, with a configurable traffic-proximity weight.
- Validates a completed candidate path against the **original road and obstacle geometry** through the shared `geometry_validation.validate_path` function.
- Rejects a candidate path if final validation fails; the current A* implementation **does not automatically replan** after rejection.
- Reports grid construction, search, validation, total attempt time, expanded nodes, path length, and approximate minimum clearance.
- Supports individual scenarios or all scenarios, including those loaded through the shared scenario system.
- Supports optional Matplotlib visualization and a headless `--no-plot` mode.

## Project layout and dependencies

The uploaded implementation is named `astar.py` and expects to live **one directory below the project root**, alongside the other planners. It adds its parent directory to Python's import path.

```text
project_root/
├── common.py                 # Road geometry, scenarios, drawing, path length
├── planning_grid.py          # PlanningGrid and grid_astar
├── geometry_validation.py    # Shared validate_path function
├── scenarios/                # Scenario data used by common.py, if configured
└── AStar/
    ├── astar.py              # This implementation
    └── README.md
```

The program requires the Python dependencies used by these shared modules and Matplotlib for plotting. Keep the shared files compatible with the imported functions and classes shown above.

## Running the planner

Run from the project root:

```bash
# All scenarios configured in common.available_scenarios()
python AStar/astar.py

# One scenario, without opening a plot window
python AStar/astar.py --scenario 4 --no-plot

# Adjust grid resolution and traffic cost weighting
python AStar/astar.py --scenario 6 --resolution 0.25 --traffic-weight 5.0 --no-plot

# Run the curved-road scenario, if present in your shared scenarios
python AStar/astar.py --scenario curved_polygon --no-plot
```

### Command-line arguments

| Argument | Default | Purpose |
|---|---|---|
| `--scenario` | All available | Select one scenario identifier, or omit to run all scenarios returned by `available_scenarios()`. |
| `--resolution` | `0.25` | Occupancy-grid cell size in meters; must be positive. |
| `--traffic-weight` | `5.0` | Traffic-cost weighting passed to `grid_astar`; must be nonnegative. |
| `--no-plot` | Off | Suppress Matplotlib path visualization. |

The exact list and geometry of scenarios are defined by the shared `common.py`/scenario data, **not hardcoded in `astar.py`**.

## How planning works

1. **Load the scenario.** `scenario(number)` returns the road, start, goal, and descriptive name.
2. **Build the grid.** `PlanningGrid(road, resolution)` precomputes the occupancy representation. Its build time is measured separately from search.
3. **Search.** `grid_astar(grid, start, goal, traffic_weight)` returns a candidate path and the number of expanded nodes.
4. **Validate.** If a candidate exists, `validate_path(road, path)` checks it against the original road/obstacle geometry, returning a validity flag, validation duration, and number of validation checks.
5. **Accept or reject.** If validation fails, the planner discards the candidate (`path = None`). It does not search for an alternative in that same call.
6. **Report and optionally plot.** The script prints timing and path statistics, then draws the scenario unless `--no-plot` is set.

### A* cost function and obstacle representation

A* prioritizes candidate grid nodes using the standard evaluation function:

$$
f(n) = g(n) + h(n)
$$

- **`g(n)` — accumulated path cost:** the cost of moving from the start to node `n`, including movement distance and any traffic-proximity penalties implemented by the shared grid search.
- **`h(n)` — heuristic cost:** an estimate of the remaining travel cost to the goal.
- **`f(n)` — total estimated cost:** the priority used to choose which node to explore next.

The **original A* README** documented the following traffic-weighted movement-cost formulation:

$$
g(n_{next}) = g(n_{current}) + d\,[1 + \lambda T(n_{next})]
$$

Here, `d` is the movement distance, `λ` is the configurable traffic weight (`--traffic-weight`, default `5.0`), and `T` is a normalized traffic-proximity penalty from `0` to `1`. At the default `0.25 m` grid resolution, cardinal moves span `0.25 m` and diagonal moves span approximately `0.354 m`. The earlier implementation also documented a Euclidean heuristic:

$$
h(n) = \sqrt{(x_n-x_{goal})^2 + (y_n-y_{goal})^2}
$$

**Implementation verification:** The uploaded `astar.py` delegates all search-cost calculations to `grid_astar(...)` in `planning_grid.py`; that shared file was not provided with this README update. Therefore, the traffic-weighted equation and Euclidean heuristic above describe the **previously documented formulation**, not independently verified behavior of the current shared implementation. Check `planning_grid.py` before citing these as the exact current equations.

The occupancy grid determines whether a location is traversable; the cost function ranks traversable alternatives. Original-geometry validation then checks the completed path against the underlying road and obstacle geometry. The shared validator may sample along path segments and should not be treated as an exact continuous swept-volume guarantee unless it explicitly implements one.

## Timing and output metrics

| Output | Meaning |
|---|---|
| `Found path` | `True` only if A* found a candidate **and** it passed geometry validation. |
| `Grid build` (`grid_build_ms`) | Time spent constructing `PlanningGrid`, measured in the planner constructor. |
| `Search` (`search_ms`) | Time spent inside `grid_astar`, excluding grid build and final validation. |
| `Validation` (`validation_ms`) | Time spent validating the completed candidate; `0.0` if no candidate was produced. |
| `validation_checks` | Number of checks reported by the shared validator. Stored on the planner, but not currently printed by the CLI. |
| `Time to valid path (cold start)` (`time_to_valid_ms`) | Sum of grid build, search, and validation times. **If the run fails, this is total attempt duration, not time to a valid path.** |
| `Expanded nodes` | Count returned by `grid_astar`; its precise counting convention is defined there. |
| `Path length` | Sum of path-segment lengths, shown only when a validated path exists. |
| `Min approximate clearance` | Minimum of `road.clearance(x, y)` over returned path points; this is sampled at path vertices and may not equal minimum clearance along the entire route. |

The implementation returns `(path, search_ms, expanded_nodes)` from `AStarPlanner.plan(...)`. Additional timing fields are accessible as attributes on the planner instance.

### Example output structure

```text
Scenario 4: <scenario name>
  Found path: True
  Grid build: <milliseconds> ms | Search: <milliseconds> ms | Validation: <milliseconds> ms
  Time to valid path (cold start): <milliseconds> ms | Expanded nodes: <count> | Valid: True
  Path length: <meters> m | Min approximate clearance: <meters> m
```

Values above are placeholders, not benchmark results.

## Benchmarking guidance

For fair comparison against RRT and A* + DWA, use the **same scenarios, safety assumptions, and original-geometry validation standard**. Report at least:

- **Cold-start attempt time:** grid build + search + validation.
- **Warm-start attempt time:** search + validation, assuming preprocessing is already available.
- **Successful time to valid path:** report only when a validated path exists.
- **Success rate:** include failed and rejected paths in the denominator.
- **Path quality:** path length, clearance, and turning, calculated only for successful validated paths.

The current planner builds a new grid for each `AStarPlanner` instance. Warm-start timing is a **component-based estimate**, not evidence of a persistent shared-grid cache.

## Limitations

- **Grid-based path, not a drivable trajectory:** A* does not enforce steering dynamics, turning radius, speed, or acceleration constraints.
- **Approximate grid occupancy:** Rasterization can differ from the original continuous geometry.
- **Final validation can reject a path:** The current planner returns failure rather than automatically replanning around the invalid segment.
- **Sampled clearance reporting:** `min_clearance` evaluates path vertices, not necessarily all points along the route.
- **Static scenario during a run:** The planner does not itself perform live sensor integration or incremental map updates.
- **Timing is machine-dependent:** Compare repeated runs under consistent conditions; do not interpret one run as a statistically robust performance result.

## Potential next steps

1. Implement repair/replanning when final geometry validation rejects an A* candidate.
2. Add segment-wide clearance measurement and, where needed, exact swept-footprint collision validation.
3. Benchmark on many randomized maps with common inputs for A*, RRT, and A* + DWA.
4. Extend to heading-aware or kinematically feasible planning if vehicle execution is required.
5. Connect pothole detections and road geometry from perception to the shared scenario/map pipeline.

## Research context

This A* planner is one component of a pothole-detection and adaptive path-planning research project. Its grid representation, traffic weighting, and geometry-validation workflow are **implementation choices**, not claims that the complete system is specified by Sistla et al.
