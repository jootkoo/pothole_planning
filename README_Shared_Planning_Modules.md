# Shared Planning Modules: Road Geometry, Occupancy Grids, and Path Validation

This document explains how `common_optimized.py` (the shared road/environment module, sometimes referred to as `common.py`), `planning_grid.py`, and `geometry_validation.py` support the project's **A\***, **RRT**, and **A\* + DWA** pothole-avoidance planners.

## Requirements
- Python 3.12 (recommended)
- `numpy`
- `matplotlib`

The code uses only Python standard-library modules beyond these two packages.

## Installation

From the project directory, optionally enter the Nix development shell:

```bash
nix develop
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

```

`common.py` contains road geometry, shared scenarios, collision checking, plotting, and the A* reference planner for DWA. All code uses NumPy and Matplotlib; Python's `heapq`, `random`, and `math` are standard-library modules.
## 1. Architecture at a glance

```text
scenarios/*.json
      |
      v
common_optimized.py   <-- shared Road model, vehicle footprint,
      |                    obstacle geometry, scenario loader, drawing
      |
      +------------------> planning_grid.py
      |                         |  occupancy + traffic-cost grids
      |                         |  optional obstacle distance field
      |                         |  shared grid_astar()
      |                         v
      |                  A* / RRT / A* + DWA
      |                         |
      +--> geometry_validation.py <-- candidate path or motion segment
                 |
                 +--> common_optimized.segment_free()
                           |
                           +--> Road.free() on original geometry
```

**Division of responsibility:** `common_optimized.py` defines *the world and what is safe*; `planning_grid.py` makes repeated planning queries *fast*; `geometry_validation.py` checks *whether a candidate actually satisfies the shared geometry model*.

## 2. Project layout

```text
project_root/
├── common_optimized.py
├── planning_grid.py
├── geometry_validation.py
├── scenarios/
│   ├── 1.json
│   ├── 2.json
│   ├── 3.json
│   ├── 4.json
│   ├── 5.json
│   ├── 6.json
│   └── curved_polygon.json
├── AStar/
│   └── astar_pothole_simulator.py
├── RRT/
│   └── rrt_pothole_simulator.py
├── DWA/
│   └── dwa_pothole_simulator.py
└── Benchmark/
    ├── benchmark.py
    └── metrics.py
```

## 3. `common_optimized.py`: shared environment and geometry

### Purpose

Defines a consistent road, obstacles, vehicle dimensions, and collision model for all planners. It also loads the same JSON scenarios for reproducible comparisons.

### Road and vehicle defaults

| Parameter | Default | Meaning |
|---|---:|---|
| `ROAD_LENGTH` | 50.0 m | Typical straight-road length |
| `ROAD_WIDTH` | 10.8 m | Three 3.6 m lanes |
| `LANE_WIDTH` | 3.6 m | Lane width |
| `CAR_LENGTH` | 4.5 m | Ego vehicle length |
| `CAR_WIDTH` | 1.8 m | Ego vehicle width |
| `SAFETY_MARGIN` | 0.25 m | Clearance added around ego vehicle |
| `RESOLUTION` | 0.25 m | Default planner-grid resolution |
| `MAP_RESOLUTION` | 0.10 m | Separate mapping-layer resolution |

The `Road` object stores the road boundary, potholes, traffic vehicles, lane boundaries, vehicle dimensions, and metadata. Potholes can be circular or polygonal; metadata can include depth, severity, and confidence. **Those metadata fields do not by themselves create a severity-weighted planner cost.**

### Core functions and methods

| Function / method | Role |
|---|---|
| `scenario(id)` | Loads JSON and returns `(road, start, goal, name)` |
| `available_scenarios()` | Lists available scenario IDs |
| `Road.add_pothole(...)` | Adds a circular or polygonal pothole |
| `Road.add_car(...)` | Adds a traffic vehicle |
| `Road.free(x, y)` | Tests whether the ego footprint is collision-free at a position |
| `Road.traffic_cost(x, y)` | Supplies the traffic-proximity penalty used by A* |
| `Road.ego_polygon(x, y)` | Builds the ego footprint with safety margin |
| `segment_free(road, a, b, step=0.12)` | Samples points along a segment and calls `Road.free()` |
| `draw(...)` | Shared visualization utility |

### Collision model

The ego vehicle is represented by an **axis-aligned rectangle** enlarged by the safety margin. The vehicle footprint is **not rotated with heading**. This is an intentional modeling approximation and applies consistently to the shared collision checks.

For the default vehicle, the enlarged footprint dimensions are:

$$
L_{\mathrm{collision}} = 4.5 + 2(0.25) = 5.0\ \mathrm{m}
$$

$$
W_{\mathrm{collision}} = 1.8 + 2(0.25) = 2.3\ \mathrm{m}
$$

Road boundaries and pothole/traffic geometry are retained for the original-geometry checks. Geometry caching and bounding-box tests help avoid unnecessary expensive polygon calculations.

### Scenario loading example

```python
from common_optimized import scenario

road, start, goal, name = scenario('4')
print(name, start, goal)
```

The coordinate system is a local world frame in **meters**, with `x` forward and `y` left.

## 4. `planning_grid.py`: shared fast planning representation

### Purpose

Constructs a **configuration-space occupancy grid**: a cell is free only if the **whole axis-aligned ego footprint**, including safety margin, is considered collision-free at that cell's center. This is **not** merely a map of whether the center point overlaps a pothole.

### `PlanningGrid`

```python
from planning_grid import PlanningGrid

grid = PlanningGrid(road, resolution=0.25)
```

At construction, the class computes:

- `grid.free`: boolean occupancy of cell centers using `road.free(x, y)`.
- `grid.traffic`: traffic-proximity costs for traversable cells using `road.traffic_cost(x, y)`.
- `grid.distance`: an **optional** obstacle distance field, enabled with `with_distance=True`.

Important operations:

| Operation | What it does |
|---|---|
| `to_cell(point)` | Converts world coordinates to grid indices |
| `to_world(cell)` | Returns the center coordinates of a cell |
| `is_free(cell)` | Checks bounds and precomputed occupancy |
| `edge_free(a, b)` | Checks endpoint occupancy and blocks diagonal corner cutting for adjacent diagonal moves |
| `sample_distance(x, y)` | Estimates obstacle clearance from the optional distance field |

**Important:** `edge_free()` is a cheap grid check, **not** full original-geometry segment validation. For RRT's longer edges, the planner checks intervening grid cells; completed paths are still checked with `validate_path()`.

### Shared A* search

`grid_astar(grid, start, goal, traffic_weight=5.0)` performs eight-neighbor A* and returns a candidate path and expanded-node count.

Its priority function is:

$$
f(n) = g(n) + h(n)
$$

The movement-cost update is:

$$
g(n_{\mathrm{next}}) = g(n) + d\left[1 + \lambda T(n_{\mathrm{next}})\right]
$$

where $d$ is the physical grid-step distance, $\lambda$ is `traffic_weight`, and $T$ is the precomputed traffic cost of the destination cell. The heuristic is Euclidean distance to the goal:

$$
h(n) = \sqrt{(x_n-x_g)^2 + (y_n-y_g)^2}
$$

At a resolution of 0.25 m, cardinal steps are 0.25 m and diagonal steps are approximately 0.354 m. The reconstructed route includes the physical start and goal plus grid-cell centers.

### Optional distance field for DWA

```python
local_grid = PlanningGrid(road, resolution=0.10, with_distance=True)
clearance_estimate = local_grid.sample_distance(x, y)
```

The field is based on eight-neighbor distances to **occupied cell centers**. `sample_distance()` subtracts a cell-diagonal allowance to produce a conservative heuristic estimate. **It is not an exact Euclidean distance or a collision-safety guarantee.** DWA performs additional checks with `Road.free()` near hazards.

## 5. `geometry_validation.py`: common final safety check

### Purpose

Validates planner-produced paths against the **original road and obstacle model** rather than trusting the occupancy grid alone.

```python
from geometry_validation import validate_path

valid, validation_ms, checked_edges = validate_path(road, path)
```

The function returns:

| Value | Meaning |
|---|---|
| `valid` | Whether every tested segment passed |
| `validation_ms` | Wall-clock validation duration in milliseconds |
| `checked_edges` | Number of path segments checked before completion or rejection |

It rejects missing paths or paths with fewer than two points. For each segment, it calls `common_optimized.segment_free()` with a default sampling step of **0.12 m**, testing interpolated positions using `Road.free()`.

### Why is this separate from the occupancy grid?

A 0.25 m grid is a discretized approximation. Grid-approved edges may cross geometry that the cell-center occupancy representation does not fully capture. Original-geometry sampling provides a second, consistent check across planners.

**Limitation:** Sampling every ~0.12 m is **not mathematically exact continuous swept-volume validation**. Very small or narrow collisions between samples may still be missed, and the footprint remains axis-aligned.

## 6. How each algorithm uses the modules

| Component | A* | RRT | A* + DWA |
|---|---|---|---|
| `common_optimized.py` | Shared road, scenario, geometry | Same | Same |
| `PlanningGrid` at 0.25 m | Global A* search | Grid-assisted RRT collision checks | Global A* reference path |
| Additional 0.10 m distance grid | No | No | Local trajectory clearance scoring |
| `grid_astar()` | Primary planner | No | Global reference planner |
| `validate_path()` | Checks final candidate path | Checks candidate goal paths | Checks reference path, motion segments, and final executed path |
| Rejection behavior | Rejects invalid result; no automatic retry | Can reject a candidate and continue sampling | Rejects invalid reference, unsafe motion, or invalid executed path |

### A* execution flow

```text
Load scenario -> Build 0.25 m PlanningGrid
              -> grid_astar() -> candidate path
              -> validate_path() -> accept or reject
```

A* uses the precomputed grid for search efficiency, including traffic costs. Its final route is accepted only after the original-geometry validator passes. The current A* wrapper does **not** automatically replan after validation failure.

### RRT execution flow

```text
Load scenario -> Build 0.25 m PlanningGrid
              -> Random sample -> Extend tree
              -> Check grid cells along proposed edge
              -> Candidate route to goal
              -> validate_path()
                   | valid   -> accept
                   | invalid -> continue searching (within iteration limit)
```

RRT samples points in continuous coordinates but uses the shared grid to accelerate repeated collision tests. A grid-accepted route is **not automatically considered safe**; the original geometry must approve it.

### A* + DWA execution flow

```text
Load scenario
   -> Build global 0.25 m PlanningGrid
   -> Build local 0.10 m PlanningGrid + distance field
   -> grid_astar() produces reference path
   -> validate_path(reference)
   -> DWA samples speed / steering commands
   -> Roll out candidate trajectories and score them
   -> Validate each chosen motion segment
   -> Validate completed executed path
```

DWA uses the global path as guidance and evaluates local trajectories using target distance, heading error, speed preference, and obstacle-clearance penalties. Its distance field accelerates local evaluation but does not replace the shared geometry validator.

## 7. Timing and fair comparisons

Grid construction, planner search, and geometry validation are conceptually distinct costs:

$$
T_{\mathrm{cold}} = T_{\mathrm{grid}} + T_{\mathrm{search}} + T_{\mathrm{validation}}
$$

$$
T_{\mathrm{warm}} = T_{\mathrm{search}} + T_{\mathrm{validation}}
$$

For A* and RRT, `T_grid` is the construction time of the shared occupancy grid. For A* + DWA, it includes **both** global and local grids, including the local distance field. This distinction matters when comparing cold-start performance.

**Do not confuse time spent on a failed attempt with time to a valid path.** A failed run has a measured duration but has not obtained a valid route. Compare successful-path times separately from success rates and failed-attempt durations.

Using the same `Road`, obstacle geometry, vehicle dimensions, and `validate_path()` across planners reduces methodological differences. However, algorithms can still differ in how often they call validation and how much preprocessing they require.

## 8. Example: shared setup

```python
from common_optimized import scenario
from planning_grid import PlanningGrid, grid_astar
from geometry_validation import validate_path

road, start, goal, name = scenario('2')
grid = PlanningGrid(road, resolution=0.25)
path, expanded = grid_astar(grid, start, goal, traffic_weight=5.0)

if path is not None:
    valid, validation_ms, checked_edges = validate_path(road, path)
    print(f'Scenario: {name}')
    print(f'Valid: {valid}, expanded: {expanded}')
    print(f'Validation: {validation_ms:.3f} ms, edges checked: {checked_edges}')
else:
    print('No grid path found')
```

Run this example from the project root so the shared modules and `scenarios/` directory can be found.

## 9. Limitations and future improvements

1. **Axis-aligned ego footprint:** Heading-aware vehicle collisions are not yet modeled.
2. **Discretization:** Grid occupancy is computed at cell centers, so its resolution affects the planning approximation.
3. **Sampled validation:** Segment sampling is not exact continuous collision checking.
4. **Static scenario snapshots:** Moving traffic and live sensor updates are not handled by these shared modules alone.
5. **Approximate DWA clearance:** The distance field is a heuristic, not an exact geometric clearance measure.
6. **No automatic A* recovery:** An invalid final A* path is rejected without an automatic alternative search.
7. **Vehicle feasibility:** Geometric validation does not itself guarantee steering, acceleration, or trajectory-tracking feasibility.

Possible extensions include orientation-aware collision geometry, exact swept-volume segment checks, incremental grid updates, and replanning after validation failures.

## 10. Key takeaway

The three modules separate **environment modeling**, **efficient planning queries**, and **independent path validation**. That separation lets A*, RRT, and A* + DWA use different search strategies while sharing the same underlying definition of obstacles and geometric safety—an important requirement for a credible algorithm benchmark.
