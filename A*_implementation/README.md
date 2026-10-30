# A* Pothole Avoidance Simulator

A Python-based, two-dimensional **grid A\*** pathfinding baseline for a three-lane road. The simulator plans a route from a start position to a goal while avoiding potholes and stationary traffic vehicles. It visualizes the planned path and reports basic planning metrics.

This implementation is **inspired by** the traffic-aware A* cost formulation discussed by Sistla et al. It is not a reproduction of the authors' complete system, nor a vehicle-dynamics simulator.

## Features

- Configurable three-lane road and occupancy grid.
- Circular potholes with inflated no-go regions.
- Stationary traffic vehicles with inflated rectangular no-go regions.
- Soft traffic-proximity costs on traversable grid cells.
- Eight-connected A* search with diagonal corner-cutting prevention.
- Euclidean heuristic, path reconstruction, and Matplotlib visualization.
- Six built-in test scenarios and summary metrics.


## Run

Save the implementation as `astar_pothole_simulator.py`, then run:

```bash
python astar_pothole_simulator.py
```

The script runs six scenarios sequentially, prints planning metrics, and opens a Matplotlib figure for each scenario. Close each figure to proceed to the next one.


## How the planner works

1. **Build the road grid.** The road is 50 m long and 10.8 m wide (three 3.6 m lanes). Each grid cell is 0.25 m across.
2. **Mark unsafe regions.** Potholes and traffic vehicles are inflated to approximate clearance for the planning vehicle. Inflated cells are blocked.
3. **Calculate traffic proximity costs.** Accessible cells near traffic receive an additional cost. Blocked cells remain impassable regardless of their cost.
4. **Search using A*.** The planner expands valid neighboring cells, retaining the lowest accumulated cost to each cell.
5. **Reconstruct the path.** On reaching the goal, the planner follows stored parent links and converts grid cells back to world coordinates.
6. **Visualize and measure.** The script plots the road and path and reports search time, expanded nodes, path length, and pothole clearance.

### Cost function

The implementation uses:

\[
f(n) = g(n) + h(n)
\]

where:

- **\(g(n)\)** is the cumulative cost of reaching node \(n\), including distance traveled and traffic penalties.
- **\(h(n)\)** is Euclidean distance from \(n\) to the goal, used as an estimate of remaining cost.

For each allowed movement from the current node to a neighboring node:

\[
g(\text{neighbor}) = g(\text{current}) + d\,[1 + \lambda T(\text{neighbor})]
\]

- \(d\): movement distance (0.25 m cardinal; approximately 0.354 m diagonal).
- \(\lambda\): `TRAFFIC_WEIGHT`, default `5.0`.
- \(T\): normalized traffic-proximity cost for the neighbor, between 0 and 1.

Traffic penalties are accumulated in \(g\), rather than added only once at the currently evaluated node. The heuristic does not include traffic penalties.

### Obstacle clearance

For potholes, the blocked radius is:

\[
R_{blocked} = R_{pothole} + \frac{W_{vehicle}}{2} + M_{safety}
\]

With a 1.8 m vehicle width and 0.25 m safety margin, a pothole with a 0.7 m radius is treated as blocked within **1.85 m** of its center.

Traffic vehicles are modeled using axis-aligned rectangles inflated by a conservative circular approximation of the planning vehicle footprint plus the safety margin. This approximation differs from the pothole inflation rule.

## Test scenarios

| # | Scenario | Expected result |
|---|---|---|
| 1 | Empty road | Direct route toward goal |
| 2 | Single pothole | Route around pothole |
| 3 | Pothole near road boundary | Route using available clearance |
| 4 | Multiple potholes | Route around multiple obstacles |
| 5 | Road fully blocked | No path found |
| 6 | Pothole with traffic | Route around pothole while avoiding the occupied traffic region |

### Scenario 6 in brief

The vehicle starts in the middle lane and approaches a pothole in its direct route. A stationary traffic vehicle occupies the upper lane. The planner rejects grid cells within both inflated obstacle regions and assigns extra costs to available cells near traffic. Since the upper side is obstructed, the lower side offers a feasible route around the pothole. After clearing the obstacles, A* heads toward the goal in the middle lane.

This is an interpretation of the configured geometry; the precise grid path and node expansion order are determined at runtime.

## Output metrics

- **Success:** Whether A* found a path to the goal.
- **Planning time (ms):** Wall-clock search duration, dependent on the computer and runtime conditions.
- **Expanded nodes:** Number of priority-queue entries processed by the current implementation; this may include repeated expansions when better paths are discovered.
- **Path length (m):** Sum of distances between successive path points.
- **Minimum pothole clearance (m):** Minimum distance between path points and the *inflated* pothole boundary. This is an approximation, not a continuous vehicle-footprint collision guarantee.
- **Overall success rate:** Fraction of the six scenarios with a path found. This is a small functional test, **not** a statistical measure of real-world avoidance success.

## Configuration

Edit constants near the top of `astar_pothole_simulator.py`:

| Setting | Default | Meaning |
|---|---:|---|
| `ROAD_LENGTH` | `50.0` | Road length in meters |
| `LANE_WIDTH` | `3.6` | Lane width in meters |
| `NUM_LANES` | `3` | Number of lanes |
| `GRID_RESOLUTION` | `0.25` | Cell size in meters |
| `VEHICLE_WIDTH` | `1.8` | Planning vehicle width in meters |
| `VEHICLE_LENGTH` | `4.5` | Planning vehicle length in meters |
| `SAFETY_MARGIN` | `0.25` | Additional clearance in meters |
| `TRAFFIC_WEIGHT` | `5.0` | Relative weight of traffic-proximity costs |
| `TRAFFIC_INFLUENCE_RADIUS` | `4.0` | Distance from traffic-vehicle footprint over which soft penalties apply |

To add or change a scenario, edit `create_scenario()` and use `env.add_pothole(...)` or `env.add_traffic_vehicle(...)`.

## Limitations

- **Grid path, not a drivable trajectory:** The search does not model heading, steering angle, turning radius, acceleration, or vehicle speed.
- **Approximate clearance:** Inflated obstacles are checked at grid-cell centers; collision-free grid paths do not guarantee continuous-space safety.
- **Static traffic:** Other vehicles do not move during planning.
- **All potholes are blocked:** The provided `depth` field is stored but does not influence A* or allow traversing shallow potholes.
- **No execution or tracking:** The program does not drive a simulated car along the path.
- **No learned perception:** Potholes are entered manually rather than supplied by YOLO or depth estimation.
- **No planner comparison yet:** RRT and DWA are not included.

## Next steps

1. Add a path-following vehicle visualization and measure tracking errors.
2. Introduce vehicle kinematics or Hybrid A* to enforce steering feasibility.
3. Add time-varying traffic and repeated replanning.
4. Integrate pothole detections, geometry, and severity estimates.
5. Implement RRT and DWA using comparable inputs, constraints, and evaluation scenarios.
6. Benchmark planning time, route length, clearance, and executed maneuver success over many randomized scenarios.

## Research context

The project explores pothole avoidance through perception, risk assessment, and adaptive planning. This A* module is an initial baseline. Its grid representation, safety inflation, traffic penalty map, and parameter choices are implementation decisions; they should not be attributed as exact, fully specified details from Sistla et al.
