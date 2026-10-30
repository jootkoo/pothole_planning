# Dynamic Window Approach (DWA) — Pothole Avoidance

## Overview

The **Dynamic Window Approach (DWA)** is a **local motion planner**. Unlike A*, which finds a route across the entire road, DWA repeatedly selects a feasible **speed and steering angle**, predicts the resulting short-term motion, and executes only the next control step.

**In this implementation, DWA follows a global reference route produced by A*.** The two algorithms work together: **A* decides where to go; DWA decides how to drive there.**

## How It Works

1. **Get the vehicle state:** position `(x, y)`, heading `θ`, speed `v`, and steering angle `δ`.
2. **Build the dynamic window:** limit candidate speeds and steering angles based on acceleration, steering rate, and vehicle limits.
3. **Predict trajectories:** simulate each candidate command using a kinematic bicycle model over a short horizon.
4. **Reject collisions:** discard trajectories that intersect potholes, traffic vehicles, or road boundaries.
5. **Score safe trajectories:** prefer progress toward an A* look-ahead waypoint, good heading alignment, higher speed, and obstacle clearance.
6. **Move and repeat:** execute the best command for one time step, update the vehicle state, and plan again.

### Vehicle Motion Model

For wheelbase `L`, time step `Δt`, speed `v`, and steering angle `δ`:

```text
x_next ≈ x + v cos(θ) Δt
y_next ≈ y + v sin(θ) Δt
θ_next = θ + (v / L) tan(δ) Δt
```

The code uses an average-heading approximation when updating position.

### Trajectory Cost

The implementation minimizes this heuristic score among collision-free candidates:

```text
J = 2.5 × distance_to_lookahead
  + 0.65 × heading_error
  + 0.65 × (maximum_speed − candidate_speed)
  + 0.12 / (minimum_clearance + 0.15)
```

This is the **simulator's custom scoring function**, not the A* `f(n) = g(n) + h(n)` function.

## Example: Scenario 6 — Pothole and Traffic

- The vehicle begins in the middle lane, approaching a pothole.
- A traffic vehicle occupies the upper lane, making that side less suitable.
- **A*** supplies a reference route around the lower side of the pothole.
- **DWA** tests reachable steering/speed commands, rejects unsafe trajectories, and executes short movements to follow that route.
- After passing the pothole, it continues toward the destination, updating its motion at each step.

## Implementation Settings

| Parameter | Value |
|---|---:|
| Control time step | 0.25 s |
| Prediction horizon | 2.0 s |
| Wheelbase | 2.7 m |
| Maximum speed | 2.5 m/s |
| Maximum acceleration | 1.5 m/s² |
| Maximum steering angle | ±35° |
| Maximum steering rate | 90°/s |
| Candidate commands per cycle | 5 speeds × 11 steering angles |
| Goal tolerance | 0.7 m |

## Running the Simulation

From the project root, install the dependencies and run:

```bash
python -m pip install -r requirements.txt
python DWA/dwa_pothole_simulator.py --scenario 6
```

If `common.py` is in the project root and `DWA/` is a separate folder, ensure the script adds the parent directory to Python's import path before `from common import ...`:

```python
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
```

## Output and Limitations

The simulation reports success/failure, executed steps, path length, and **combined A* planning plus DWA simulation time**. It plots the A* reference route and the DWA-driven path.

DWA can become trapped in local minima; the A* reference helps but does not eliminate this risk. Traffic is stationary in these scenarios. The implementation is a simplified simulation, not a validated vehicle controller or emergency-braking system.
