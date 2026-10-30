# Rapidly-exploring Random Tree (RRT) — Pothole Avoidance

RRT is a **sampling-based global path planner**. Rather than evaluating a road grid like A*, it grows a tree of collision-free connections from the vehicle's starting position until it can connect to the goal. It finds a *feasible* path, not necessarily the shortest one.

![RRT exploration and final path](assets/rrt_exploration.png)

## Scenario 6: Pothole + traffic

| Item | Position |
| --- | --- |
| Start | `(2.0, 5.4)` |
| Goal | `(47.0, 5.4)` |
| Pothole | `(22.0, 5.4)`, radius `0.7 m` |
| Traffic vehicle | `(22.0, 9.0)` |
| Road | `50 m`, three lanes |

The pothole blocks the direct middle-lane route; traffic makes the upper route less accessible. RRT explores collision-free alternatives, potentially passing below the pothole.

## How RRT works

1. **Initialize:** Create a tree with the start node, `q_start`.
2. **Sample:** Choose a random point `q_rand` on the road. Occasionally sample the goal instead (*goal bias*).
3. **Nearest node:** Find the tree node `q_near` closest to `q_rand`.
4. **Steer:** Extend toward the sample by at most a fixed step size `η`:

   `q_new = q_near + min(η, ||q_rand − q_near||) · (q_rand − q_near) / ||q_rand − q_near||`

5. **Collision check:** Add `q_new` and its parent connection only if the *entire edge* avoids potholes, traffic, and road boundaries (including safety margins).
6. **Connect to goal:** When close enough, attempt a final collision-free edge to the goal.
7. **Reconstruct:** Follow parent pointers from the goal back to the start, then reverse the sequence.

![RRT extension step](assets/rrt_extension.png)

### Worked extension example

With `q_near = (2.0, 5.4)`, `q_rand = (12.0, 3.0)`, and step size `η = 2.0 m`:

- Distance to sample: `sqrt(10² + (−2.4)²) ≈ 10.28 m`
- Unit direction: approximately `(0.973, −0.233)`
- New node: `q_new ≈ (3.95, 4.93)`

RRT advances **2 meters toward** the random sample; it does not jump directly to `(12, 3)`.

## What happens at an obstacle?

If an attempted edge crosses a pothole or traffic vehicle, RRT **rejects that edge** and keeps the existing tree. The next sample may expand a different branch. If it reaches the maximum iteration limit without connecting to the goal, it reports **no path found**—which does not prove no path exists.

## Using this project

Project layout:

```text
project/
├── common.py
├── requirements.txt
└── RRT/
    ├── rrt_pothole_simulator.py
    ├── README.md
    └── assets/
```

From the project root, install dependencies and run:

```bash
python -m pip install -r requirements.txt
python RRT/rrt_pothole_simulator.py --scenario 6
```

> If your script has a different name or CLI options, adjust the command accordingly. The planner uses the shared `common.py` for road geometry, scenarios, and collision checks.

## RRT vs. A* vs. DWA

| Planner | Core approach | Main limitation |
| --- | --- | --- |
| **A*** | Heuristic search over a grid | Grid path may not be drivable |
| **RRT** | Randomly grow a collision-free tree | Variable results; no shortest-path guarantee |
| **DWA** | Score short-horizon vehicle trajectories | Can get stuck in local minima |

**Research considerations:** Run RRT with multiple random seeds and report success rate, planning time, path length, and clearance. Basic geometric RRT does **not** enforce steering radius, acceleration, or smoothness; a controller or kinodynamic planner is needed for vehicle-feasible motion.

**Takeaway:** *A* searches for a minimum-cost route on its graph; RRT explores until it finds a feasible route; DWA chooses how to move locally.*
