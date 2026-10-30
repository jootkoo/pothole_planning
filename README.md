# Pothole Avoidance: RRT and DWA

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

If you do not have a requirements file yet, install the current dependencies directly:

```bash
python -m pip install numpy matplotlib
```
Two runnable planners using the **same six static test scenarios** as the earlier A* demonstration:

- **RRT** (`rrt_pothole_simulator.py`): samples a geometric tree to find a collision-free route.
- **DWA** (`dwa_pothole_simulator.py`): samples short-horizon bicycle-model commands and follows an **A*-computed reference path**.

Read [README_RRT.md](README_RRT.md) and [README_DWA.md](README_DWA.md) for the algorithm steps, assumptions, metrics, and limitations.

## Setup (inside `nix develop` and your activated `.venv`)
```bash
python -m pip install -r requirements.txt
python rrt_pothole_simulator.py --scenario 6
python dwa_pothole_simulator.py --scenario 6
```

`common.py` contains road geometry, shared scenarios, collision checking, plotting, and the A* reference planner for DWA. All code uses NumPy and Matplotlib; Python's `heapq`, `random`, and `math` are standard-library modules.

## Research note
The two algorithms are **educational, independently specified implementations**, not the authors' original code. RRT is a global geometric planner; DWA is a local motion planner, so their time metrics and success criteria differ. These examples do not establish real-vehicle safety.
