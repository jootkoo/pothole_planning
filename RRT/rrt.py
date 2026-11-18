"""RRT in continuous space with a shared occupancy grid and exact path validation.

The grid is a fast, conservative filter; accepted solutions are checked against
Road's original continuous polygon/circle geometry before being reported.
"""
import argparse
import math
import random
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import Road, available_scenarios, draw, path_length, scenario, segment_free
from planning_grid import PlanningGrid
from geometry_validation import validate_path


def grid_segment_free(grid, a, b):
    """Check all grid cells touched by segment a->b (supercover traversal).

    Includes cells touched at corners/boundaries, avoiding unsafe shortcuts.
    The grid represents the inflated, axis-aligned ego footprint at cell centers;
    continuous validation is still required for exact collision safety.
    """
    x0 = (a[0] - grid.xmin) / grid.resolution
    y0 = (a[1] - grid.ymin) / grid.resolution
    x1 = (b[0] - grid.xmin) / grid.resolution
    y1 = (b[1] - grid.ymin) / grid.resolution
    if not all(map(math.isfinite, (x0, y0, x1, y1))):
        return False

    ix, iy = math.floor(x0), math.floor(y0)
    ex, ey = math.floor(x1), math.floor(y1)
    if not grid.is_free((ix, iy)) or not grid.is_free((ex, ey)):
        return False

    dx, dy = x1 - x0, y1 - y0
    step_x = (1 if dx > 0 else -1) if dx else 0
    step_y = (1 if dy > 0 else -1) if dy else 0
    tx_delta = abs(1.0 / dx) if dx else math.inf
    ty_delta = abs(1.0 / dy) if dy else math.inf
    tx_max = ((ix + 1 - x0) / dx if dx > 0 else (x0 - ix) / -dx) if dx else math.inf
    ty_max = ((iy + 1 - y0) / dy if dy > 0 else (y0 - iy) / -dy) if dy else math.inf

    # If a segment lies exactly along a cell boundary, check cells on both sides.
    vertical_boundary = dx == 0 and x0 == math.floor(x0)
    horizontal_boundary = dy == 0 and y0 == math.floor(y0)

    def safe(i, j):
        for ox in ((0, -1) if vertical_boundary else (0,)):
            for oy in ((0, -1) if horizontal_boundary else (0,)):
                if not grid.is_free((i + ox, j + oy)):
                    return False
        return True

    if not safe(ix, iy):
        return False
    while (ix, iy) != (ex, ey):
        if abs(tx_max - ty_max) <= 1e-12:
            # Segment passes through a grid corner: include both side cells.
            if not safe(ix + step_x, iy) or not safe(ix, iy + step_y):
                return False
            ix += step_x
            iy += step_y
            tx_max += tx_delta
            ty_max += ty_delta
        elif tx_max < ty_max:
            ix += step_x
            tx_max += tx_delta
        else:
            iy += step_y
            ty_max += ty_delta
        if not safe(ix, iy):
            return False
    return True


class RRTPlanner:
    def __init__(self, road: Road, step_size=0.8, goal_bias=0.15,
                 max_iterations=12000, seed=42, resolution=0.25):
        self.road = road
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.rng = random.Random(seed)
        began = time.perf_counter()
        self.grid = PlanningGrid(road, resolution=resolution)
        self.grid_build_ms = (time.perf_counter() - began) * 1000
        self.rejected_exact = 0

    def _candidate_path(self, nodes, parents, goal_parent, goal):
        path = [goal]
        idx = goal_parent
        while idx >= 0:
            path.append(nodes[idx])
            idx = parents[idx]
        path.reverse()
        return path

    def _finish(self, path, began, iteration):
        self.search_ms = max(0.0, (time.perf_counter() - began) * 1000 - self.validation_ms)
        self.time_to_valid_ms = self.grid_build_ms + self.search_ms + self.validation_ms
        return path, self.search_ms, iteration

    def plan(self, start, goal):
        began = time.perf_counter()
        self.rejected_exact = 0
        self.validation_ms = 0.0
        self.validation_checks = 0
        self.validated_candidates = 0
        if (not self.road.free(*start) or not self.road.free(*goal)
                or not self.grid.is_free(self.grid.to_cell(start))
                or not self.grid.is_free(self.grid.to_cell(goal))):
            return self._finish(None, began, 0)

        nodes = [tuple(start)]
        parents = [-1]
        for iteration in range(1, self.max_iterations + 1):
            target = tuple(goal) if self.rng.random() < self.goal_bias else (
                self.rng.uniform(self.road.bounds[0], self.road.bounds[2]),
                self.rng.uniform(self.road.bounds[1], self.road.bounds[3]))
            nearest_index = min(range(len(nodes)), key=lambda i: math.dist(nodes[i], target))
            nearest = nodes[nearest_index]
            distance = math.dist(nearest, target)
            if distance < 1e-10:
                continue
            scale = min(1.0, self.step_size / distance)
            new = (nearest[0] + (target[0] - nearest[0]) * scale,
                   nearest[1] + (target[1] - nearest[1]) * scale)
            if not grid_segment_free(self.grid, nearest, new):
                continue
            nodes.append(new)
            parents.append(nearest_index)
            if (math.dist(new, goal) <= self.step_size
                    and grid_segment_free(self.grid, new, goal)):
                path = self._candidate_path(nodes, parents, len(nodes) - 1, tuple(goal))
                valid, validation_ms, checks = validate_path(self.road, path)
                self.validation_ms += validation_ms
                self.validation_checks += checks
                self.validated_candidates += 1
                if valid:
                    return self._finish(path, began, iteration)
                # A grid approximation may admit an edge that fails exact
                # geometry. Continue searching; never report that unsafe path.
                self.rejected_exact += 1
        return self._finish(None, began, self.max_iterations)


def main():
    parser = argparse.ArgumentParser(description='RRT using shared occupancy grid and exact path validation')
    parser.add_argument('--scenario', type=str, help='Run only this scenario')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--resolution', type=float, default=0.25, help='Shared planning grid resolution (meters)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    for sid in ([args.scenario] if args.scenario else available_scenarios()):
        road, start, goal, name = scenario(sid)
        planner = RRTPlanner(road, seed=args.seed + sum(map(ord, str(sid))),
                             max_iterations=args.iterations, resolution=args.resolution)
        path, ms, iterations = planner.plan(start, goal)
        print(f'Scenario {sid}: {name}')
        print(f'  Found path: {path is not None} | Exact-geometry rejected candidates: {planner.rejected_exact}')
        print(f'  Grid build: {planner.grid_build_ms:.2f} ms | Search: {planner.search_ms:.2f} ms | Validation: {planner.validation_ms:.2f} ms')
        print(f'  Time to valid path (cold start): {planner.time_to_valid_ms:.2f} ms | Validated candidates: {planner.validated_candidates} | Valid: {path is not None}')
        print(f'  Iterations: {iterations} | Path length: {path_length(path):.2f} m')
        draw(road, start, goal, path, f'RRT | Scenario {sid}: {name}')
        if args.no_show:
            plt.close()
        else:
            plt.show()


if __name__ == '__main__':
    main()
