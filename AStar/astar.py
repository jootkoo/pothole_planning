"""Standalone A* planner using the shared Road model in ../common.py."""
import argparse
import heapq
import math
import sys
import time
from pathlib import Path

# Works when run from any working directory; common.py is in the parent folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import draw, path_length, scenario
from planning_grid import PlanningGrid, grid_astar
from geometry_validation import validate_path


class AStarPlanner:
    def __init__(self, road, resolution=0.25, traffic_weight=5.0):
        self.road = road
        self.resolution = resolution
        self.traffic_weight = traffic_weight
        # One-time precomputation; excluded from search time, reported separately.
        began = time.perf_counter()
        self.grid = PlanningGrid(road, resolution)
        self.grid_build_ms = (time.perf_counter() - began) * 1000

    def plan(self, start, goal):
        began = time.perf_counter()
        path, expanded = grid_astar(self.grid, start, goal, self.traffic_weight)
        self.search_ms = (time.perf_counter() - began) * 1000
        self.validation_ms = 0.0
        self.validation_checks = 0
        if path is not None:
            valid, self.validation_ms, self.validation_checks = validate_path(self.road, path)
            if not valid:
                path = None  # Never report a geometrically invalid path as successful.
        self.time_to_valid_ms = self.grid_build_ms + self.search_ms + self.validation_ms
        return path, self.search_ms, expanded


def min_clearance(path, road):
    return min((road.clearance(*p) for p in path), default=None) if path else None


def main():
    parser = argparse.ArgumentParser(description='A* pothole avoidance using shared common.py')
    parser.add_argument('--scenario', type=str, help='Run one scenario (default: all)')
    parser.add_argument('--resolution', type=float, default=0.25)
    parser.add_argument('--traffic-weight', type=float, default=5.0)
    parser.add_argument('--no-plot', action='store_true', help='Run without displaying plots')
    args = parser.parse_args()
    if args.resolution <= 0 or args.traffic_weight < 0:
        parser.error('resolution must be positive and traffic-weight nonnegative')

    import matplotlib.pyplot as plt
    from common import available_scenarios
    for number in ([args.scenario] if args.scenario else available_scenarios()):
        road, start, goal, name = scenario(number)
        planner = AStarPlanner(road, args.resolution, args.traffic_weight)
        path, elapsed_ms, expanded = planner.plan(start, goal)
        clearance = min_clearance(path, road)
        print(f'Scenario {number}: {name}')
        print(f'  Found path: {path is not None}')
        print(f'  Grid build: {planner.grid_build_ms:.2f} ms | Search: {planner.search_ms:.2f} ms | Validation: {planner.validation_ms:.2f} ms')
        print(f'  Time to valid path (cold start): {planner.time_to_valid_ms:.2f} ms | Expanded nodes: {expanded} | Valid: {path is not None}')
        if path:
            print(f'  Path length: {path_length(path):.2f} m | Min approximate clearance: {clearance:.3f} m')
        if not args.no_plot:
            draw(road, start, goal, path, f'A* — Scenario {number}: {name}')
            plt.show()


if __name__ == '__main__':
    main()
