"""Standalone RRT planner on the six shared static pothole scenarios."""
import argparse
import math
import random
import time
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Works when run from any working directory; common.py is in the parent folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import ROAD_LENGTH, ROAD_WIDTH, Road, draw, path_length, scenario, segment_free

class RRTPlanner:
    def __init__(self, road: Road, step_size=.8, goal_bias=.15, max_iterations=12000, seed=42):
        self.road = road
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iterations = max_iterations
        self.rng = random.Random(seed)

    def plan(self, start, goal):
        started = time.perf_counter()
        if not self.road.free(*start) or not self.road.free(*goal):
            return None, (time.perf_counter() - started) * 1000, 0
        nodes = [start]
        parents = [-1]
        for iteration in range(1, self.max_iterations + 1):
            target = goal if self.rng.random() < self.goal_bias else (
                self.rng.uniform(0, ROAD_LENGTH), self.rng.uniform(0, ROAD_WIDTH))
            nearest_index = min(range(len(nodes)), key=lambda i: math.dist(nodes[i], target))
            nearest = nodes[nearest_index]
            distance = math.dist(nearest, target)
            if distance < 1e-10:
                continue
            scale = min(1.0, self.step_size / distance)
            new = (nearest[0] + (target[0] - nearest[0]) * scale,
                   nearest[1] + (target[1] - nearest[1]) * scale)
            if not segment_free(self.road, nearest, new):
                continue
            nodes.append(new)
            parents.append(nearest_index)
            if math.dist(new, goal) <= self.step_size and segment_free(self.road, new, goal):
                nodes.append(goal)
                parents.append(len(nodes) - 2)
                result = []
                idx = len(nodes) - 1
                while idx >= 0:
                    result.append(nodes[idx])
                    idx = parents[idx]
                return result[::-1], (time.perf_counter() - started) * 1000, iteration
        return None, (time.perf_counter() - started) * 1000, self.max_iterations


def main():
    parser = argparse.ArgumentParser(description='RRT pothole avoidance simulator')
    parser.add_argument('--scenario', type=int, choices=range(1, 7), help='Run only this scenario')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--iterations', type=int, default=12000)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    for sid in ([args.scenario] if args.scenario else range(1, 7)):
        road, start, goal, name = scenario(sid)
        planner = RRTPlanner(road, seed=args.seed + sid, max_iterations=args.iterations)
        path, ms, iterations = planner.plan(start, goal)
        print(f'Scenario {sid}: {name}')
        print(f'  Found path: {path is not None}')
        print(f'  Planning time: {ms:.1f} ms | Iterations: {iterations} | Path Length:{path_length(path):.2f}m')
        draw(road, start, goal, path, f'RRT | Scenario {sid}: {name}')
        if args.no_show:
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    main()
