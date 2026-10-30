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

from common import ROAD_LENGTH, ROAD_WIDTH, RESOLUTION, draw, path_length, scenario, segment_free


class AStarPlanner:
    def __init__(self, road, resolution=RESOLUTION, traffic_weight=5.0):
        self.road = road
        self.resolution = resolution
        self.traffic_weight = traffic_weight
        self.cols = math.ceil(ROAD_LENGTH / resolution)
        self.rows = math.ceil(ROAD_WIDTH / resolution)
        self.moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if (dx, dy) != (0, 0)]

    def world_to_grid(self, point):
        return int(point[0] / self.resolution), int(point[1] / self.resolution)

    def grid_to_world(self, node):
        return ((node[0] + 0.5) * self.resolution, (node[1] + 0.5) * self.resolution)

    def heuristic(self, node, goal):
        return math.dist(self.grid_to_world(node), self.grid_to_world(goal))

    def valid(self, node):
        x, y = node
        return (0 <= x < self.cols and 0 <= y < self.rows
                and self.road.free(*self.grid_to_world(node)))

    def plan(self, start, goal):
        began = time.perf_counter()
        s, g = self.world_to_grid(start), self.world_to_grid(goal)
        if not self.valid(s) or not self.valid(g):
            return None, (time.perf_counter() - began) * 1000, 0
        # Ensure that the requested physical endpoints can reach their cell centers.
        if not (segment_free(self.road, start, self.grid_to_world(s))
                and segment_free(self.road, self.grid_to_world(g), goal)):
            return None, (time.perf_counter() - began) * 1000, 0

        heap = [(self.heuristic(s, g), 0, s)]
        best = {s: 0.0}
        parent = {}
        counter = expanded = 0
        while heap:
            f, _, cur = heapq.heappop(heap)
            if f > best[cur] + self.heuristic(cur, g) + 1e-9:
                continue  # stale queue entry
            expanded += 1
            if cur == g:
                nodes = [cur]
                while nodes[-1] in parent:
                    nodes.append(parent[nodes[-1]])
                nodes.reverse()
                path = [self.grid_to_world(n) for n in nodes]
                # Include requested physical endpoints if they differ from cell centers.
                path = [start] + path + [goal]
                return path, (time.perf_counter() - began) * 1000, expanded

            for dx, dy in self.moves:
                nb = (cur[0] + dx, cur[1] + dy)
                if not self.valid(nb):
                    continue
                # Continuous collision check also prevents diagonal corner cutting.
                if not segment_free(self.road, self.grid_to_world(cur), self.grid_to_world(nb)):
                    continue
                distance = self.resolution * math.hypot(dx, dy)
                traffic = self.road.traffic_cost(*self.grid_to_world(nb))
                candidate = best[cur] + distance * (1 + self.traffic_weight * traffic)
                if candidate < best.get(nb, math.inf):
                    best[nb] = candidate
                    parent[nb] = cur
                    counter += 1
                    heapq.heappush(heap, (candidate + self.heuristic(nb, g), counter, nb))
        return None, (time.perf_counter() - began) * 1000, expanded


def min_clearance(path, road):
    return min((road.clearance(*p) for p in path), default=None) if path else None


def main():
    parser = argparse.ArgumentParser(description='A* pothole avoidance using shared common.py')
    parser.add_argument('--scenario', type=int, choices=range(1, 7), help='Run one scenario (default: all)')
    parser.add_argument('--resolution', type=float, default=RESOLUTION)
    parser.add_argument('--traffic-weight', type=float, default=5.0)
    parser.add_argument('--no-plot', action='store_true', help='Run without displaying plots')
    args = parser.parse_args()
    if args.resolution <= 0 or args.traffic_weight < 0:
        parser.error('resolution must be positive and traffic-weight nonnegative')

    import matplotlib.pyplot as plt
    for number in ([args.scenario] if args.scenario else range(1, 7)):
        road, start, goal, name = scenario(number)
        planner = AStarPlanner(road, args.resolution, args.traffic_weight)
        path, elapsed_ms, expanded = planner.plan(start, goal)
        clearance = min_clearance(path, road)
        print(f'Scenario {number}: {name}')
        print(f'  Found path: {path is not None}')
        print(f'  Planning time: {elapsed_ms:.2f} ms | Expanded nodes: {expanded}')
        if path:
            print(f'  Path length: {path_length(path):.2f} m | Min approximate clearance: {clearance:.3f} m')
        if not args.no_plot:
            draw(road, start, goal, path, f'A* — Scenario {number}: {name}')
            plt.show()


if __name__ == '__main__':
    main()
