"""Shared precomputed configuration-space grid and obstacle distance field.

Cells represent whether the *entire axis-aligned ego footprint*, including the
Road safety margin, is collision-free at the cell center. This is NOT the raw
point-occupancy layer from Road.grid_layers(). The vehicle's heading is not
modeled in the footprint, matching the supplied Road implementation.
"""
import heapq
import math
import numpy as np


class PlanningGrid:
    def __init__(self, road, resolution=0.25, with_distance=False):
        if not (math.isfinite(resolution) and resolution > 0):
            raise ValueError('Planning resolution must be finite and positive')
        self.road = road
        self.resolution = float(resolution)
        self.xmin, self.ymin, xmax, ymax = road.bounds
        self.cols = math.ceil((xmax - self.xmin) / self.resolution)
        self.rows = math.ceil((ymax - self.ymin) / self.resolution)
        self.free = np.zeros((self.rows, self.cols), dtype=np.bool_)
        self.traffic = np.zeros((self.rows, self.cols), dtype=np.float32)
        for j in range(self.rows):
            for i in range(self.cols):
                x, y = self.to_world((i, j))
                self.free[j, i] = road.free(x, y)
                if self.free[j, i]:
                    self.traffic[j, i] = road.traffic_cost(x, y)
        self.distance = self._distance_field() if with_distance else None

    def in_bounds(self, cell):
        i, j = cell
        return 0 <= i < self.cols and 0 <= j < self.rows

    def to_cell(self, point):
        return (math.floor((point[0] - self.xmin) / self.resolution),
                math.floor((point[1] - self.ymin) / self.resolution))

    def to_world(self, cell):
        i, j = cell
        return (self.xmin + (i + .5) * self.resolution,
                self.ymin + (j + .5) * self.resolution)

    def is_free(self, cell):
        return self.in_bounds(cell) and bool(self.free[cell[1], cell[0]])

    def edge_free(self, a, b):
        """Cheap no-corner-cut check; validate the returned path continuously."""
        if not self.is_free(a) or not self.is_free(b):
            return False
        dx, dy = b[0] - a[0], b[1] - a[1]
        if abs(dx) == 1 and abs(dy) == 1:
            return self.is_free((a[0] + dx, a[1])) and self.is_free((a[0], a[1] + dy))
        return True

    def _distance_field(self):
        """8-neighbor distance transform in meters, to occupied *cell centers*.

        Distances are conservative for collision-free scoring after subtracting
        a cell-diagonal allowance. This is not exact Euclidean clearance.
        """
        dist = np.full(self.free.shape, np.inf, dtype=np.float32)
        queue = []
        for j, i in np.argwhere(~self.free):
            dist[j, i] = 0.0
            heapq.heappush(queue, (0.0, int(i), int(j)))
        moves = [(dx, dy, self.resolution * math.hypot(dx, dy))
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
        while queue:
            d, i, j = heapq.heappop(queue)
            if d > float(dist[j, i]) + 1e-5:
                continue
            for dx, dy, cost in moves:
                ni, nj = i + dx, j + dy
                if 0 <= ni < self.cols and 0 <= nj < self.rows:
                    nd = d + cost
                    if nd + 1e-5 < dist[nj, ni]:
                        dist[nj, ni] = nd
                        heapq.heappush(queue, (nd, ni, nj))
        return dist

    def sample_distance(self, x, y):
        """Fast conservative estimate of vehicle-footprint clearance.

        For cells near hazards, the caller must confirm using road.free().
        Unknown/outside-map positions are unsafe.
        """
        if self.distance is None:
            raise RuntimeError('Distance field was not precomputed')
        cell = self.to_cell((x, y))
        if not self.is_free(cell):
            return -1.0
        i, j = cell
        # Distance to an occupied center minus maximum center-to-point and
        # occupied-cell half diagonal. Used as a heuristic, NOT safety proof.
        return max(0.0, float(self.distance[j, i]) - math.sqrt(2) * self.resolution)


def grid_astar(grid, start, goal, traffic_weight=5.0):
    """A* using precomputed grid lookups. Returns physical endpoints + cell centers.

    A returned route is validated against continuous Road geometry. If the
    coarse route is invalid it returns None, rather than reporting an unsafe path.
    """
    from common import segment_free
    road = grid.road
    s, g = grid.to_cell(start), grid.to_cell(goal)
    if not grid.is_free(s) or not grid.is_free(g):
        return None, 0
    if not (road.free(*start) and road.free(*goal)
            and segment_free(road, start, grid.to_world(s))
            and segment_free(road, grid.to_world(g), goal)):
        return None, 0
    def heuristic(c):
        return math.dist(grid.to_world(c), grid.to_world(g))
    queue = [(heuristic(s), 0, s)]
    best = {s: 0.0}
    parents = {}
    counter = expanded = 0
    moves = [(dx, dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1) if dx or dy]
    while queue:
        f, _, cur = heapq.heappop(queue)
        if f > best[cur] + heuristic(cur) + 1e-8:
            continue
        expanded += 1
        if cur == g:
            nodes = [cur]
            while nodes[-1] in parents:
                nodes.append(parents[nodes[-1]])
            nodes.reverse()
            path = [start] + [grid.to_world(n) for n in nodes] + [goal]
            # Continuous validation guards against coarse-grid aliasing.
            if all(segment_free(road, a, b) for a, b in zip(path, path[1:])):
                return path, expanded
            return None, expanded
        for dx, dy in moves:
            nb = (cur[0] + dx, cur[1] + dy)
            if not grid.edge_free(cur, nb):
                continue
            step = grid.resolution * math.hypot(dx, dy)
            cost = best[cur] + step * (1 + traffic_weight * float(grid.traffic[nb[1], nb[0]]))
            if cost < best.get(nb, math.inf):
                best[nb] = cost
                parents[nb] = cur
                counter += 1
                heapq.heappush(queue, (cost + heuristic(nb), counter, nb))
    return None, expanded
