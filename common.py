"""Shared static road scenarios, geometry, grid A* reference, and plotting."""
from dataclasses import dataclass, field
import heapq
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle

ROAD_LENGTH = 50.0
ROAD_WIDTH = 10.8
LANE_WIDTH = 3.6
RESOLUTION = 0.25
CAR_LENGTH = 4.5
CAR_WIDTH = 1.8
SAFETY_MARGIN = 0.25
# Conservative full-vehicle circular collision footprint for both planners.
CAR_RADIUS = math.hypot(CAR_LENGTH / 2, CAR_WIDTH / 2)

@dataclass
class Road:
    potholes: list = field(default_factory=list)
    cars: list = field(default_factory=list)

    def add_pothole(self, x, y, radius, depth=0.1):
        self.potholes.append((float(x), float(y), float(radius), float(depth)))

    def add_car(self, x, y, length=CAR_LENGTH, width=CAR_WIDTH):
        self.cars.append((float(x), float(y), float(length), float(width)))

    def clearance(self, x, y):
        """Approximate axis-aligned rectangular ego footprint, plus safety margin.

        This is a 2D planning approximation, not a heading-aware collision test.
        """
        hx = CAR_LENGTH / 2 + SAFETY_MARGIN
        hy = CAR_WIDTH / 2 + SAFETY_MARGIN
        d = min(y - hy, ROAD_WIDTH - y - hy, x - hx, ROAD_LENGTH - x - hx)
        for px, py, pr, _ in self.potholes:
            dx = max(abs(x - px) - hx, 0.0)
            dy = max(abs(y - py) - hy, 0.0)
            d = min(d, math.hypot(dx, dy) - pr)
        for cx, cy, length, width in self.cars:
            dx = abs(x - cx) - (length / 2 + hx)
            dy = abs(y - cy) - (width / 2 + hy)
            outside = math.hypot(max(dx, 0.0), max(dy, 0.0))
            inside = min(max(dx, dy), 0.0)
            d = min(d, outside + inside)
        return d

    def free(self, x, y):
        return self.clearance(x, y) >= 0

    def traffic_cost(self, x, y, radius=4.0):
        cost = 0.0
        for cx, cy, length, width in self.cars:
            dx = max(abs(x - cx) - length / 2, 0.0)
            dy = max(abs(y - cy) - width / 2, 0.0)
            dist = math.hypot(dx, dy)
            cost = max(cost, max(0.0, 1.0 - dist / radius))
        return cost


def scenario(number):
    road = Road()
    start, goal = (2.8, 5.4), (47.0, 5.4)
    names = {1: 'Empty road', 2: 'Single pothole', 3: 'Pothole near boundary',
             4: 'Multiple potholes', 5: 'Road blocked', 6: 'Pothole and traffic'}
    if number == 2:
        road.add_pothole(22, 5.4, .6)
    elif number == 3:
        road.add_pothole(22, 1.0, .8)
        # Keep center sufficiently far from boundary for conservative circular footprint.
        start, goal = (2.8, 3.0), (47.0, 3.0)
    elif number == 4:
        for x, y, r in [(15, 5.4, .7), (25, 3.6, .8), (35, 7.2, .9)]:
            road.add_pothole(x, y, r)
    elif number == 5:
        for y in np.arange(0, ROAD_WIDTH + .5, .8):
            road.add_pothole(25, y, .7)
    elif number == 6:
        road.add_pothole(22, 5.4, .7)
        road.add_car(22, 9.0)
    elif number != 1:
        raise ValueError('Scenario must be between 1 and 6')
    return road, start, goal, names[number]


def segment_free(road, a, b, step=0.12):
    dist = math.dist(a, b)
    for t in np.linspace(0, 1, max(2, math.ceil(dist / step) + 1)):
        if not road.free(a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])):
            return False
    return True


def reference_astar(road, start, goal, resolution=.5, traffic_weight=5.0):
    """Static global reference path used ONLY by the DWA local controller."""
    cols, rows = math.ceil(ROAD_LENGTH / resolution), math.ceil(ROAD_WIDTH / resolution)
    def cell(p):
        return (min(cols - 1, int(p[0] / resolution)), min(rows - 1, int(p[1] / resolution)))
    def world(c):
        return ((c[0] + .5) * resolution, (c[1] + .5) * resolution)
    s, g = cell(start), cell(goal)
    if not road.free(*world(s)) or not road.free(*world(g)):
        return None
    def h(c):
        return math.dist(world(c), world(g))
    queue = [(h(s), 0, s)]
    best, parent = {s: 0.0}, {}
    counter = 0
    while queue:
        f, _, cur = heapq.heappop(queue)
        if f > best[cur] + h(cur) + 1e-8:
            continue
        if cur == g:
            result = [world(cur)]
            while cur in parent:
                cur = parent[cur]
                result.append(world(cur))
            return result[::-1]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == dy == 0:
                    continue
                nb = (cur[0] + dx, cur[1] + dy)
                if not (0 <= nb[0] < cols and 0 <= nb[1] < rows):
                    continue
                if not segment_free(road, world(cur), world(nb)):
                    continue
                d = math.hypot(dx, dy) * resolution
                cost = best[cur] + d * (1 + traffic_weight * road.traffic_cost(*world(nb)))
                if cost < best.get(nb, math.inf):
                    best[nb], parent[nb] = cost, cur
                    counter += 1
                    heapq.heappush(queue, (cost + h(nb), counter, nb))
    return None


def path_length(path):
    return sum(math.dist(a, b) for a, b in zip(path, path[1:])) if path else 0.0


def draw(road, start, goal, path, title, reference=None):
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.add_patch(Rectangle((0, 0), ROAD_LENGTH, ROAD_WIDTH, color='#333333'))
    for y in (LANE_WIDTH, 2 * LANE_WIDTH):
        ax.axhline(y, linestyle='--', color='white', alpha=.65)
    for x, y, radius, _ in road.potholes:
        ax.add_patch(Circle((x, y), radius, color='black', zorder=4))
        ax.add_patch(Circle((x, y), radius + CAR_WIDTH / 2 + SAFETY_MARGIN,
                            fill=False, color='orange', linestyle=':', zorder=3))
    for x, y, length, width in road.cars:
        ax.add_patch(Rectangle((x - length / 2, y - width / 2), length, width,
                               color='red', zorder=4))
    if reference:
        ax.plot(*zip(*reference), linestyle='--', color='yellow', alpha=.65, label='A* reference')
    if path:
        ax.plot(*zip(*path), color='cyan', linewidth=2.2, label='Planned path', zorder=5)
    ax.scatter(*start, color='lime', s=85, label='Start', zorder=6)
    ax.scatter(*goal, color='yellow', marker='*', s=120, label='Goal', zorder=6)
    ax.set(xlim=(0, ROAD_LENGTH), ylim=(0, ROAD_WIDTH), xlabel='Longitudinal x (m)',
           ylabel='Lateral y (m)', title=title)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    fig.tight_layout()
    return fig
