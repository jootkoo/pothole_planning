import heapq
import math
import time

import numpy as np
import matplotlib.pyplot as plt

from matplotlib.patches import Circle, Rectangle


# ============================================================
# CONFIGURATION
# ============================================================

ROAD_LENGTH = 50.0
LANE_WIDTH = 3.6
NUM_LANES = 3
ROAD_WIDTH = LANE_WIDTH * NUM_LANES

GRID_RESOLUTION = 0.25

VEHICLE_WIDTH = 1.8
VEHICLE_LENGTH = 4.5

SAFETY_MARGIN = 0.25

TRAFFIC_WEIGHT = 5.0
TRAFFIC_INFLUENCE_RADIUS = 4.0

# Conservative circular vehicle footprint.
VEHICLE_RADIUS = math.hypot(
    VEHICLE_LENGTH / 2,
    VEHICLE_WIDTH / 2
)

# For initial 2D A*, use vehicle half-width plus margin
# for pothole clearance. Full footprint checks come later.
POTHOLE_CLEARANCE = VEHICLE_WIDTH / 2 + SAFETY_MARGIN


# ============================================================
# ROAD ENVIRONMENT
# ============================================================

class RoadEnvironment:

    def __init__(self):

        self.length = ROAD_LENGTH
        self.width = ROAD_WIDTH
        self.resolution = GRID_RESOLUTION

        self.cols = math.ceil(self.length / self.resolution)
        self.rows = math.ceil(self.width / self.resolution)

        self.grid = np.zeros(
            (self.rows, self.cols),
            dtype=np.uint8
        )

        self.traffic_cost = np.zeros(
            (self.rows, self.cols),
            dtype=float
        )

        self.potholes = []
        self.traffic_vehicles = []

    def world_to_grid(self, x, y):

        gx = int(x / self.resolution)
        gy = int(y / self.resolution)

        return gx, gy

    def grid_to_world(self, gx, gy):

        x = (gx + 0.5) * self.resolution
        y = (gy + 0.5) * self.resolution

        return x, y

    def add_pothole(self, x, y, radius, depth):

        pothole = {
            "x": x,
            "y": y,
            "radius": radius,
            "depth": depth
        }

        self.potholes.append(pothole)

        inflated_radius = radius + POTHOLE_CLEARANCE

        for gy in range(self.rows):
            for gx in range(self.cols):

                wx, wy = self.grid_to_world(gx, gy)

                distance = math.hypot(
                    wx - x,
                    wy - y
                )

                if distance <= inflated_radius:
                    self.grid[gy, gx] = 1

    def add_traffic_vehicle(self, x, y, width=1.8, length=4.5):

        vehicle = {
            "x": x,
            "y": y,
            "width": width,
            "length": length
        }

        self.traffic_vehicles.append(vehicle)

        half_width = width / 2
        half_length = length / 2

        # Conservative rectangular obstacle inflated by
        # the planning vehicle's circular footprint.
        inflation = VEHICLE_RADIUS + SAFETY_MARGIN

        for gy in range(self.rows):
            for gx in range(self.cols):

                wx, wy = self.grid_to_world(gx, gy)

                dx = abs(wx - x)
                dy = abs(wy - y)

                # Hard collision region.
                if (
                    dx <= half_length + inflation
                    and dy <= half_width + inflation
                ):
                    self.grid[gy, gx] = 1

                # Soft traffic penalty region.
                nearest_x = max(
                    dx - half_length,
                    0.0
                )

                nearest_y = max(
                    dy - half_width,
                    0.0
                )

                distance = math.hypot(
                    nearest_x,
                    nearest_y
                )

                if distance < TRAFFIC_INFLUENCE_RADIUS:

                    penalty = (
                        1.0
                        - distance / TRAFFIC_INFLUENCE_RADIUS
                    )

                    self.traffic_cost[gy, gx] = max(
                        self.traffic_cost[gy, gx],
                        penalty
                    )


# ============================================================
# A* PATHFINDING
# ============================================================

class AStarPlanner:

    def __init__(self, environment):

        self.env = environment

        self.movements = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1)
        ]

    def heuristic(self, node, goal):

        dx = (goal[0] - node[0]) * self.env.resolution
        dy = (goal[1] - node[1]) * self.env.resolution

        return math.hypot(dx, dy)

    def is_valid(self, node):

        x, y = node

        if x < 0 or x >= self.env.cols:
            return False

        if y < 0 or y >= self.env.rows:
            return False

        if self.env.grid[y, x] == 1:
            return False

        return True

    def reconstruct_path(self, came_from, current):

        path = [current]

        while current in came_from:

            current = came_from[current]
            path.append(current)

        path.reverse()

        return [
            self.env.grid_to_world(x, y)
            for x, y in path
        ]

    def plan(self, start_world, goal_world):

        start_time = time.perf_counter()

        start = self.env.world_to_grid(*start_world)
        goal = self.env.world_to_grid(*goal_world)

        if not self.is_valid(start):
            return None, 0.0, 0

        if not self.is_valid(goal):
            return None, 0.0, 0

        open_heap = []
        counter = 0

        heapq.heappush(
            open_heap,
            (self.heuristic(start, goal), counter, start)
        )

        came_from = {}

        g_score = {
            start: 0.0
        }

        expanded_nodes = 0

        while open_heap:

            current_f, _, current = heapq.heappop(open_heap)

            expected_f = (
                g_score[current]
                + self.heuristic(current, goal)
            )

            # Skip outdated queue entries.
            if current_f > expected_f + 1e-9:
                continue

            expanded_nodes += 1

            if current == goal:

                path = self.reconstruct_path(
                    came_from,
                    current
                )

                elapsed = (
                    time.perf_counter() - start_time
                ) * 1000

                return path, elapsed, expanded_nodes

            for dx, dy in self.movements:

                neighbor = (
                    current[0] + dx,
                    current[1] + dy
                )

                if not self.is_valid(neighbor):
                    continue

                # Prevent diagonal corner cutting.
                if dx != 0 and dy != 0:

                    side_a = (
                        current[0] + dx,
                        current[1]
                    )

                    side_b = (
                        current[0],
                        current[1] + dy
                    )

                    if (
                        not self.is_valid(side_a)
                        or not self.is_valid(side_b)
                    ):
                        continue

                movement_cost = (
                    self.env.resolution
                    * math.hypot(dx, dy)
                )

                nx, ny = neighbor

                traffic_penalty = (
                    TRAFFIC_WEIGHT
                    * self.env.traffic_cost[ny, nx]
                    * movement_cost
                )

                tentative_g = (
                    g_score[current]
                    + movement_cost
                    + traffic_penalty
                )

                if tentative_g < g_score.get(
                    neighbor,
                    math.inf
                ):

                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    f_score = (
                        tentative_g
                        + self.heuristic(neighbor, goal)
                    )

                    counter += 1

                    heapq.heappush(
                        open_heap,
                        (f_score, counter, neighbor)
                    )

        elapsed = (
            time.perf_counter() - start_time
        ) * 1000

        return None, elapsed, expanded_nodes


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_path_length(path):

    if path is None or len(path) < 2:
        return 0.0

    return sum(
        math.dist(path[i], path[i + 1])
        for i in range(len(path) - 1)
    )


def calculate_min_clearance(path, environment):

    if path is None:
        return None

    if not environment.potholes:
        return None

    minimum = math.inf

    for px, py in path:

        for pothole in environment.potholes:

            distance = math.hypot(
                px - pothole["x"],
                py - pothole["y"]
            )

            clearance = (
                distance
                - pothole["radius"]
                - POTHOLE_CLEARANCE
            )

            minimum = min(
                minimum,
                clearance
            )

    return minimum


# ============================================================
# VISUALIZATION
# ============================================================

def visualize(environment, path, start, goal, title):

    fig, ax = plt.subplots(figsize=(14, 5))

    # Road background.
    ax.add_patch(
        Rectangle(
            (0, 0),
            environment.length,
            environment.width,
            facecolor="#333333",
            edgecolor="black"
        )
    )

    # Lane dividers.
    for lane in range(1, NUM_LANES):

        y = lane * LANE_WIDTH

        ax.plot(
            [0, environment.length],
            [y, y],
            linestyle="--",
            color="white",
            linewidth=1.2,
            alpha=0.7
        )

    # Potholes.
    for pothole in environment.potholes:

        ax.add_patch(
            Circle(
                (pothole["x"], pothole["y"]),
                pothole["radius"],
                color="black",
                zorder=4
            )
        )

        ax.add_patch(
            Circle(
                (pothole["x"], pothole["y"]),
                pothole["radius"] + POTHOLE_CLEARANCE,
                fill=False,
                edgecolor="orange",
                linestyle=":",
                linewidth=1.5,
                zorder=3
            )
        )

    # Traffic vehicles.
    for vehicle in environment.traffic_vehicles:

        ax.add_patch(
            Rectangle(
                (
                    vehicle["x"] - vehicle["length"] / 2,
                    vehicle["y"] - vehicle["width"] / 2
                ),
                vehicle["length"],
                vehicle["width"],
                color="red",
                zorder=5
            )
        )

    # Planned path.
    if path is not None:

        xs = [point[0] for point in path]
        ys = [point[1] for point in path]

        ax.plot(
            xs,
            ys,
            color="cyan",
            linewidth=2.5,
            label="A* Path",
            zorder=6
        )

    # Start and goal.
    ax.scatter(
        start[0],
        start[1],
        color="lime",
        s=120,
        marker="o",
        label="Start",
        zorder=7
    )

    ax.scatter(
        goal[0],
        goal[1],
        color="yellow",
        s=150,
        marker="*",
        label="Goal",
        zorder=7
    )

    ax.set_xlim(0, environment.length)
    ax.set_ylim(0, environment.width)

    ax.set_aspect("equal")

    ax.set_xlabel("Road Distance (m)")
    ax.set_ylabel("Lateral Position (m)")

    ax.set_title(title)

    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# TEST SCENARIOS
# ============================================================

def create_scenario(scenario_id):

    env = RoadEnvironment()

    start = (2.0, 5.4)
    goal = (47.0, 5.4)

    if scenario_id == 1:

        title = "Scenario 1: Empty Road"

    elif scenario_id == 2:

        title = "Scenario 2: Single Pothole"

        env.add_pothole(
            x=22.0,
            y=5.4,
            radius=0.6,
            depth=0.10
        )

    elif scenario_id == 3:

        title = "Scenario 3: Pothole Near Road Boundary"

        env.add_pothole(
            x=22.0,
            y=1.0,
            radius=0.8,
            depth=0.12
        )

        start = (2.0, 1.8)
        goal = (47.0, 1.8)

    elif scenario_id == 4:

        title = "Scenario 4: Multiple Potholes"

        env.add_pothole(
            x=15.0,
            y=5.4,
            radius=0.7,
            depth=0.10
        )

        env.add_pothole(
            x=25.0,
            y=3.6,
            radius=0.8,
            depth=0.08
        )

        env.add_pothole(
            x=35.0,
            y=7.2,
            radius=0.9,
            depth=0.15
        )

    elif scenario_id == 5:

        title = "Scenario 5: Road Completely Blocked"

        # Multiple overlapping obstacles span the road.
        for y in np.arange(0.0, ROAD_WIDTH + 0.5, 0.8):

            env.add_pothole(
                x=25.0,
                y=float(y),
                radius=0.7,
                depth=0.15
            )

    elif scenario_id == 6:

        title = "Scenario 6: Pothole With Traffic"

        env.add_pothole(
            x=22.0,
            y=5.4,
            radius=0.7,
            depth=0.12
        )

        env.add_traffic_vehicle(
            x=22.0,
            y=9.0
        )

    else:
        raise ValueError("Unknown scenario ID")

    return env, start, goal, title


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    print("=" * 65)
    print("A* POTHOLE AVOIDANCE SIMULATOR")
    print("=" * 65)

    results = []

    for scenario_id in range(1, 7):

        env, start, goal, title = create_scenario(
            scenario_id
        )

        planner = AStarPlanner(env)

        path, planning_time, expanded_nodes = planner.plan(
            start,
            goal
        )

        success = path is not None

        path_length = calculate_path_length(path)

        clearance = calculate_min_clearance(
            path,
            env
        )

        print(f"\n{title}")

        print(f"Success: {success}")

        print(
            f"Planning Time: {planning_time:.3f} ms"
        )

        print(
            f"Expanded Nodes: {expanded_nodes}"
        )

        if success:

            print(
                f"Path Length: {path_length:.2f} m"
            )

            if clearance is not None:

                print(
                    f"Minimum Pothole Clearance: "
                    f"{clearance:.2f} m"
                )

        results.append({
            "scenario": scenario_id,
            "success": success,
            "planning_time_ms": planning_time,
            "expanded_nodes": expanded_nodes,
            "path_length_m": path_length,
            "minimum_clearance_m": clearance
        })

        visualize(
            env,
            path,
            start,
            goal,
            title
        )

    # Overall performance.
    success_count = sum(
        result["success"]
        for result in results
    )

    success_rate = (
        success_count / len(results)
    ) * 100

    average_time = np.mean([
        result["planning_time_ms"]
        for result in results
    ])

    print("\n" + "=" * 65)
    print("OVERALL RESULTS")
    print("=" * 65)

    print(
        f"Success Rate: {success_rate:.2f}%"
    )

    print(
        f"Average Planning Time: {average_time:.3f} ms"
    )

    print(
        f"Total Scenarios: {len(results)}"
    )


if __name__ == "__main__":
    main()