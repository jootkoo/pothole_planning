"""Bicycle-model Dynamic Window Approach with an A* global reference path.

DWA is a LOCAL trajectory sampler. A* supplies route guidance around dead ends.
"""
import argparse
import math
import time
import numpy as np
import matplotlib.pyplot as plt
import sys
from pathlib import Path

# Works when run from any working directory; common.py is in the parent folder.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import Road, draw, path_length, scenario, segment_free
from planning_grid import PlanningGrid, grid_astar
from geometry_validation import validate_path

DT = 0.25
HORIZON = 2.0
WHEELBASE = 2.7
MAX_SPEED = 2.5
MAX_ACCEL = 1.5
MAX_STEER = math.radians(35)
MAX_STEER_RATE = math.radians(90)


def angle_wrap(a):
    return math.atan2(math.sin(a), math.cos(a))


def advance(state, speed, steer, dt):
    x, y, theta, _, _ = state
    theta_next = angle_wrap(theta + speed / WHEELBASE * math.tan(steer) * dt)
    return (x + speed * math.cos(theta + theta_next) * dt / 2,
            y + speed * math.sin(theta + theta_next) * dt / 2,
            theta_next, speed, steer)


class DWAPlanner:
    def __init__(self, road: Road, planning_resolution=0.25, field_resolution=0.10):
        self.road = road
        self.planning_resolution = planning_resolution
        began = time.perf_counter()
        self.global_grid = PlanningGrid(road, planning_resolution)
        self.local_grid = PlanningGrid(road, field_resolution, with_distance=True)
        self.grid_build_ms = (time.perf_counter() - began) * 1000

    def local_step(self, state, target):
        """Sample reachable (speed, steering angle) commands and roll them out."""
        v_min = max(0.0, state[3] - MAX_ACCEL * DT)
        v_max = min(MAX_SPEED, state[3] + MAX_ACCEL * DT)
        s_min = max(-MAX_STEER, state[4] - MAX_STEER_RATE * DT)
        s_max = min(MAX_STEER, state[4] + MAX_STEER_RATE * DT)
        speeds = np.linspace(v_min, v_max, 5)
        steers = np.linspace(s_min, s_max, 11)
        best = None
        best_score = math.inf
        for v in speeds:
            for steer in steers:
                test = state
                min_clearance = math.inf
                safe = True
                for _ in range(round(HORIZON / DT)):
                    test = advance(test, float(v), float(steer), DT)
                    clearance = self.local_grid.sample_distance(test[0], test[1])
                    # The grid is an approximation. Confirm actual geometry
                    # near occupied cells and whenever the field reports danger.
                    if clearance < 0 or (clearance < 0.6 and not self.road.free(test[0], test[1])):
                        safe = False
                        break
                    min_clearance = min(min_clearance, clearance)
                if not safe:
                    continue
                distance = math.dist(test[:2], target)
                desired_heading = math.atan2(target[1] - test[1], target[0] - test[0])
                heading_error = abs(angle_wrap(desired_heading - test[2]))
                # End-point distance, heading, slower speeds, and obstacle proximity.
                score = (2.5 * distance + .65 * heading_error +
                         .65 * (MAX_SPEED - v) + .12 / (min_clearance + .15))
                if score < best_score:
                    best_score = score
                    best = (float(v), float(steer))
        return best

    def plan(self, start, goal, max_steps=550):
        began = time.perf_counter()
        self.validation_ms = 0.0
        self.validation_checks = 0
        self.search_ms = 0.0
        self.validated_candidates = 0
        reference, _ = grid_astar(self.global_grid, start, goal)
        if reference is not None:
            valid, ms, checks = validate_path(self.road, reference)
            self.validation_ms += ms
            self.validation_checks += checks
            self.validated_candidates += 1
            if not valid:
                reference = None
        if reference is None:
            return self._finish(None, None, began, 0)
        state = (start[0], start[1], 0.0, 0.0, 0.0)
        executed = [start]
        progress = 0
        for step in range(max_steps):
            if math.dist(state[:2], goal) < .7:
                valid, ms, checks = validate_path(self.road, executed)
                self.validation_ms += ms
                self.validation_checks += checks
                self.validated_candidates += 1
                return self._finish(executed if valid else None, reference, began, step)
            near = min(range(progress, len(reference)),
                       key=lambda i: math.dist(state[:2], reference[i]))
            progress = max(progress, near)
            lookahead = progress
            while (lookahead < len(reference) - 1 and
                   math.dist(reference[progress], reference[lookahead]) < 3.5):
                lookahead += 1
            target = reference[lookahead]
            command = self.local_step(state, target)
            if command is None:
                return self._finish(None, reference, began, step)
            next_state = advance(state, *command, DT)
            # Safety-critical online validation is measured separately too.
            valid, ms, checks = validate_path(self.road, [state[:2], next_state[:2]])
            self.validation_ms += ms
            self.validation_checks += checks
            if not valid:
                return self._finish(None, reference, began, step)
            state = next_state
            executed.append(state[:2])
        return self._finish(None, reference, began, max_steps)

    def _finish(self, path, reference, began, steps):
        self.search_ms = max(0.0, (time.perf_counter() - began) * 1000 - self.validation_ms)
        self.time_to_valid_ms = self.grid_build_ms + self.search_ms + self.validation_ms
        return path, reference, self.search_ms, steps


def main():
    parser = argparse.ArgumentParser(description='DWA pothole avoidance simulator')
    parser.add_argument('--scenario', type=str)
    parser.add_argument('--steps', type=int, default=550)
    parser.add_argument('--resolution', type=float, default=0.25, help='Global A* planning resolution (m)')
    parser.add_argument('--field-resolution', type=float, default=0.10, help='DWA distance field resolution (m)')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if args.resolution <= 0 or args.field_resolution <= 0:
        parser.error("resolutions must be positive")
    from common import available_scenarios
    for sid in ([args.scenario] if args.scenario else available_scenarios()):
        road, start, goal, name = scenario(sid)
        planner = DWAPlanner(road, args.resolution, args.field_resolution)
        path, reference, ms, steps = planner.plan(start, goal, args.steps)
        print(f'Scenario {sid}: {name}')
        print(f'  Found path: {path is not None}')
        print(f'  Grid build: {planner.grid_build_ms:.2f} ms | Search/control: {planner.search_ms:.2f} ms | Validation: {planner.validation_ms:.2f} ms')
        print(f'  Time to valid path (cold start): {planner.time_to_valid_ms:.2f} ms | Steps: {steps} | Path length: {path_length(path):.2f}m | Valid: {path is not None}')
        draw(road, start, goal, path, f'DWA | Scenario {sid}: {name}', reference=reference)
        if args.no_show:
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    main()
