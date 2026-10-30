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

from common import Road, draw, path_length, reference_astar, scenario

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
    def __init__(self, road: Road):
        self.road = road

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
                    clearance = self.road.clearance(test[0], test[1])
                    if clearance < 0:
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
        reference = reference_astar(self.road, start, goal)
        if reference is None:
            return None, None, (time.perf_counter() - began) * 1000, 0
        state = (start[0], start[1], 0.0, 0.0, 0.0)
        executed = [start]
        progress = 0
        for step in range(max_steps):
            if math.dist(state[:2], goal) < .7:
                return executed, reference, (time.perf_counter() - began) * 1000, step
            # Never move the reference index backward; target ~3 m ahead.
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
                return None, reference, (time.perf_counter() - began) * 1000, step
            next_state = advance(state, *command, DT)
            # Dense check of executed segment; rollout checks are at DT intervals.
            from common import segment_free
            if not segment_free(self.road, state[:2], next_state[:2]):
                return None, reference, (time.perf_counter() - began) * 1000, step
            state = next_state
            executed.append(state[:2])
        return None, reference, (time.perf_counter() - began) * 1000, max_steps


def main():
    parser = argparse.ArgumentParser(description='DWA pothole avoidance simulator')
    parser.add_argument('--scenario', type=int, choices=range(1, 7))
    parser.add_argument('--steps', type=int, default=550)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    for sid in ([args.scenario] if args.scenario else range(1, 7)):
        road, start, goal, name = scenario(sid)
        path, reference, ms, steps = DWAPlanner(road).plan(start, goal, args.steps)
        print(f'Scenario {sid}: {name}')
        print(f'  Found path: {path is not None}')
        print(f'  Planning time: {ms:.1f} ms | Steps: {steps} | Path Length:{path_length(path):.2f}m')
        draw(road, start, goal, path, f'DWA | Scenario {sid}: {name}', reference=reference)
        if args.no_show:
            plt.close()
        else:
            plt.show()

if __name__ == '__main__':
    main()
