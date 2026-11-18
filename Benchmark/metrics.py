"""Shared path-quality metrics; only compute after planner has accepted geometry validation."""
from __future__ import annotations
import math

def path_length(path):
    return sum(math.dist(a[:2], b[:2]) for a, b in zip(path, path[1:])) if path else None

def minimum_clearance(path, road, spacing=0.10):
    if not path: return None
    best = math.inf
    for a, b in zip(path, path[1:]):
        n = max(1, math.ceil(math.dist(a[:2], b[:2]) / spacing))
        for i in range(n + 1):
            t = i / n
            best = min(best, road.clearance(a[0] + (b[0]-a[0])*t, a[1] + (b[1]-a[1])*t))
    if len(path) == 1: best = road.clearance(*path[0][:2])
    return best

def turn_metrics(path):
    headings = [math.atan2(b[1]-a[1], b[0]-a[0]) for a,b in zip(path,path[1:]) if math.dist(a[:2],b[:2])>1e-9]
    turns = [abs(math.atan2(math.sin(b-a), math.cos(b-a))) for a,b in zip(headings,headings[1:])]
    return sum(turns), max(turns,default=0.0)

def calculate_metrics(path, road):
    if not path: return dict(path_length_m=None,min_clearance_m=None,total_turn_rad=None,max_turn_rad=None,waypoints=None)
    total_turn,max_turn = turn_metrics(path)
    return dict(path_length_m=path_length(path), min_clearance_m=minimum_clearance(path,road),total_turn_rad=total_turn,max_turn_rad=max_turn,waypoints=len(path))
