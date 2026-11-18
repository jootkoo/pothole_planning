"""Identical continuous-geometry validation for A*, RRT, and A*+DWA.

Uses the original Road.free() model via common.segment_free().
Sampling is not a mathematically exact swept-volume collision guarantee.
"""
import time
from common import segment_free

VALIDATION_STEP = 0.12

def validate_path(road, path, step=VALIDATION_STEP):
    """Return (is_valid, elapsed_ms, checked_edges) for one candidate path."""
    began = time.perf_counter()
    checked = 0
    valid = path is not None and len(path) >= 2
    if valid:
        for a, b in zip(path, path[1:]):
            checked += 1
            if not segment_free(road, a, b, step=step):
                valid = False
                break
    return bool(valid), (time.perf_counter() - began) * 1000, checked
