"""Trajectory generation from route polylines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np

Point = Tuple[float, float]


@dataclass
class TimedTrajectory:
    """Per-player trajectory with uniform-speed interpolation over global timeline."""

    player_id: str
    t: np.ndarray
    positions: np.ndarray  # shape: [N, 2]

    def position(self, time_s: float) -> Point:
        """Get interpolated position at time t."""
        x = np.interp(time_s, self.t, self.positions[:, 0])
        y = np.interp(time_s, self.t, self.positions[:, 1])
        return float(x), float(y)


def _polyline_length(path: np.ndarray) -> float:
    if len(path) < 2:
        return 0.0
    seg = np.diff(path, axis=0)
    return float(np.sum(np.linalg.norm(seg, axis=1)))


def _resample_polyline(path: np.ndarray, num_samples: int) -> np.ndarray:
    """Resample polyline by arc length into evenly spaced samples."""
    if len(path) == 0:
        return np.zeros((num_samples, 2), dtype=np.float32)
    if len(path) == 1:
        return np.repeat(path.astype(np.float32), num_samples, axis=0)

    distances = np.zeros(len(path), dtype=np.float32)
    distances[1:] = np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    total = distances[-1]
    if total <= 1e-6:
        return np.repeat(path[:1].astype(np.float32), num_samples, axis=0)

    targets = np.linspace(0, total, num_samples)
    x = np.interp(targets, distances, path[:, 0])
    y = np.interp(targets, distances, path[:, 1])
    return np.column_stack([x, y]).astype(np.float32)


def build_trajectories(
    players_json: Dict,
    speed_units_per_s: float = 7.0,
    duration_s: float | None = None,
    fps: int = 30,
) -> List[TimedTrajectory]:
    """Create time-based trajectories for all players on one shared time axis."""
    raw_paths: List[Tuple[str, np.ndarray]] = []
    route_durations: List[float] = []

    for p in players_json["players"]:
        path = np.array(p["path"], dtype=np.float32)
        if len(path) == 0:
            path = np.array([p["start"]], dtype=np.float32)
        length = _polyline_length(path)
        d = max(0.1, length / max(speed_units_per_s, 1e-6))
        raw_paths.append((p["id"], path))
        route_durations.append(d)

    if duration_s is None:
        # Clamp to required MVP window.
        duration_s = float(np.clip(max(route_durations) if route_durations else 3.0, 3.0, 5.0))

    frame_count = max(2, int(duration_s * fps))
    t = np.linspace(0.0, duration_s, frame_count)

    trajectories: List[TimedTrajectory] = []
    for (pid, path), route_time in zip(raw_paths, route_durations):
        route_frames = max(2, int((route_time / duration_s) * frame_count))
        motion = _resample_polyline(path, route_frames)
        if route_frames < frame_count:
            # Player waits at final point after completing route.
            end = np.repeat(motion[-1][None, :], frame_count - route_frames, axis=0)
            pos = np.vstack([motion, end])
        else:
            pos = motion[:frame_count]
        trajectories.append(TimedTrajectory(player_id=pid, t=t, positions=pos.astype(np.float32)))

    return trajectories
