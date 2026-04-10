"""Simulation engine for synchronized multi-player movement."""

from __future__ import annotations

from typing import Dict, List

import numpy as np

from trajectory import TimedTrajectory


def simulate_frames(trajectories: List[TimedTrajectory]) -> Dict:
    """Generate frame-by-frame player coordinates.

    Returns a dict with:
      - t: array of timestamps
      - frames: list of {player_id: (x, y)}
    """
    if not trajectories:
        return {"t": np.array([]), "frames": []}

    t = trajectories[0].t
    frames = []
    for i in range(len(t)):
        frame = {tr.player_id: (float(tr.positions[i, 0]), float(tr.positions[i, 1])) for tr in trajectories}
        frames.append(frame)

    return {"t": t, "frames": frames}
