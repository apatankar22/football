"""Rendering utilities for route animation output (GIF/MP4)."""

from __future__ import annotations

from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

try:
    from matplotlib.animation import FFMpegWriter
except Exception:  # pragma: no cover
    FFMpegWriter = None


Color = Tuple[float, float, float]


def _field_bounds(frames: List[Dict[str, Tuple[float, float]]], pad: float = 30.0):
    all_pts = np.array([pt for f in frames for pt in f.values()], dtype=np.float32)
    if len(all_pts) == 0:
        return 0, 100, 0, 53
    min_x, min_y = np.min(all_pts, axis=0)
    max_x, max_y = np.max(all_pts, axis=0)
    return min_x - pad, max_x + pad, min_y - pad, max_y + pad


def render_animation(
    sim: Dict,
    output_path: str,
    fps: int = 30,
    show_trails: bool = True,
    title: str = "Flag Football Route Simulation",
):
    """Render simulation frames to GIF/MP4."""
    t = sim["t"]
    frames = sim["frames"]
    if len(frames) == 0:
        raise ValueError("No frames to render")

    x0, x1, y0, y1 = _field_bounds(frames)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_title(title)
    ax.set_facecolor("#2d7f3f")
    ax.set_xlim(x0, x1)
    ax.set_ylim(y1, y0)  # top-down image coordinates
    ax.set_aspect("equal")

    # Field grid lines (simple MVP visualization)
    for frac in np.linspace(0, 1, 11):
        x = x0 + frac * (x1 - x0)
        ax.plot([x, x], [y0, y1], color="white", alpha=0.2, linewidth=1)

    player_ids = list(frames[0].keys())
    cmap = plt.get_cmap("tab10")
    colors: Dict[str, Color] = {pid: cmap(i % 10)[:3] for i, pid in enumerate(player_ids)}

    scatters = {
        pid: ax.plot([], [], marker="o", markersize=10, color=colors[pid], linestyle="None", label=pid)[0]
        for pid in player_ids
    }
    trails = {
        pid: ax.plot([], [], color=colors[pid], linewidth=1.8, alpha=0.9)[0]
        for pid in player_ids
    }

    ax.legend(loc="upper right")

    def update(frame_idx: int):
        current = frames[frame_idx]
        for pid in player_ids:
            x, y = current[pid]
            scatters[pid].set_data([x], [y])
            if show_trails:
                xs = [frames[j][pid][0] for j in range(frame_idx + 1)]
                ys = [frames[j][pid][1] for j in range(frame_idx + 1)]
                trails[pid].set_data(xs, ys)
        ax.set_xlabel(f"t = {t[frame_idx]:.2f}s")
        artists = list(scatters.values()) + list(trails.values())
        return artists

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)

    if output_path.lower().endswith(".gif"):
        anim.save(output_path, writer=PillowWriter(fps=fps))
    elif output_path.lower().endswith(".mp4") and FFMpegWriter is not None:
        anim.save(output_path, writer=FFMpegWriter(fps=fps, bitrate=1800))
    else:
        # Fallback if extension is unknown or ffmpeg unavailable.
        fallback = output_path.rsplit(".", 1)[0] + ".gif"
        anim.save(fallback, writer=PillowWriter(fps=fps))

    plt.close(fig)
