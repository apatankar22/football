"""End-to-end pipeline: image -> route extraction -> trajectory -> simulation -> animation."""

from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np

from image_processing import process_route_image, to_json_dict
from render import render_animation
from simulation import simulate_frames
from trajectory import build_trajectories


def create_synthetic_route_image(path: str, width: int = 900, height: int = 500) -> None:
    """Generate a dummy route diagram image for quick testing."""
    img = np.full((height, width, 3), 255, dtype=np.uint8)

    starts = [(120, 120), (120, 200), (120, 280), (120, 360), (120, 440)]

    # Draw player markers (mix of circles and X symbols)
    for i, (x, y) in enumerate(starts):
        if i % 2 == 0:
            cv2.circle(img, (x, y), 9, (0, 0, 0), 2)
        else:
            cv2.line(img, (x - 8, y - 8), (x + 8, y + 8), (0, 0, 0), 2)
            cv2.line(img, (x + 8, y - 8), (x - 8, y + 8), (0, 0, 0), 2)

    # Routes: slant, go, out, post, drag-like curve
    cv2.line(img, starts[0], (340, 60), (0, 0, 0), 2)
    cv2.line(img, starts[1], (120, 30), (0, 0, 0), 2)
    cv2.line(img, starts[2], (360, 280), (0, 0, 0), 2)
    cv2.line(img, (360, 280), (460, 240), (0, 0, 0), 2)
    cv2.line(img, starts[3], (320, 220), (0, 0, 0), 2)
    cv2.line(img, (320, 220), (470, 90), (0, 0, 0), 2)

    # Bezier-like polyline for curved route.
    curve_pts = np.array([[120, 440], [190, 430], [280, 380], [360, 320], [460, 310]], dtype=np.int32)
    cv2.polylines(img, [curve_pts], False, (0, 0, 0), 2)

    cv2.imwrite(path, img)


def run_pipeline(image_path: str, output_path: str, expected_players: int, fps: int, speed: float, duration: float | None):
    detection = process_route_image(image_path, expected_players=expected_players)
    json_ir = to_json_dict(detection.assignments)

    trajectories = build_trajectories(
        json_ir,
        speed_units_per_s=speed,
        duration_s=duration,
        fps=fps,
    )
    sim = simulate_frames(trajectories)
    render_animation(sim, output_path=output_path, fps=fps, show_trails=True)

    return json_ir


def main():
    parser = argparse.ArgumentParser(description="Flag football route diagram to animation MVP")
    parser.add_argument("--image", type=str, default="synthetic_routes.png", help="Input PNG/JPG diagram path")
    parser.add_argument("--output", type=str, default="routes.gif", help="Output animation path (.gif or .mp4)")
    parser.add_argument("--json", type=str, default="routes.json", help="Output structured JSON file")
    parser.add_argument("--expected-players", type=int, default=5)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--speed", type=float, default=7.0, help="Route speed in units/sec")
    parser.add_argument("--duration", type=float, default=None, help="Optional fixed duration in seconds")
    parser.add_argument("--make-synthetic", action="store_true", help="Create and use a synthetic test diagram")
    args = parser.parse_args()

    if args.make_synthetic or not os.path.exists(args.image):
        create_synthetic_route_image(args.image)

    json_ir = run_pipeline(
        image_path=args.image,
        output_path=args.output,
        expected_players=args.expected_players,
        fps=args.fps,
        speed=args.speed,
        duration=args.duration,
    )

    with open(args.json, "w", encoding="utf-8") as f:
        json.dump(json_ir, f, indent=2)

    print(f"Wrote JSON IR to {args.json}")
    print(f"Rendered animation to {args.output}")


if __name__ == "__main__":
    main()
