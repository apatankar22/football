"""Image processing utilities for extracting flag-football routes from a diagram image.

MVP approach (classical CV only):
1) Detect marker-like player starts (circles/X blobs) using connected components.
2) Detect route ink via threshold + morphology + thinning-like contour tracing.
3) Split route network into path candidates and associate each player with nearest path.
4) Order points in each route path from player's start using nearest-neighbor walk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence, Tuple

import cv2
import numpy as np


Point = Tuple[float, float]


@dataclass
class PlayerRoute:
    """Structured route representation for one player."""

    player_id: str
    start: Point
    path: List[Point]


@dataclass
class DetectionResult:
    """Output of image processing stage."""

    players: List[Point]
    route_paths: List[np.ndarray]
    assignments: List[PlayerRoute]


def load_image(path: str) -> np.ndarray:
    """Load BGR image from disk."""
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Could not read image at '{path}'")
    return image


def _binary_ink_mask(image: np.ndarray) -> np.ndarray:
    """Create robust binary mask for dark route drawing and markers."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Adaptive threshold handles uneven lighting in scanned/phone images.
    thr = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31,
        C=7,
    )
    # Morphological cleanup for noisy lines.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(thr, cv2.MORPH_OPEN, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=1)
    return cleaned


def detect_player_starts(image: np.ndarray, expected_min: int = 4, expected_max: int = 5) -> List[Point]:
    """Detect likely player start markers using connected components heuristics."""
    mask = _binary_ink_mask(image)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)

    candidates: List[Tuple[float, float, float]] = []
    h, w = mask.shape
    image_area = h * w
    for i in range(1, num_labels):
        x, y, bw, bh, area = stats[i]
        if area < 20 or area > image_area * 0.02:
            continue
        aspect = bw / max(bh, 1)
        if not 0.4 <= aspect <= 2.5:
            continue
        # Compactness proxy to prefer marker-like blobs over long route segments.
        component = (labels == i).astype(np.uint8)
        component_contours, _ = cv2.findContours(component, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not component_contours:
            continue
        perimeter = cv2.arcLength(component_contours[0], True)
        circularity = (4 * np.pi * area) / (perimeter * perimeter + 1e-8)
        score = abs(circularity - 0.6)  # circles/X tend to have medium-high circularity
        cx, cy = centroids[i]
        candidates.append((score, float(cx), float(cy)))

    # Best N candidates. Fallback to strongest corners if too few detections.
    candidates.sort(key=lambda c: c[0])
    chosen = [(c[1], c[2]) for c in candidates[:expected_max]]

    if len(chosen) < expected_min:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners = cv2.goodFeaturesToTrack(gray, maxCorners=10, qualityLevel=0.02, minDistance=20)
        if corners is not None:
            for c in corners[:, 0, :]:
                pt = (float(c[0]), float(c[1]))
                if all(np.hypot(pt[0] - p[0], pt[1] - p[1]) > 15 for p in chosen):
                    chosen.append(pt)
                if len(chosen) >= expected_min:
                    break

    # Sort left-to-right for stable IDs.
    chosen.sort(key=lambda p: (p[0], p[1]))
    return chosen


def detect_route_lines(image: np.ndarray, min_length: int = 40) -> List[np.ndarray]:
    """Extract route polyline candidates from binary route mask via contours."""
    mask = _binary_ink_mask(image)
    # Emphasize line structure.
    edges = cv2.Canny(mask, 40, 120)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    paths: List[np.ndarray] = []
    for cnt in contours:
        if len(cnt) < min_length:
            continue
        epsilon = 1.5
        approx = cv2.approxPolyDP(cnt, epsilon, False)
        pts = approx[:, 0, :].astype(np.float32)
        if len(pts) >= 8:
            paths.append(pts)

    # Keep largest paths first.
    paths.sort(key=lambda p: cv2.arcLength(p.reshape(-1, 1, 2), False), reverse=True)
    return paths


def _order_route_points(start: Point, route: np.ndarray, step_limit: int = 400) -> List[Point]:
    """Order unordered contour points into a traversal path starting near the player."""
    points = route.copy()
    used = np.zeros(len(points), dtype=bool)
    start_arr = np.array(start, dtype=np.float32)
    current_idx = int(np.argmin(np.linalg.norm(points - start_arr, axis=1)))

    ordered: List[Point] = []
    for _ in range(min(step_limit, len(points))):
        used[current_idx] = True
        p = points[current_idx]
        ordered.append((float(p[0]), float(p[1])))

        remaining = np.where(~used)[0]
        if len(remaining) == 0:
            break

        dists = np.linalg.norm(points[remaining] - points[current_idx], axis=1)
        next_rel = int(np.argmin(dists))
        if dists[next_rel] > 30:  # stop if disconnected jump
            break
        current_idx = int(remaining[next_rel])

    # Light decimation to reduce jitter and keep path compact.
    simplified: List[Point] = []
    for p in ordered:
        if not simplified or np.hypot(p[0] - simplified[-1][0], p[1] - simplified[-1][1]) >= 4:
            simplified.append(p)
    return simplified


def associate_routes_to_players(player_starts: Sequence[Point], routes: Sequence[np.ndarray]) -> List[PlayerRoute]:
    """Associate each player to nearest unassigned route; fallback to nearest route if overlap."""
    assignments: List[PlayerRoute] = []
    unused = set(range(len(routes)))

    for i, start in enumerate(player_starts):
        best_idx = None
        best_dist = float("inf")
        candidate_pool = unused if unused else set(range(len(routes)))
        for r_idx in candidate_pool:
            route = routes[r_idx]
            d = np.min(np.linalg.norm(route - np.array(start, dtype=np.float32), axis=1))
            if d < best_dist:
                best_dist = float(d)
                best_idx = r_idx

        if best_idx is None:
            path = [start]
        else:
            path = _order_route_points(start, routes[best_idx])
            if best_idx in unused:
                unused.remove(best_idx)

        assignments.append(PlayerRoute(player_id=f"WR{i+1}", start=start, path=path))

    return assignments


def process_route_image(image_path: str, expected_players: int = 5) -> DetectionResult:
    """Main image-processing pipeline for route diagram extraction."""
    image = load_image(image_path)
    players = detect_player_starts(image, expected_min=max(1, expected_players - 1), expected_max=expected_players)
    routes = detect_route_lines(image)
    assignments = associate_routes_to_players(players, routes)
    return DetectionResult(players=players, route_paths=routes, assignments=assignments)


def to_json_dict(assignments: Sequence[PlayerRoute]) -> dict:
    """Convert assignments to the requested intermediate JSON structure."""
    return {
        "players": [
            {
                "id": a.player_id,
                "start": [float(a.start[0]), float(a.start[1])],
                "path": [[float(x), float(y)] for x, y in a.path],
            }
            for a in assignments
        ]
    }
