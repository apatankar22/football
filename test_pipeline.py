"""Simple smoke test for the full MVP pipeline using synthetic input."""

from __future__ import annotations

import json
from pathlib import Path

from main import create_synthetic_route_image, run_pipeline


def test_synthetic_pipeline(tmp_dir: str = ".tmp_test") -> None:
    out = Path(tmp_dir)
    out.mkdir(parents=True, exist_ok=True)

    image_path = out / "synthetic.png"
    anim_path = out / "synthetic.gif"
    json_path = out / "synthetic.json"

    create_synthetic_route_image(str(image_path))
    ir = run_pipeline(
        image_path=str(image_path),
        output_path=str(anim_path),
        expected_players=5,
        fps=20,
        speed=7.0,
        duration=4.0,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(ir, f, indent=2)

    assert anim_path.exists(), "Animation file was not created"
    assert len(ir["players"]) >= 4, "Expected at least 4 player routes"


if __name__ == "__main__":
    test_synthetic_pipeline()
    print("Synthetic pipeline smoke test passed")
