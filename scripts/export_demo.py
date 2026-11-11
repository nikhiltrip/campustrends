"""Regenerate the committed dashboard fixture from synthetic posts only."""

import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.pipeline import run_pipeline
from src.sample_data import generate_sample, CALENDAR


def export_demo(destination=ROOT / "frontend/public/demo-report.json"):
    with tempfile.TemporaryDirectory(prefix="campus-synthetic-") as directory:
        root = Path(directory)
        generate_sample(root / "posts.json")
        report = run_pipeline(
            root / "posts.json", root / "analysis", CALENDAR, "synthetic"
        )
        if not report["is_sample"] or any(
            not p["id"].startswith("sample-") for p in report["posts"]
        ):
            raise ValueError(
                "The public fixture must contain generated sample posts only."
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(
            f"Exported {report['summary']['total_posts']} fictional posts to {destination}"
        )


if __name__ == "__main__":
    export_demo()
