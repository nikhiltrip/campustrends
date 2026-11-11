"""Run a local analysis: python run_pipeline.py --sample."""

import argparse
import json
from pathlib import Path
from src.pipeline import run_pipeline
from src.sample_data import generate_sample, CALENDAR

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample", action="store_true")
    source.add_argument(
        "--input", help="Local JSON or CSV with post_id,text,upvotes,timestamp,school"
    )
    parser.add_argument("--output-dir", default="reports/latest")
    parser.add_argument(
        "--calendar", help="JSON list of terms with name,start,end (Los Angeles dates)"
    )
    args = parser.parse_args()
    calendar = json.loads(Path(args.calendar).read_text()) if args.calendar else None
    if args.sample:
        input_path = Path(args.output_dir) / "sample_posts.json"
        generate_sample(input_path)
        calendar = calendar or CALENDAR
    else:
        input_path = args.input
    report = run_pipeline(
        input_path,
        args.output_dir,
        calendar=calendar,
        provenance="synthetic" if args.sample else "user-supplied",
    )
    print(
        json.dumps(
            {
                "posts": report["summary"]["total_posts"],
                "model": report["evaluation"]["selected_model"],
                "test": report["evaluation"]["test"],
                "report": str(Path(args.output_dir) / "report.json"),
                "synthetic": report["is_sample"],
            },
            indent=2,
        )
    )
