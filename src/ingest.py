"""Validate local post exports. No collection or network access."""

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd

REQUIRED = ("post_id", "text", "upvotes", "timestamp", "school")


class PostDataIngestor:
    def __init__(self, school_filter="UCLA"):
        self.school_filter = school_filter
        self.quality = {}

    def load_json(self, path):
        with open(path, encoding="utf-8") as handle:
            records = json.load(handle)
        if isinstance(records, dict):
            records = records.get("posts")
        if not isinstance(records, list):
            raise ValueError(
                "JSON must contain a list of posts or an object with a posts list."
            )
        return pd.DataFrame(records)

    def load_csv(self, path):
        return pd.read_csv(path, dtype={"post_id": "string"})

    def validate_schema(self, df):
        missing = sorted(set(REQUIRED) - set(df.columns))
        if missing:
            raise ValueError("Missing columns: " + ", ".join(missing))
        return True

    def clean_data(self, df):
        self.validate_schema(df)
        out = df[list(REQUIRED) + (["platform"] if "platform" in df else [])].copy()
        counts = {"input_rows": len(out)}
        # Drop reasons are sequential: each rejected row counts once.
        valid_text = out.text.map(
            lambda x: isinstance(x, str) and bool(x.strip()) and len(x) <= 20000
        )
        counts["invalid_text"] = int((~valid_text).sum())
        out = out[valid_text].copy()
        out["text"] = out.text.str.strip()
        valid_id = out.post_id.notna() & out.post_id.astype(str).str.strip().ne("")
        counts["missing_id"] = int((~valid_id).sum())
        out = out[valid_id].copy()
        out["post_id"] = out.post_id.astype(str).str.strip()
        votes = pd.to_numeric(out.upvotes, errors="coerce")
        valid_votes = (
            np.isfinite(votes) & votes.ge(0) & votes.le(1e9) & votes.mod(1).eq(0)
        )
        counts["invalid_upvotes"] = int((~valid_votes).sum())
        out = out[valid_votes].copy()
        out["upvotes"] = votes[valid_votes].astype(int)
        stamps = pd.to_datetime(
            out.timestamp, errors="coerce", utc=True, format="mixed"
        )
        counts["invalid_timestamp"] = int(stamps.isna().sum())
        out = out[stamps.notna()].copy()
        out["timestamp"] = stamps[stamps.notna()]
        selected = (
            out.school.astype(str)
            .str.strip()
            .str.casefold()
            .eq(self.school_filter.casefold())
        )
        counts["other_school"] = int((~selected).sum())
        out = out[selected].copy()
        out["school"] = self.school_filter
        counts["duplicate_id"] = int(out.post_id.duplicated().sum())
        out = out.drop_duplicates("post_id", keep="first")
        out["platform"] = (
            out.get("platform", pd.Series("unspecified", index=out.index))
            .fillna("unspecified")
            .astype(str)
            .str.slice(0, 80)
        )
        out = out.sort_values(["timestamp", "post_id"]).reset_index(drop=True)
        counts["accepted_rows"] = len(out)
        self.quality = counts
        if out.empty:
            raise ValueError("No valid UCLA posts remain after cleaning.")
        return out

    def ingest(self, input_path, output_path=None):
        path = Path(input_path)
        loader = self.load_csv if path.suffix.lower() == ".csv" else self.load_json
        df = self.clean_data(loader(path))
        if output_path:
            self.save_clean_data(df, output_path)
        return df

    @staticmethod
    def save_clean_data(df, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    ingestor = PostDataIngestor()
    ingestor.ingest(args.input, args.output)
    print(json.dumps(ingestor.quality, indent=2))
