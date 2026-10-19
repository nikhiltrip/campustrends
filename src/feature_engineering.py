"""Features available at posting time and past-only engagement targets."""

import argparse
from datetime import date
import numpy as np
import pandas as pd

POSITIVE = {"love", "great", "good", "happy", "best", "amazing", "win", "excited"}
NEGATIVE = {"hate", "bad", "stress", "stressed", "worst", "broken", "tired", "awful"}
NUMERIC = [
    "hour_sin",
    "hour_cos",
    "day_sin",
    "day_cos",
    "year_sin",
    "year_cos",
    "week_of_term",
    "in_term",
    "post_length",
    "word_count",
    "sentiment_score",
]


def validate_calendar(calendar):
    if calendar is not None and not isinstance(calendar, list):
        raise ValueError("Academic calendar must be a JSON list of terms.")
    result = []
    for term in calendar or []:
        if not isinstance(term, dict) or not all(
            key in term for key in ("name", "start", "end")
        ):
            raise ValueError("Each term needs name, start, and end fields.")
        try:
            start, end = (
                date.fromisoformat(term["start"]),
                date.fromisoformat(term["end"]),
            )
        except (ValueError, TypeError):
            raise ValueError("Term dates must use YYYY-MM-DD.") from None
        if not isinstance(term["name"], str) or not term["name"].strip():
            raise ValueError("Each term needs a nonempty name.")
        if end < start:
            raise ValueError("Academic term end must follow its start.")
        if result and start <= date.fromisoformat(result[-1]["end"]):
            raise ValueError("Academic terms must be sorted and must not overlap.")
        result.append(
            {"name": term["name"], "start": start.isoformat(), "end": end.isoformat()}
        )
    return result


def features(df, calendar=None):
    out = df.copy()
    stamp = pd.to_datetime(out.timestamp, utc=True, format="mixed").dt.tz_convert(
        "America/Los_Angeles"
    )
    for field, values, cycle in [
        ("hour", stamp.dt.hour, 24),
        ("day", stamp.dt.dayofweek, 7),
        ("year", stamp.dt.dayofyear, 365.25),
    ]:
        out[field + "_sin"] = np.sin(2 * np.pi * values / cycle)
        out[field + "_cos"] = np.cos(2 * np.pi * values / cycle)
    out["hour_of_day"], out["day_of_week"] = stamp.dt.hour, stamp.dt.dayofweek
    out["week_of_term"], out["in_term"], out["term"] = 0, 0, "Outside configured terms"
    local_days = stamp.dt.tz_localize(None).dt.normalize()
    for term in validate_calendar(calendar):
        start, end = pd.Timestamp(term["start"]), pd.Timestamp(term["end"])
        mask = local_days.between(start, end)
        out.loc[mask, "week_of_term"] = (
            (local_days[mask] - start).dt.days // 7 + 1
        ).clip(upper=15)
        out.loc[mask, "in_term"] = 1
        out.loc[mask, "term"] = term["name"]
    out["post_length"] = out.text.str.len().clip(upper=20000) / 1000
    out["word_count"] = out.text.str.split().str.len().clip(upper=5000) / 100

    def sentiment(text):
        words = set(text.lower().split())
        pos, neg = len(words & POSITIVE), len(words & NEGATIVE)
        return (pos - neg) / max(1, pos + neg)

    out["sentiment_score"] = out.text.map(sentiment)
    return out


def label_engagement(df, days=7, min_history=20):
    """Compare observed upvotes to the prior 7 days, excluding equal timestamps.

    A target uses historical outcome counts, never the post's future neighbors.
    Assume each imported count has the same observation age (e.g. 48 hours).
    Importing unequal-age live counts would make this target misleading.
    """
    out = df.sort_values(["timestamp", "post_id"]).reset_index(drop=True).copy()
    series = out.set_index("timestamp").upvotes
    rolling = series.rolling(f"{days}D", closed="left", min_periods=min_history)
    out["engagement_threshold"] = rolling.quantile(0.90).to_numpy()
    out["history_count"] = (
        series.rolling(f"{days}D", closed="left")
        .count()
        .fillna(0)
        .to_numpy()
        .astype(int)
    )
    out["is_high_engagement"] = np.where(
        out.engagement_threshold.notna(),
        (out.upvotes > out.engagement_threshold).astype(float),
        np.nan,
    )
    return out


class FeatureEngineer:
    def __init__(self, engagement_threshold=0.10):
        if engagement_threshold != 0.10:
            raise ValueError(
                "This pipeline defines high engagement as the historical top decile."
            )

    def engineer_features(self, df, text_column="cleaned_text", extract_tfidf=False):
        if extract_tfidf:
            raise ValueError(
                "TF-IDF is fitted on the training split by EngagementModel, not on all posts."
            )
        out = df.copy()
        out["timestamp"] = pd.to_datetime(out.timestamp, utc=True, format="mixed")
        return label_engagement(features(out))

    def label_high_engagement(self, df):
        return label_engagement(df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    FeatureEngineer().engineer_features(pd.read_csv(args.input)).to_csv(
        args.output, index=False
    )
