"""Train local models and export a self-contained dashboard report."""

import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from .ingest import PostDataIngestor
from .preprocess import TextPreprocessor
from .feature_engineering import features, label_engagement, validate_calendar
from .topic_modeling import TopicModel, trend_report
from .engagement_model import EngagementModel, chronological_split, metrics
from .archetypes import ArchetypeModel

SCHEMA_VERSION = 1


def run_pipeline(input_path, output_dir, calendar=None, provenance="user-supplied"):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ingestor = PostDataIngestor()
    df = ingestor.ingest(input_path)
    # Only our generated fixture declares synthetic provenance. Uploaded metadata is ignored by the API.
    calendar = validate_calendar(calendar)
    df = TextPreprocessor().preprocess_dataframe(df)
    df = features(df, calendar)
    df = label_engagement(df)
    labeled = df[df.is_high_engagement.notna()].copy()
    train, validation, test = chronological_split(labeled)
    topics = TopicModel().fit(train.cleaned_text)
    df["topic_id"] = topics.transform(df.cleaned_text).argmax(axis=1)
    predictor = EngagementModel().fit_features(train, topics).fit(train, validation)
    test_scores = predictor.predict(test)
    trend = trend_report(df, topics.topics)
    archetypes = ArchetypeModel().fit(train, predictor)
    split_ids = {
        p: set(part.post_id)
        for p, part in [("train", train), ("validation", validation), ("test", test)]
    }
    df["split"] = df.post_id.map(
        lambda post_id: next(
            (name for name, ids in split_ids.items() if post_id in ids),
            "warm-up / embargo",
        )
    )
    df["score"] = np.nan
    # Scores in the explorer are held-out test scores only, not in-sample fit scores.
    df.loc[df.post_id.isin(test.post_id), "score"] = test_scores
    days = (
        df.assign(
            date=df.timestamp.dt.tz_convert("America/Los_Angeles").dt.strftime(
                "%Y-%m-%d"
            )
        )
        .groupby(["date", "topic_id"])
        .size()
    )
    timeline = [
        {"date": date, "topic_id": int(topic), "count": int(count)}
        for (date, topic), count in days.items()
    ]
    posts = []
    for row in df.itertuples():
        posts.append(
            {
                "id": row.post_id,
                "text": row.text,
                "upvotes": int(row.upvotes),
                "timestamp": row.timestamp.isoformat(),
                "platform": row.platform,
                "topic_id": int(row.topic_id),
                "term": row.term,
                "week_of_term": int(row.week_of_term),
                "high_engagement": bool(row.is_high_engagement)
                if pd.notna(row.is_high_engagement)
                else None,
                "threshold": float(row.engagement_threshold)
                if pd.notna(row.engagement_threshold)
                else None,
                "split": row.split,
                "score": float(row.score) if pd.notna(row.score) else None,
            }
        )
    report = {
        "schema_version": SCHEMA_VERSION,
        "title": "CampusTrendsML",
        "provenance": provenance,
        "is_sample": provenance == "synthetic",
        "school": "UCLA",
        "data_start": df.timestamp.min().isoformat(),
        "data_end": df.timestamp.max().isoformat(),
        "calendar": calendar,
        "quality": ingestor.quality,
        "summary": {
            "total_posts": len(df),
            "labeled_posts": len(labeled),
            "high_posts": int(labeled.is_high_engagement.sum()),
            "topic_count": len(topics.topics),
            "platforms": sorted(df.platform.unique().tolist()),
        },
        "trends": trend,
        "timeline": timeline,
        "posts": posts,
        "archetypes": archetypes.summarize(df, pd.Timestamp(trend["recent_start"])),
        "evaluation": {
            "selected_model": predictor.name,
            "selection_metric": "validation average precision",
            "validation": predictor.validation,
            "test": metrics(test.is_high_engagement, test_scores),
            "baseline": metrics(
                test.is_high_engagement, np.full(len(test), predictor.training_prior)
            ),
            "splits": [
                {
                    "name": name,
                    "count": len(part),
                    "start": part.timestamp.min().isoformat(),
                    "end": part.timestamp.max().isoformat(),
                    "positive_rate": float(part.is_high_engagement.mean()),
                }
                for name, part in [
                    ("train", train),
                    ("validation", validation),
                    ("test", test),
                ]
            ],
            "embargo_days": 2,
            "feature_weights": predictor.top_features(),
            "limitations": [
                "Synthetic results demonstrate software behavior, not predictive accuracy on campus posts."
                if provenance == "synthetic"
                else "One chronological holdout does not establish future accuracy.",
                "Collect upvotes at a consistent observation age, such as 48 hours. Snapshot exports with unequal ages bias the target.",
                "Scores are uncalibrated model outputs, not a verified probability of going viral.",
                "LDA keyword topics and TF-IDF archetypes need human interpretation. They do not establish student group membership.",
                "The small word-list sentiment feature does not understand sarcasm.",
            ],
        },
    }
    artifact = {
        "predictor": predictor,
        "calendar": calendar,
        "provenance": provenance,
        "report": report,
    }
    joblib.dump(artifact, output_dir / "model.joblib")
    df.to_csv(output_dir / "features.csv", index=False)
    (output_dir / "report.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def score_post(artifact, text, timestamp):
    if not isinstance(text, str) or not 5 <= len(text.strip()) <= 20000:
        raise ValueError("Draft text must contain 5 to 20,000 characters.")
    stamp = pd.Timestamp(timestamp)
    if stamp.tzinfo is None:
        raise ValueError("Choose a posting time with a time zone.")
    df = pd.DataFrame([{"text": text.strip(), "timestamp": stamp.tz_convert("UTC")}])
    df = features(TextPreprocessor().preprocess_dataframe(df), artifact["calendar"])
    predictor = artifact["predictor"]
    weights = predictor.topic_model.transform(df.cleaned_text)[0]
    topic = predictor.topic_model.topics[int(weights.argmax())]
    vocab = predictor.vectorizer.vocabulary_
    tokens = df.cleaned_text.iloc[0].split()
    known = sum(word in vocab for word in tokens)
    return {
        "score": float(predictor.predict(df)[0]),
        "topic": topic,
        "term": df.term.iloc[0],
        "week_of_term": int(df.week_of_term.iloc[0]),
        "known_word_fraction": known / max(1, len(tokens)),
        "is_sample": artifact["provenance"] == "synthetic",
        "model": predictor.name,
        "outside_training_dates": not (
            pd.Timestamp(artifact["report"]["evaluation"]["splits"][0]["start"])
            <= stamp
            <= pd.Timestamp(artifact["report"]["evaluation"]["splits"][0]["end"])
        ),
        "note": "Uncalibrated model score. A sample-trained model cannot predict real campus engagement."
        if artifact["provenance"] == "synthetic"
        else "Uncalibrated model score. Topic shifts and unseen wording can reduce reliability.",
    }
