import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from src.ingest import PostDataIngestor
from src.preprocess import TextPreprocessor
from src.feature_engineering import label_engagement, features, validate_calendar
from src.engagement_model import chronological_split, metrics
from src.pipeline import run_pipeline, score_post
from src.sample_data import generate_sample, CALENDAR
from src.topic_modeling import trend_report


@pytest.fixture(scope="session")
def analysis(tmp_path_factory):
    root = tmp_path_factory.mktemp("analysis")
    path = root / "posts.json"
    generate_sample(path)
    report = run_pipeline(
        path, root / "output", calendar=CALENDAR, provenance="synthetic"
    )
    return root, report, joblib.load(root / "output/model.joblib")


def post(**kw):
    return {
        "post_id": "a",
        "text": "parking on campus",
        "upvotes": 2,
        "timestamp": "2026-01-01T12:00:00Z",
        "school": "UCLA",
        **kw,
    }


def test_ingest_rejects_invalid_and_deduplicates():
    records = [
        post(),
        post(),
        post(post_id="b", upvotes="bad"),
        post(post_id="c", timestamp="bad"),
        post(post_id="d", upvotes=np.inf),
        post(post_id="e", upvotes=2.3),
        post(post_id="f", text="  "),
        post(post_id="g", school="USC"),
        post(post_id=None),
        post(post_id="good", upvotes="7", timestamp="2026-01-02"),
    ]
    reader = PostDataIngestor()
    out = reader.clean_data(pd.DataFrame(records))
    assert out.post_id.tolist() == ["a", "good"]
    assert out.upvotes.tolist() == [2, 7]
    assert reader.quality["accepted_rows"] == 2
    assert (
        sum(
            n
            for k, n in reader.quality.items()
            if k not in ["input_rows", "accepted_rows"]
        )
        == 8
    )
    assert str(out.timestamp.dtype) == "datetime64[ns, UTC]"


def test_schema_and_empty_fail_actionably():
    with pytest.raises(ValueError, match="Missing columns"):
        PostDataIngestor().clean_data(pd.DataFrame([{"text": "only text"}]))
    with pytest.raises(ValueError, match="No valid"):
        PostDataIngestor().clean_data(pd.DataFrame([post(text="")]))


def test_text_cleanup_preserves_negation_and_unicode():
    value = TextPreprocessor().preprocess_text(
        "Not great! café @somebody https://example.com #finals"
    )
    assert "not" in value and "café" in value and "finals" in value
    assert "somebody" not in value and "https" not in value


def test_targets_only_prior_window_and_equal_times():
    df = pd.DataFrame(
        [
            post(
                post_id=str(i),
                upvotes=i,
                timestamp=pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(hours=i),
            )
            for i in range(25)
        ]
    )
    df["timestamp"] = pd.to_datetime(df.timestamp, utc=True)
    now = pd.Timestamp("2026-01-03", tz="UTC")
    extra = pd.DataFrame(
        [
            post(post_id="x", upvotes=100000, timestamp=now),
            post(post_id="y", upvotes=1, timestamp=now),
        ]
    )
    combined = pd.concat([df, extra], ignore_index=True)
    labeled = label_engagement(combined)
    a, b = labeled.tail(2).engagement_threshold
    assert a == b == pytest.approx(np.quantile(df.upvotes, 0.9))
    original = label_engagement(df)
    assert labeled.iloc[: len(df)].engagement_threshold.equals(
        original.engagement_threshold
    )
    assert original.iloc[:20].is_high_engagement.isna().all()
    assert original.iloc[20].is_high_engagement == 1


def test_window_excludes_posts_older_than_seven_days():
    frame = pd.DataFrame(
        [
            post(
                post_id=str(i),
                upvotes=100,
                timestamp=pd.Timestamp("2026-01-01", tz="UTC")
                + pd.Timedelta(minutes=i),
            )
            for i in range(21)
        ]
        + [post(post_id="late", timestamp=pd.Timestamp("2026-01-12", tz="UTC"))]
    )
    assert pd.isna(label_engagement(frame).iloc[-1].is_high_engagement)


def test_calendar_uses_los_angeles_date():
    frame = pd.DataFrame(
        [
            {"text": "great class", "timestamp": "2026-03-30T01:00:00Z"},
            {"text": "not good", "timestamp": "2026-04-06T10:00:00Z"},
        ]
    )
    out = features(frame, CALENDAR)
    assert out.week_of_term.tolist() == [0, 2]
    assert out.in_term.tolist() == [0, 1]
    assert out.sentiment_score.between(-1, 1).all()
    with pytest.raises(ValueError):
        validate_calendar([{"name": "bad", "start": "2026-02-03", "end": "2026-02-01"}])


def test_trend_share_is_volume_normalized():
    rows = []
    for day, n in [("2026-01-01", 20), ("2026-01-08", 40)]:
        rows.extend(
            {
                "timestamp": pd.Timestamp(day, tz="UTC"),
                "topic_id": i % 2,
                "is_high_engagement": 0,
                "text": "test post",
            }
            for i in range(n)
        )
    report = trend_report(
        pd.DataFrame(rows), [{"id": 0}, {"id": 1}], as_of="2026-01-09T00:00:00Z"
    )
    assert report["recent_total"] == 40 and report["previous_total"] == 20
    assert all(
        t["share_change"] == 0 and t["status"] == "steady" for t in report["topics"]
    )


def test_sample_pipeline_exports_and_model_reload(analysis):
    root, report, artifact = analysis
    assert report["is_sample"] and report["summary"]["total_posts"] == 1512
    assert sum(report["quality"][k] for k in ["accepted_rows"]) == 1512
    loaded = json.loads((root / "output/report.json").read_text())
    assert loaded == report
    frame = pd.read_csv(root / "output/features.csv")
    frame["timestamp"] = pd.to_datetime(frame.timestamp, utc=True)
    train, val, test = chronological_split(frame[frame.is_high_engagement.notna()])
    assert train.timestamp.max() + pd.Timedelta(days=2) < val.timestamp.min()
    assert val.timestamp.max() + pd.Timedelta(days=2) < test.timestamp.min()
    actual = artifact["predictor"].predict(test)
    assert metrics(test.is_high_engagement, actual) == report["evaluation"]["test"]
    assert np.isfinite(actual).all()
    assert np.allclose(actual, test.score)
    assert not frame[frame.split == "train"].score.notna().any()
    assert (
        "upvotes" not in artifact["predictor"].features
        and "is_high_engagement" not in artifact["predictor"].features
    )
    text = "zzzzfuturetoken parking campus"
    scored = score_post(artifact, text, "2026-10-18T18:00:00-07:00")
    assert 0 <= scored["score"] <= 1 and scored["is_sample"]
    assert "zzzzfuturetoken" not in artifact["predictor"].vectorizer.vocabulary_
    with pytest.raises(ValueError, match="time zone"):
        score_post(artifact, text, "2026-10-18T18:00:00")


def test_train_only_vocabulary(analysis):
    _, _, artifact = analysis
    predictor = artifact["predictor"]
    before = set(predictor.vectorizer.vocabulary_)
    frame = features(
        TextPreprocessor().preprocess_dataframe(
            pd.DataFrame(
                [{"text": "uniquewordonlyintest", "timestamp": "2026-10-18T12:00:00Z"}]
            )
        )
    )
    predictor.predict(frame)
    assert set(predictor.vectorizer.vocabulary_) == before
    assert "uniquewordonlyintest" not in predictor.topic_model.vectorizer.vocabulary_


def test_api_rejects_invalid_and_foreign_origins(analysis, monkeypatch):
    import src.api as api

    monkeypatch.setitem(api.state, "artifact", analysis[2])
    client = TestClient(api.app)
    assert client.get("/api/report").json()["is_sample"]
    assert (
        client.post(
            "/api/score",
            json={"text": "campus parking advice", "timestamp": "2026-10-18T12:00:00Z"},
        ).status_code
        == 200
    )
    assert (
        client.post(
            "/api/score", json={"text": "campus parking advice", "timestamp": "bad"}
        ).status_code
        == 422
    )
    assert (
        client.post("/api/analyze", files={"file": ("bad.exe", b"bad")}).status_code
        == 422
    )
    assert (
        client.post("/api/analyze", files={"file": ("bad.json", b"[]")}).status_code
        == 422
    )
    assert (
        client.post("/api/analyze", files={"file": ("bad.json", b"{")}).status_code
        == 422
    )
    assert (
        client.post(
            "/api/analyze", files={"file": ("large.csv", b"x" * (api.MAX_UPLOAD + 1))}
        ).status_code
        == 413
    )
    assert (
        client.get("/api/report", headers={"host": "evil.example"}).status_code == 403
    )
    assert (
        client.post(
            "/api/reset", headers={"origin": "https://evil.example"}
        ).status_code
        == 403
    )


def test_api_import_runs_real_pipeline(analysis, monkeypatch):
    import src.api as api

    monkeypatch.setitem(api.state, "artifact", analysis[2])
    client = TestClient(api.app)
    content = (analysis[0] / "posts.json").read_bytes()
    response = client.post(
        "/api/analyze",
        files={"file": ("posts.json", content)},
        data={"calendar": json.dumps(CALENDAR)},
    )
    assert response.status_code == 200, response.text
    assert response.json()["provenance"] == "user-supplied"
    assert client.get("/api/report").json()["summary"]["total_posts"] == 1512
    assert (
        client.post(
            "/api/score",
            json={
                "text": "Great game day Bruins basketball crowd",
                "timestamp": "2026-10-18T12:00:00Z",
            },
        ).status_code
        == 200
    )


def test_posting_features_ignore_outcomes(analysis):
    artifact = analysis[2]
    rows = pd.DataFrame([post(), post(post_id="different", upvotes=999999)])
    frame = features(TextPreprocessor().preprocess_dataframe(rows))
    assert (
        artifact["predictor"].predict(frame)[0]
        == artifact["predictor"].predict(frame)[1]
    )


@pytest.mark.parametrize(
    "calendar",
    [
        {},
        ["bad"],
        [{"name": "", "start": "2026-01-01", "end": "2026-01-02"}],
        [{"name": "bad", "start": "NaT", "end": "2026-01-02"}],
    ],
)
def test_invalid_calendars_raise(calendar):
    with pytest.raises(ValueError):
        validate_calendar(calendar)
