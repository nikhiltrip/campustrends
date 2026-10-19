"""Local LDA topics and volume-normalized, adjacent-window comparisons."""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS


class TopicModel:
    def __init__(self, n_topics=6):
        topic_stopwords = sorted(
            set(ENGLISH_STOP_WORDS)
            | {
                "anyone",
                "campus",
                "today",
                "weekend",
                "question",
                "asking",
                "friend",
                "just",
                "think",
                "update",
                "honest",
                "tips",
                "else",
                "like",
                "new",
                "good",
                "great",
                "best",
                "really",
                "does",
                "anybody",
                "near",
                "looking",
                "finally",
                "getting",
                "day",
            }
        )
        self.vectorizer = CountVectorizer(
            max_features=4000, min_df=2, max_df=0.80, stop_words=topic_stopwords
        )
        self.model = LatentDirichletAllocation(
            n_components=n_topics,
            max_iter=15,
            random_state=42,
            learning_method="batch",
            n_jobs=1,
        )

    def fit(self, texts):
        matrix = self.vectorizer.fit_transform(texts)
        if matrix.shape[1] < 3:
            raise ValueError(
                "Not enough distinct words to discover topics. Supply more varied posts."
            )
        self.model.fit(matrix)
        vocab = self.vectorizer.get_feature_names_out()
        self.topics = []
        for i, weights in enumerate(self.model.components_):
            keywords = vocab[np.argsort(weights)[-7:][::-1]].tolist()
            self.topics.append(
                {"id": i, "name": " / ".join(keywords[:2]), "keywords": keywords}
            )
        return self

    def transform(self, texts):
        return self.model.transform(self.vectorizer.transform(texts))


def trend_report(df, topics, window_days=7, as_of=None):
    # Half-open windows end just after the latest post, so that it is included.
    end = (
        pd.Timestamp(as_of)
        if as_of
        else df.timestamp.max() + pd.Timedelta(microseconds=1)
    )
    recent_start = end - pd.Timedelta(days=window_days)
    previous_start = recent_start - pd.Timedelta(days=window_days)
    recent = df[(df.timestamp >= recent_start) & (df.timestamp < end)]
    previous = df[(df.timestamp >= previous_start) & (df.timestamp < recent_start)]
    result = []
    for topic in topics:
        now = recent[recent.topic_id == topic["id"]]
        old = previous[previous.topic_id == topic["id"]]
        now_share, old_share = (
            len(now) / max(1, len(recent)),
            len(old) / max(1, len(previous)),
        )
        result.append(
            {
                **topic,
                "recent_count": len(now),
                "previous_count": len(old),
                "recent_share": now_share,
                "previous_share": old_share,
                "share_change": now_share - old_share,
                "status": "insufficient data"
                if len(recent) < 20 or len(previous) < 20 or len(now) < 5
                else (
                    "new in window"
                    if not len(old)
                    else ("rising" if now_share - old_share >= 0.03 else "steady")
                ),
                "recent_high_count": int(now.is_high_engagement.fillna(0).sum()),
                "examples": now.tail(2).text.tolist(),
            }
        )
    return {
        "window_days": window_days,
        "end": end.isoformat(),
        "recent_start": recent_start.isoformat(),
        "previous_start": previous_start.isoformat(),
        "recent_total": len(recent),
        "previous_total": len(previous),
        "topics": sorted(result, key=lambda t: t["share_change"], reverse=True),
    }
