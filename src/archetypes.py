"""Cluster training-set high performers and summarize later occurrences."""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances


class ArchetypeModel:
    def fit(self, train, engagement_model):
        self.vectorizer = engagement_model.vectorizer
        high = train[train.is_high_engagement == 1].copy()
        matrix = self.vectorizer.transform(high.cleaned_text)
        self.model = KMeans(
            n_clusters=min(4, max(1, len(high) // 8)), random_state=42, n_init=10
        ).fit(matrix)
        vocab = self.vectorizer.get_feature_names_out()
        self.groups = []
        for i, centroid in enumerate(self.model.cluster_centers_):
            keywords = vocab[np.argsort(centroid)[-5:][::-1]].tolist()
            members = np.flatnonzero(self.model.labels_ == i)
            if not len(members):
                continue
            closest = members[
                np.argsort(
                    pairwise_distances(matrix[members], centroid.reshape(1, -1)).ravel()
                )[:2]
            ]
            self.groups.append(
                {
                    "id": i,
                    "name": " / ".join(keywords[:2]),
                    "keywords": keywords,
                    "training_high_posts": len(members),
                    "examples": high.iloc[closest].text.tolist(),
                }
            )
        return self

    def summarize(self, df, recent_start):
        assignments = self.model.predict(self.vectorizer.transform(df.cleaned_text))
        result = []
        for group in self.groups:
            recent = df[(assignments == group["id"]) & (df.timestamp >= recent_start)]
            count = int(recent.is_high_engagement.notna().sum())
            result.append(
                {
                    **group,
                    "recent_posts": len(recent),
                    "recent_labeled_posts": count,
                    "recent_high_posts": int(recent.is_high_engagement.fillna(0).sum()),
                    "recent_high_rate": float(recent.is_high_engagement.mean())
                    if count
                    else None,
                }
            )
        return result
