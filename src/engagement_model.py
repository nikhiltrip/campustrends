"""Chronological model selection with train-only feature fitting."""

import math
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    precision_score,
    recall_score,
)
from xgboost import XGBClassifier
from .feature_engineering import NUMERIC


def chronological_split(df, embargo_days=2):
    times = np.sort(df.timestamp.unique())
    if len(times) < 30:
        raise ValueError(
            "Need at least 30 distinct posting timestamps for chronological evaluation."
        )
    a, b = times[int(len(times) * 0.60)], times[int(len(times) * 0.80)]
    gap = pd.Timedelta(days=embargo_days)
    masks = [
        df.timestamp < a - gap,
        (df.timestamp >= a) & (df.timestamp < b - gap),
        df.timestamp >= b,
    ]
    parts = [df[mask].copy() for mask in masks]
    if any(len(p) < 20 or p.is_high_engagement.nunique() < 2 for p in parts):
        raise ValueError(
            "Each chronological split needs at least 20 labeled posts and both engagement classes. Add more data across time."
        )
    return parts


def metrics(y, scores):
    y, scores = np.asarray(y, int), np.asarray(scores)
    k = max(1, math.ceil(len(y) * 0.10))
    ranked = np.argsort(-scores, kind="stable")[:k]
    predicted = scores >= 0.5
    return {
        "posts": len(y),
        "positive_rate": float(y.mean()),
        "precision_at_10": float(y[ranked].mean()),
        "top_k": k,
        "roc_auc": float(roc_auc_score(y, scores)) if len(np.unique(y)) == 2 else None,
        "average_precision": float(average_precision_score(y, scores)),
        "precision_at_half": float(precision_score(y, predicted, zero_division=0)),
        "recall_at_half": float(recall_score(y, predicted, zero_division=0)),
        "confusion_matrix": confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    }


class EngagementModel:
    def fit_features(self, df, topic_model):
        self.topic_model = topic_model
        self.vectorizer = TfidfVectorizer(
            max_features=4000, min_df=2, ngram_range=(1, 2), sublinear_tf=True
        )
        self.vectorizer.fit(df.cleaned_text)
        self.scaler = StandardScaler().fit(df[NUMERIC])
        self.features = (
            self.vectorizer.get_feature_names_out().tolist()
            + NUMERIC
            + [f"topic_{i}" for i in range(topic_model.model.n_components)]
        )
        return self

    def transform(self, df):
        return hstack(
            [
                self.vectorizer.transform(df.cleaned_text),
                csr_matrix(self.scaler.transform(df[NUMERIC])),
                csr_matrix(self.topic_model.transform(df.cleaned_text)),
            ],
            format="csr",
        )

    def fit(self, train, validation):
        x, y = self.transform(train), train.is_high_engagement.astype(int)
        candidates = {
            "Logistic regression": LogisticRegression(
                C=1, max_iter=1200, random_state=42
            ),
            "XGBoost": XGBClassifier(
                n_estimators=160,
                max_depth=3,
                learning_rate=0.04,
                subsample=0.85,
                colsample_bytree=0.8,
                reg_lambda=3,
                tree_method="hist",
                eval_metric="logloss",
                n_jobs=2,
                random_state=42,
            ),
        }
        self.validation = {}
        for name, candidate in candidates.items():
            candidate.fit(x, y)
            self.validation[name] = metrics(
                validation.is_high_engagement,
                candidate.predict_proba(self.transform(validation))[:, 1],
            )
        # Pick once on validation average precision. Never select using test metrics.
        self.name = max(
            self.validation, key=lambda name: self.validation[name]["average_precision"]
        )
        self.classifier = candidates[self.name]
        self.training_prior = float(y.mean())
        return self

    def predict(self, df):
        return self.classifier.predict_proba(self.transform(df))[:, 1]

    def top_features(self):
        weights = (
            self.classifier.coef_[0]
            if hasattr(self.classifier, "coef_")
            else self.classifier.feature_importances_
        )
        indices = np.argsort(weights)[-12:][::-1]
        return [
            {"feature": self.features[i], "weight": float(weights[i])} for i in indices
        ]
