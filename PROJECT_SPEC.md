# CampusTrendsML specification

## Purpose

Analyze anonymous campus post exports for UCLA to detect emerging conversations, rank likely high-engagement posts, and identify recurring high-performing patterns. Include posting-time and academic-calendar context. Run locally with open-source tools and no paid APIs.

## Current implementation

1. Validate `post_id`, `text`, `upvotes`, `timestamp`, `school`, and optional `platform`. Reject malformed rows and duplicate IDs; retain only UCLA posts.
2. Normalize text, strip URLs and handles, retain negation, and derive cyclical hour/day/year features, optional academic week, length, and a basic word-list sentiment feature.
3. Label posts whose observed upvotes exceed the 90th percentile in the preceding seven days. Exclude equal timestamps; require 20 earlier posts. The target assumes a consistent count observation age.
4. Split labeled timestamps into chronological training/validation/test periods with a two-day gap before the later periods. Fit preprocessing and LDA on training data only.
5. Train logistic regression and XGBoost with TF-IDF, numeric context, and topic proportions. Select by validation average precision; evaluate once on the held-out test period.
6. Compare topic shares in adjacent seven-day windows anchored to the dataset's latest timestamp. Report counts, denominators, share changes, and low-evidence states.
7. Fit K-means archetypes on high-engagement training vectors. Show keywords, representative training posts, and recent assignments/high-engagement rates.
8. Export report JSON, feature CSV, and a local joblib model. Serve a React dashboard with a loopback FastAPI service for uploads and draft scoring.

The dashboard includes Overview, Topics & trends, Post archetypes, Draft lab, Evaluation, and Dataset. The static build supports the sample report and report import/export; model training/scoring require the local Python service.

## Research targets and open work

The original design proposed transformer embeddings, BERTopic or LDA, topic-coherence studies, archetype stability, and optional generated examples. This version implements the local LDA/TF-IDF path. Transformer comparisons, quantified topic quality/stability, calibration, generation, and platform adapters remain future work.

A goal of greater than 70% precision among top-decile predictions remains a research target. It is not an achieved result. The repository contains only synthetic demonstration data, so no real UCLA prediction claim follows from its metrics.

## Scope

One campus, text and posting metadata, local research. No individual tracking, exact-upvote regression, paid APIs, live collection, or public multi-user backend. Free text can contain personal information; the importer cannot guarantee anonymity. Clean and permission source data before use. Read [README.md](README.md) for setup, schema, metric definitions, and limitations.
