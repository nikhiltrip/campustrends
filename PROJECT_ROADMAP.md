# CampusTrendsML roadmap

## Implemented

- CSV/JSON cleaning with rejected-row counts
- Text/time/academic-week features and past-only seven-day engagement labels
- Train-only LDA topic discovery with topic-share changes across adjacent weeks
- Logistic regression and XGBoost comparison on chronological validation data
- Held-out evaluation, saved model, and reproducible report export
- TF-IDF/K-means archetypes fitted on high-engagement training posts
- React dashboard, local upload analysis, post inspection, draft scoring, report import/export
- Synthetic sample data, regression tests, and local setup instructions

## Next research work

1. Obtain a permissioned UCLA dataset with comparable post-age engagement measurements.
2. Review topics and archetype labels with real examples; measure coherence and stability.
3. Run repeated chronological backtests and examine performance by source and term.
4. Compare LDA/TF-IDF with local sentence embeddings or BERTopic.
5. Calibrate and validate scores before interpreting them as probabilities.
6. Consider optional local template/LLM generation after validating the analysis.

Live collection, user tracking, and cross-campus prediction remain outside the current scope.
