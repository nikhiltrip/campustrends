# CampusTrendsML

I built CampusTrendsML to study how campus conversations change: which topics are growing, which kinds of posts gain attention, and how those patterns depend on timing. The project processes local exports of anonymous UCLA posts from Reddit-style or Yik Yak-style sources.

The repo now includes an end-to-end Python pipeline and a React dashboard. **The included demo uses 1,512 fictional posts and artificial upvotes. No real campus dataset or live platform connection is included.** Sample metrics demonstrate the software, not real-world predictive performance.

## What works

- Clean CSV/JSON exports, validate timestamps and upvotes, remove duplicate IDs, and report rejected rows.
- Discover six keyword topics with LDA and compare each topic's share of posts across adjacent seven-day windows.
- Train logistic regression and XGBoost on text, time-of-day, season, optional academic-week features, and topic proportions. Select a model on chronological validation data and evaluate it on a later test period.
- Cluster high-engagement training posts with TF-IDF and K-means to inspect recurring post archetypes and their recent frequency.
- Explore posts, topics, trends, and evaluation results in a responsive dashboard. Filter by topic, source, date window, or text, and inspect individual posts.
- Score a draft and posting time with the saved model in the local app. Scores are uncalibrated model outputs, not verified probabilities of virality.
- Export a report, reopen it in the browser, or analyze a new local CSV/JSON file.

The interface uses a newspaper-style masthead, navy type, paper background, and cobalt charts. It has no external font or chart-service dependency.

## Run the dashboard

Requires Python 3.11+ and Node.js 22.12+. XGBoost needs an OpenMP runtime. On macOS, install it with `brew install libomp` before running Python.

```bash
git clone https://github.com/nikhiltrip/campustrends.git
cd campustrends
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
npm --prefix frontend ci
./scripts/start.sh
```

Open **http://127.0.0.1:8031/**. The startup script generates the sample analysis if needed, builds the dashboard, and serves it with the local Python API. The app listens on this computer only. Stop it with Ctrl+C.

For frontend development, run the Python app above and `npm --prefix frontend run dev` in another terminal. Vite uses port 8030 and proxies API requests to 8031.

### Use your own data

```bash
python run_pipeline.py --input data/raw/posts.csv --calendar examples/calendar.json --output-dir reports/my-analysis
CAMPUS_REPORT_DIR=reports/my-analysis .venv/bin/python -m uvicorn src.api:app --host 127.0.0.1 --port 8031
```

The calendar is optional. Omit `--calendar` if you do not have verified term dates; the pipeline disables academic-week features. `examples/calendar.json` contains **illustrative demo dates**, not an authoritative UCLA calendar. Provide dates from the academic calendar for your dataset.

The dashboard can also analyze uploads smaller than 5 MB / 10,000 rows. It keeps the uploaded analysis in memory until reset or restart. Export its report to preserve it; use the CLI to retain a trained model and features. The CLI needs enough history for all three chronological splits to contain both classes; the app gives a validation error if the dataset is too small.

Required columns:

```json
{
  "post_id": "anonymous-post-001",
  "text": "Where are people studying for finals?",
  "upvotes": 42,
  "timestamp": "2026-06-01T18:00:00-07:00",
  "school": "UCLA",
  "platform": "local-export"
}
```

`platform` is optional. JSON accepts an array of posts or `{ "posts": [...] }`. CSV uses the same column names. The importer keeps only these fields. Supply timestamps with a timezone; naive timestamps currently use UTC. Only nonnegative integer upvote counts qualify. Reddit net scores that can be negative are not interchangeable with this schema's upvote count; do not silently substitute them.

Use upvote counts collected at a consistent post age, such as 48 hours. A fresh post and a week-old post have had different opportunities to collect votes. The pipeline cannot correct unequal-age snapshots after import.

## Target and evaluation

A post qualifies as high engagement when its observed upvotes **exceed** the 90th percentile of counts from posts in the preceding seven days. The reference window excludes equal-time and future posts. At least 20 historical posts are required; earlier posts have no target. Ties count as negative. This rolling-reference definition does not force exactly 10% of the dataset to be positive.

The pipeline divides labeled timestamps into approximately 60% training, 20% validation, and 20% test, with a two-day gap before validation and test for a 48-hour outcome horizon. It fits vocabulary, numeric scaling, LDA, and archetype clusters on training data only. It trains two classifiers and selects one using validation average precision. The final model does not refit on validation or test data.

Reports include precision among the highest-scored 10% of test posts, average precision, ROC-AUC, base positive rate, and a confusion matrix at a 0.5 threshold. A constant-prior baseline supplies a reference; its top-decile ranking breaks ties by row order. Use the positive rate as the expected precision of random selection.

Trending topics compare **shares of total posts**, not raw counts: a topic rising from 20% to 30% gains 10 percentage points. A rising flag requires at least 20 posts in each window, five topic posts in the recent window, and a three-point share increase. New topics receive a separate status. The windows end at the latest dataset timestamp, not the computer's current date. Keyword topic labels need human review. They do not establish membership in actual campus subcultures.

The original specification's 70% precision goal remains an unverified research target. Real predictive usefulness needs permissioned data, consistent observation horizons, multiple time-based backtests, source-specific analysis, and human review of topic quality. This implementation uses local LDA and TF-IDF rather than transformer embeddings or BERTopic. Short posts, sarcasm, repeated text, and platform differences can distort results. The small sentiment word list is only a basic feature.

## Static preview

```bash
npm --prefix frontend run build
```

Serve `dist/` with a static web server. It includes only the fictional sample report, filters, charts, and report import/export. Raw-data training and draft scoring require the local Python app. Opening a saved report does not load a model; analyze its source data locally to enable draft scoring.

Private preview: https://nikhil-campus-trends.cometpond2.chatgpt.site

Do not copy a real-data report into `frontend/public/`. Reports contain post text. Regenerate the committed fixture with `.venv/bin/python scripts/export_demo.py`; this command generates synthetic source data itself.

## Verify

```bash
.venv/bin/python -m pytest -q
npm --prefix frontend test
npm --prefix frontend run build
python run_pipeline.py --sample --output-dir reports/latest
```

Tests cover invalid data, historical-window boundaries, duplicate timestamps, time zones, train-only features, saved model predictions, volume-normalized trends, API upload analysis, and local API access checks.

## Files

| Path | Purpose |
| --- | --- |
| `run_pipeline.py` | CLI for sample or imported data |
| `src/ingest.py`, `preprocess.py` | Input validation and text cleanup |
| `src/feature_engineering.py` | Past-only targets and posting-time features |
| `src/topic_modeling.py` | Train-fitted LDA and weekly topic comparisons |
| `src/engagement_model.py` | Temporal splitting, classification, and metrics |
| `src/archetypes.py` | High-performer clustering |
| `src/pipeline.py`, `api.py` | Report/model export and local dashboard API |
| `frontend/` | React/TypeScript interface |
| `reports/` | Ignored local outputs and model artifacts |
| `notebooks/` | Exploration and evaluation of locally generated reports |

## Data handling and remaining work

The app does not collect posts, track users, or request platform credentials. Remove identifying information from free text before importing; dropping username columns does not anonymize the text itself. Treat saved reports as sensitive if the source content is sensitive. Load only model artifacts you created locally because joblib files can execute code.

Not implemented: Reddit/Yik Yak collection adapters, transformer embeddings, quantitative topic-coherence or archetype-stability studies, model calibration, automated post generation, and public multi-user hosting. The core scope remains one campus and local research. Optional template/local-LLM generation from the original plan can follow after real-data validation.

Method references: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html), [time-aware evaluation](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split), [XGBoost Python API](https://xgboost.readthedocs.io/en/stable/python/python_api.html).
