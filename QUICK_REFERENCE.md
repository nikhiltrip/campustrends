# Quick reference

```bash
# Generate fictional sample posts, train, and export a report/model.
.venv/bin/python run_pipeline.py --sample

# Analyze your own export with optional term dates.
.venv/bin/python run_pipeline.py --input data/raw/posts.csv --calendar examples/calendar.json --output-dir reports/my-analysis

# Build and run the local dashboard.
./scripts/start.sh

# Verify the implementation.
.venv/bin/python -m pytest -q
npm --prefix frontend run build
```

Read [README.md](README.md) for the target definition, time-based splits, input schema, limitations, and local/static differences. The example calendar contains illustrative dates; replace them for real data.
