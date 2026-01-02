# CampusTrendML Quick Reference

## Command Cheatsheet

### Pipeline Execution
```bash
# Full pipeline (recommended)
python run_pipeline.py --input data/raw/posts.json

# Individual stages
python src/ingest.py --input data/raw/posts.json --output data/processed/clean.csv
python src/preprocess.py --input data/processed/clean.csv --output data/processed/preprocessed.csv
python src/feature_engineering.py --input data/processed/preprocessed.csv --output data/processed/features.csv
```

### Jupyter Notebooks
```bash
# Start Jupyter
jupyter notebook

# Open exploration notebook
jupyter notebook notebooks/exploration.ipynb
```

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | Project overview and documentation |
| `PROJECT_SPEC.md` | Detailed technical specification |
| `GETTING_STARTED.md` | Setup and installation guide |
| `requirements.txt` | Python dependencies |
| `run_pipeline.py` | One-command pipeline runner |

## Data Schema

### Required Fields
```json
{
  "post_id": "string",      // Unique identifier
  "text": "string",          // Post content
  "upvotes": int,           // Engagement metric
  "timestamp": "ISO-8601",  // When posted
  "school": "UCLA"          // School name
}
```

### Generated Features

**Temporal Features:**
- `hour_of_day` (0-23)
- `day_of_week` (0-6, Mon=0)
- `week_of_year` (1-52)
- `is_weekend` (0 or 1)

**Text Features:**
- `char_count` (integer)
- `word_count` (integer)
- `sentiment_score` (-1.0 to +1.0)
- `cleaned_text` (processed text)

**Target Label:**
- `is_high_engagement` (0 or 1, top 10%)

## Common Parameters

### TextPreprocessor
```python
TextPreprocessor(
    remove_stopwords=True  # Remove common words
)
```

### FeatureEngineer
```python
FeatureEngineer(
    engagement_threshold=0.10  # Top 10% threshold
)
```

## Directory Structure

```
campustrends/
├── data/
│   ├── raw/              # Input: Your data goes here
│   └── processed/        # Output: Pipeline results
├── src/                  # Source code modules
├── notebooks/            # Jupyter notebooks
├── models/              # Saved ML models
└── run_pipeline.py      # Pipeline orchestrator
```

## Python API Examples

### Basic Usage
```python
from src.ingest import PostDataIngestor
from src.preprocess import TextPreprocessor
from src.feature_engineering import FeatureEngineer
import pandas as pd

# Load data
ingestor = PostDataIngestor()
df = ingestor.ingest('data/raw/posts.json', 'data/processed/clean.csv')

# Preprocess
preprocessor = TextPreprocessor()
df = preprocessor.preprocess_dataframe(df)

# Engineer features
engineer = FeatureEngineer()
df = engineer.engineer_features(df)

# Save
df.to_csv('data/processed/features.csv', index=False)
```

### Custom Parameters
```python
# Keep stopwords
preprocessor = TextPreprocessor(remove_stopwords=False)

# Change engagement threshold to top 5%
engineer = FeatureEngineer(engagement_threshold=0.05)

# Different school
ingestor = PostDataIngestor(school_filter='USC')
```

## Machine Learning Workflow

### 1. Data Preparation
```python
# Load features
df = pd.read_csv('data/processed/features.csv')

# Separate features and target
feature_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 
                'char_count', 'word_count', 'sentiment_score']
X = df[feature_cols]
y = df['is_high_engagement']
```

### 2. Train-Test Split
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 3. Model Training (Example)
```python
from sklearn.ensemble import RandomForestClassifier

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
from sklearn.metrics import classification_report
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
```

## Debugging Tips

### Check Data Quality
```python
# View first few rows
df.head()

# Check for missing values
df.isnull().sum()

# Check data types
df.dtypes

# Summary statistics
df.describe()
```

### Validate Schema
```python
required_columns = ['post_id', 'text', 'upvotes', 'timestamp', 'school']
missing = [col for col in required_columns if col not in df.columns]
if missing:
    print(f"Missing columns: {missing}")
```

### Test Pipeline Stages
```python
# Test on small sample first
df_sample = df.head(100)
preprocessor = TextPreprocessor()
df_processed = preprocessor.preprocess_dataframe(df_sample)
```

## Performance Tips

1. **Start Small**: Test with 100-1000 posts first
2. **Monitor Memory**: Large datasets may need chunking
3. **Save Intermediate Results**: Don't re-run stages unnecessarily
4. **Use Logging**: Check logs for warnings/errors
5. **Vectorize Operations**: Pandas is faster than loops

## Common Issues

### "Module not found"
- Ensure you're in the project root: `cd campustrends`
- Activate virtual environment: `source venv/bin/activate`

### "File not found"
- Check file paths are relative to project root
- Use absolute paths if unsure

### "Empty DataFrame"
- Check that input data matches required schema
- Verify filtering criteria (e.g., school='UCLA')

### Timestamp parsing errors
- Ensure timestamps are ISO-8601 format
- Check for null/invalid dates

## Key Concepts Reference

### Binary Classification
Predicting one of two categories (high vs. low engagement)

### Feature Engineering
Creating new variables from raw data to improve model performance

### TF-IDF
"Term Frequency-Inverse Document Frequency" - measures word importance

### Stopwords
Common words (the, is, at) that don't add meaning

### Sentiment Analysis
Measuring emotional tone of text (positive/negative)

### Rolling Window
Comparing data within a time range (e.g., ±7 days)

### Engagement Threshold
Percentile cutoff for "high engagement" (e.g., top 10%)

## Useful Resources

- **pandas docs**: https://pandas.pydata.org/docs/
- **scikit-learn docs**: https://scikit-learn.org/
- **matplotlib gallery**: https://matplotlib.org/stable/gallery/
- **spaCy docs**: https://spacy.io/usage

---

**Quick Help**: For any module, use `--help`:
```bash
python src/ingest.py --help
python run_pipeline.py --help
```
