# CampusTrendsML

A local, offline machine learning system that analyzes anonymous campus social media posts (YikYak-style data) to detect trending topics, predict high-engagement posts, and identify recurring post archetypes—all for UCLA.

## Objectives

This project demonstrates:
- **Trend Detection**: Identify trending topics and campus events over time
- **Engagement Prediction**: Predict whether a post will be high-engagement (top 10% by upvotes)
- **Archetype Discovery**: Find recurring patterns in high-performing posts
- **Local Processing**: All analysis runs locally using open-source tools (no paid APIs)

## Scope

- **Single School**: UCLA only
- **Data Type**: Text + metadata (no user tracking)
- **Task**: Binary classification for engagement prediction
- **Analysis**: Time-aware topic modeling and clustering
- **Ethics**: Fully anonymous, aggregated analysis only

## Tech Stack

- **Python 3.10+**
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, XGBoost
- **NLP**: sentence-transformers, spaCy/NLTK
- **Topic Modeling**: BERTopic or LDA
- **Visualization**: matplotlib, seaborn

## 📁 Project Structure

```
CampusTrendML/
│
├── data/
│   ├── raw/              # Original post data
│   └── processed/        # Cleaned and feature-engineered data
│
├── src/
│   ├── ingest.py                # Load and validate raw data
│   ├── preprocess.py            # Clean and normalize text
│   ├── feature_engineering.py   # Extract features from posts
│   ├── topic_modeling.py        # Detect trending topics
│   ├── engagement_model.py      # Predict high-engagement posts
│   ├── archetypes.py           # Cluster post patterns
│   └── generate.py             # (Optional) Generate example posts
│
├── notebooks/
│   ├── exploration.ipynb   # Data exploration and analysis
│   └── evaluation.ipynb    # Model evaluation
│
├── models/
│   ├── engagement_model.pkl  # Trained engagement predictor
│   └── topic_model/          # Topic modeling artifacts
│
├── README.md
└── PROJECT_SPEC.md         # Detailed technical specification
```

## Getting Started

### Installation

```bash
# Clone the repository
git clone https://github.com/nikhiltrip/campustrends.git
cd campustrends

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install pandas numpy scikit-learn sentence-transformers bertopic xgboost spacy matplotlib seaborn nltk
python -m spacy download en_core_web_sm
```

### Quick Start

```python
# 1. Ingest raw data
python src/ingest.py --input data/raw/posts.json --output data/processed/clean_posts.csv

# 2. Preprocess text
python src/preprocess.py --input data/processed/clean_posts.csv --output data/processed/preprocessed.csv

# 3. Engineer features
python src/feature_engineering.py --input data/processed/preprocessed.csv --output data/processed/features.csv

# 4. Train engagement model
python src/engagement_model.py --input data/processed/features.csv --output models/engagement_model.pkl

# 5. Analyze topics
python src/topic_modeling.py --input data/processed/features.csv --output models/topic_model/
```

## Pipeline Overview

### Stage 1: Ingestion
Load JSON/CSV post data, validate schema, remove duplicates

### Stage 2: Preprocessing
Clean text: lowercase, remove URLs/emojis, tokenize, remove stopwords

### Stage 3: Feature Engineering
Extract:
- TF-IDF features and sentence embeddings
- Temporal features (hour, day, week)
- Sentiment scores
- Post length metrics

Label posts as high-engagement (top 10% within 7-day rolling window)

### Stage 4: Topic Modeling
Fit topic model, assign labels, track frequency over time, detect spikes

### Stage 5: Engagement Prediction
Train classifier using text embeddings + topic + time + sentiment
Evaluate with Precision@Top10% and ROC-AUC

### Stage 6: Archetype Extraction
Cluster high-engagement posts, extract representative phrases, label patterns

### Stage 7 (Optional): Post Generation
Generate example posts using templates or local LLM (Ollama)

## Evaluation Metrics

- **Precision on High-Engagement Prediction**: How accurately we identify viral posts
- **Topic Coherence**: How meaningful the discovered topics are
- **Archetype Stability**: Consistency of patterns across time periods

## Ethical Constraints

- No usernames or personal identifiers
- No individual user tracking
- Aggregated analysis only
- Research and demonstration use

## Data Schema

Each post record contains:
```json
{
  "post_id": "unique identifier",
  "text": "post content",
  "upvotes": 42,
  "timestamp": "2026-01-02T10:30:00Z",
  "school": "UCLA"
}
```

## Contributing

This is a research/educational project. Feel free to fork and experiment!

## License

MIT License - See LICENSE file for details

## Learning Resources

This project is designed to teach:
- Text preprocessing and feature engineering
- Topic modeling with transformers
- Time-series analysis of social data
- Binary classification for engagement prediction
- Ethical considerations in social media analysis

---
