# CampusTrendML — Project Specification

## 1. Objective

Build a local, offline machine learning system that analyzes anonymous campus social media posts (YikYak-style data) for a single school (UCLA) to:

1. **Detect trending topics and campus events over time**
2. **Predict whether a post will be high-engagement** (top 10% by upvotes)
3. **Identify recurring high-performing post archetypes**
4. **(Optional) Generate example posts** based on active events and archetypes

**Key Constraint**: All processing must run locally using open-source tools. No paid APIs.

---

## 2. Scope & Constraints

### In Scope
- ✅ One school (UCLA)
- ✅ Text + metadata only (no users)
- ✅ Engagement prediction as binary classification
- ✅ Time-aware analysis
- ✅ Archetype clustering on top-performing posts

### Out of Scope
- ❌ User tracking
- ❌ Real-time deployment
- ❌ Cross-school generalization
- ❌ Exact upvote prediction (regression)
- ❌ Paid APIs

---

## 3. Tech Stack

| Category | Tools |
|----------|-------|
| **Language** | Python 3.10+ |
| **Data Processing** | pandas, numpy |
| **Machine Learning** | scikit-learn, XGBoost/RandomForest |
| **NLP/Embeddings** | sentence-transformers |
| **Topic Modeling** | BERTopic or LDA |
| **Visualization** | matplotlib, seaborn |
| **Text Processing** | spaCy or NLTK |

---

## 4. Data Schema

### Raw Post Record
```json
{
  "post_id": "string",
  "text": "string",
  "upvotes": "int",
  "timestamp": "ISO-8601",
  "school": "UCLA"
}
```

### Derived Features
After feature engineering, each post will have:

| Feature | Type | Description |
|---------|------|-------------|
| `hour_of_day` | int | 0-23 |
| `day_of_week` | int | 0-6 (Monday=0) |
| `week_of_term` | int | Academic week number |
| `post_length` | int | Character count |
| `sentiment_score` | float | -1.0 to 1.0 |
| `topic_id` | int | Assigned topic cluster |
| `is_high_engagement` | bool | Top 10% flag |

---

## 5. Directory Structure

```
CampusTrendML/
│
├── data/
│   ├── raw/              # Original JSON/CSV files
│   └── processed/        # Cleaned and engineered datasets
│
├── src/
│   ├── ingest.py                # Load and validate raw data
│   ├── preprocess.py            # Clean and normalize text
│   ├── feature_engineering.py   # Extract temporal/text features
│   ├── topic_modeling.py        # BERTopic or LDA pipeline
│   ├── engagement_model.py      # Train/evaluate classifier
│   ├── archetypes.py           # Cluster high-engagement posts
│   └── generate.py             # (Optional) Post generation
│
├── notebooks/
│   ├── exploration.ipynb   # EDA and visualizations
│   └── evaluation.ipynb    # Model performance analysis
│
├── models/
│   ├── engagement_model.pkl  # Trained classifier
│   └── topic_model/          # BERTopic artifacts
│
├── README.md
└── PROJECT_SPEC.md
```

---

## 6. Pipeline Stages

### Stage 1: Ingestion
**Module**: `ingest.py`

**Tasks**:
- Load JSON/CSV post data
- Validate schema (required fields present)
- Remove empty or duplicate posts
- Store clean data in `data/processed/`

**Output**: `clean_posts.csv`

---

### Stage 2: Preprocessing
**Module**: `preprocess.py`

**Tasks**:
- Lowercase all text
- Remove URLs, emojis, special characters
- Tokenize text
- Remove stopwords
- Save cleaned text

**Output**: `preprocessed.csv` with `cleaned_text` column

---

### Stage 3: Feature Engineering
**Module**: `feature_engineering.py`

**Generate**:
- **Text Features**:
  - TF-IDF vectors
  - Sentence embeddings (sentence-transformers)
- **Temporal Features**:
  - `hour_of_day`
  - `day_of_week`
  - `week_of_term`
- **Content Features**:
  - Sentiment score (polarity)
  - Post length (characters, words)

**Label**:
- `is_high_engagement = 1` if post is in **top 10% of upvotes** within a rolling **7-day window**

**Output**: `features.csv` with all derived columns

---

### Stage 4: Topic Modeling
**Module**: `topic_modeling.py`

**Tasks**:
- Fit topic model (BERTopic recommended) on all posts
- Assign `topic_id` to each post
- Track topic frequency over time
- Identify topic spikes (unusual frequency)

**Output**:
- `topic_assignments.csv`
- `models/topic_model/` artifacts

**Deliverables**:
- List of top topics with keywords
- Time-series plot of topic trends

---

### Stage 5: Engagement Prediction
**Module**: `engagement_model.py`

**Task**: Train a binary classifier to predict `is_high_engagement`

**Features**:
- Text embeddings (from sentence-transformers)
- `topic_id`
- Temporal features (`hour_of_day`, `day_of_week`, etc.)
- `sentiment_score`
- `post_length`

**Model Options**:
- XGBoost
- Random Forest
- Logistic Regression

**Evaluation Metrics**:
- **Precision @ Top 10%**: Of predicted high-engagement posts, how many are actually high-engagement?
- **ROC-AUC**: Overall classifier performance
- **Confusion Matrix**

**Output**:
- `models/engagement_model.pkl`
- Performance report

---

### Stage 6: Archetype Extraction
**Module**: `archetypes.py`

**Tasks**:
1. Filter to **high-engagement posts only**
2. Cluster embeddings (K-Means or HDBSCAN)
3. For each cluster:
   - Extract representative phrases (TF-IDF or BERTopic keywords)
   - Manually label archetype (stored as metadata)

**Example Archetypes**:
- "Complaining about parking"
- "Excited about game day"
- "Finals week stress"
- "Dining hall roasts"

**Output**:
- `archetypes.csv` with cluster labels and keywords
- Cluster visualization

---

### Stage 7 (Optional): Post Generation
**Module**: `generate.py`

**Given**:
- Active topics (from Stage 4)
- Time context (hour, day)
- Archetype (from Stage 6)

**Generate**:
- Example posts using:
  - **Template-based generation** (fill-in-the-blank with keywords)
  - **Local LLM** via Ollama (e.g., Llama 2)

**Output**:
- `generated_posts.json` with synthetic examples

---

## 7. Evaluation Metrics

| Metric | Purpose |
|--------|---------|
| **Precision @ Top 10%** | How accurately we identify viral posts |
| **ROC-AUC** | Overall engagement model quality |
| **Topic Coherence** | Are discovered topics meaningful? |
| **Archetype Stability** | Do clusters remain consistent across weeks? |

---

## 8. Ethical Constraints

This project adheres to strict ethical guidelines:

- ✅ **No usernames**: Only anonymous post text and metadata
- ✅ **No individual tracking**: Aggregate patterns only
- ✅ **Research/demo use**: Not for surveillance or targeting
- ✅ **Transparency**: All methods are documented and open-source

**Privacy by Design**: This system cannot identify individuals and does not attempt to.

---

## 9. Deliverables

1. ✅ **Reproducible pipeline** (all stages automated)
2. ✅ **Saved models** (engagement predictor, topic model)
3. ✅ **README** explaining setup and usage
4. ✅ **Example analysis notebook** with visualizations
5. ⭐ **(Optional) Generated post examples**

---

## 10. First Tasks (Implementation Checklist)

- [x] Create directory structure
- [x] Write `README.md`
- [x] Write `PROJECT_SPEC.md`
- [ ] Implement `ingest.py` (data loading and validation)
- [ ] Implement `preprocess.py` (text cleaning)
- [ ] Implement `feature_engineering.py` (feature extraction)
- [ ] Create `notebooks/exploration.ipynb` (EDA)
- [ ] Implement `topic_modeling.py` (BERTopic pipeline)
- [ ] Implement `engagement_model.py` (classifier)
- [ ] Implement `archetypes.py` (clustering)
- [ ] (Optional) Implement `generate.py` (post generation)
- [ ] Create `notebooks/evaluation.ipynb` (model evaluation)

---

## 11. Success Criteria

The project is successful if:

1. **Engagement model** achieves **>70% precision** at identifying top 10% posts
2. **Topics are interpretable** (human can understand what each topic represents)
3. **Archetypes are stable** (similar patterns emerge week-to-week)
4. **All code is documented** with teaching-quality comments
5. **Pipeline runs locally** without external APIs

---

## 12. Future Extensions

Once the core pipeline is working, potential enhancements:

- Multi-school comparison (UCLA vs USC vs UCSD)
- Real-time dashboard (Streamlit or Dash)
- Temporal anomaly detection (unusual post patterns)
- Cross-topic engagement analysis
- Sentiment trend tracking over academic calendar

---

**Last Updated**: January 2, 2026
