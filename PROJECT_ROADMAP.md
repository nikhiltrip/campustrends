# CampusTrendML - Complete Project Analysis & Roadmap

## FILE-BY-FILE BREAKDOWN

### Documentation Files

#### 1. **README.md**
- **Purpose**: Project overview, setup instructions, and quick reference
- **What it does**: Explains the project goals, tech stack, and how to get started
- **When you use it**: First file anyone reads to understand the project

#### 2. **PROJECT_SPEC.md**
- **Purpose**: Detailed technical specification
- **What it does**: Defines data schema, pipeline stages, evaluation metrics, and success criteria
- **When you use it**: Reference for implementation decisions and technical details

#### 3. **GETTING_STARTED.md**
- **Purpose**: Step-by-step setup and installation guide
- **What it does**: Walks through environment setup, dependency installation, and first run
- **When you use it**: When setting up the project for the first time

#### 4. **QUICK_REFERENCE.md**
- **Purpose**: Command cheatsheet and API reference
- **What it does**: Quick lookup for commands, parameters, and common operations
- **When you use it**: During development when you need to remember syntax

#### 5. **requirements.txt**
- **Purpose**: Python package dependencies
- **What it does**: Lists all required libraries with versions
- **When you use it**: Run `pip install -r requirements.txt` to install everything

#### 6. **.gitignore**
- **Purpose**: Tells git which files to ignore
- **What it does**: Prevents committing large data files, models, and system files
- **When you use it**: Automatic - git uses it when committing

---

### Core Pipeline Modules (src/)

#### 7. **src/ingest.py** (426 lines)
**What it does:**
- Loads raw post data from JSON or CSV files
- Validates that required fields exist (post_id, text, upvotes, timestamp, school)
- Removes duplicates, empty posts, and invalid data
- Filters to UCLA posts only
- Outputs clean CSV file

**Key class:** `PostDataIngestor`

**Main methods:**
- `load_json()` - Load from JSON
- `load_csv()` - Load from CSV
- `validate_schema()` - Check required fields
- `clean_data()` - Remove bad data
- `ingest()` - Run full pipeline

**Run it:**
```bash
python src/ingest.py --input data/raw/posts.json --output data/processed/clean.csv
```

#### 8. **src/preprocess.py** (489 lines)
**What it does:**
- Cleans text: lowercase, remove URLs, emojis, special characters
- Removes @mentions and #hashtags
- Tokenizes text (splits into words)
- Removes stopwords (common words like "the", "is", "at")
- Outputs preprocessed CSV with "cleaned_text" column

**Key class:** `TextPreprocessor`

**Main methods:**
- `lowercase()` - Convert to lowercase
- `remove_urls()` - Strip URLs
- `remove_emojis()` - Remove emoji characters
- `remove_special_characters()` - Keep only letters/numbers
- `tokenize()` - Split into words
- `remove_stopwords_from_tokens()` - Remove common words
- `preprocess_text()` - Run all steps
- `preprocess_dataframe()` - Process entire dataset

**Run it:**
```bash
python src/preprocess.py --input data/processed/clean.csv --output data/processed/preprocessed.csv
```

#### 9. **src/feature_engineering.py** (657 lines)
**What it does:**
- Extracts temporal features: hour_of_day, day_of_week, week_of_year, is_weekend
- Extracts text features: character count, word count, sentiment score
- Labels posts as high-engagement (top 10% within 7-day rolling window)
- Can extract TF-IDF features (converts text to numbers)

**Key class:** `FeatureEngineer`

**Main methods:**
- `extract_hour_of_day()` - Get posting hour (0-23)
- `extract_day_of_week()` - Get day (0=Monday, 6=Sunday)
- `extract_week_of_year()` - Get week number (1-52)
- `extract_is_weekend()` - Binary weekend indicator
- `extract_post_length()` - Character and word counts
- `extract_sentiment_simple()` - Positive/negative sentiment score
- `label_high_engagement()` - Mark top 10% posts
- `extract_tfidf_features()` - Convert text to TF-IDF vectors
- `engineer_features()` - Run all feature extraction

**Run it:**
```bash
python src/feature_engineering.py --input data/processed/preprocessed.csv --output data/processed/features.csv
```

#### 10. **src/__init__.py**
- **Purpose**: Makes src/ a proper Python package
- **What it does**: Allows importing modules like `from src.ingest import PostDataIngestor`

---

### Orchestration Scripts

#### 11. **run_pipeline.py** (181 lines)
**What it does:**
- Runs all three stages in sequence automatically
- Calls ingest.py → preprocess.py → feature_engineering.py
- One-command execution of the entire pipeline
- Provides progress updates and saves intermediate files

**Run it:**
```bash
python run_pipeline.py --input data/raw/posts.json
```

**Output files:**
- `data/processed/clean_posts.csv`
- `data/processed/preprocessed.csv`
- `data/processed/features.csv`

---

### Analysis Notebooks

#### 12. **notebooks/exploration.ipynb**
**What it does:**
- Exploratory Data Analysis (EDA)
- Generates sample data for testing
- Creates visualizations:
  - Upvote distribution
  - Posting patterns by hour/day
  - Text length vs engagement
  - Sentiment analysis
  - Feature correlations
- Educational explanations of each concept

**When to use it:**
- After running the pipeline
- To understand your data before modeling
- To visualize trends and patterns
- To learn about data science concepts

---

## HOW THE ENGAGEMENT PREDICTION MODEL WORKS

### Current State: **Feature Engineering Only**

Right now, the repo **doesn't have a trained model yet**. It only prepares the data. Here's what exists vs what's needed:

### What's Implemented

**Stage 1: Data Preparation**
```
Raw Posts → Clean Data → Preprocessed Text → Feature Vectors
```

The pipeline creates these features for each post:
- **Temporal**: hour_of_day, day_of_week, week_of_year, is_weekend
- **Text**: char_count, word_count, sentiment_score
- **Target**: is_high_engagement (0 or 1)

### What's NOT Implemented Yet

**Stage 2-4: Model Training, Topic Modeling, Archetypes**

These files are mentioned but don't exist:
- `src/topic_modeling.py` - NOT CREATED YET
- `src/engagement_model.py` - NOT CREATED YET
- `src/archetypes.py` - NOT CREATED YET
- `notebooks/evaluation.ipynb` - NOT CREATED YET

### How the Model WILL Work (When Implemented)

Here's the complete flow:

#### **Binary Classification Problem**
- **Input**: Post features (hour, day, length, sentiment, text embeddings)
- **Output**: Prediction (0 = low engagement, 1 = high engagement)
- **"High engagement"** = Top 10% of posts by upvotes within a 7-day window

#### **Algorithm Options** (from PROJECT_SPEC.md):
1. **XGBoost** (gradient boosting) - Best for tabular data
2. **Random Forest** - Good baseline, interpretable
3. **Logistic Regression** - Simple, fast, interpretable

#### **Training Process**:
```python
# Pseudocode for what engagement_model.py will do:

# 1. Load feature data
df = pd.read_csv('data/processed/features.csv')

# 2. Separate features and target
X = df[['hour_of_day', 'day_of_week', 'is_weekend', 
        'char_count', 'word_count', 'sentiment_score']]
y = df['is_high_engagement']

# 3. Split train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 4. Train model
model = XGBClassifier()
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# 6. Save model
pickle.dump(model, open('models/engagement_model.pkl', 'wb'))
```

#### **Why This Works**:
The model learns patterns like:
- Posts at 9pm get more engagement than 3am
- Weekend posts behave differently
- Medium-length posts (50-100 chars) perform best
- Positive sentiment correlates with upvotes
- Certain topics (finals, events) spike engagement

---

## DATA COLLECTION FOR MOBILE APP

Since you mentioned "mobile app", here's what you need to know:

### Option 1: Backend API + Database (RECOMMENDED)

**Architecture:**
```
Mobile App → Backend Server → Database → ML Pipeline
```

**Flow:**
1. Users post on your mobile app
2. App sends POST request to your backend
3. Backend saves to database (PostgreSQL, MongoDB, etc.)
4. Separate script exports data periodically
5. ML pipeline processes exported data

**Pros:**
- Clean separation of concerns
- App doesn't know about ML
- Can scale easily
- Secure (no direct DB access from app)

**Cons:**
- Need to build/maintain backend
- More infrastructure

**Implementation:**
```python
# Backend API (Flask/FastAPI)
@app.post("/api/posts")
def create_post(post_data):
    # Save to database
    db.posts.insert({
        'post_id': generate_id(),
        'text': post_data['text'],
        'upvotes': 0,
        'timestamp': datetime.now(),
        'school': 'UCLA'
    })
    return {'status': 'success'}

# Separate export script (runs daily/weekly)
def export_for_ml():
    posts = db.posts.find({'school': 'UCLA'})
    df = pd.DataFrame(posts)
    df.to_json('data/raw/posts.json')
```

### Option 2: Direct Mobile → ML Pipeline (NOT RECOMMENDED)

**Architecture:**
```
Mobile App → Local File Storage → ML Pipeline
```

**Why NOT recommended:**
- No scalability
- Data loss risk
- Hard to update ML models
- Requires ML code on device (large)

### Option 3: Middleware/Data Lake (PROFESSIONAL)

**Architecture:**
```
Mobile App → API Gateway → Message Queue → Data Lake → ML Pipeline
```

**Tools:**
- API Gateway: AWS API Gateway, Kong
- Message Queue: Kafka, RabbitMQ, AWS SQS
- Data Lake: S3, Google Cloud Storage
- Processing: Apache Airflow, AWS Glue

**When to use:**
- High volume (1000+ posts/day)
- Multiple data sources
- Real-time requirements
- Production deployment

### RECOMMENDED APPROACH FOR YOU

**Start Simple:**

1. **Development Phase**:
   - Use sample data or manually collected posts
   - Store as JSON/CSV files
   - Run ML pipeline locally

2. **MVP Phase**:
   - Build simple Flask/FastAPI backend
   - Store posts in SQLite or PostgreSQL
   - Export script runs weekly
   - ML pipeline processes exports

3. **Production Phase**:
   - Add proper authentication
   - Scale database (PostgreSQL, MongoDB)
   - Real-time data pipeline
   - Model deployment as API

**Minimal Backend Example:**
```python
# backend.py
from flask import Flask, request, jsonify
import sqlite3
from datetime import datetime

app = Flask(__name__)

@app.route('/api/posts', methods=['POST'])
def create_post():
    data = request.json
    conn = sqlite3.connect('posts.db')
    conn.execute('''
        INSERT INTO posts (text, upvotes, timestamp, school)
        VALUES (?, ?, ?, ?)
    ''', (data['text'], 0, datetime.now().isoformat(), 'UCLA'))
    conn.commit()
    return jsonify({'success': True})

@app.route('/api/export', methods=['GET'])
def export_data():
    conn = sqlite3.connect('posts.db')
    df = pd.read_sql('SELECT * FROM posts', conn)
    df.to_json('data/raw/posts.json', orient='records')
    return jsonify({'exported': len(df)})
```

---

## PROJECT TIMELINE & PHASES

### **PHASE 1: Foundation (COMPLETED) - Week 1-2**

**Goal**: Set up infrastructure and data pipeline

**Completed:**
- ✅ Project structure
- ✅ Data ingestion module
- ✅ Text preprocessing module
- ✅ Feature engineering module
- ✅ Documentation
- ✅ Exploration notebook

**Deliverables:**
- Working pipeline: raw data → features
- Clean, documented code
- EDA notebook

---

### **PHASE 2: Model Development (NEXT) - Week 3-4**

**Goal**: Build and train engagement prediction model

**Tasks:**
1. **Create `src/engagement_model.py`** (Week 3)
   - Load feature data
   - Train/test split
   - Train XGBoost classifier
   - Evaluate with metrics
   - Save trained model

2. **Create `notebooks/evaluation.ipynb`** (Week 3)
   - Model performance analysis
   - Feature importance
   - Confusion matrix
   - ROC curve
   - Precision-recall curves

3. **Hyperparameter Tuning** (Week 4)
   - Grid search
   - Cross-validation
   - Optimize for precision@10%

**Deliverables:**
- Trained model file (`models/engagement_model.pkl`)
- Evaluation metrics (>70% precision target)
- Feature importance analysis

**Code to Write:**
```python
# src/engagement_model.py
class EngagementPredictor:
    def train(self, X, y):
        # Train XGBoost
        pass
    
    def predict(self, X):
        # Predict high engagement
        pass
    
    def evaluate(self, X_test, y_test):
        # Calculate metrics
        pass
```

---

### **PHASE 3: Topic Modeling (Week 5-6)**

**Goal**: Discover trending topics over time

**Tasks:**
1. **Create `src/topic_modeling.py`**
   - Implement BERTopic pipeline
   - Assign topics to posts
   - Track topic frequency over time
   - Detect topic spikes

2. **Visualization**
   - Topic word clouds
   - Topic trends over time
   - Topic-engagement correlation

**Deliverables:**
- Topic assignments for all posts
- Topic trend visualizations
- Top 10 recurring topics

**Code to Write:**
```python
# src/topic_modeling.py
class TopicModeler:
    def fit_topics(self, texts):
        # Use BERTopic
        pass
    
    def assign_topics(self, texts):
        # Assign to posts
        pass
    
    def track_trends(self, df):
        # Time series of topics
        pass
```

---

### **PHASE 4: Archetype Extraction (Week 7-8)**

**Goal**: Identify post patterns that drive engagement

**Tasks:**
1. **Create `src/archetypes.py`**
   - Filter high-engagement posts
   - Cluster using embeddings
   - Extract representative phrases
   - Label archetypes manually

2. **Analysis**
   - Identify 5-10 archetypes
   - Examples: "Finals stress", "Parking complaints", "Game day hype"

**Deliverables:**
- Archetype clusters
- Keyword extraction per archetype
- Example posts for each archetype

---

### **PHASE 5: Data Collection (Week 9-10)**

**Goal**: Set up real data pipeline

**Tasks:**
1. **Build Backend API** (Week 9)
   - Flask/FastAPI server
   - POST /posts endpoint
   - SQLite database
   - Export functionality

2. **Mobile App Integration** (Week 10)
   - Connect app to backend
   - Test post creation
   - Verify data flow

3. **Data Export Scheduler**
   - Cron job or scheduler
   - Export to JSON daily/weekly
   - Trigger ML pipeline

**Deliverables:**
- Working backend API
- Database with real posts
- Automated export script

---

### **PHASE 6: Real Data Analysis (Week 11-12)**

**Goal**: Run ML pipeline on real campus data

**Tasks:**
1. **Collect Data** (Week 11)
   - Gather 500-1000 posts minimum
   - Ensure diverse time periods
   - Include weekdays and weekends

2. **Retrain Models** (Week 12)
   - Run pipeline on real data
   - Retrain engagement model
   - Retrain topic model
   - Update archetypes

3. **Validate Results**
   - Check model accuracy
   - Verify topics make sense
   - Validate archetypes

**Deliverables:**
- Models trained on real data
- Performance metrics
- Insights report

---

### **PHASE 7: (Optional) Post Generation (Week 13-14)**

**Goal**: Generate example posts using patterns

**Tasks:**
1. **Create `src/generate.py`**
   - Template-based generation
   - OR local LLM (Ollama)
   - Input: topic + archetype + time
   - Output: synthetic post

2. **Quality Control**
   - Manual review of generated posts
   - Ensure they're realistic

**Deliverables:**
- Post generation module
- 50-100 generated examples

---

### **PHASE 8: Deployment & Iteration (Week 15-16)**

**Goal**: Make system production-ready

**Tasks:**
1. **Model API**
   - Serve model as REST API
   - Real-time predictions
   - Batch predictions

2. **Dashboard** (Optional)
   - Streamlit or Dash
   - Live topic trends
   - Engagement metrics
   - Top posts

3. **Documentation**
   - API documentation
   - Deployment guide
   - User manual

**Deliverables:**
- Deployed model API
- (Optional) Live dashboard
- Complete documentation

---

## IMMEDIATE NEXT STEPS

### Priority 1: Build Engagement Model
```bash
# Create this file next:
touch src/engagement_model.py
```

**What to implement:**
1. Load features.csv
2. Split train/test (80/20)
3. Train XGBoost classifier
4. Evaluate with precision, recall, F1
5. Save model

### Priority 2: Get Sample Data
Two options:

**Option A: Generate Synthetic Data**
- Use the exploration notebook (already has sample generation)
- Create 1000 fake posts for testing

**Option B: Manual Collection**
- Browse YikYak/similar apps
- Manually record 100-200 posts
- Save as JSON

### Priority 3: Test Full Pipeline
```bash
# Once you have data:
python run_pipeline.py --input data/raw/posts.json
python src/engagement_model.py --input data/processed/features.csv
```

---

## CHECKLIST FOR SUCCESS

### Data Requirements
- [ ] Minimum 500 posts (1000+ ideal)
- [ ] Date range: At least 4-6 weeks
- [ ] Diverse time periods (weekdays, weekends, different hours)
- [ ] Mix of high and low engagement posts
- [ ] All required fields present

### Model Requirements
- [ ] Precision @ Top 10% > 70%
- [ ] ROC-AUC > 0.75
- [ ] Feature importance analysis done
- [ ] Model saved and loadable

### Topic Modeling Requirements
- [ ] 10-20 interpretable topics
- [ ] Topics remain stable across weeks
- [ ] Topic trends visualized

### Production Requirements
- [ ] Backend API working
- [ ] Database configured
- [ ] Data export automated
- [ ] ML pipeline scheduled

---

## CRITICAL QUESTIONS TO ANSWER

1. **Do you have real campus post data already?**
   - If YES: Skip to Phase 2
   - If NO: Focus on Phase 5 first (data collection)

2. **Is the mobile app already built?**
   - If YES: Add backend integration
   - If NO: Build data collection infrastructure first

3. **Timeline pressure?**
   - Tight deadline: Focus on engagement model only
   - Flexible: Complete all phases

4. **Deployment target?**
   - Local analysis: Keep current setup
   - Public tool: Need API + hosting

---

## LEARNING PATH

**If you're new to ML:**
1. Week 1-2: Understand the existing code (READ ALL COMMENTS)
2. Week 3-4: Build simple engagement model (start with logistic regression)
3. Week 5-6: Try XGBoost, tune parameters
4. Week 7+: Add advanced features (topic modeling, etc.)

**If you're experienced:**
1. Week 1: Build engagement model
2. Week 2: Add topic modeling
3. Week 3: Extract archetypes
4. Week 4+: Deploy and iterate

---

This document should give you a complete picture of where the project is and where it's going. Let me know which phase you want to tackle first!
