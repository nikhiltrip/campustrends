# Getting Started with CampusTrendML

Welcome! This guide will help you get the project up and running.

## Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Git (already installed if you cloned this repo)

## Quick Start

### 1. Set Up Your Environment

```bash
# Navigate to project directory
cd campustrends

# Create a virtual environment (recommended)
python3 -m venv venv

# Activate the virtual environment
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download spaCy language model
python -m spacy download en_core_web_sm
```

### 2. Prepare Your Data

Place your raw post data in `data/raw/`. The data should be in JSON or CSV format with these fields:

```json
{
  "post_id": "unique_identifier",
  "text": "post content here",
  "upvotes": 42,
  "timestamp": "2026-01-02T10:30:00Z",
  "school": "UCLA"
}
```

**Don't have real data yet?** No problem! The exploration notebook includes sample data generation.

### 3. Run the Pipeline

#### Option A: Run the Complete Pipeline (Easiest)

```bash
# Process your data in one command
python run_pipeline.py --input data/raw/your_posts.json
```

This will:
- Load and validate your data
- Clean and preprocess text
- Extract features
- Save results to `data/processed/features.csv`

#### Option B: Run Stages Individually

```bash
# Stage 1: Ingest raw data
python src/ingest.py \
  --input data/raw/posts.json \
  --output data/processed/clean_posts.csv

# Stage 2: Preprocess text
python src/preprocess.py \
  --input data/processed/clean_posts.csv \
  --output data/processed/preprocessed.csv

# Stage 3: Engineer features
python src/feature_engineering.py \
  --input data/processed/preprocessed.csv \
  --output data/processed/features.csv
```

### 4. Explore Your Data

```bash
# Open the exploration notebook
jupyter notebook notebooks/exploration.ipynb
```

The notebook includes:
- Data visualization
- Statistical analysis
- Engagement patterns
- Feature correlations

## 📂 Project Structure

```
campustrends/
├── data/
│   ├── raw/              # Place your input data here
│   └── processed/        # Pipeline outputs go here
│
├── src/
│   ├── ingest.py                # Stage 1: Load & validate
│   ├── preprocess.py            # Stage 2: Clean text
│   ├── feature_engineering.py   # Stage 3: Extract features
│   ├── topic_modeling.py        # (Coming soon)
│   ├── engagement_model.py      # (Coming soon)
│   └── archetypes.py           # (Coming soon)
│
├── notebooks/
│   └── exploration.ipynb   # Data exploration & analysis
│
├── models/                 # Trained models saved here
├── run_pipeline.py        # Convenience script to run everything
├── requirements.txt       # Python dependencies
└── README.md             # Project overview
```

## Learning the Code

All code is heavily commented to teach ML concepts. Each module includes:

- **What**: What the code does
- **Why**: Why this step is important
- **How**: How the algorithm works
- **Examples**: Concrete examples with sample data

Start with:
1. `src/ingest.py` - Learn about data validation
2. `src/preprocess.py` - Learn about text cleaning
3. `src/feature_engineering.py` - Learn about feature extraction
4. `notebooks/exploration.ipynb` - Learn about EDA

## 📊 Example Workflow

```python
# In a Python script or notebook:

from src.ingest import PostDataIngestor
from src.preprocess import TextPreprocessor
from src.feature_engineering import FeatureEngineer

# Load data
ingestor = PostDataIngestor(school_filter='UCLA')
df = ingestor.ingest('data/raw/posts.json', 'data/processed/clean.csv')

# Preprocess
preprocessor = TextPreprocessor(remove_stopwords=True)
df = preprocessor.preprocess_dataframe(df)

# Engineer features
engineer = FeatureEngineer(engagement_threshold=0.10)
df = engineer.engineer_features(df)

# Now df contains all features ready for modeling!
print(df.head())
```

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the right directory
cd campustrends

# Make sure virtual environment is activated
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate  # Windows

# Reinstall dependencies
pip install -r requirements.txt
```

### Missing Data
The pipeline expects specific data fields. Check that your input data matches the schema in the README.

### Module Not Found
If you get "module not found" errors when running scripts:
```bash
# Run from the project root directory
cd campustrends
python src/ingest.py --help
```

## Next Steps

Once you've processed your data:

1. **Explore patterns**: Use `notebooks/exploration.ipynb`
2. **Train models**: (Coming in next phase)
3. **Analyze topics**: (Coming in next phase)
4. **Extract archetypes**: (Coming in next phase)

## Tips

- **Start small**: Test with a small dataset first (100-1000 posts)
- **Read the comments**: The code is designed to teach - read the explanations!
- **Experiment**: Try different parameters and see what happens
- **Visualize**: Use the notebooks to understand your data before modeling

## Need Help?

- Check the detailed comments in each source file
- Review the PROJECT_SPEC.md for technical details
- Look at example outputs in the exploration notebook

## What's Implemented

Complete data ingestion pipeline  
Text preprocessing with stopword removal  
Feature engineering (temporal + text)  
Engagement labeling  
Exploratory data analysis notebook  

## Coming Soon

⏳ Topic modeling (BERTopic)  
⏳ Engagement prediction model  
⏳ Archetype clustering  
⏳ Post generation (optional)  

---

**Happy analyzing!**

*Remember: Understanding your data is more important than fancy algorithms.*
