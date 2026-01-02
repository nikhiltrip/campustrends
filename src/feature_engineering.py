"""
Feature Engineering Module for CampusTrendML

This module extracts features from preprocessed text to prepare data for machine learning.
Think of features as "measurements" or "characteristics" that help our model understand
and predict engagement.

What is feature engineering?
-----------------------------
Feature engineering is the process of transforming raw data into numerical features that
machine learning models can use. Text can't be fed directly into models - we need to
convert it into numbers that capture meaning, time patterns, and other signals.

Why is this important?
----------------------
Good features are the foundation of good machine learning models. The phrase "garbage in,
garbage out" applies here. Well-engineered features help models learn patterns more
effectively and make better predictions.

Author: CampusTrendML Team
Date: January 2, 2026
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Tuple
from datetime import datetime, timedelta
from pathlib import Path

# For text features, we'll use scikit-learn's TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    A class to extract various features from social media posts.
    
    Feature categories:
    -------------------
    1. Temporal features: Time-based patterns (hour, day, week)
    2. Text features: Content-based features (length, TF-IDF)
    3. Engagement labels: Binary classification target (high vs low engagement)
    
    Each category captures different aspects of what makes a post successful.
    """
    
    def __init__(self, engagement_threshold: float = 0.10):
        """
        Initialize the feature engineer.
        
        Parameters:
        -----------
        engagement_threshold : float
            Percentile threshold for high engagement (default: 0.10 = top 10%)
            
        What is a threshold?
        --------------------
        Instead of predicting exact upvote counts (hard!), we predict whether
        a post will be in the top 10% (high engagement) or not (low engagement).
        This is called binary classification and is often more reliable.
        """
        self.engagement_threshold = engagement_threshold
        self.tfidf_vectorizer = None  # Will be initialized when we fit
        
        logger.info(f"Initialized FeatureEngineer with top {engagement_threshold*100}% threshold")
    
    # ==================== TEMPORAL FEATURES ====================
    
    def extract_hour_of_day(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Extract the hour of day from timestamps (0-23).
        
        Why is hour important?
        ----------------------
        Campus posts have different engagement at different times:
        - Morning (7-11am): Commute time, checking updates
        - Afternoon (12-5pm): Between classes
        - Evening (6-10pm): Peak social media time
        - Late night (11pm-2am): Study sessions, bored scrolling
        
        Our model can learn that posts at certain hours tend to get more engagement.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with timestamp column
        timestamp_column : str
            Name of the timestamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new 'hour_of_day' column (0-23)
            
        Example:
        --------
        "2026-01-02 14:30:00" → hour_of_day = 14
        """
        logger.info("Extracting hour of day...")
        
        # Ensure timestamps are datetime objects
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # .dt accessor gives us datetime properties
        df['hour_of_day'] = df[timestamp_column].dt.hour
        
        logger.info(f"Hour distribution: {df['hour_of_day'].value_counts().sort_index().to_dict()}")
        return df
    
    def extract_day_of_week(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Extract the day of week from timestamps (0=Monday, 6=Sunday).
        
        Why is day of week important?
        ------------------------------
        Engagement patterns differ across the week:
        - Monday-Thursday: Regular school week, consistent patterns
        - Friday: Weekend excitement, going out posts
        - Saturday-Sunday: Lower overall volume but different content
        
        Students behave differently on weekdays vs weekends!
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with timestamp column
        timestamp_column : str
            Name of the timestamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new 'day_of_week' column (0=Monday to 6=Sunday)
            
        Example:
        --------
        Thursday → day_of_week = 3
        """
        logger.info("Extracting day of week...")
        
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # .dayofweek gives us 0=Monday through 6=Sunday
        df['day_of_week'] = df[timestamp_column].dt.dayofweek
        
        # Log distribution
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_dist = df['day_of_week'].value_counts().sort_index()
        logger.info(f"Day distribution: {dict(zip(day_names, day_dist))}")
        
        return df
    
    def extract_week_of_year(self, df: pd.DataFrame, timestamp_column: str = 'timestamp') -> pd.DataFrame:
        """
        Extract the week number of the year (1-52).
        
        Why is week of year important?
        -------------------------------
        Campus life has seasonal patterns:
        - Week 1-2: Start of quarter/semester (orientation posts)
        - Week 5-6: Midterms (stress posts)
        - Week 10-11: Finals (peak stress)
        - Week 15+: Break time (travel, relaxation)
        
        The academic calendar creates predictable content patterns!
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with timestamp column
        timestamp_column : str
            Name of the timestamp column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new 'week_of_year' column (1-52)
            
        Example:
        --------
        January 2, 2026 → week_of_year = 1
        """
        logger.info("Extracting week of year...")
        
        df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        # .isocalendar().week gives us ISO week number
        df['week_of_year'] = df[timestamp_column].dt.isocalendar().week
        
        logger.info(f"Week range: {df['week_of_year'].min()} to {df['week_of_year'].max()}")
        return df
    
    def extract_is_weekend(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create a binary feature for weekend (Saturday/Sunday).
        
        Why separate weekends?
        ----------------------
        Weekend behavior is often very different from weekday behavior.
        This binary feature helps the model treat weekends as a special case.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with 'day_of_week' column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with new 'is_weekend' column (0 or 1)
            
        Example:
        --------
        Saturday (day_of_week=5) → is_weekend = 1
        Tuesday (day_of_week=1) → is_weekend = 0
        """
        logger.info("Creating weekend indicator...")
        
        # Days 5 and 6 are Saturday and Sunday
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        weekend_pct = df['is_weekend'].mean() * 100
        logger.info(f"Weekend posts: {weekend_pct:.1f}%")
        
        return df
    
    # ==================== TEXT FEATURES ====================
    
    def extract_post_length(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Extract the length of posts in characters and words.
        
        Why is length important?
        ------------------------
        Post length can affect engagement:
        - Very short posts (< 20 chars): Might lack substance
        - Medium posts (20-150 chars): Sweet spot for social media
        - Long posts (> 150 chars): Might be too much to read
        
        We extract both character count and word count as they capture different things.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with text column
        text_column : str
            Name of the text column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with 'char_count' and 'word_count' columns
            
        Example:
        --------
        "love this campus" → char_count = 17, word_count = 3
        """
        logger.info("Extracting post length features...")
        
        # Character count: total number of characters
        df['char_count'] = df[text_column].str.len()
        
        # Word count: number of space-separated words
        df['word_count'] = df[text_column].str.split().str.len()
        
        logger.info(f"Average character count: {df['char_count'].mean():.1f}")
        logger.info(f"Average word count: {df['word_count'].mean():.1f}")
        
        return df
    
    def extract_sentiment_simple(self, df: pd.DataFrame, text_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Extract a simple sentiment score based on positive/negative words.
        
        What is sentiment?
        ------------------
        Sentiment measures the emotional tone of text:
        - Positive: "love", "great", "amazing", "best"
        - Negative: "hate", "terrible", "worst", "awful"
        - Neutral: Most other words
        
        Why is sentiment important?
        ---------------------------
        Posts with strong emotions (positive or negative) might get more engagement.
        People react to emotional content more than neutral factual statements.
        
        How we calculate it:
        --------------------
        sentiment_score = (positive_words - negative_words) / total_words
        Range: -1.0 (very negative) to +1.0 (very positive)
        
        Note: This is a simple approach. For production, consider using:
        - VADER (Valence Aware Dictionary and sEntiment Reasoner)
        - TextBlob
        - Hugging Face sentiment models
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with text column
        text_column : str
            Name of the text column
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with 'sentiment_score' column (-1.0 to 1.0)
        """
        logger.info("Extracting sentiment scores...")
        
        # Simple positive and negative word lists
        # In production, use a comprehensive sentiment lexicon
        positive_words = {
            'love', 'great', 'best', 'awesome', 'amazing', 'excellent', 'good',
            'wonderful', 'fantastic', 'happy', 'fun', 'cool', 'nice', 'perfect',
            'beautiful', 'brilliant', 'exciting', 'enjoyed', 'appreciate'
        }
        
        negative_words = {
            'hate', 'worst', 'terrible', 'awful', 'bad', 'horrible', 'sucks',
            'boring', 'annoying', 'disappointing', 'frustrated', 'angry', 'sad',
            'stressed', 'tired', 'difficult', 'hard', 'problem', 'issue', 'fail'
        }
        
        def calculate_sentiment(text):
            """Calculate sentiment for a single text."""
            words = text.lower().split()
            
            if len(words) == 0:
                return 0.0
            
            # Count positive and negative words
            pos_count = sum(1 for word in words if word in positive_words)
            neg_count = sum(1 for word in words if word in negative_words)
            
            # Calculate sentiment score
            # Normalize by total words to account for post length
            sentiment = (pos_count - neg_count) / len(words)
            
            return sentiment
        
        # Apply sentiment calculation to each post
        df['sentiment_score'] = df[text_column].apply(calculate_sentiment)
        
        # Log statistics
        logger.info(f"Average sentiment: {df['sentiment_score'].mean():.3f}")
        logger.info(f"Positive posts: {(df['sentiment_score'] > 0).sum()} ({(df['sentiment_score'] > 0).mean()*100:.1f}%)")
        logger.info(f"Negative posts: {(df['sentiment_score'] < 0).sum()} ({(df['sentiment_score'] < 0).mean()*100:.1f}%)")
        
        return df
    
    # ==================== ENGAGEMENT LABELING ====================
    
    def label_high_engagement(self, df: pd.DataFrame, upvote_column: str = 'upvotes',
                             window_days: int = 7) -> pd.DataFrame:
        """
        Label posts as high-engagement (top 10%) within a rolling time window.
        
        Why use a rolling window?
        -------------------------
        Upvote counts vary over time. A post with 20 upvotes might be exceptional
        in a slow week but average in a busy week. By using a rolling window,
        we compare each post to others from around the same time period.
        
        What is a rolling window?
        -------------------------
        For each post, we look at all posts within ±window_days and calculate
        the engagement threshold based on that local context.
        
        Example:
        --------
        If window_days=7 and we're looking at a post from Jan 5:
        - Consider all posts from Jan 1 to Jan 12
        - Find the 90th percentile of upvotes in that window
        - If our post is >= 90th percentile → high_engagement = 1
        
        Why top 10%?
        ------------
        This creates a balanced classification problem. If we tried to predict
        "viral" posts (top 1%), we'd have very few positive examples. Top 10%
        gives us enough examples to learn from.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with upvotes and timestamp
        upvote_column : str
            Name of the upvote column
        window_days : int
            Size of rolling window in days (default: 7)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with 'is_high_engagement' column (0 or 1)
        """
        logger.info(f"Labeling high engagement (top {self.engagement_threshold*100}% in {window_days}-day window)...")
        
        # Ensure timestamp is datetime and sort by time
        df = df.sort_values('timestamp').copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Initialize the engagement label column
        df['is_high_engagement'] = 0
        
        # For each post, calculate the threshold within its time window
        for idx in df.index:
            post_time = df.loc[idx, 'timestamp']
            
            # Define the time window: ±window_days from this post
            window_start = post_time - timedelta(days=window_days)
            window_end = post_time + timedelta(days=window_days)
            
            # Get all posts within this window
            window_mask = (df['timestamp'] >= window_start) & (df['timestamp'] <= window_end)
            window_posts = df[window_mask]
            
            # Calculate the engagement threshold (e.g., 90th percentile)
            threshold_percentile = 1 - self.engagement_threshold  # 0.90 for top 10%
            threshold_value = window_posts[upvote_column].quantile(threshold_percentile)
            
            # Label this post as high engagement if it meets/exceeds threshold
            if df.loc[idx, upvote_column] >= threshold_value:
                df.loc[idx, 'is_high_engagement'] = 1
        
        # Log statistics
        high_eng_count = df['is_high_engagement'].sum()
        high_eng_pct = df['is_high_engagement'].mean() * 100
        
        logger.info(f"High engagement posts: {high_eng_count} ({high_eng_pct:.1f}%)")
        logger.info(f"Average upvotes (high): {df[df['is_high_engagement']==1][upvote_column].mean():.1f}")
        logger.info(f"Average upvotes (low): {df[df['is_high_engagement']==0][upvote_column].mean():.1f}")
        
        return df
    
    # ==================== TF-IDF FEATURES ====================
    
    def extract_tfidf_features(self, df: pd.DataFrame, text_column: str = 'cleaned_text',
                               max_features: int = 100, save_path: str = None) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Extract TF-IDF features from text.
        
        What is TF-IDF?
        ---------------
        TF-IDF stands for "Term Frequency-Inverse Document Frequency". It's a way
        to convert text into numbers that captures word importance.
        
        - TF (Term Frequency): How often a word appears in a document
        - IDF (Inverse Document Frequency): How rare a word is across all documents
        
        TF-IDF score = TF × IDF
        
        Why is this useful?
        -------------------
        Words that appear frequently in one post but rarely in others are likely
        important for distinguishing that post. For example:
        - "finals" might be important during finals week
        - "parking" might be important for parking complaints
        - "the" appears everywhere, so it gets a low TF-IDF score
        
        How does it work?
        -----------------
        1. Count how many times each word appears in each post (TF)
        2. Calculate how many posts contain each word (DF)
        3. IDF = log(total_posts / DF) - rare words get higher scores
        4. Multiply TF × IDF for each word in each post
        
        Example:
        --------
        Post: "love the campus"
        → TF-IDF vector: [0.0, 0.7, 0.5, 0.0, ...]
                         (100-dimensional vector, one per vocabulary word)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with text column
        text_column : str
            Name of the text column
        max_features : int
            Maximum number of TF-IDF features to extract (vocabulary size)
        save_path : str
            Optional path to save the vectorizer for later use
            
        Returns:
        --------
        Tuple[pd.DataFrame, np.ndarray]
            - Updated DataFrame (unchanged, for chaining)
            - TF-IDF feature matrix (n_posts × max_features)
        """
        logger.info(f"Extracting TF-IDF features (max_features={max_features})...")
        
        # Initialize the TF-IDF vectorizer
        # ngram_range=(1,2) means we consider single words and pairs of words
        # min_df=2 means ignore words that appear in fewer than 2 documents
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),  # Unigrams and bigrams
            min_df=2,  # Minimum document frequency
            max_df=0.8,  # Ignore words in more than 80% of posts (too common)
            strip_accents='unicode',
            lowercase=True
        )
        
        # Fit the vectorizer on our text and transform to TF-IDF matrix
        # fit_transform learns the vocabulary and converts text to numbers
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(df[text_column])
        
        # Log vocabulary information
        vocabulary = self.tfidf_vectorizer.get_feature_names_out()
        logger.info(f"Vocabulary size: {len(vocabulary)}")
        logger.info(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        logger.info(f"Sample features: {list(vocabulary[:20])}")
        
        # Optionally save the vectorizer for future use
        # This allows us to transform new posts using the same vocabulary
        if save_path:
            import pickle
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'wb') as f:
                pickle.dump(self.tfidf_vectorizer, f)
            logger.info(f"Saved TF-IDF vectorizer to {save_path}")
        
        # Convert sparse matrix to dense array for easier handling
        tfidf_array = tfidf_matrix.toarray()
        
        return df, tfidf_array
    
    # ==================== MAIN PIPELINE ====================
    
    def engineer_features(self, df: pd.DataFrame, text_column: str = 'cleaned_text',
                         extract_tfidf: bool = False) -> pd.DataFrame:
        """
        Apply the complete feature engineering pipeline.
        
        This orchestrator method runs all feature extraction steps in sequence.
        It's the main entry point for feature engineering.
        
        Pipeline:
        ---------
        1. Extract temporal features (hour, day, week, weekend)
        2. Extract text features (length, sentiment)
        3. Label high-engagement posts
        4. Optionally extract TF-IDF features (separate, as they're high-dimensional)
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with preprocessed posts
        text_column : str
            Name of the cleaned text column
        extract_tfidf : bool
            Whether to extract TF-IDF features (creates separate matrix)
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all engineered features
        """
        logger.info("=" * 60)
        logger.info("Starting feature engineering pipeline")
        logger.info("=" * 60)
        
        # Temporal features
        df = self.extract_hour_of_day(df)
        df = self.extract_day_of_week(df)
        df = self.extract_week_of_year(df)
        df = self.extract_is_weekend(df)
        
        # Text features
        df = self.extract_post_length(df, text_column)
        df = self.extract_sentiment_simple(df, text_column)
        
        # Engagement label
        df = self.label_high_engagement(df)
        
        logger.info("=" * 60)
        logger.info("Feature engineering complete!")
        logger.info("=" * 60)
        
        # Display feature summary
        logger.info("\nFeature Summary:")
        logger.info(f"- Temporal features: hour_of_day, day_of_week, week_of_year, is_weekend")
        logger.info(f"- Text features: char_count, word_count, sentiment_score")
        logger.info(f"- Target label: is_high_engagement")
        logger.info(f"- Total feature columns: {len(df.columns)}")
        
        return df
    
    def save_features(self, df: pd.DataFrame, output_path: str):
        """
        Save the feature-engineered DataFrame to a CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with engineered features
        output_path : str
            Path to save the CSV file
        """
        logger.info(f"Saving features to {output_path}")
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} posts with features")


def main():
    """
    Example usage of the FeatureEngineer.
    
    Run with: python feature_engineering.py --input data/processed/preprocessed.csv
                                            --output data/processed/features.csv
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Engineer features from preprocessed campus posts'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to preprocessed CSV file'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save feature-engineered data (CSV)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='cleaned_text',
        help='Name of the cleaned text column (default: cleaned_text)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.10,
        help='Engagement threshold percentile (default: 0.10 for top 10%%)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    
    # Ensure timestamp is parsed
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    logger.info(f"Loaded {len(df)} posts")
    
    # Create feature engineer and extract features
    engineer = FeatureEngineer(engagement_threshold=args.threshold)
    df = engineer.engineer_features(df, text_column=args.text_column)
    
    # Save results
    engineer.save_features(df, args.output)
    
    print(f"\n✓ Successfully engineered features for {len(df)} posts")
    print(f"✓ Feature data saved to {args.output}")
    
    # Display sample
    print("\nSample of engineered features:")
    print("-" * 80)
    feature_cols = ['hour_of_day', 'day_of_week', 'is_weekend', 'char_count', 
                   'word_count', 'sentiment_score', 'is_high_engagement']
    print(df[feature_cols].head(10).to_string())


if __name__ == "__main__":
    main()
