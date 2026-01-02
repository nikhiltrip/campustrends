"""
Data Ingestion Module for CampusTrendML

This module handles loading raw social media post data and performing initial validation.
Think of this as the "doorway" to our machine learning pipeline - it's where data enters
and gets its first quality check before we do any analysis.

What is data ingestion?
-----------------------
Data ingestion is the process of obtaining and importing data for immediate use or storage.
In our case, we're loading anonymous campus social media posts from files and making sure
they have all the required information before we analyze them.

Why do we need this?
--------------------
Real-world data is messy! Posts might be missing text, have invalid timestamps, or contain
duplicates. This module catches these issues early, so our machine learning models only
work with clean, valid data.

Author: CampusTrendML Team
Date: January 2, 2026
"""

import pandas as pd
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

# Set up logging so we can track what's happening in our pipeline
# Logging is like keeping a diary of your program - it helps debug issues
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class PostDataIngestor:
    """
    A class to handle loading and validating campus social media posts.
    
    What is a class?
    ----------------
    A class is like a blueprint or template. This PostDataIngestor class
    contains all the methods (functions) we need to load and validate data.
    By organizing code into a class, we can reuse it easily and keep related
    functionality together.
    
    Required fields for each post:
    - post_id: Unique identifier for the post
    - text: The actual content of the post
    - upvotes: Number of upvotes (engagement metric)
    - timestamp: When the post was created
    - school: Which school (we're focusing on UCLA)
    """
    
    def __init__(self, school_filter: str = "UCLA"):
        """
        Initialize the ingestor.
        
        Parameters:
        -----------
        school_filter : str
            Only keep posts from this school (default: "UCLA")
            
        What is __init__?
        -----------------
        This is a special method called a "constructor" that runs when you
        create a new PostDataIngestor object. It sets up the initial state.
        """
        self.school_filter = school_filter
        self.required_fields = ['post_id', 'text', 'upvotes', 'timestamp', 'school']
        logger.info(f"Initialized PostDataIngestor for school: {school_filter}")
    
    def load_json(self, file_path: str) -> pd.DataFrame:
        """
        Load posts from a JSON file.
        
        What is JSON?
        -------------
        JSON (JavaScript Object Notation) is a popular format for storing data.
        It looks like a list of dictionaries: [{"key": "value"}, {"key": "value"}]
        It's human-readable and commonly used in web applications.
        
        Parameters:
        -----------
        file_path : str
            Path to the JSON file containing posts
            
        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame (think of it as a table/spreadsheet) with all posts
            
        Example JSON structure:
        ----------------------
        [
            {
                "post_id": "abc123",
                "text": "Finals week is brutal",
                "upvotes": 42,
                "timestamp": "2026-01-02T10:30:00Z",
                "school": "UCLA"
            }
        ]
        """
        logger.info(f"Loading JSON data from {file_path}")
        
        try:
            # Read the JSON file
            # 'with open()' automatically closes the file when done
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Convert the list of dictionaries to a pandas DataFrame
            # DataFrames are powerful - they let us filter, sort, and analyze data easily
            df = pd.DataFrame(data)
            logger.info(f"Successfully loaded {len(df)} posts from JSON")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON format: {e}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error loading JSON: {e}")
            raise
    
    def load_csv(self, file_path: str) -> pd.DataFrame:
        """
        Load posts from a CSV file.
        
        What is CSV?
        ------------
        CSV (Comma-Separated Values) is another popular format for tabular data.
        Each row is a post, and columns are separated by commas.
        Think of it like a simplified Excel spreadsheet.
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file containing posts
            
        Returns:
        --------
        pd.DataFrame
            A pandas DataFrame with all posts
        """
        logger.info(f"Loading CSV data from {file_path}")
        
        try:
            # pandas can read CSV files directly
            df = pd.read_csv(file_path)
            logger.info(f"Successfully loaded {len(df)} posts from CSV")
            return df
            
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading CSV: {e}")
            raise
    
    def validate_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check that the DataFrame has all required fields.
        
        Why validate?
        -------------
        Before we invest time analyzing data, we need to make sure it has
        all the necessary columns. Missing columns could crash our pipeline later.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The DataFrame to validate
            
        Returns:
        --------
        pd.DataFrame
            The same DataFrame (if validation passes)
            
        Raises:
        -------
        ValueError
            If required fields are missing
        """
        logger.info("Validating data schema...")
        
        # Check if all required fields are present
        missing_fields = [field for field in self.required_fields 
                         if field not in df.columns]
        
        if missing_fields:
            error_msg = f"Missing required fields: {missing_fields}"
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info("Schema validation passed ✓")
        return df
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove invalid, duplicate, or empty posts.
        
        What makes a post invalid?
        --------------------------
        1. Empty or null text (a post with no content isn't useful)
        2. Missing upvote count (we need this for engagement prediction)
        3. Invalid timestamp (we can't analyze trends without proper time data)
        4. Duplicate post_id (same post appearing multiple times)
        
        This is called "data cleaning" - it's a crucial step in any ML project.
        Garbage in, garbage out! Clean data leads to better models.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The raw DataFrame
            
        Returns:
        --------
        pd.DataFrame
            A cleaned DataFrame with only valid posts
        """
        logger.info("Starting data cleaning...")
        original_count = len(df)
        
        # Step 1: Remove posts with empty or null text
        # .isna() finds missing values, .str.strip() removes whitespace
        df = df[df['text'].notna() & (df['text'].str.strip() != '')]
        logger.info(f"Removed {original_count - len(df)} posts with empty text")
        
        # Step 2: Remove posts with missing upvotes
        # We need upvotes to calculate engagement metrics
        current_count = len(df)
        df = df[df['upvotes'].notna()]
        logger.info(f"Removed {current_count - len(df)} posts with missing upvotes")
        
        # Step 3: Ensure upvotes are non-negative integers
        # Sometimes data entry errors can create negative numbers
        current_count = len(df)
        df = df[df['upvotes'] >= 0]
        logger.info(f"Removed {current_count - len(df)} posts with negative upvotes")
        
        # Step 4: Remove duplicate post_ids
        # .duplicated() marks duplicates, keep='first' keeps the first occurrence
        current_count = len(df)
        df = df.drop_duplicates(subset=['post_id'], keep='first')
        logger.info(f"Removed {current_count - len(df)} duplicate posts")
        
        # Step 5: Parse and validate timestamps
        # We convert string timestamps to datetime objects for proper time analysis
        try:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            logger.info("Successfully parsed timestamps")
        except Exception as e:
            logger.error(f"Error parsing timestamps: {e}")
            raise
        
        # Step 6: Filter by school
        # We're only analyzing UCLA posts for this project
        if self.school_filter:
            current_count = len(df)
            df = df[df['school'] == self.school_filter]
            logger.info(f"Filtered to {len(df)} posts from {self.school_filter}")
        
        logger.info(f"Data cleaning complete: {original_count} → {len(df)} posts")
        return df
    
    def get_data_summary(self, df: pd.DataFrame) -> Dict:
        """
        Generate a summary of the dataset for quick inspection.
        
        Why is a summary useful?
        ------------------------
        Before diving into complex analysis, it's good to understand your data:
        - How many posts do we have?
        - What's the date range?
        - What's the engagement distribution?
        
        This helps catch issues early and gives context for our analysis.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The cleaned DataFrame
            
        Returns:
        --------
        Dict
            A dictionary with summary statistics
        """
        summary = {
            'total_posts': len(df),
            'date_range': {
                'start': df['timestamp'].min().isoformat(),
                'end': df['timestamp'].max().isoformat()
            },
            'upvotes': {
                'mean': float(df['upvotes'].mean()),
                'median': float(df['upvotes'].median()),
                'max': int(df['upvotes'].max()),
                'min': int(df['upvotes'].min())
            },
            'school': self.school_filter,
            'avg_post_length': float(df['text'].str.len().mean())
        }
        
        return summary
    
    def save_processed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save the cleaned data to a CSV file.
        
        Why save intermediate results?
        ------------------------------
        Our ML pipeline has multiple stages. By saving cleaned data, we:
        1. Don't need to re-run cleaning every time
        2. Can inspect the data between stages
        3. Create checkpoints in case something goes wrong later
        
        Parameters:
        -----------
        df : pd.DataFrame
            The cleaned DataFrame
        output_path : str
            Where to save the CSV file
        """
        logger.info(f"Saving processed data to {output_path}")
        
        # Create the directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV with index=False (we don't need row numbers)
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} posts to {output_path}")
    
    def ingest(self, input_path: str, output_path: str) -> pd.DataFrame:
        """
        Main method to run the complete ingestion pipeline.
        
        This is the "orchestrator" method - it calls all the other methods
        in the right order to load, validate, clean, and save data.
        
        Pipeline steps:
        ---------------
        1. Load data (JSON or CSV)
        2. Validate schema
        3. Clean data
        4. Generate summary
        5. Save processed data
        
        Parameters:
        -----------
        input_path : str
            Path to the raw data file (JSON or CSV)
        output_path : str
            Path where cleaned data should be saved
            
        Returns:
        --------
        pd.DataFrame
            The cleaned and validated DataFrame
        """
        logger.info("=" * 60)
        logger.info("Starting data ingestion pipeline")
        logger.info("=" * 60)
        
        # Determine file type and load accordingly
        if input_path.endswith('.json'):
            df = self.load_json(input_path)
        elif input_path.endswith('.csv'):
            df = self.load_csv(input_path)
        else:
            raise ValueError("Input file must be .json or .csv")
        
        # Run validation and cleaning
        df = self.validate_schema(df)
        df = self.clean_data(df)
        
        # Generate and display summary
        summary = self.get_data_summary(df)
        logger.info("Data Summary:")
        logger.info(json.dumps(summary, indent=2))
        
        # Save processed data
        self.save_processed_data(df, output_path)
        
        logger.info("=" * 60)
        logger.info("Data ingestion complete!")
        logger.info("=" * 60)
        
        return df


def main():
    """
    Example usage of the PostDataIngestor.
    
    This function shows how to use the ingestor from the command line.
    You can run this script with: python ingest.py
    """
    import argparse
    
    # argparse lets us accept command-line arguments
    # This makes our script flexible - users can specify their own file paths
    parser = argparse.ArgumentParser(
        description='Ingest and clean campus social media posts'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input file (JSON or CSV)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save cleaned data (CSV)'
    )
    parser.add_argument(
        '--school',
        type=str,
        default='UCLA',
        help='School to filter for (default: UCLA)'
    )
    
    args = parser.parse_args()
    
    # Create an ingestor and run the pipeline
    ingestor = PostDataIngestor(school_filter=args.school)
    df = ingestor.ingest(args.input, args.output)
    
    print(f"\n✓ Successfully processed {len(df)} posts")
    print(f"✓ Cleaned data saved to {args.output}")


# This is a Python idiom: only run main() if this script is executed directly
# If someone imports this module, main() won't run automatically
if __name__ == "__main__":
    main()
