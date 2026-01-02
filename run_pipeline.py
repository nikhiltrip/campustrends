#!/usr/bin/env python3
"""
Pipeline Runner for CampusTrendML

This script runs the complete data processing pipeline from raw data to features.
It's a convenience wrapper that executes each stage in sequence.

Usage:
    python run_pipeline.py --input data/raw/posts.json

Author: CampusTrendML Team
Date: January 2, 2026
"""

import argparse
import logging
import sys
from pathlib import Path

# Import our modules
# Note: This assumes you're running from the project root
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from ingest import PostDataIngestor
from preprocess import TextPreprocessor
from feature_engineering import FeatureEngineer

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_full_pipeline(input_path: str, output_dir: str = 'data/processed'):
    """
    Run the complete data processing pipeline.
    
    What does this do?
    ------------------
    This function orchestrates all three stages of our pipeline:
    1. Ingestion: Load and validate raw data
    2. Preprocessing: Clean and normalize text
    3. Feature Engineering: Extract features for modeling
    
    Think of it as a "one-click" solution to go from raw data to
    model-ready features!
    
    Parameters:
    -----------
    input_path : str
        Path to raw data file (JSON or CSV)
    output_dir : str
        Directory to save processed data (default: 'data/processed')
    
    Returns:
    --------
    str
        Path to the final feature-engineered CSV file
    """
    logger.info("="*70)
    logger.info("STARTING CAMPUSTRENDML PIPELINE")
    logger.info("="*70)
    
    # Define output paths for each stage
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    clean_path = output_dir / 'clean_posts.csv'
    preprocessed_path = output_dir / 'preprocessed.csv'
    features_path = output_dir / 'features.csv'
    
    try:
        # ========== STAGE 1: INGESTION ==========
        logger.info("\n" + "="*70)
        logger.info("STAGE 1/3: DATA INGESTION")
        logger.info("="*70)
        
        ingestor = PostDataIngestor(school_filter='UCLA')
        df = ingestor.ingest(input_path, str(clean_path))
        
        logger.info(f"✓ Stage 1 complete: {len(df)} posts ingested")
        
        # ========== STAGE 2: PREPROCESSING ==========
        logger.info("\n" + "="*70)
        logger.info("STAGE 2/3: TEXT PREPROCESSING")
        logger.info("="*70)
        
        preprocessor = TextPreprocessor(remove_stopwords=True)
        df = preprocessor.preprocess_dataframe(df, text_column='text')
        preprocessor.save_preprocessed_data(df, str(preprocessed_path))
        
        logger.info(f"✓ Stage 2 complete: {len(df)} posts preprocessed")
        
        # ========== STAGE 3: FEATURE ENGINEERING ==========
        logger.info("\n" + "="*70)
        logger.info("STAGE 3/3: FEATURE ENGINEERING")
        logger.info("="*70)
        
        engineer = FeatureEngineer(engagement_threshold=0.10)
        df = engineer.engineer_features(df, text_column='cleaned_text')
        engineer.save_features(df, str(features_path))
        
        logger.info(f"✓ Stage 3 complete: {len(df)} posts with features")
        
        # ========== PIPELINE COMPLETE ==========
        logger.info("\n" + "="*70)
        logger.info("PIPELINE COMPLETE!")
        logger.info("="*70)
        logger.info(f"\nOutput files:")
        logger.info(f"  1. Clean data: {clean_path}")
        logger.info(f"  2. Preprocessed: {preprocessed_path}")
        logger.info(f"  3. Features: {features_path}")
        logger.info(f"\n✓ Ready for modeling! Use {features_path} to train models.")
        
        return str(features_path)
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed: {e}")
        raise


def main():
    """
    Command-line interface for the pipeline.
    """
    parser = argparse.ArgumentParser(
        description='Run the CampusTrendML data processing pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run pipeline on JSON data
  python run_pipeline.py --input data/raw/posts.json
  
  # Run pipeline on CSV data with custom output directory
  python run_pipeline.py --input data/raw/posts.csv --output data/processed_v2
  
  # Get help
  python run_pipeline.py --help
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to raw data file (JSON or CSV)'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default='data/processed',
        help='Output directory for processed data (default: data/processed)'
    )
    
    args = parser.parse_args()
    
    # Validate input file exists
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input file not found: {args.input}")
        logger.error(f"Please check the file path and try again.")
        sys.exit(1)
    
    # Run the pipeline
    try:
        features_path = run_full_pipeline(args.input, args.output)
        print(f"\n🎉 Success! Feature data saved to: {features_path}")
        print(f"\n📊 Next steps:")
        print(f"   1. Explore the data: jupyter notebook notebooks/exploration.ipynb")
        print(f"   2. Train a model: python src/engagement_model.py --input {features_path}")
        print(f"   3. Analyze topics: python src/topic_modeling.py --input {features_path}")
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
