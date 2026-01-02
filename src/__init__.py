"""
CampusTrendML - Source Package

This package contains all the core modules for analyzing campus social media posts.

Modules:
--------
- ingest: Data loading and validation
- preprocess: Text cleaning and normalization
- feature_engineering: Feature extraction for ML
- topic_modeling: Topic discovery and tracking
- engagement_model: Engagement prediction
- archetypes: Pattern identification
- generate: Post generation (optional)

Usage:
------
    from src.ingest import PostDataIngestor
    from src.preprocess import TextPreprocessor
    from src.feature_engineering import FeatureEngineer

Author: CampusTrendML Team
Date: January 2, 2026
"""

__version__ = '0.1.0'
__author__ = 'CampusTrendML Team'

# Make key classes easily importable
try:
    from .ingest import PostDataIngestor
    from .preprocess import TextPreprocessor
    from .feature_engineering import FeatureEngineer
except ImportError:
    # Modules might not be in the path yet
    pass
