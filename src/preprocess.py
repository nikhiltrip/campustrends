"""
Text Preprocessing Module for CampusTrendML

This module handles cleaning and normalizing social media post text for analysis.
Think of this as "preparing ingredients before cooking" - we need clean, standardized
text before we can extract meaningful features or train models.

What is text preprocessing?
----------------------------
Raw text from social media is messy: it has URLs, emojis, irregular capitalization,
and lots of noise. Preprocessing transforms this messy text into a clean, standardized
format that machine learning algorithms can understand better.

Why do we need this?
--------------------
Machine learning models work better with clean, consistent input. For example:
- "LOVE this campus!!!" and "love this campus" should be treated as similar
- URLs like "https://example.com" don't add meaning to topic analysis
- Stopwords like "the", "is", "at" are common but don't distinguish topics

Author: CampusTrendML Team
Date: January 2, 2026
"""

import pandas as pd
import re
import logging
from typing import List, Set
import unicodedata

# Set up logging to track what's happening
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TextPreprocessor:
    """
    A class to handle all text cleaning and normalization operations.
    
    What does preprocessing include?
    --------------------------------
    1. Lowercasing: "HELLO" → "hello" (standardizes capitalization)
    2. URL removal: Removes links that don't add semantic meaning
    3. Emoji removal: Strips emoji characters (optional: could analyze separately)
    4. Special character removal: Keeps only letters, numbers, and spaces
    5. Whitespace normalization: Removes extra spaces
    6. Tokenization: Splits text into individual words
    7. Stopword removal: Removes common words that don't add meaning
    
    These steps make text cleaner and more uniform for analysis.
    """
    
    def __init__(self, remove_stopwords: bool = True):
        """
        Initialize the preprocessor.
        
        Parameters:
        -----------
        remove_stopwords : bool
            Whether to remove common English stopwords (default: True)
            
        What are stopwords?
        -------------------
        Stopwords are very common words like "the", "is", "at", "on" that appear
        frequently but don't help distinguish between different topics or sentiments.
        Removing them helps our models focus on meaningful content words.
        """
        self.remove_stopwords = remove_stopwords
        
        # Define a basic set of English stopwords
        # In production, you might use NLTK or spaCy's stopword lists
        self.stopwords = self._get_stopwords()
        
        logger.info(f"Initialized TextPreprocessor (remove_stopwords={remove_stopwords})")
    
    def _get_stopwords(self) -> Set[str]:
        """
        Get a set of common English stopwords.
        
        What's a set?
        -------------
        A set is a Python data structure that stores unique items and allows
        fast lookups. Perfect for checking if a word is a stopword!
        
        Returns:
        --------
        Set[str]
            A set of lowercase stopwords
            
        Note: This is a basic list. For production use, consider NLTK's stopwords:
            from nltk.corpus import stopwords
            return set(stopwords.words('english'))
        """
        # Common English stopwords
        # These don't usually add meaning in social media analysis
        return {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'will', 'with', 'she', 'they', 'we', 'you', 'this',
            'have', 'had', 'but', 'or', 'if', 'not', 'so', 'what', 'when',
            'where', 'who', 'which', 'why', 'how', 'all', 'each', 'their',
            'there', 'been', 'were', 'would', 'could', 'should', 'do', 'does',
            'did', 'am', 'can', 'his', 'her', 'my', 'our', 'me', 'him', 'them',
            'us', 'i'
        }
    
    def lowercase(self, text: str) -> str:
        """
        Convert text to lowercase.
        
        Why lowercase?
        --------------
        "Happy", "HAPPY", and "happy" should be treated as the same word.
        Lowercasing standardizes text so we don't have multiple versions
        of the same word.
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Lowercase version of the text
            
        Example:
        --------
        "I LOVE UCLA!!!" → "i love ucla!!!"
        """
        return text.lower()
    
    def remove_urls(self, text: str) -> str:
        """
        Remove URLs from text.
        
        Why remove URLs?
        ----------------
        URLs like "https://example.com/article?id=123" don't tell us about
        the content of a post. They're just noise in our text analysis.
        Plus, they're all unique, which would create sparse features.
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Text with URLs removed
            
        Example:
        --------
        "Check this out https://example.com" → "Check this out"
        
        Technical note:
        ---------------
        We use a regular expression (regex) to find URLs. The pattern matches:
        - http:// or https:// at the start
        - Followed by any non-whitespace characters
        """
        # Regular expression pattern to match URLs
        # \S+ means "one or more non-whitespace characters"
        url_pattern = r'https?://\S+'
        
        # re.sub() replaces all matches with an empty string
        return re.sub(url_pattern, '', text)
    
    def remove_mentions(self, text: str) -> str:
        """
        Remove @mentions from text.
        
        Why remove mentions?
        --------------------
        In campus posts, @mentions might refer to specific users or accounts.
        Since we're doing anonymous analysis, we don't need these.
        They also don't help with topic modeling.
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Text with @mentions removed
            
        Example:
        --------
        "@uclaalumni this is great!" → "this is great!"
        """
        # Match @ followed by word characters (letters, numbers, underscores)
        mention_pattern = r'@\w+'
        return re.sub(mention_pattern, '', text)
    
    def remove_hashtags(self, text: str) -> str:
        """
        Remove hashtags from text.
        
        Why remove hashtags?
        --------------------
        Hashtags like #UCLA or #finals could be informative, but they're often
        redundant with the post content. We remove them for consistency.
        
        Alternative approach: You could keep hashtags and remove just the #
        symbol, treating them as regular words.
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Text with hashtags removed
            
        Example:
        --------
        "Finals week #UCLA #stress" → "Finals week"
        """
        hashtag_pattern = r'#\w+'
        return re.sub(hashtag_pattern, '', text)
    
    def remove_emojis(self, text: str) -> str:
        """
        Remove emoji characters from text.
        
        Why remove emojis?
        ------------------
        Emojis can convey emotion (😊, 😢) but are tricky to process consistently.
        For simplicity, we remove them. Advanced projects might analyze them separately.
        
        How it works:
        -------------
        We use Unicode categories to identify emoji characters. Most emojis fall
        into specific Unicode ranges that we can filter out.
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Text with emojis removed
            
        Example:
        --------
        "I love UCLA 😍🎓" → "I love UCLA"
        """
        # Filter out characters in emoji Unicode ranges
        # This is a simplified approach; full emoji detection is complex
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags
            "\U00002702-\U000027B0"  # dingbats
            "\U000024C2-\U0001F251"
            "]+",
            flags=re.UNICODE
        )
        return emoji_pattern.sub('', text)
    
    def remove_special_characters(self, text: str) -> str:
        """
        Remove special characters, keeping only letters, numbers, and spaces.
        
        Why remove special characters?
        ------------------------------
        Characters like !, ?, @, #, $ don't add semantic meaning for topic modeling.
        We keep letters and numbers because they form actual words and meaningful
        content (like "CS101" or "Route66").
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Text with only alphanumeric characters and spaces
            
        Example:
        --------
        "This is great!!!" → "This is great"
        
        Technical note:
        ---------------
        The pattern [^a-zA-Z0-9\s] means:
        - ^ = NOT
        - a-zA-Z = letters
        - 0-9 = numbers
        - \s = whitespace
        So we replace everything that's NOT a letter, number, or space.
        """
        # Keep only alphanumeric characters and whitespace
        return re.sub(r'[^a-zA-Z0-9\s]', '', text)
    
    def normalize_whitespace(self, text: str) -> str:
        """
        Replace multiple spaces with a single space and trim.
        
        Why normalize whitespace?
        -------------------------
        After removing URLs, emojis, etc., we often end up with extra spaces:
        "This  is    great" → "This is great"
        
        Consistent spacing makes text cleaner and easier to tokenize.
        
        Parameters:
        -----------
        text : str
            The input text
            
        Returns:
        --------
        str
            Text with normalized whitespace
            
        Example:
        --------
        "This   is    great  " → "This is great"
        """
        # \s+ matches one or more whitespace characters
        text = re.sub(r'\s+', ' ', text)
        # .strip() removes leading/trailing whitespace
        return text.strip()
    
    def tokenize(self, text: str) -> List[str]:
        """
        Split text into individual words (tokens).
        
        What is tokenization?
        ---------------------
        Tokenization is the process of breaking text into individual units (tokens),
        usually words. It's a fundamental step in NLP.
        
        "I love UCLA" → ["I", "love", "UCLA"]
        
        Why tokenize?
        -------------
        Most NLP algorithms work with individual words, not full sentences.
        Tokenization lets us analyze word frequencies, remove stopwords, etc.
        
        Parameters:
        -----------
        text : str
            The cleaned text
            
        Returns:
        --------
        List[str]
            A list of words (tokens)
            
        Example:
        --------
        "i love ucla" → ["i", "love", "ucla"]
        
        Note: This is a simple tokenization (split by spaces). For more advanced
        tokenization, consider using NLTK or spaCy, which handle contractions,
        punctuation, and edge cases better.
        """
        # Simple tokenization: split by whitespace
        return text.split()
    
    def remove_stopwords_from_tokens(self, tokens: List[str]) -> List[str]:
        """
        Remove common stopwords from a list of tokens.
        
        Why remove stopwords?
        ---------------------
        Words like "the", "is", "at" appear in almost every post but don't help
        distinguish topics. Removing them makes our models focus on content words.
        
        Example:
        --------
        ["i", "love", "the", "campus"] → ["love", "campus"]
        
        Parameters:
        -----------
        tokens : List[str]
            List of word tokens
            
        Returns:
        --------
        List[str]
            Tokens with stopwords removed
            
        Technical note:
        ---------------
        We use a list comprehension: [x for x in tokens if condition]
        This creates a new list containing only tokens that aren't stopwords.
        """
        if not self.remove_stopwords:
            return tokens
        
        # Keep only tokens that are NOT in the stopwords set
        # Set lookup is very fast (O(1) complexity)
        return [token for token in tokens if token not in self.stopwords]
    
    def preprocess_text(self, text: str, return_tokens: bool = False) -> str:
        """
        Apply the full preprocessing pipeline to a single text.
        
        This is the main method that combines all cleaning steps in the right order.
        
        Pipeline order matters!
        -----------------------
        1. Lowercase first (makes pattern matching easier)
        2. Remove URLs, mentions, hashtags
        3. Remove emojis
        4. Remove special characters
        5. Normalize whitespace
        6. Optionally tokenize and remove stopwords
        
        Parameters:
        -----------
        text : str
            Raw text to preprocess
        return_tokens : bool
            If True, return a list of tokens; if False, return cleaned text string
            
        Returns:
        --------
        str or List[str]
            Cleaned text (string) or list of tokens
            
        Example:
        --------
        Input: "I LOVE UCLA!!! 😍 https://example.com #UCLA"
        Output: "love ucla" (or ["love", "ucla"] if return_tokens=True)
        """
        # Apply preprocessing steps in sequence
        text = self.lowercase(text)
        text = self.remove_urls(text)
        text = self.remove_mentions(text)
        text = self.remove_hashtags(text)
        text = self.remove_emojis(text)
        text = self.remove_special_characters(text)
        text = self.normalize_whitespace(text)
        
        # If we want tokens (list of words), tokenize and optionally remove stopwords
        if return_tokens:
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords_from_tokens(tokens)
            return tokens
        
        # Otherwise, return cleaned text as a string
        # (We still apply stopword removal at the string level)
        if self.remove_stopwords:
            tokens = self.tokenize(text)
            tokens = self.remove_stopwords_from_tokens(tokens)
            text = ' '.join(tokens)
        
        return text
    
    def preprocess_dataframe(self, df: pd.DataFrame, text_column: str = 'text',
                            output_column: str = 'cleaned_text') -> pd.DataFrame:
        """
        Apply preprocessing to all texts in a DataFrame.
        
        Why process a whole DataFrame?
        -------------------------------
        In our pipeline, we have thousands of posts stored in a pandas DataFrame.
        This method efficiently processes all of them at once.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing posts
        text_column : str
            Name of the column with raw text (default: 'text')
        output_column : str
            Name for the new column with cleaned text (default: 'cleaned_text')
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with a new column containing cleaned text
            
        Technical note:
        ---------------
        We use .apply() which applies our preprocess_text function to each row.
        This is more efficient than a for loop in pandas.
        """
        logger.info(f"Preprocessing {len(df)} posts...")
        
        # Apply preprocessing to each post
        # .apply() runs our function on each value in the column
        df[output_column] = df[text_column].apply(self.preprocess_text)
        
        # Log statistics
        avg_original_length = df[text_column].str.len().mean()
        avg_cleaned_length = df[output_column].str.len().mean()
        
        logger.info(f"Average original length: {avg_original_length:.1f} characters")
        logger.info(f"Average cleaned length: {avg_cleaned_length:.1f} characters")
        logger.info(f"Compression ratio: {avg_cleaned_length/avg_original_length:.2%}")
        
        # Remove posts that became empty after cleaning
        # (This can happen if a post was all emojis or URLs)
        original_count = len(df)
        df = df[df[output_column].str.len() > 0]
        removed = original_count - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} posts that became empty after preprocessing")
        
        logger.info("Preprocessing complete ✓")
        return df
    
    def save_preprocessed_data(self, df: pd.DataFrame, output_path: str):
        """
        Save the preprocessed DataFrame to a CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            The preprocessed DataFrame
        output_path : str
            Where to save the CSV file
        """
        logger.info(f"Saving preprocessed data to {output_path}")
        df.to_csv(output_path, index=False)
        logger.info(f"Successfully saved {len(df)} posts")


def main():
    """
    Example usage of the TextPreprocessor.
    
    This function demonstrates how to use the preprocessor from the command line.
    Run with: python preprocess.py --input data/processed/clean_posts.csv
                                   --output data/processed/preprocessed.csv
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Preprocess campus social media post text'
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input CSV file (cleaned posts)'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to save preprocessed data (CSV)'
    )
    parser.add_argument(
        '--text-column',
        type=str,
        default='text',
        help='Name of the text column (default: text)'
    )
    parser.add_argument(
        '--keep-stopwords',
        action='store_true',
        help='Keep stopwords (default: remove them)'
    )
    
    args = parser.parse_args()
    
    # Load data
    logger.info(f"Loading data from {args.input}")
    df = pd.read_csv(args.input)
    logger.info(f"Loaded {len(df)} posts")
    
    # Create preprocessor and process data
    preprocessor = TextPreprocessor(remove_stopwords=not args.keep_stopwords)
    df = preprocessor.preprocess_dataframe(df, text_column=args.text_column)
    
    # Save results
    preprocessor.save_preprocessed_data(df, args.output)
    
    print(f"\n✓ Successfully preprocessed {len(df)} posts")
    print(f"✓ Cleaned data saved to {args.output}")
    
    # Show a few examples
    print("\nExample preprocessed posts:")
    print("-" * 60)
    for i in range(min(5, len(df))):
        print(f"Original: {df.iloc[i][args.text_column][:80]}...")
        print(f"Cleaned:  {df.iloc[i]['cleaned_text'][:80]}...")
        print()


if __name__ == "__main__":
    main()
