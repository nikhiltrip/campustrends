"""Deterministic text cleanup without model downloads."""

import argparse
import re
import unicodedata
import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


class TextPreprocessor:
    def __init__(self, remove_stopwords=True):
        self.remove_stopwords = remove_stopwords
        self.stopwords = set(ENGLISH_STOP_WORDS) - {"no", "not", "never"}

    def preprocess_text(self, text):
        text = unicodedata.normalize("NFKC", str(text)).lower()
        text = re.sub(r"https?://\S+|www\.\S+|@[\w.-]+", " ", text)
        tokens = re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        return " ".join(
            t for t in tokens if not self.remove_stopwords or t not in self.stopwords
        )

    def preprocess_dataframe(self, df, text_column="text"):
        out = df.copy()
        out["cleaned_text"] = out[text_column].map(self.preprocess_text)
        return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    TextPreprocessor().preprocess_dataframe(pd.read_csv(args.input)).to_csv(
        args.output, index=False
    )
