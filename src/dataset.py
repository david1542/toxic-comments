import os
import re
import string
from functools import partial
from pathlib import Path
from typing import List

import nltk
import numpy as np
import pandas as pd
from datasets import Dataset
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizer

tqdm.pandas()

from utils import hash_split

stop_words = stopwords.words("english")
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
lemmatizer = WordNetLemmatizer()


def remove_non_english_characters(comments: pd.Series):
    # Create a string with all English characters
    english_characters = string.ascii_letters + string.digits + string.punctuation + " "
    pattern = r"[^" + english_characters + "]"

    # Use a generator comprehension to remove any characters that are not in the english_characters string
    _comments = comments.str.replace(pattern, "", regex=True)
    return _comments


def remove_stop_words(comments: pd.Series):
    pattern = r"\b(?:{})\b".format("|".join(stop_words))
    _comments = comments.str.replace(pattern, "", regex=True)
    return _comments


def preprocess_df(df):
    # Download packages from NLTK
    nltk.download("stopwords")
    nltk.download("wordnet")

    df_copy = df.copy()
    df_copy["comment_text"] = preprocess_comments(df_copy["comment_text"])
    df_copy = df_copy[df_copy["comment_text"] != ""]
    df_copy = df_copy.drop_duplicates(["comment_text"])
    return df_copy


# Define mapping functions
def preprocess_comments(comments: List[str] | pd.Series) -> pd.Series:
    if not isinstance(comments, pd.Series):
        comments = pd.Series(comments)

    # Lowercase
    comments = comments.str.lower()
    # Remove stopwords
    comments = remove_stop_words(comments)
    # Remove non-english characters
    comments = remove_non_english_characters(comments)
    # Remove punctuation & backslash
    comments = comments.str.replace("[{}]".format(string.punctuation), "", regex=True)
    comments = comments.str.replace("\\", "", regex=True)
    # Remove URLs
    comments = comments.str.replace(r"http\S+", "", regex=True)
    # Remove newlines
    comments = comments.str.replace("\n", "")
    # Remove numbers
    comments = comments.str.replace(r"\d+", "", regex=True)
    # Remove whitespaces
    comments = comments.str.strip()
    comments = comments.str.replace(r"\s+", " ", regex=True)
    # Lemmatizing
    # comments = comments.progress_apply(lambda text: ' '.join([stemmer.stem(word) for word in text.split()]))
    comments = comments.progress_apply(lambda text: " ".join([lemmatizer.lemmatize(word) for word in text.split()]))
    comments = comments.progress_apply(
        lambda text: " ".join([lemmatizer.lemmatize(word, pos="v") for word in text.split()])
    )

    return comments


def create_dataset(df: pd.DataFrame, tokenizer, combine_labels=True):
    def _tokenize(example):
        return tokenizer(example["text"], truncation=True)

    def _merge_labels(example):
        example["labels"] = np.vstack([example[label] for label in labels]).T.astype("float")
        return example

    ds = Dataset.from_pandas(df)
    ds = ds.rename_column("comment_text", "text")
    if combine_labels:
        ds = ds.map(_merge_labels, batched=True)
    ds = ds.remove_columns(labels)
    ds = ds.map(_tokenize, batched=True)
    return ds


def load_csv(csv_path: str, preprocess: bool = False, use_cache: bool = True):
    cache_path = Path(csv_path).parent / f"{Path(csv_path).stem}_cached.csv"
    if use_cache and os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
    else:
        df = pd.read_csv(csv_path)
        if preprocess:
            df = preprocess_df(df)
        df.to_csv(cache_path, index=False)

    return df


def hashed_train_test_split(df: pd.DataFrame, valid_size: float):
    ids = df["id"].apply(lambda id: hash_split(id, [1 - valid_size, valid_size]))
    df_train = df[ids == 0].reset_index(drop=True)
    df_valid = df[ids == 1].reset_index(drop=True)
    return df_train, df_valid
