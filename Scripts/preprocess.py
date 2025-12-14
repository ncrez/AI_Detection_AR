# preprocess.py

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.isri import ISRIStemmer
from datasets import load_dataset

from utils import (
    remove_diacritics,
    normalize,
    simple_word_tokenize,
    sentence_tokenize,
    paragraph_tokenize
)

nltk.download("stopwords")

def load_raw_dataset():
    return load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")

def process_split(split, split_name):
    data = []

    for text in split["original_abstract"]:
        data.append({
            "text": text,
            "label": 0,
            "generated_by": "human",
            "source_split": split_name
        })

    ai_sources = {
        "allam": split["allam_generated_abstract"],
        "jais": split["jais_generated_abstract"],
        "llama": split["llama_generated_abstract"],
        "openai": split["openai_generated_abstract"],
    }

    for model, texts in ai_sources.items():
        for text in texts:
            data.append({
                "text": text,
                "label": 1,
                "generated_by": model,
                "source_split": split_name
            })

    return pd.DataFrame(data)

stop_words = set(stopwords.words("arabic"))
stemmer = ISRIStemmer()

def preprocess_text(text):
    text = str(text)
    text = remove_diacritics(text)
    text = normalize(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [stemmer.stem(w) for w in tokens]
    return " ".join(tokens)

def apply_preprocessing(df):
    df = df.copy()

    df["clean_text"] = df["text"].apply(preprocess_text)

    df["tokens"] = df["clean_text"].apply(
        lambda t: [tok for tok in simple_word_tokenize(t) if tok.strip()]
    )

    df["words"] = df["tokens"].apply(
        lambda toks: [tok for tok in toks if tok.isalpha()]
    )

    df["sentences"] = df["text"].apply(sentence_tokenize)
    df["paragraphs"] = df["text"].apply(paragraph_tokenize)

    return df
