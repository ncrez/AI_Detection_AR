# utils.py

import re
import regex as re2

def remove_diacritics(text):
    diacritics = re.compile(r'[\u0617-\u061A\u064B-\u0652]')
    return re.sub(diacritics, '', text)

def normalize(text):
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ؤ", "و", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ئ", "ي", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("[^؀-ۿ ]+", " ", text)
    return text

def simple_word_tokenize(text):
    return re2.findall(r"\p{Arabic}+|\w+|[^\s\w]", text, flags=re2.VERSION1)

def sentence_tokenize(text):
    if not isinstance(text, str):
        return []
    parts = re.split(r'(?<=[\.\?\!\u061F\u061B])\s+', text)
    return [p.strip() for p in parts if p.strip()]

def paragraph_tokenize(text):
    if not isinstance(text, str):
        return []
    paragraphs = re.split(r'\s*\n\s*\n\s*|\s*\r\n\s*\r\n\s*', text.strip())
    return [p.strip() for p in paragraphs if p.strip()]
