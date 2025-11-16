import string
import json
from nltk.stem import PorterStemmer
from .constants import (
    STOPWORDS_FILE_PATH,
    MOVIES_DATA_PATH
)

def remove_all_punctuation_lowercase(text: str) -> str:
    tt = str.maketrans("", "", string.punctuation)
    return text.translate(tt).lower()

def tokenize(text: str) -> list[str]:
    return text.lower().split()

def compare_token_lists(query_tokens: list[str], data_tokens: list[str]) -> bool:
    for qt in query_tokens:
        for dt in data_tokens:
            if qt in dt:
                return True
    return False 


def remove_stop_words(tokens: list[str]) -> list[str]:
    with open(STOPWORDS_FILE_PATH, "r") as f:
        words = f.read()
    
    stop_words  = words.splitlines()

    result = []
    for token in tokens:
        if token in stop_words:
            continue
        result.append(token)

    return stem_tokens(result)

def stem_tokens(tokens: list[str]) -> list[str]:
    stemmer = PorterStemmer()
    return list(map(lambda token: stemmer.stem(token), tokens))

def process_text_to_tokens(text: str) -> list[str]:
    tokens = remove_all_punctuation_lowercase(text)
    tokens = tokenize(tokens)
    tokens = remove_stop_words(tokens)
    tokens = stem_tokens(tokens)
    return tokens


def get_movie_data_from_file() -> list[dict]:
    with open(MOVIES_DATA_PATH, "r") as f:
        movies_dict = json.load(f)
    return movies_dict["movies"]

