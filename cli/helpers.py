import string
from pathlib import Path

def remove_all_punctuation_lowercase(text: str) -> str:
    tt = str.maketrans("", "", string.punctuation)
    return text.translate(tt).lower()

def tokenize(text: str) -> list[str]:
    return text.split()

def compare_token_lists(query_tokens: list[str], data_tokens: list[str]) -> bool:
    for qt in query_tokens:
        for dt in data_tokens:
            if qt in dt:
                return True
    return False 

def remove_stop_words(tokens: list[str]) -> list[str]:
    stop_words_file = Path(__file__).resolve().parent.parent / "data" / "stopwords.txt"
    with open(stop_words_file, "r") as f:
        words = f.read()
    
    stop_words  = words.splitlines()

    result = []
    for token in tokens:
        if token in stop_words:
            continue
        result.append(token)

    return result