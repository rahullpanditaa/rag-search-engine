import string

def remove_all_punctuation_lowercase(text: str) -> str:
    tt = str.maketrans("", "", string.punctuation)
    return text.translate(tt).lower()