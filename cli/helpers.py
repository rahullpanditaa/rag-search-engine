import string

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