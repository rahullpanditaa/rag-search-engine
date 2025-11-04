import json
from pathlib import Path
from helpers import remove_all_punctuation_lowercase, tokenize, compare_token_lists, remove_stop_words

def keyword_search(search_query: str) -> list:
    movies_data_path = Path(__file__).resolve().parent.parent / "data" / "movies.json"
    with open(movies_data_path, "r") as f:
        movies_dict = json.load(f)

    results = []
    query_tokens = tokenize(search_query)
    query_tokens = remove_stop_words(query_tokens)
    movies = movies_dict["movies"]
    for movie in sorted(movies, key=lambda m: m["id"]):
        title = remove_all_punctuation_lowercase(movie["title"])
        title_tokens = tokenize(title)
        title_tokens = remove_stop_words(title_tokens)
        if compare_token_lists(query_tokens=query_tokens, data_tokens=title_tokens):
            results.append(movie["title"])

    for i, movie in enumerate(results):
        if i > 4:
            break
        else:
            print(f"{i+1}. {movie}")
    return results
    




    