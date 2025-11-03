import json
from pathlib import Path

def search_title_in_movies(search_query: str) -> list:
    movies_data_path = Path(__file__).resolve().parent.parent / "data" / "movies.json"
    with open(movies_data_path, "r") as f:
        movies_dict = json.load(f)

    results = []
    movies = movies_dict["movies"]
    for movie in sorted(movies, key=lambda m: m["id"]):
        if search_query in movie["title"]:
            results.append(movie["title"])

    for i, movie in enumerate(results):
        if i > 4:
            break
        else:
            print(f"{i+1}. {movie}")
    return results
    



    