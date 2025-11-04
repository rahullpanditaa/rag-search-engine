from helpers import process_text_to_tokens, get_movie_data_from_file, remove_all_punctuation_lowercase
from pathlib import Path
import pickle

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, dict[str, object]]

    cache_dir_path = Path(__file__).resolve().parent.parent / "cache"
    index_file_path = cache_dir_path / "index.pkl"
    docmap_file_path = cache_dir_path / "docmap.pkl"

    def __init__(self):
        # dict mapping tokens(str) to sets of doc ids
        self.index = {}

        # dict mapping doc id to full doc objects {id, title, desciption}
        self.docmap = {}

    def __add_document(self, doc_id: int, text: str):
        # tokenize input text
        tokens = process_text_to_tokens(text=text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            
    
    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term.lower())
        if not doc_ids:
            return []
        return sorted(doc_ids)
    
    
    def build(self):
        movies_list = get_movie_data_from_file()
        for movie in movies_list:
            self.__add_document(movie["id"], f'{movie["title"]} {movie["description"]}')
            self.docmap[movie["id"]] = movie

    def save(self):
        type(self).cache_dir_path.mkdir(parents=True, exist_ok=True)
        with type(self).index_file_path.open("wb") as index_dump, type(self).docmap_file_path.open("wb") as docmap_dump:
            pickle.dump(self.index, index_dump)
            pickle.dump(self.docmap, docmap_dump)

    def load(self):
        with type(self).index_file_path.open("rb") as index_dump, type(self).docmap_file_path.open("rb") as docmap_dump:
            self.index = pickle.load(index_dump)
            self.docmap = pickle.load(docmap_dump)
       

def build_command() -> None:
    index = InvertedIndex()
    index.build()
    index.save()
        
def search_command(search_query: str) -> list[dict]:
    index = InvertedIndex()
    index.load()

    query_tokens = process_text_to_tokens(search_query)

    results, doc_ids_seen = [], set()
    for token in query_tokens:
        doc_ids = index.get_documents(token)
        for doc_id in doc_ids:
            if doc_id not in doc_ids_seen:
                doc_ids_seen.add(doc_id)
                movie_doc = index.docmap[doc_id]
                results.append(movie_doc)
                if len(results) == 5:
                    return results
                
    return results
