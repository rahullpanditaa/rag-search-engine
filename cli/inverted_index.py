from helpers import process_text_to_tokens, get_movie_data_from_file, BM25_K1
from pathlib import Path
import pickle
from collections import Counter
from math import log

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, dict[str, object]]
    term_frequencies: dict[int, Counter[str]]

    cache_dir_path = Path(__file__).resolve().parent.parent / "cache"
    index_file_path = cache_dir_path / "index.pkl"
    docmap_file_path = cache_dir_path / "docmap.pkl"
    term_frequencies_file_path = cache_dir_path / "term_frequencies.pkl"

    def __init__(self):
        # dict mapping tokens(str) to sets of doc ids
        self.index = {}

        # dict mapping doc id to full doc objects {id, title, desciption}
        self.docmap = {}

        # dict mapping doc ids to Counter objects
        # doc id -> Counter keeping track of how often each term appears in doc
        self.term_frequencies = {}

    def __add_document(self, doc_id: int, text: str):
        # tokenize input text
        tokens = process_text_to_tokens(text=text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        # if doc_id not in self.term_frequencies:
        self.term_frequencies[doc_id] = Counter(tokens)   

    def get_tf(self, doc_id: int, term: str) -> int:
        term_token = process_text_to_tokens(term)
        if len(term_token) != 1:
            raise ValueError("multiple tokens given, need a single token")
        tf = self.term_frequencies[doc_id][term_token[0]]
        return tf
    
    def get_bm25_idf(self, term: str) -> float:
        tokenized_term = process_text_to_tokens(term)
        if len(tokenized_term) != 1:
            raise ValueError(f"Given term {term} has multiple tokens, single required")
        
        # term appears in how many docs?
        df = 0
        for _, term_count in self.term_frequencies.items():
            if tokenized_term[0] in term_count:
                df += 1
        
        # N - total docs
        n = len(self.term_frequencies)

        return log((n - df + 0.5) / (df + 0.5) + 1)
    
    def get_bm25_tf(self, doc_id: int, term: str, k1=BM25_K1) -> float:
        # raw tf
        raw_tf = self.get_tf(doc_id=doc_id, term=term)

        # saturate tf
        return (raw_tf * (k1 + 1)) / (raw_tf + k1)
            
    
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
        with type(self).index_file_path.open("wb") as index_dump, type(self).docmap_file_path.open("wb") as docmap_dump, type(self).term_frequencies_file_path.open("wb") as term_frequencies_dump:
            pickle.dump(self.index, index_dump)
            pickle.dump(self.docmap, docmap_dump)
            pickle.dump(self.term_frequencies, term_frequencies_dump)

    def load(self):
        with type(self).index_file_path.open("rb") as index_dump, type(self).docmap_file_path.open("rb") as docmap_dump, type(self).term_frequencies_file_path.open("rb") as term_frequencies_dump:
            self.index = pickle.load(index_dump)
            self.docmap = pickle.load(docmap_dump)
            self.term_frequencies = pickle.load(term_frequencies_dump)
       

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

def tf_command(doc_id: int, search_term: str) -> int:
    index = InvertedIndex()
    index.load()

    return index.get_tf(doc_id=doc_id, term=search_term)