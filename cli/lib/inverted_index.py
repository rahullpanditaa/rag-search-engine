from cli.lib.utils import process_text_to_tokens, get_movie_data_from_file, BM25_K1, BM25_B
from pathlib import Path
import pickle
from collections import Counter
from math import log

class InvertedIndex:
    index: dict[str, set[int]]
    docmap: dict[int, dict[str, object]]
    term_frequencies: dict[int, Counter[str]]
    doc_lengths: dict[int, int]

    cache_dir_path = Path(__file__).resolve().parent.parent / "cache"
    index_file_path = cache_dir_path / "index.pkl"
    docmap_file_path = cache_dir_path / "docmap.pkl"
    term_frequencies_file_path = cache_dir_path / "term_frequencies.pkl"
    doc_lengths_path = cache_dir_path / "doc_lengths.pkl"

    def __init__(self):
        # dict mapping tokens(str) to sets of doc ids
        self.index = {}

        # dict mapping doc id to full doc objects {id, title, desciption}
        self.docmap = {}

        # dict mapping doc ids to Counter objects
        # doc id -> Counter keeping track of how often each term appears in doc
        self.term_frequencies = {}

        self.doc_lengths = {}

    def __add_document(self, doc_id: int, text: str):
        tokens = process_text_to_tokens(text=text)
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

        # if doc_id not in self.term_frequencies:
        self.term_frequencies[doc_id] = Counter(tokens)  

        # total number of tokens in each doc
        for doc_id, tokens_counter in self.term_frequencies.items():
            self.doc_lengths[doc_id] = tokens_counter.total()

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
    
    def get_bm25_tf(self, doc_id: int, term: str, k1: float=BM25_K1, b: float=BM25_B) -> float:
        # raw tf
        raw_tf = self.get_tf(doc_id=doc_id, term=term)

        # given doc ids length
        doc_length = self.doc_lengths[doc_id]

        avg_doc_length = self.__get_avg_doc_length()

        ratio = doc_length / avg_doc_length
        
        length_norm = 1 - b + b * (ratio)

        # saturate tf
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def bm25(self, doc_id: int, term: str) -> float:
        bm25tf = self.get_bm25_tf(doc_id=doc_id, term=term)
        bm25idf = self.get_bm25_idf(term=term)
        return bm25tf * bm25idf    
    
    def bm25_search(self, query: str, limit: int):
        query_tokens = process_text_to_tokens(query)
        docs_bm25_scores: dict[int, int] = {}

        for token in query_tokens:
            docs_having_token = self.index.get(token, {})
            for doc_id in docs_having_token:
                score = self.bm25(doc_id=doc_id, term=token)
                docs_bm25_scores[doc_id] = docs_bm25_scores.get(doc_id, 0.0) + score
        
        # list of tuples(doc_id, bm25 score)
        sorted_scores = sorted(docs_bm25_scores.items(), key=lambda d: d[1], reverse=True)
        return sorted_scores[:limit]
    
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
        with type(self).index_file_path.open("wb") as index_dump, type(self).docmap_file_path.open("wb") as docmap_dump, type(self).term_frequencies_file_path.open("wb") as term_frequencies_dump, type(self).doc_lengths_path.open("wb") as doc_lengths_dump:
            pickle.dump(self.index, index_dump)
            pickle.dump(self.docmap, docmap_dump)
            pickle.dump(self.term_frequencies, term_frequencies_dump)
            pickle.dump(self.doc_lengths, doc_lengths_dump)

    def load(self):
        with type(self).index_file_path.open("rb") as index_dump, type(self).docmap_file_path.open("rb") as docmap_dump, type(self).term_frequencies_file_path.open("rb") as term_frequencies_dump, type(self).doc_lengths_path.open("rb") as doc_lengths_dump:
            self.index = pickle.load(index_dump)
            self.docmap = pickle.load(docmap_dump)
            self.term_frequencies = pickle.load(term_frequencies_dump)
            self.doc_lengths = pickle.load(doc_lengths_dump)

    def __get_avg_doc_length(self) -> float:
        # calculate avg doc length
        # sum of length of all docs / number of docs
        number_of_docs = len(self.doc_lengths)
        if number_of_docs == 0:
            # no docs
            return 0.0

        sum_of_lengths = sum(self.doc_lengths.values())

        avg = sum_of_lengths / number_of_docs
        return avg
        
       

