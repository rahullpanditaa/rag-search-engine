from pathlib import Path

from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .constants import INDEX_FILE_PATH

class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents=documents)

        self.idx = InvertedIndex()
        if not Path.exists(INDEX_FILE_PATH):
            self.idx.build()
            self.idx.save()

    def _bm25_search(self, query, limit):
        self.idx.load()
        return self.idx.bm25_search(query=query, limit=limit)
    
    def weighted_search(self, query, alpha, limit=5):
        raise NotImplementedError("Weighted hybrid search is not implemented yet.")
    
    def rrf_search(self, query, k, limit=10):
        raise NotImplementedError("RRF hybrid search is not implemented yet.")

def normalize_scores(scores: list[float]) -> list[float]:
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)

    if min_score == max_score:
        return [1.0 for i in range(len(scores))]
    
    normalized_scores = []
    for score in scores:
        value = (score - min_score) / (max_score - min_score)
        normalized_scores.append(value)
    return normalized_scores

def normalize_command(scores: list[float]) -> None:
    normalized_scores = normalize_scores(scores=scores)
    if not normalized_scores:
        print("No scores given to normalize.")
        return
    for score in normalized_scores:
        print(f"* {score:.4f}")