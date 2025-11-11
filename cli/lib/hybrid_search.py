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
        # list of tuples - (doc_id, bm25 score) sorted by score 
        bm25_results = self._bm25_search(query=query, limit=limit*500)
        bm25_results = sorted(bm25_results, key=lambda item: item[0])
        # sorted by doc_id -> 1, 2, ...

        # list of dict - {doc_id, doc_title, doc_description, score, doc_metadata or {}}
        semantic_results = self.semantic_search.search_chunks(query=query, limit=limit*500)
        semantic_results = sorted(semantic_results, key=lambda item: item["doc_id"])

        bm_25_scores = []
        for _, score in bm25_results:
            bm_25_scores.append(score)
        normalized_bm25_scores = normalize_scores(bm_25_scores)

        semantic_scores = []
        for result in semantic_results:
            semantic_scores.append(result["score"])
        normalized_semantic_scores = normalize_scores(semantic_scores)

        results = []
        for doc_id, doc in enumerate(self.documents, 1):
            keyword_score = normalized_bm25_scores.index(doc_id-1)
            semantic_score = normalized_semantic_scores.index(doc_id-1)
            hybrid_score_doc = hybrid_score(bm25_score=keyword_score, semantic_score=semantic_score, alpha=alpha)
            results.append({
                doc_id: doc,
                "bm25_score": keyword_score,
                "semantic_score": semantic_score,
                "hybrid_score": hybrid_score_doc
            })
        sorted_scores = sorted(results, key=lambda result: result["hybrid_score"], reverse=True)
        return sorted_scores
    
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

def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score