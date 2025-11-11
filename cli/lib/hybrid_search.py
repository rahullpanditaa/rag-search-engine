from pathlib import Path
from .utils import get_movie_data_from_file
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
    
    def weighted_search(self, query, alpha, limit=5) -> list[dict]:
        # list of tuples - (doc_id, bm25 score) sorted by score 
        bm25_results = self._bm25_search(query=query, limit=limit*500)
        bm25_results = sorted(bm25_results, key=lambda item: item[0])
        # sorted by doc_id -> 1, 2, ...
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}

        # list of dict - {doc_id, doc_title, doc_description, score, doc_metadata or {}}
        semantic_results = self.semantic_search.search_chunks(query=query, limit=limit*500)
        semantic_results = sorted(semantic_results, key=lambda item: item["id"])
        semantic_dict = {result["id"]: result["score"] for result in semantic_results}

        all_doc_ids = set(bm25_dict.keys()) | set(semantic_dict.keys())
        
        bm_25_scores = [bm25_dict.get(doc_id, 0.0) for doc_id in all_doc_ids]
        normalized_bm25_scores = normalize_scores(bm_25_scores)        

        semantic_scores = [semantic_dict.get(doc_id, 0.0) for doc_id in all_doc_ids]
        normalized_semantic_scores = normalize_scores(semantic_scores)
    
        document_map = {doc["id"]: doc for doc in self.documents}
        doc_scores = {}        
        for i, doc_id in enumerate(all_doc_ids):
            bm25_score = normalized_bm25_scores[i]
            semantic_score = normalized_semantic_scores[i]
            hybrid = hybrid_score(bm25_score=bm25_score,
                                  semantic_score=semantic_score,
                                  alpha=alpha)
            
            # get matching doc from self.documents, or None
            doc = document_map.get(doc_id)
            if doc:
                doc_scores[doc_id] = {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "document": doc.get("description", "")[:100],
                    "bm25_score": bm25_score,
                    "semantic_score": semantic_score,
                    "hybrid_score": hybrid
                }
        sorted_results = sorted(doc_scores.values(), key=lambda d: d["hybrid_score"], reverse=True)
        return sorted_results[:limit]

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

def weighted_search(query: str, alpha: float=0.5, limit: int=5) -> list[dict]:
    movies = get_movie_data_from_file()
    searcher = HybridSearch(movies)
    return searcher.weighted_search(query=query, alpha=alpha, limit=limit)

def weighted_search_command(query: str, alpha: float=0.5, limit: int=5) -> None:
    print(f"Searching for '{query}'. Generating upto {limit} results...")
    results = weighted_search(query=query, alpha=alpha, limit=limit)    
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"Hybrid Score: {result['hybrid_score']:.4f}")
        print(f"BM25: {result['bm25_score']:.4f}, Semantic: {result['semantic_score']:.4f}")
        print(f"{result['document']}")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score