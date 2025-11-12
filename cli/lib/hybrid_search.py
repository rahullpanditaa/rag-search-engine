from pathlib import Path
from .utils import get_movie_data_from_file
from .inverted_index import InvertedIndex
from .chunked_semantic_search import ChunkedSemanticSearch
from .constants import INDEX_FILE_PATH
from typing import Optional
from .enhance_query import enhance_query

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

    def rrf_search(self, query, k, limit=10) -> list[dict]:
        bm25_results = self._bm25_search(query=query, limit=limit*500)
        bm25_ranks = [doc_id for doc_id, _ in bm25_results]

        semantic_results = self.semantic_search.search_chunks(query=query, limit=limit*500)
        semantic_ranks = [result["id"] for result in semantic_results]
        # all doc ids in both results
        all_doc_ids = set(bm25_ranks) | set(semantic_ranks)
        document_map = {doc["id"]: doc for doc in self.documents}

        doc_scores_map = {}

        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.index(doc_id) + 1 if doc_id in bm25_ranks else None
            semantic_rank = semantic_ranks.index(doc_id) + 1 if doc_id in semantic_ranks else None

            rrf = 0.0
            if bm25_rank is not None:
                rrf += rrf_score(rank=bm25_rank, k=k)
            if semantic_rank is not None:
                rrf += rrf_score(rank=semantic_rank, k=k)
            
            doc = document_map.get(doc_id)
            if doc:
                doc_scores_map[doc_id] = {
                    "id": doc_id,
                    "title": doc.get("title", ""),
                    "document": doc.get("description", "")[:100],
                    "bm25_rank": bm25_rank,
                    "semantic_rank": semantic_rank,
                    "rrf_score": rrf
                }
        sorted_results = sorted(doc_scores_map.values(), key=lambda d: d["rrf_score"], reverse=True)
        return sorted_results[:limit]

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

def rrf_search(query: str, k: int=60, limit: int=5, enhance: Optional[str]=None) -> dict:
    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)
    
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query=query, method=enhance)
    
    query_to_use = enhanced_query if enhanced_query else query
    return {
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query_used": query_to_use,
        "results": searcher.rrf_search(query=query_to_use, k=k, limit=limit)
    }

def rrf_search_command(query: str, k: int=60, limit: int=5, enhance: Optional[str]=None) -> None:
    print(f"Searching for '{query}'. Generating upto {limit} results...")
    results = rrf_search(query=query, k=k, limit=limit, enhance= enhance)
    
    if results["enhanced_query"] is not None and results["enhanced_query"] != query:
        print(f"Enhanced query ({enhance}): '{query}' -> '{results['enhanced_query']}'\n")

    
    for i, result in enumerate(results["results"], 1):
        print(f"\n{i}. {result['title']}")
        print(f"RRF Score: {result['rrf_score']:.4f}")
        print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
        print(f"{result['document']}...")


def hybrid_score(bm25_score, semantic_score, alpha=0.5):
    return alpha * bm25_score + (1 - alpha) * semantic_score

def rrf_score(rank, k=60):
    return 1 / (k + rank)