from pathlib import Path
from lib.inverted_index import InvertedIndex
from lib.chunked_semantic_search.logic import ChunkedSemanticSearch
from lib.constants import INDEX_FILE_PATH

class HybridSearch:
    """
    Combines BM25 keyword search and chunked semantic search to produce
    hybrid ranked retrieval results.

    The class initializes:
    - A chunked semantic search engine (precomputes or loads chunk embeddings)
    - A BM25 inverted index (creates it if missing)

    Two fusion strategies are supported:
    1. Weighted combination of normalized BM25 and semantic scores
    2. Reciprocal Rank Fusion (RRF)

    Attributes:
        documents (list[dict]):
            Collection of movie documents used for both searches.
        semantic_search (ChunkedSemanticSearch):
            Engine performing chunk-level semantic similarity search.
        idx (InvertedIndex):
            Inverted index used for BM25 keyword scoring."""
    def __init__(self, documents):
        """
        Initialize hybrid search by preparing the chunked semantic search engine and
        constructing or loading the BM25 inverted index.

        Arguments:
            documents (list[dict]):
                Static list of movie dictionaries with keys:
                {id, title, description}."""
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents=documents)

        self.idx = InvertedIndex()
        if not INDEX_FILE_PATH.exists():
            self.idx.build()
            self.idx.save()        

    def _bm25_search(self, query, limit):
        """
        Run BM25 search on the inverted index.

        Arguments:
            query (str): Search query.
            limit (int): Number of results to return.

        Returns:
            list[tuple[int, float]]: (doc_id, bm25_score) pairs sorted by score."""
        self.idx.load()
        return self.idx.bm25_search(query=query, limit=limit)
    
    def weighted_search(self, query, alpha, limit=5) -> list[dict]:
        """
        Perform hybrid search using a weighted combination of
        BM25 and chunked semantic scores.

        Workflow:
            • Compute BM25 scores for the query.
            • Compute semantic chunk scores for the query.
            • Normalize both score sets to [0, 1].
            • Combine scores using: alpha*bm25 + (1 - alpha)*semantic.
            • Sort results by hybrid score.

        Arguments:
            query (str): Search query.
            alpha (float): Weight for BM25 (range 0–1).
            limit (int): Number of results to return.

        Returns:
            list[dict]: Ranked results containing:
                id, title, snippet, bm25_score, semantic_score, hybrid_score."""
        
        # list of tuples - (doc_id, bm25 score) sorted by score 
        bm25_results = self._bm25_search(query=query, limit=limit*500)
        bm25_results = sorted(bm25_results, key=lambda item: item[0])
        # sorted by doc_id -> 1, 2, ...
        bm25_dict = {doc_id: score for doc_id, score in bm25_results}

        # list of dict - {doc_id, doc_title, doc_description, score, doc_metadata or {}}
        semantic_results = self.semantic_search.search_chunks(query=query, limit=limit*500)
        semantic_results = sorted(semantic_results, key=lambda item: item["id"])
        semantic_dict = {result["id"]: result["score"] for result in semantic_results}

        all_doc_ids = sorted(set(bm25_dict.keys()) | set(semantic_dict.keys()))
        
        bm_25_scores = [bm25_dict.get(doc_id, 0.0) for doc_id in all_doc_ids]
        normalized_bm25_scores = normalize_scores(bm_25_scores)        

        semantic_scores = [semantic_dict.get(doc_id, 0.0) for doc_id in all_doc_ids]
        normalized_semantic_scores = normalize_scores(semantic_scores)
    
        document_map = {doc["id"]: doc for doc in self.documents}
        doc_scores = {}        
        for i, doc_id in enumerate(all_doc_ids):
            bm25_score = normalized_bm25_scores[i]
            semantic_score = normalized_semantic_scores[i]
            hybrid = _hybrid_score(bm25_score=bm25_score,
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

    def rrf_search(self, query, k: int=60, limit=10) -> list[dict]:
        """
        Perform Reciprocal Rank Fusion (RRF) between BM25 results and
        chunked semantic search results.

        RRF combines rankings rather than raw scores, making it robust
        when the two systems use very different scoring scales.

        Formula:
            RRF(rank) = 1 / (k + rank)

        Arguments:
            query (str): Search query.
            k (int): RRF smoothing constant (default 60).
            limit (int): Number of results to return.

        Returns:
            list[dict]: Ranked result dictionaries containing:
                id, title, snippet, bm25_rank, semantic_rank, rrf_score."""
        bm25_results = self._bm25_search(query=query, limit=limit*100)
        bm25_ranks = [doc_id for doc_id, _ in bm25_results]

        semantic_results = self.semantic_search.search_chunks(query=query, limit=limit*100)
        semantic_ranks = [result["id"] for result in semantic_results]
        # all doc ids in both results
        all_doc_ids = list(dict.fromkeys(bm25_ranks + semantic_ranks))
        document_map = {doc["id"]: doc for doc in self.documents}

        doc_scores_map = {}

        for doc_id in all_doc_ids:
            bm25_rank = bm25_ranks.index(doc_id) + 1 if doc_id in bm25_ranks else len(bm25_ranks) + 1
            semantic_rank = semantic_ranks.index(doc_id) + 1 if doc_id in semantic_ranks else len(semantic_ranks) + 1

            rrf = 0.0
            if bm25_rank is not None:
                rrf += _rrf_score(rank=bm25_rank, k=k)
            if semantic_rank is not None:
                rrf += _rrf_score(rank=semantic_rank, k=k)
            
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
    """
    Normalize a list of scores to the range [0, 1].

    If all scores are identical, returns all 1.0.

    Arguments:
        scores (list[float]): Raw scores.

    Returns:
        list[float]: Normalized scores."""
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

def _hybrid_score(bm25_score, semantic_score, alpha=0.5):
    """
      Linear hybrid scoring function.

    Computes:
        alpha * bm25_score + (1 - alpha) * semantic_score

    Arguments:
        bm25_score (float): Normalized BM25 score.
        semantic_score (float): Normalized semantic score.
        alpha (float): Weight for BM25.

    Returns:
        float: Hybrid score."""
    return alpha * bm25_score + (1 - alpha) * semantic_score

def _rrf_score(rank, k=60):
    """
    Compute Reciprocal Rank Fusion (RRF) score for a given rank.

    Formula:
        RRF = 1 / (k + rank)

    Arguments:
        rank (int): Rank position of the document (1 = highest).
        k (int): Smoothing constant to avoid large swings.

    Returns:
        float: RRF score."""
    return 1 / (k + rank)