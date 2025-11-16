
from lib.utils import get_movie_data_from_file
from .logic import HybridSearch
from typing import Optional
from lib.enhance_query import enhance_query
from lib.re_rank_results import re_rank_scores
from .utils import llm_evaluation_prompt, generate_response_evaluate_results


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

def rrf_search(query: str, k: int=60, 
               limit: int=5, 
               enhance: Optional[str]=None,
               re_rank: Optional[str]= None) -> dict:
    movies = get_movie_data_from_file()
    searcher = HybridSearch(documents=movies)

    print(f"Original query: {query}")
    
    enhanced_query = None
    if enhance:
        enhanced_query = enhance_query(query=query, method=enhance)
    
    query_to_use = enhanced_query if enhanced_query else query
    if query_to_use == enhanced_query:
        print(f"Enhanced query ({enhance}): '{query}' -> '{query_to_use}'")

    # initial rrf search (list of docs ranked by rrf score)
    results = searcher.rrf_search(query=query_to_use, k=k, limit=limit)
    print(f"Initial RRF search before re-ranking (list of docs ranked by RRF score):")
    for i, doc in enumerate(results, 1):
        print(f"{i}. Movie: {doc['title']}, RRF Score: {doc['rrf_score']}")
    
    # results ranked by re_rank method
    re_ranked_results = re_rank_scores(query=query_to_use, scores=results, method=re_rank)
    # print(f"Docs/Movies re-ranked by '{re_rank}' method:")
    # for i, doc in enumerate(re_ranked_results, 1):
    #     print(f"{i}. {doc['title']}")

    return {
        "enhanced_query": enhanced_query,
        "enhance_method": enhance,
        "query_used": query_to_use,
        "results": re_ranked_results
    }


def rrf_search_command(query: str, k: int=60, limit: int=5, 
                       enhance: Optional[str]=None,
                       re_rank: Optional[str]=None,
                       evaluate: Optional[bool]=None) -> None:    
    search_limit = limit
    if re_rank:
        search_limit *= 5

    results = rrf_search(query=query, k=k, limit=search_limit, enhance= enhance, re_rank=re_rank)   

    if re_rank:
        print(f"Reranking top {limit} results using {re_rank} method...\n")
    print(f"Reciprocal Rank Fusion results for '{results['query_used']}' (k = {k}):")
    print()
    for i, result in enumerate(results["results"][:limit], 1):
        print(f"\n{i}. {result['title']}")
        if re_rank == "individual":
            re_rank_score = result["re_rank_score"]
            print(f"Rerank Score: {re_rank_score:.3f}/10")
        if re_rank == "batch":
            rank = result["rank"]
            print(f"Rerank Rank: {rank}")
        if re_rank == "cross_encoder":
            cross_score = result["cross_encoder_score"]
            print(f"Cross Encoder Score: {cross_score:.3f}")
        print(f"RRF Score: {result['rrf_score']:.3f}")
        print(f"BM25 Rank: {result['bm25_rank']}, Semantic Rank: {result['semantic_rank']}")
        print(f"{result['document']}...")
        print()
    
    if evaluate:
        _evaluate_results(query=results["query_used"], results=results["results"][:limit])

def _evaluate_results(query: str, results: list[dict]) -> None:
    prompt = llm_evaluation_prompt(query=query, results=results)
    scores = generate_response_evaluate_results(prompt=prompt)

    evaluation_results = []
    for i, score in enumerate(scores):
        evaluation_results.append({
            "title": results[i]['title'],
            "score": score
        })
        # print(f"{i+1}. {results[i]['title']}: {score}/3")
    sorted_eval_results = sorted(evaluation_results, key=lambda d: d['score'], reverse=True)
    for i, eval_result in enumerate(sorted_eval_results, 1):
        print(f"{i}. {eval_result['title']}: {eval_result['score']}/3")