import argparse
from lib.hybrid_search.logic import normalize_command
from lib.hybrid_search.commands import (
    weighted_search_command,
    rrf_search_command
)
from dotenv import load_dotenv

def main() -> None:
    parser = argparse.ArgumentParser(description="Hybrid Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    normalize_parser = subparsers.add_parser("normalize", help="Accept a list of scores, print the normalized scores")
    normalize_parser.add_argument("scores", type=float, nargs='+', help="List of scores to be normalized")
    
    weighted_search_parser = subparsers.add_parser("weighted-search", help="Search for query using weighted hybrid score (BM25 + Semantic Score)")
    weighted_search_parser.add_argument("query", type=str, help="Query to search for")
    weighted_search_parser.add_argument("--alpha", type=float, nargs='?', default=0.5, help="Constant used to dynamically control weighing between 2 scores")
    weighted_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return from the search")    
    
    rrf_search_parser = subparsers.add_parser("rrf-search", help="Search for query using RRF scores")
    rrf_search_parser.add_argument("query", type=str, help="Query to search for")
    rrf_search_parser.add_argument("--k", type=int, nargs='?', default=60, help="Tunable k parameter to control weight given to higher vs lower ranks")
    rrf_search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")
    rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
    rrf_search_parser.add_argument("--rerank-method", type=str,choices=["individual", "batch", "cross_encoder"], help="Re-ranking search results" )
    rrf_search_parser.add_argument("--evaluate", type=bool, nargs='?', default=False, help="Rate the search results")

    args = parser.parse_args()

    match args.command:
        case "normalize":
            normalize_command(scores=args.scores)
        case "weighted-search":
            weighted_search_command(query=args.query, alpha=args.alpha, limit=args.limit)
        case "rrf-search":
            rrf_search_command(query=args.query, 
                               k=args.k, limit=args.limit, 
                               enhance=args.enhance, 
                               re_rank=args.rerank_method, 
                               evaluate=args.evaluate)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()