#!/usr/bin/env python3

import argparse
from lib.keyword_search_commands import (
    search_command, 
    build_command, 
    tf_command, 
    idf_command, 
    tfidf_command, 
    bm25_idf_command,
    bm25_tf_command, 
    bm25_search_command
)
from lib.constants import BM25_K1, BM25_B

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index, save it to disk")
    
    tf_parser = subparsers.add_parser("tf", help="Print the term frequency of given term in given doc ID")
    tf_parser.add_argument("document_ID", type=int, help="Document ID")
    tf_parser.add_argument("search_term", help="Term to search for in doc")
    
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for given term")
    idf_parser.add_argument("term", type=str, help="Term whose IDF is to be calculated")
    
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate the TF-IDF score for given document and term")
    tfidf_parser.add_argument("document_ID", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Search term whose TF-IDF score is to be calculated")
    
    bm25_idf_parser = subparsers.add_parser("bm25idf", help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term whose BM25 IDF score is to be calculated")
    
    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("document_ID", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")
    
    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    
    args = parser.parse_args()

    match args.command:
        case "search":
            search_command(search_query=args.query)                
        case "build":
            build_command()
        case "tf":
            tf_command(doc_id=args.document_ID, search_term=args.search_term)
        case "idf":
            idf_command(term=args.term)
        case "tfidf":
            tfidf_command(doc_id=args.document_ID, term=args.term)        
        case "bm25idf":
            bm25_idf_command(term=args.term)
        case "bm25tf":
            bm25_tf_command(doc_id=args.document_ID, term=args.term, k1=args.k1, b=args.b)        
        case "bm25search":
            bm25_search_command(query=args.query)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()