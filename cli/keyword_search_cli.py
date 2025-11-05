#!/usr/bin/env python3

import argparse
from inverted_index import search_command, build_command, tf_command
from commands import idf_command, tfidf_command, bm25_idf_command

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
    args = parser.parse_args()

    match args.command:
        case "search":
            print("Building inverted index...")
            try:
                movie_docs = search_command(args.query)
                print("Inverted Index built!!")
            except FileNotFoundError:
                print("Index not built.")
                return
            
            for i, movie in enumerate(movie_docs):
                print(f"{i+1}. Movie ID: {movie["id"]}, Movie Title: {movie["title"]}")
                
        case "build":
            build_command()

        case "tf":
            print(f"Searching for {args.search_term} in document {args.document_ID}...")
            tf = tf_command(args.document_ID, args.search_term)
            print(f"Result:")
            print(f"Term: {args.search_term}, Doc ID: {args.document_ID}, Term Frequency: {tf}")
        case "idf":
            print(f"Calculating the IDF score of {args.term}...")
            idf = idf_command(args.term)
            print(f"Inverse document frequency of '{args.term}': {idf:.2f}")

        case "tfidf":
            doc_id = args.document_ID
            term = args.term
            print(f"Finding TF-IDF score of '{term} in document '{doc_id}...")
            tf_idf = tfidf_command(doc_id=doc_id, term=term)
            print(f"TF-IDF score of '{term}' in document '{doc_id}': {tf_idf:.2f}")
        
        case "bm25idf":
            term = args.term
            print(f"Calculating the BM25 IDF score of '{term}'...")
            bm25idf = bm25_idf_command(term=term)
            print(f"BM25 IDF score of '{term}': {bm25idf:.2f}")

        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()