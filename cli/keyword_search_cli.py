#!/usr/bin/env python3

import argparse
from keyword_search import keyword_search
from helpers import remove_all_punctuation_lowercase
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index, save it to disk")
    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            query = remove_all_punctuation_lowercase(args.query)
            keyword_search(query)
        case "build":
            inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            
            # doc ids for token merida
            merida_doc_ids = inverted_index.get_documents("merida")
            print(f"First document for token 'merida' = {merida_doc_ids[0]}")
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()