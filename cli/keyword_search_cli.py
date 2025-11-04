#!/usr/bin/env python3

import argparse
from keyword_search import keyword_search
from helpers import remove_all_punctuation_lowercase, process_text_to_tokens, tokenize
from inverted_index import InvertedIndex


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index, save it to disk")
    args = parser.parse_args()

    inverted_index = InvertedIndex()
    match args.command:
        case "search":
            try:
                inverted_index.load()
            except FileNotFoundError:
                print("Index not built.")
                return
            query_tokens = process_text_to_tokens(args.query)

            results = []
            doc_ids_matched = set()
            for token in query_tokens:
                doc_ids = inverted_index.get_documents(token)
                for doc_id in doc_ids:
                    if doc_id not in doc_ids_matched:
                        doc_ids_matched.add(doc_id)
                        results.append(doc_id)
            
            print(f"Total results: {len(results)}")

            for doc_id in results:
                movie = inverted_index.docmap[doc_id]
                if "Klansman" in movie["title"]:
                    print(f"Found Klansman at ID: {doc_id}")
                    break

            for doc_id in results:
                movie = inverted_index.docmap[doc_id]
                if "Madrasapattinam" in movie["title"]:
                    print(f"Found Madrasapattinam at ID: {doc_id}")
                    break
            for doc_id in results[:5]:
                movie = inverted_index.docmap[doc_id]
                print(f"Movie ID: {doc_id}, Title: {movie["title"]}")
                
        case "build":
            # inverted_index = InvertedIndex()
            inverted_index.build()
            inverted_index.save()
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()