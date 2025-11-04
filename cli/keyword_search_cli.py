#!/usr/bin/env python3

import argparse
from cli.keyword_search import keyword_search
from helpers import remove_all_punctuation_lowercase


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            print(f"Searching for: {args.query}")
            query = remove_all_punctuation_lowercase(args.query)
            keyword_search(query)
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()