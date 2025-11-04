#!/usr/bin/env python3

import argparse
from inverted_index import search_command, build_command


def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    build_parser = subparsers.add_parser("build", help="Build the inverted index, save it to disk")
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
            
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()