#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text, verify_embeddings, embed_query_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available semantic search commands")

    verify_parser = subparsers.add_parser("verify", help="Print information about embedding model used.")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Generates an embedding for given string")
    embed_text_parser.add_argument("text", type=str, help="Single string argument to generate embedding for.")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Print information about movie docs embeddings")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Create embedding for argument query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embeddings for")
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)
        case "verify_embeddings":
            verify_embeddings()
        case "embedquery":
            embed_query_text(args.query)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()