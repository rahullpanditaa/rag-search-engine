#!/usr/bin/env python3

import argparse
from lib.semantic_search import verify_model, embed_text

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available semantic search commands")

    verify_parser = subparsers.add_parser("verify", help="Print information about embedding model used.")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Generates an embedding for given string")
    embed_text_parser.add_argument("text", type=str, help="Single string argument to generate embedding for.")
    
    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()