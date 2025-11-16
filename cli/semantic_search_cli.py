#!/usr/bin/env python3

import argparse

from lib.semantic_search.commands import (
    verify_model_command,
    embed_text_command,
    verify_embeddings_command,
    embed_query_text_command,
    search_command,
    chunk_command
)
from lib.chunked_semantic_search import (
    semantic_chunk_command, 
    embed_chunks_command,
    search_chunked_command
)

def main():
    parser = argparse.ArgumentParser(description="Semantic Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available semantic search commands")

    verify_parser = subparsers.add_parser("verify", help="Print information about embedding model used.")
    
    embed_text_parser = subparsers.add_parser("embed_text", help="Generates an embedding for given string")
    embed_text_parser.add_argument("text", type=str, help="Single string argument to generate embedding for.")
    
    verify_embeddings_parser = subparsers.add_parser("verify_embeddings", help="Print information about movie docs embeddings")
    
    embed_query_parser = subparsers.add_parser("embedquery", help="Create embedding for argument query")
    embed_query_parser.add_argument("query", type=str, help="Query to generate embeddings for")
    
    search_parser = subparsers.add_parser("search", help="Search for argument query, generate similarity score for each document")
    search_parser.add_argument("query", type=str, help="Query to compare with each doc and calculate similarity scores for")
    search_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")
    
    chunk_parser = subparsers.add_parser("chunk", help="Chunk input text")
    chunk_parser.add_argument("text", type=str, help="Text to chunk")
    chunk_parser.add_argument("--chunk-size", type=int, nargs='?', default=200, help="Size of an individual chunk in words")
    chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="How many words should overlap between consecutive chunks")

    semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Split text on sentence boundaries to preserve meaning")
    semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
    semantic_chunk_parser.add_argument("--max-chunk-size", type=int, nargs='?', default=4, help="Maximum size of a chunk in sentences")
    semantic_chunk_parser.add_argument("--overlap", type=int, nargs='?', default=0, help="How many sentences should overlap between chunks")
    
    embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Embed all the chunks in a doc")
    
    search_chunked_parser = subparsers.add_parser("search_chunked", help="Search for a query in all the chunks, generate similarity score for each chunk and arg query")
    search_chunked_parser.add_argument("query", type=str, help="Search query")
    search_chunked_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")

    args = parser.parse_args()

    match args.command:
        case "verify":
            verify_model_command()

        case "embed_text":
            embed_text_command(args.text)
        case "verify_embeddings":
            verify_embeddings_command()
        case "embedquery":
            embed_query_text_command(args.query)

        case "search":
            # query = args.query
            # limit = args.limit
            # print(f"Calculating similarity scores for given query '{query}' for {limit} docs...")
            search_command(query=args.query, limit=args.limit)
        
        case "chunk":
            chunk_command(args.text, chunk_size=args.chunk_size, overlap=args.overlap)

        case "semantic_chunk":
            chunks = semantic_chunk_command(args.text, args.max_chunk_size, args.overlap)
            print(f"Semantically chunking {len(args.text)} characters")
            for i, ch in enumerate(chunks, 1):
                print(f"{i}. {ch}")
        case "embed_chunks":
            embed_chunks_command()
        
        case "search_chunked":
            print(f"Searching for '{args.query}'. Generating upto {args.limit} results...")
            search_chunked_command(query=args.query, limit=args.limit)
        case _:
            parser.print_help()

if __name__ == "__main__":
    main()