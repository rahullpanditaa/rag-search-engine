import argparse
from lib.augmented_generation import (
    rag_command, 
    summarize_command,
    citations_command
)


def main():
    parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    rag_parser = subparsers.add_parser(
        "rag", help="Perform RAG (search + generate answer)"
    )
    rag_parser.add_argument("query", type=str, help="Search query for RAG")

    summarize_parser = subparsers.add_parser("summarize", help="Summarize search results")
    summarize_parser.add_argument("query", type=str, help="Search query for RAG")
    summarize_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results")
    
    citations_parser = subparsers.add_parser("citations", help="Generated answer will reference its sources")
    citations_parser.add_argument("query", type=str, help="Search query")
    citations_parser.add_argument("--limit", type=int, nargs='?', default=5, help="Number of results to return")
    
    args = parser.parse_args()

    match args.command:
        case "rag":
            query = args.query
            rag_command(query=query)
        case "summarize":
            summarize_command(query=args.query, limit=args.limit)
        case "citations":
            citations_command(query=args.query, limit=args.limit)
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()