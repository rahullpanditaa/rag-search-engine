from lib.constants import (
    DEFAULT_SEMANTIC_CHUNK_SIZE,
    DEFAULT_CHUNK_OVERLAP
)
from .logic import ChunkedSemanticSearch, semantic_chunk
from lib.utils import get_movie_data_from_file
import re

def semantic_chunk(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    text = text.strip()
    if not text:
        return []
    
    sentences = re.split(r"(?<=[.!?])\s+", text)

    # only one sentence in text, and it does not end with a . or ! or ?
    if len(sentences) == 1 and not sentences[0].endswith((".", "!", "?")):
        sentences = [text]

    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    i = 0
    num_of_sentences = len(sentences)
    while i < num_of_sentences:
        sentences_in_chunk = sentences[i: i+max_chunk_size]
        if chunks and len(sentences_in_chunk) <= overlap:
            break
        chunks.append(" ".join(sentences_in_chunk))
        i += max_chunk_size - overlap
    return chunks

def semantic_chunk_command(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    chunks = semantic_chunk(text=text, max_chunk_size=max_chunk_size, overlap=overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks, 1):
        print(f"{i}. {chunk}")


def embed_chunks_command():
    movies_list = get_movie_data_from_file()
    chunked_sem_search = ChunkedSemanticSearch()
    chunk_embeddings = chunked_sem_search.load_or_create_chunk_embeddings(movies_list)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")

def search_chunked_command(query: str, limit: int=5) -> list[dict]:
    movies_list = get_movie_data_from_file()
    searcher = ChunkedSemanticSearch()
    chunk_embeddings = searcher.load_or_create_chunk_embeddings(documents=movies_list)
    results = searcher.search_chunks(query=query, limit=limit)

    print(f"Searching for '{query}'. Generating upto {limit} results...")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")