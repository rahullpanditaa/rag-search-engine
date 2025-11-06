import numpy as np
from lib.semantic_search import SemanticSearch
from lib.utils import get_movie_data_from_file

def verify_model():
    sem_search = SemanticSearch()
    print(f"Model loaded: {sem_search.model}")
    print(f"Max sequence length: {sem_search.model.max_seq_length}")

def embed_text(text: str):
    sem_search = SemanticSearch()
    embedding = sem_search.generate_embedding(text=text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    sem_search = SemanticSearch()
    movies_list_docs = get_movie_data_from_file()
    embeddings = sem_search.load_or_create_embeddings(movies_list_docs)
    print(f"Number of docs: {len(movies_list_docs)}")
    print(f"Embeddings of shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions")    

def embed_query_text(query: str):
    sem_search = SemanticSearch()
    query_embedding = sem_search.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {query_embedding[:5]}")
    print(f"Shape: {query_embedding.shape}")

def search_command(query: str, limit: int=5):
    sem_search = SemanticSearch()
    movies_list = get_movie_data_from_file()
    movie_embeddings = sem_search.load_or_create_embeddings(documents=movies_list)
    results = sem_search.search(query=query, limit=limit)
    for i, r in enumerate(results):
        print(f"{i+1}. {r["title"]} (score: {r["score"]})")
        print(f"{r["description"]}")
        print()

def chunk_command(text: str, chunk_size: int=200):
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)

    print(f"Chunking {len(text)} characters")
    for i, ch in enumerate(chunks, 1):     
        print(f"{i}. {ch}")
