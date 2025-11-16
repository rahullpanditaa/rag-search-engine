from lib.semantic_search.logic import SemanticSearch, cosine_similarity
from pathlib import Path
from lib.utils import get_movie_data_from_file
import numpy as np
import json
import re
from .constants import (
    CACHE_DIR_PATH,
    CHUNK_EMBEDDINGS_PATH, 
    CHUNK_METADATA_PATH, 
    DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    SCORE_PRECISION
)
# from .semantic_search import cosine_similarity

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None
        self.list_of_all_chunks = []

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        # documents -> list of {movie_id, title, description}
        self.documents = documents
        all_chunks = []
        chunks_metadata: list[dict] = []
        self.document_map = {}
        for doc_index, doc in enumerate(self.documents):
            self.document_map[doc["id"]] = doc
            if not doc["description"].strip():
                continue
            # split the description text for doc/movie into chunks
            doc_description_chunks = semantic_chunk_command(text=doc["description"], max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, 
                                                            overlap=DEFAULT_CHUNK_OVERLAP)
            for chunk_index, description_chunk in enumerate(doc_description_chunks):
                all_chunks.append(description_chunk)
                self.list_of_all_chunks.append({
                    "movie_doc_index" : doc["id"],
                    "chunk_index_inside_doc": chunk_index,
                    "chunk_text": description_chunk
                })
                chunks_metadata.append(
                    {
                        "movie_idx": doc["id"],
                        "chunk_idx": chunk_index,
                        "total_chunks": len(doc_description_chunks)
                    }
                )
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        CACHE_DIR_PATH.mkdir(parents=True, exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if CHUNK_EMBEDDINGS_PATH.exists() and CHUNK_METADATA_PATH.exists():
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            return self.chunk_embeddings
        return self.build_chunk_embeddings(documents=documents)
    
    def search_chunks(self, query: str, limit: int=10):
        query_embedding = self.generate_embedding(text=query)
        chunk_scores: list[dict] = []
        for chunk_idx, ch_embedding in enumerate(self.chunk_embeddings):
            similarity_score = cosine_similarity(ch_embedding, query_embedding)            
            chunk_scores.append({
                "chunk_idx": chunk_idx,
                "movie_idx": self.chunk_metadata[chunk_idx]["movie_idx"],
                "score": similarity_score
            })        
        # {movie_index/doc_id -> score}
        movie_scores = {}

        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_scores = dict(
            list(sorted(movie_scores.items(), key=lambda item: item[1], reverse=True))[:limit]
        )


        results: list[dict] = []
        for doc_id, doc_score in sorted_scores.items():
            doc_title = self.document_map[doc_id]["title"]
            doc_description = self.document_map[doc_id]["description"][:100]
            metadata = self.document_map[doc_id].get("metadata", {})
            results.append({
                "id": doc_id,
                "title": doc_title,
                "document": doc_description,
                "score": round(doc_score, SCORE_PRECISION),
                "metadata": metadata
            })

        return results

def semantic_chunk_command(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if len(sentences) == 1 and not sentences[0].endswith(('.', '!', '?')):
        sentences = [text]
    
    sentences = [s.strip() for s in sentences if s.strip()]
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        chunks.append(" ".join(chunk_sentences))
        i += max_chunk_size - overlap
    return chunks


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

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['document']}...")