from lib.semantic_search import SemanticSearch
from pathlib import Path
from lib.utils import get_movie_data_from_file
import numpy as np
import json
import re
from .constants import (
    CACHE_DIR_PATH,
    CHUNK_EMBEDDINGS_PATH, 
    CHUNK_METADATA_PATH, 
    DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
)

from .semantic_search import cosine_similarity

class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

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
                chunks_metadata.append(
                    {
                        "movie_idx": doc_index,
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
            print(f"Loaded chunks: {len(self.chunk_embeddings)}")
            print(f"loaded meta: {len(self.chunk_metadata)}")
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

        for score in chunk_scores:
            mv_idx = score.get("movie_idx")
            if mv_idx is None:
                continue
            current_score = movie_scores.get(mv_idx, float("-inf"))
            new_score = score.get("score")
            if new_score > current_score:
                movie_scores[mv_idx] = new_score
            
        sorted_scores = dict(sorted(movie_scores.items(), key=lambda item: item[1], reverse=True))
        sorted_scores = dict(list(sorted_scores.items())[:limit])

        results = []
        for doc_id, doc_score in sorted_scores.items():
            doc_title = self.document_map[doc_id]["title"]
            doc_description = self.document_map[doc_id]["description"][:100]
            metadata = self.document_map[doc_id]
            results.append({
                "id": doc_id,
                "title": doc_title,
                "document": doc_description,
                "score": round(doc_score, 2),
                "metadata": metadata or {}
            })

        return results



def semantic_chunk_command(text: str, max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text)
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