import re
import numpy as np
import json
from lib.constants import (
    CACHE_DIR_PATH,
    CHUNK_EMBEDDINGS_PATH, 
    CHUNK_METADATA_PATH, 
    DEFAULT_SEMANTIC_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP,
    SCORE_PRECISION
)
from lib.semantic_search.logic import SemanticSearch, cosine_similarity

class ChunkedSemanticSearch(SemanticSearch):
    """
    Performs semantic search over documents by splitting each document’s
    description into overlapping text chunks and embedding each chunk
    independently.

    This approach increases retrieval granularity by allowing search over
    fine-grained text segments rather than entire documents. Chunk embeddings
    and chunk-level metadata are stored on disk under `cache/`.

    ChunkedSemanticSearch extends `SemanticSearch` and adds:

        • Chunk generation using a sliding window of sentences
        • Per-chunk embedding creation and caching
        • Search that scores individual chunks, then aggregates scores
          at the document level
    Attributes:
        chunk_embeddings (np.ndarray | None):
            Embedding matrix of all text chunks generated from all documents.

        chunk_metadata (list[dict] | None):
            Metadata objects describing each chunk, including:
            {
                "movie_idx": <document id>,
                "chunk_idx": <chunk index within the document>,
                "total_chunks": <total chunks for this document>
            }
"""

    def __init__(self, model_name="all-MiniLM-L6-v2") -> None:
        super().__init__(model_name)
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        """
        Iterate over every single document in ```documents``` attribute which contains a ```list[dict]```, each dict 
        havings keys ```id, title, description```.

        1. Populate the ```document_map``` attribute. 
        2. For every document, split the description text for doc into a list of chunks.
        3. For every chunk in a doc's list of chunks, append the chunk to a list containing all chunks from all docs.
        4. Then, append to the ```chunk_metadata``` attribute a ```dict``` containing metadata about the current chunk.
        5. Encode (create embeddings vector list) for the list containing all chunks, store in ```chunk_embeddings```attribute.
        6. Store the ```list[dict]``` of **chunks_metadata** in ```chunks_metadata``` attribute.
        7. Create **cache/** directory in project root if it does not exist.
        8. Save the ```chunk_embeddings``` to a **.npy** file, ```chunks_metadata``` to a **.json file** in the **cache/** directory.
        """
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
            doc_description_chunks = semantic_chunk(text=doc["description"], max_chunk_size=DEFAULT_SEMANTIC_CHUNK_SIZE, 
                                                            overlap=DEFAULT_CHUNK_OVERLAP)
            for chunk_index, description_chunk in enumerate(doc_description_chunks):
                all_chunks.append(description_chunk)
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
        """
        Populate the ```document_map``` attribute. Load precomputed **chunk embeddings** and **chunks metadata** into 
        the ```chunk_embeddings``` and ```chunk_metadata``` attributes respectively. If the data does not exist,
        call ```build_chunk_embeddings(document)``` to build embeddings."""
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
        """Search for a user's query in the chunks generated for all docs.
        Given a query by the user, generate an **embedding vector** for it by calling the super class's ```generate_embedding(query)```
        method. Iterate over every chunk embedding in ```chunk_embeddings``` attribute, compute the **cosine_similarity score** for
        the query and the chunk, append to a list of chunk scores. Iterating over all chunk scores, create a dictionary 
        where **key=doc_id** and **value=highest score amongst all chunks in that doc**. 
         """
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