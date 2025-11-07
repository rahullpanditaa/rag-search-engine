from lib.semantic_search import SemanticSearch
from lib.semantic_search_commands import semantic_chunk_command
from pathlib import Path
from lib.utils import get_movie_data_from_file
import numpy as np
import json

class ChunkedSemanticSearch(SemanticSearch):
    cache_dir_path = Path(__file__).resolve().parent.parent.parent / "cache"
    chunk_embeddings_path = cache_dir_path / "chunk_embeddings.npy"
    chunk_metadata_path = cache_dir_path / "chunk_metadata.json"

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
            doc_description_chunks = semantic_chunk_command(text=doc["description"], max_chunk_size=4, overlap=1)
            for chunk_index, description_chunk in enumerate(doc_description_chunks):
                all_chunks.append(description_chunk)
                chunks_metadata.append(
                    {
                        "movie_idx": doc_index,
                        "chunk_idx": chunk_index,
                        "total_chunks": len(doc_description_chunks)
                    }
                )
        print("chunks:", len(all_chunks))
        print("meta:", len(chunks_metadata))
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunks_metadata
        np.save(type(self).chunk_embeddings_path, self.chunk_embeddings)
        
        with open(type(self).chunk_metadata_path, "w") as f:
            json.dump({"chunks": self.chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2)
        
        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if type(self).chunk_embeddings_path.exists() and type(self).chunk_metadata_path.exists():
            self.chunk_embeddings = np.load(type(self).chunk_embeddings_path)
            with open(type(self).chunk_metadata_path, "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata["chunks"]
            print(f"Loaded chunks: {len(self.chunk_embeddings)}")
            print(f"loaded meta: {len(self.chunk_metadata)}")
            return self.chunk_embeddings
        else:
            self.chunk_embeddings = self.build_chunk_embeddings(documents=documents)
            return self.chunk_embeddings
        
def embed_chunks_command():
    movies_list = get_movie_data_from_file()
    chunked_sem_search = ChunkedSemanticSearch()
    chunk_embeddings = chunked_sem_search.load_or_create_chunk_embeddings(movies_list)
    print(f"Generated {len(chunk_embeddings)} chunked embeddings")