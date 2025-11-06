import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

class SemanticSearch:
    cache_dir_path = Path(__file__).resolve().parent.parent.parent / "cache"
    movie_embeddings_path = cache_dir_path / "movie_embeddings.py"

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-V2')
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    def generate_embedding(self, text: str):
        if text == "" or text.isspace():
            raise ValueError("Given text is empty or contains only whitespce")
        
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        all_docs_str = []
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            all_docs_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(all_docs_str, show_progress_bar=True)
        np.save(type(self).movie_embeddings_path, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if type(self).movie_embeddings_path.exists():
            self.embeddings = np.load(type(self).movie_embeddings_path, "r")
        if len(self.embeddings) == documents:
            return self.embeddings
        return self.build_embeddings(documents=documents)


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