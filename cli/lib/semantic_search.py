import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from .constants import MOVIE_EMBEDDINGS_PATH

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
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
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if MOVIE_EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents=documents)
    
    def search(self, query: str, limit: int) -> list[dict]:
        if self.embeddings is None or len(self.embeddings) == 0:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(text=query)

        docs_similarity_scores: list[tuple] = []

        for i, doc_embedding in enumerate(self.embeddings):
            doc = self.documents[i]
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            docs_similarity_scores.append((similarity_score, doc))
        
        sorted_scores = sorted(docs_similarity_scores, key=lambda t: t[0], reverse=True)

        search_results = []
        for doc_score in sorted_scores[:limit]:
            result = {"score": doc_score[0], 
                      "title": doc_score[1]["title"],
                      "description": doc_score[1]["description"]}
            search_results.append(result)
        return search_results
        
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


