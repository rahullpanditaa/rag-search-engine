import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
from lib.utils import get_movie_data_from_file

class SemanticSearch:
    cache_dir_path = Path(__file__).resolve().parent.parent.parent / "cache"
    movie_embeddings_path = cache_dir_path / "movie_embeddings.npy"

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
            self.embeddings = np.load(type(self).movie_embeddings_path)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents=documents)
    
    def search(self, query: str, limit: int):
        if self.embeddings == None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        
        query_embedding = self.generate_embedding(text=query)

        docs_similarity_scores: list[tuple] = []

        for i, doc_embedding in enumerate(self.embeddings):
            doc = self.documents[i]
            similarity_score = cosine_similarity(query_embedding, doc_embedding)
            docs_similarity_scores.append(tuple(similarity_score, doc))
        
        sorted_scores = sorted(docs_similarity_scores, key=lambda t: t[0], reverse=True)

        search_results = []
        for doc_score in sorted_scores[:limit]:
            result = {"score": doc_score[0], 
                      "title": doc_score[1]["title"],
                      "description": doc_score[1]["description"]}
            search_results.append(result)
        return search_results
        


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

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)