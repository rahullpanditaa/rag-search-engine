import numpy as np
from sentence_transformers import SentenceTransformer
from lib.constants import MOVIE_EMBEDDINGS_PATH

class SemanticSearch:
    """
    Performs semantic similarity search over a static collection of documents
    using SentenceTransformer vector embeddings.
    
    The class encodes all documents (using title + description) into vector embeddings, stores them on disk
    in ```cache/```, and computes the **cosine similariy score**
    between the vector embedding for a query and the document embeddings.
    Results for search are ranked by semantic closeness rather than keyword
    overlap.
    
    Attributes:
        model (SentenceTransformer):
            The sentence transformer model used to generate embeddings.
        embeddings (np.ndarray | None):
            Matrix of document embeddings; shape = (num_docs, embedding_dimension).
        documents (list[dict] | None): 
            Raw document objects used to build embeddings.
        document_map (dict[int, dict]):
            Mapping from document IDs to full document dictionaries."""
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = dict()

    
    def generate_embedding(self, text: str):
        """Generate an embedding vector for a single input string."""
        if text == "" or text.isspace():
            raise ValueError("Given text is empty or contains only whitespace")
        
        embeddings = self.model.encode([text])
        return embeddings[0]

    def build_embeddings(self, documents: list[dict]):
        """Create a list of embedding vectors for all documents,
        save them to disk
        
        Keyword arguments:
        documents -- a list of dictionaries where each dict 
        represents a doc having keys ```id, title, description```"""
        # self.model.na
        all_docs_str = []
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc
            all_docs_str.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(all_docs_str, show_progress_bar=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents: list[dict]):
        """Populate the document_map attribute, a dictionary where
        the key is the doc id and the value is the doc object itself. 
        
        Loads precomputed vector embeddings if they exist, else 
        call ```build_embeddings(documents)``` to generate the embeddings."""
        
        self.documents = documents
        for doc in self.documents:
            self.document_map[doc["id"]] = doc

        if MOVIE_EMBEDDINGS_PATH.exists():
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents=documents)
    
    def search(self, query: str, limit: int) -> list[dict]:
        """Call the ```generate_embedding(query)``` method to generate
        an embedding vector for user's query. 
        
        Compute the cosine similarity score between the query 
        embedding and each document embedding stored in ```self.embeddings```,
        sort the results by similarity score and return the top ```limit``` results."""
        
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
    """Computes the cosine similrity score between 2 vectors."""
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


