from PIL import Image
from sentence_transformers import SentenceTransformer, util
from pathlib import Path
from .utils import get_movie_data_from_file

class MultimodalSearch():
    def __init__(self, documents: list[dict], model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)
        self.documents = documents
        self.texts = [f"{doc['title']}: {doc['description']}" 
                      for doc in documents]
        self.text_embeddings = self.model.encode(self.texts, show_progress_bar=True)

    def embed_image(self, image: str):
        img_path = Path(image).resolve()
        if not img_path.exists():
            raise ValueError(f"Given image {image} not found or does not exist")
        img = Image.open(img_path).convert("RGB")
        image_embeddings = self.model.encode([img], show_progress_bar=True)
        return image_embeddings[0]
    
    def search_with_image(self, img_path: str):
        img_embedding = self.embed_image(image=img_path)

        results = []

        for i, txt_emb in enumerate(self.text_embeddings):
            cosine_sim = util.cos_sim(img_embedding, txt_emb).item()
            results.append({
                "doc_id": i+1,
                "title": self.documents[i]["title"],
                "description": self.documents[i]["description"],
                "similarity_score": cosine_sim
            })
        
        sorted_results = sorted(results, key=lambda d: d["similarity_score"], reverse=True)
        return sorted_results[:5]
    
def verify_image_embedding_command(img_path):
    img = Path(img_path).resolve()

    searcher = MultimodalSearch()
    img_embedding = searcher.embed_image(image=img)
    print(f"Embedding shape: {img_embedding.shape[0]} dimensions")

def _image_search(image_path: str):
    movies = get_movie_data_from_file()
    searcher = MultimodalSearch(documents=movies)
    results = searcher.search_with_image(image_path)
    return results

def image_search_command(image_path: str):
    results = _image_search(image_path=image_path)

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (similarity: {result['similarity_score']})")
        print(f"{result['description']}\n")

# def cosine_similarity(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm1 = np.linalg.norm(vec1)
#     norm2 = np.linalg.norm(vec2)

#     if norm1 == 0 or norm2 == 0:
#         return 0.0
    
#     return dot_product / (norm1 * norm2)
