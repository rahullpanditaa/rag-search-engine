from PIL import Image
from sentence_transformers import SentenceTransformer
from pathlib import Path

class MultimodalSearch():
    def __init__(self, model_name="clip-ViT-B-32"):
        self.model = SentenceTransformer(model_name)

    def embed_image(self, image: str):
        img_path = Path(image).resolve()
        if not img_path.exists():
            raise ValueError(f"Given image {image} not found or does not exist")
        img = Image.open(img_path)
        image_embeddings = self.model.encode([img], show_progress_bar=True)
        return image_embeddings[0]
    
def verify_image_embedding(img_path):
    img = Path(img_path).resolve()

    searcher = MultimodalSearch()
    img_embedding = searcher.embed_image(image=img_path)
    print(f"Embedding shape: {img_embedding.shape[0]} dimensions")