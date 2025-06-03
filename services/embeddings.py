from sentence_transformers import SentenceTransformer
from fastapi import HTTPException

model = SentenceTransformer("paraphrase-MiniLM-L6-v2")
def generate_embedding(text: str):
    try:
        return model.encode(text).tolist()
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors de la génération des embeddings")
