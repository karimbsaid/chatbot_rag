import chromadb
import uuid
from app_config import CHROMA_DB_PATH
from services.embeddings import generate_embedding

chroma_client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
collection = chroma_client.get_or_create_collection(name="documents")

def store_documents(split_sections: list[dict]):
    for section in split_sections:
        text = f"{section['title']}\n{section['content']}"
        embedding = generate_embedding(text)
        collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            metadatas=[section]
        )

def retrieve_optimized(query: str, top_k: int = 3, score_threshold: float = 0.7) -> list[str]:
    query_embedding = generate_embedding(query)
    title_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["metadatas", "distances"]
    )
    unique_titles = {
        meta["title"] for meta, score in zip(title_results["metadatas"][0], title_results["distances"][0])
        if score < score_threshold
    }
    final_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"title": {"$in": list(unique_titles)}} if unique_titles else None
    )
    return [doc["content"] for doc in final_results["metadatas"][0]]
