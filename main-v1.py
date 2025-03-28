from fastapi import FastAPI, HTTPException
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


class Item(BaseModel):
    question: str


# Initialiser FastAPI
app = FastAPI()

# Charger le modèle d'embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connexion à ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")


# 📌 Fonction pour générer l'embedding d'un texte
def generate_embedding(text: str):
    try:
        return model.encode(text).tolist()
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors de la génération des embeddings")


# 📌 Route pour stocker un texte en ChromaDB
@app.post("/store")
async def store_text(text: str):
    try:
        embedding = generate_embedding(text)

        # Ajouter l'entrée dans ChromaDB
        collection.add(
            ids=[str(collection.count())],  # Utilisation de count() pour éviter les doublons d'IDs
            embeddings=[embedding],
            metadatas=[{"content": text}]
        )

        return {"message": "Texte stocké avec succès dans ChromaDB"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 📌 Configuration d'Ollama et du prompt
OLLAMA_API_URL = "http://localhost:11434/"
LLAMA_MODEL = "llama3.1"

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use three sentences maximum and keep the answer concise:
    Question: {question}
    Documents: {documents}
    Answer:
    """,
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    base_url="http://localhost:11434",  # Correction ici
    model=LLAMA_MODEL,
    temperature=0.3,
    top_p=0.9,
    num_ctx=4096
)


# 📌 Fonction pour récupérer des documents similaires de ChromaDB
def retrieve_similar_documents(query: str, top_k: int = 3):
    embedding = generate_embedding(query)
    print("embedding :" , embedding)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=top_k
    )
    
    # Vérification pour éviter l'accès à une liste vide
    if results.get("metadatas") and results["metadatas"][0]:
        return [doc["content"] for doc in results["metadatas"][0]]
    
    return []


# 📌 Route pour poser une question à Llama
@app.post("/ask")
async def ask_question(item: Item):
    try:
        # Récupération des documents pertinents
        retrieved_docs = retrieve_similar_documents(item.question)
        
        # Création du prompt avec les documents récupérés
        formatted_prompt = prompt.format(question=item.question, documents="\n".join(retrieved_docs))
        
        # Utilisation du modèle pour générer une réponse
        output_parser = StrOutputParser()
        response = llm.invoke(formatted_prompt)
        answer = response.content  # ✅ Récupère uniquement le texte de la réponse

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 📌 Lancer l'API avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
