from fastapi import FastAPI, HTTPException , UploadFile,File,Form
from sentence_transformers import SentenceTransformer
import chromadb
from pydantic import BaseModel
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama
import time
import uuid
import json
import os
from pdf_extractor import extract_sections_by_font
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from langchain.text_splitter import RecursiveCharacterTextSplitter

class Item(BaseModel):
    question: str


# Initialiser FastAPI
app = FastAPI()

# Charger le modèle d'embeddings
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Connexion à ChromaDB
chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="documents")
allowed_origins = [
    "http://localhost:5173",  # URL de votre frontend React
    # Ajoutez d'autres origines si nécessaire
]

# Ajout du middleware CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,  # Origines autorisées
    allow_credentials=True,
    allow_methods=["*"],  # Méthodes HTTP autorisées
    allow_headers=["*"],  # En-têtes autorisés
)


splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n##", "\n#", "\n•", ". ", "\n", " "]
)
# 📌 Fonction pour générer l'embedding d'un texte
def generate_embedding(text: str):
    try:
        return model.encode(text).tolist()
    except Exception:
        raise HTTPException(status_code=500, detail="Erreur lors de la génération des embeddings")

class Section(BaseModel):
    title: str
    content: str
# 📌 Route pour stocker un texte en ChromaDB
class DocumentRequest(BaseModel):
    title: str  # Nouveau champ pour le titre
    content: str  # Contenu séparé du titre
@app.get("/")
async def root():
    return {"message": "L'API fonctionne correctement !"}

# Modification de la route de stockage
# @app.post("/store")
# async def store_document(document: list[DocumentRequest]):
#     print(document)
#     try:
#         # Combiner titre + contenu pour l'embedding
#         full_text = f"{document.title}\n{document.content}"
#         print(full_text)
#         embedding = generate_embedding(full_text)
#         print(embedding)
        
#         collection.add(
#             ids=[str(collection.count())],
#             embeddings=[embedding],
#             metadatas=[{
#                 "title": document.title,
#                 "content": document.content
#             }]
#         )
#         return {"message": "Document stocké avec succès"}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))


# @app.post("/store")
# async def store_documents(sections: list[Section]):
#     try:
#         # Découper chaque section en morceaux
#         split_sections = []
#         for section in sections:
#             heading = section.title
#             content = section.content.strip()

#             # Supprimer un éventuel point au début
#             if content.startswith("."):
#                 content = content[1:].strip()

#             # Découper le contenu
#             chunks = splitter.split_text(content)

#             # Ajouter chaque segment découpé
#             for chunk in chunks:
#                 if chunk.startswith("."):
#                     chunk = chunk[1:].strip()

#                 split_sections.append({
#                     "title": heading,
#                     "content": chunk
#                 })

#         # Ajouter les sections découpées dans ChromaDB
#         for section in split_sections:
#             full_text = f"{section['title']}\n{section['content']}"
#             embedding = generate_embedding(full_text)

#             collection.add(
#                 ids=[str(uuid.uuid4())],
#                 embeddings=[embedding],
#                 metadatas=[{
#                     "title": section['title'],
#                     "content": section['content']
#                 }]
#             )

#         return {"message": f"{len(split_sections)} sections stockées avec succès."}

#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))







# import json
# import uuid
# from fastapi import HTTPException
# from fastapi.responses import JSONResponse

@app.post("/store")
async def store_documents(sections: list[Section]):
    try:
        # Découper chaque section en morceaux
        split_sections = []
        for section in sections:
            heading = section.title
            content = section.content.strip()

            # Supprimer un éventuel point au début
            if content.startswith("."):
                content = content[1:].strip()

            # Découper le contenu
            chunks = splitter.split_text(content)

            # Ajouter chaque segment découpé
            for chunk in chunks:
                if chunk.startswith("."):
                    chunk = chunk[1:].strip()

                split_sections.append({
                    "title": heading,
                    "content": chunk
                })

        # Enregistrer split_sections dans final_split.json
        with open("final_split.json", "w", encoding="utf-8") as json_file:
            json.dump(split_sections, json_file, ensure_ascii=False, indent=2)

        # Ajouter les sections découpées dans ChromaDB
        for section in split_sections:
            full_text = f"{section['title']}\n{section['content']}"
            embedding = generate_embedding(full_text)

            collection.add(
                ids=[str(uuid.uuid4())],
                embeddings=[embedding],
                metadatas=[{
                    "title": section['title'],
                    "content": section['content']
                }]
            )

        return {"message": f"{len(split_sections)} sections stockées avec succès."}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# 📌 Configuration d'Ollama et du prompt
OLLAMA_API_URL = "http://localhost:11434/"
LLAMA_MODEL = "mistral"

prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks.
    Use the following documents to answer the question.
    If you don't know the answer, just say that you don't know.
    Use five sentences maximum and keep the answer concise:
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
    num_ctx=2048
)


# # 📌 Fonction pour récupérer des documents similaires de ChromaDB
# def retrieve_optimized(query: str, top_k: int = 3):
#     # Recherche en deux phases
#     query_embedding = generate_embedding(query)
#     print("generation d'embedding - ",timestamp)
    
#     # Phase 1: Recherche de titres pertinents
#     title_results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=5,
#         include=["metadatas"]
#     )
#     print("result - " , timestamp)
    
#     # Extraction des titres uniques
#     unique_titles = {m["title"] for m in title_results["metadatas"][0]}
#     print("extraction title unique - ",unique_titles)
    
#     # Phase 2: Recherche filtrée sur les titres pertinents
#     final_results = collection.query(
#         query_embeddings=[query_embedding],
#         n_results=top_k,
#         where={"title": {"$in": list(unique_titles)}} if unique_titles else None
#     )
#     print("final results - ",timestamp)
    
#     return [doc["content"] for doc in final_results["metadatas"][0]]



def retrieve_optimized(query: str, top_k: int = 3, score_threshold=0.7):
    query_embedding = generate_embedding(query)

    # 🔹 Phase 1: Trouver les titres les plus pertinents
    title_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,  # Augmenter pour capturer plus de titres similaires
        include=["metadatas", "distances"]
    )

    # 🔹 Filtrer les titres ayant une bonne similarité
    unique_titles = {
        meta["title"] for meta, score in zip(title_results["metadatas"][0], title_results["distances"][0])
        if score < score_threshold  # Garder les titres avec une distance faible (haute similarité)
    }

    print("Titres pertinents extraits:", unique_titles)

    # 🔹 Phase 2: Récupérer les documents associés aux titres trouvés
    final_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"title": {"$in": list(unique_titles)}} if unique_titles else None
    )

    return [doc["content"] for doc in final_results["metadatas"][0]]


@app.post("/store-from-file")
async def store_document(
    file: UploadFile = File(...),
    subheading_size: float = Form(12.0),
    debut_de_document: int = Form(0),
    space_threshold: float = Form(1.5)
):
    print(subheading_size,debut_de_document,space_threshold)
    try:
        # Sauvegarder temporairement le fichier reçu
        pdf_path = f"temp_{file.filename}"
        with open(pdf_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        # Extraire les sections du fichier PDF
        sections = extract_sections_by_font(pdf_path, subheading_size, space_threshold, debut_de_document)

        # ✅ Enregistrer dans extracted_section.json
        with open("extracted_section.json", "w", encoding="utf-8") as json_file:
            json.dump(sections, json_file, ensure_ascii=False, indent=2)

        return JSONResponse(content=sections)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")



# @app.post("/store-from-file")
# async def store_document(file: UploadFile = File(...) , subheading_size: float = Form(12.0),
#     debut_de_document: int = Form(0),space_threshold :float = Form(1.5) ):
#     # print(file)
#     # print(subheading_size)
#     # print(debut_de_document)
#     # print(space_threshold)
#     try:
#         # Sauvegarder temporairement le fichier reçu
#         pdf_path = f"temp_{file.filename}"
#         with open(pdf_path, "wb") as f:
#             content = await file.read()
#             f.write(content)
        
#         # Extraire les sections du fichier PDF
#         sections = extract_sections_by_font(pdf_path , subheading_size , space_threshold , debut_de_document)
#         return JSONResponse(content=sections)

#         # Traitement des sections et stockage dans la collection
        
    
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Erreur: {str(e)}")

# 📌 Route pour poser une question à Llama
@app.post("/ask")
async def ask_question(item: Item):
    try:
        print("le question est pose ",int(time.time()) )
        # Récupération des documents pertinents
        retrieved_docs = retrieve_optimized(item.question)
        print("le question est retrievé ",int(time.time()) )

        
        # Création du prompt avec les documents récupérés
        formatted_prompt = prompt.format(question=item.question, documents="\n".join(retrieved_docs))
        print(formatted_prompt)
        
        # Utilisation du modèle pour générer une réponse
        output_parser = StrOutputParser()
        response = llm.invoke(formatted_prompt)
        print("la reponse depuis llama est",int(time.time()) )
        answer = response.content  # ✅ Récupère uniquement le texte de la réponse

        return {"response": answer}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# 📌 Lancer l'API avec Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
