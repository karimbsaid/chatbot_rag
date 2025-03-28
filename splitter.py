import json
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Charger le fichier JSON
with open("sections_by_font.json", "r", encoding="utf-8") as f:
    sections = json.load(f)

# Configurer le text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n##", "\n#", "\n•", ". ", "\n", " "]  # Séparateurs pour découper le texte
)

# Fonction pour découper chaque section
split_sections = []

for section in sections:
    heading = section.get("heading", "Untitled")  # Gérer les sections sans titre
    content = section.get("content", "").strip()

    # Supprimer le point au début si présent
    if content.startswith("."):
        content = content[1:].strip()

    # Découper le contenu
    chunks = splitter.split_text(content)

    # Ajouter chaque segment découpé
    for idx, chunk in enumerate(chunks):
        # Vérifier et supprimer un éventuel point au début après découpage
        if chunk.startswith("."):
            chunk = chunk[1:].strip()

        split_sections.append({
            "title": heading,
            "content": chunk
        })

# Sauvegarder le nouveau fichier JSON
with open("split_sections.json", "w", encoding="utf-8") as f:
    json.dump(split_sections, f, ensure_ascii=False, indent=4)

print("✅ Découpage terminé. Résultats enregistrés dans split_sections.json")
