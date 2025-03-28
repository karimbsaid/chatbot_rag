# # # import pdfplumber

# # # def extract_text_and_titles(pdf_path):
# # #     extracted_data = []

# # #     with pdfplumber.open(pdf_path) as pdf:
# # #         for page in pdf.pages:
# # #             text = page.extract_text()
# # #             if text:
# # #                 lines = text.split("\n")
# # #                 for line in lines:
# # #                     print(line)
# # #                     # Supposons que les titres sont en majuscules ou en gras dans le PDF
# # #                     if line.isupper() or len(line) < 50:  
# # #                         extracted_data.append({"title": line.strip()})
# # #                     else:
# # #                         extracted_data.append({"content": line.strip()})

# # #     return extracted_data

# # # # 🔹 Test avec un fichier PDF
# # # pdf_path = "Course notes - Convolutional Neural Networks.pdf" 
# # # data = extract_text_and_titles(pdf_path)

# # # # 🔹 Affichage des résultats
# # # for item in data:
# # #     print(item)


# # import pdfplumber

# # def extract_bold_text(pdf_path):
# #     bold_texts = []
    
# #     with pdfplumber.open(pdf_path) as pdf:
# #         for page in pdf.pages:
# #             words = page.extract_words(extra_attrs=["fontname"])  # Récupère les noms des polices
            
# #             for word in words:
# #                 fontname = word["fontname"]
# #                 text = word["text"]
                
# #                 # Vérification si le mot est en gras (dépend de la police)
# #                 if "Bold" in fontname or "Black" in fontname:
# #                     bold_texts.append(text)
    
# #     return bold_texts

# # # 🔹 Test avec un fichier PDF
# # pdf_path = "Course notes - Convolutional Neural Networks.pdf"  # Remplace par le chemin de ton fichier
# # bold_words = extract_bold_text(pdf_path)

# # # 🔹 Affichage des mots en gras
# # print("Mots en gras détectés :", bold_words)

# # import pdfplumber

# # pdf = pdfplumber.open("Course notes - Convolutional Neural Networks.pdf")

# # print(set(char["fontname"] for char in pdf.chars))

# import pdfplumber

# def extract_lines_with_font(pdf_path, target_fonts):
#     # Liste pour stocker les lignes qui contiennent la police spécifiée
#     matching_lines = []
    
#     with pdfplumber.open(pdf_path) as pdf:
#         for page in pdf.pages:
#             # Extraire tous les mots avec leurs propriétés (y compris la police)
#             words = page.extract_words(extra_attrs=["fontname", "doctop"])
            
#             # Liste pour construire les lignes à partir des mots
#             current_line = []
            
#             # Parcourir chaque mot extrait
#             for word in words:
#                 # Si la police du mot correspond à l'une des polices cibles, l'ajouter à la ligne
#                 if word["fontname"] in target_fonts:
#                     current_line.append(word["text"])
#                 else:
#                     # Si une nouvelle ligne commence, enregistrer la ligne précédente (si elle existe)
#                     if current_line:
#                         matching_lines.append(" ".join(current_line))
#                         current_line = []
                    
#             # Ajouter la dernière ligne si elle existe
#             if current_line:
#                 matching_lines.append(" ".join(current_line))

#     return matching_lines

# # 🔹 Liste des polices que tu veux rechercher
# target_fonts = {'BCDEEE+Montserrat-Regular', 'BCDGEE+Montserrat-Regular', 'BCDFEE+CambriaMath', 'BCDHEE+Cambria'}

# # Chemin du PDF
# pdf_path = "Course notes - Convolutional Neural Networks.pdf"

# # Extraction des lignes avec les polices cibles
# lines_with_font = extract_lines_with_font(pdf_path, target_fonts)

# # Affichage des lignes extraites
# for line in lines_with_font:
#     print(line)


import pdfplumber

def extract_sections_with_titles(pdf_path):
    sections = []
    current_section = None
    last_font_size = None  # Pour suivre la taille de la police du texte précédent
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["fontname", "size", "doctop"])

            for word in words:
                text = word.get("text")
                font_size = word.get("size")
                print((font_size , text))
                
                # Définir un seuil de taille de police pour détecter les titres (ex: 14 pt ou plus)
                if font_size > 20:  # Ajuste cette valeur selon ton document
                    # Si on trouve un titre, on sauvegarde la section précédente et démarre une nouvelle section
                    if current_section:
                        sections.append(current_section)

                    # Commencer une nouvelle section avec le titre trouvé
                    current_section = {"title": text, "content": ""}
                else:
                    # Ajouter le texte au contenu de la section actuelle
                    if current_section:
                        current_section["content"] += " " + text

        # Ajouter la dernière section trouvée
        if current_section:
            sections.append(current_section)

    return sections


# 📂 Utilisation du code pour extraire les sections depuis le PDF
pdf_path = "Course notes - Convolutional Neural Networks.pdf"  # Remplace par ton fichier PDF
sections = extract_sections_with_titles(pdf_path)

# Afficher les sections extraites
# for section in sections:
#     print(f"Title: {section['title']}")
#     print(f"Content: {section['content']}")
#     print("=" * 80)
