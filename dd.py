import pdfplumber
import json

def is_section(current_line, next_line):
    """
    Détermine si la ligne actuelle est un titre de section.
    Une ligne est une section si elle a une taille plus grande que la suivante
    et que leur style de police est différent.
    """
    return current_line['size'] > next_line['size'] and current_line['fontname'] != next_line['fontname']

def extract_lines_with_attributes(pdf_path):
    """
    Extrait les lignes d'un PDF avec leurs attributs (texte, taille, police) et corrige les espaces entre les mots.
    """
    lines_with_attrs = []
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            chars = page.chars
            if not chars:
                continue

            # Regrouper les caractères par ligne
            lines = {}
            for char in chars:
                line_key = round(char['top'], 1)
                if line_key not in lines:
                    lines[line_key] = []
                lines[line_key].append(char)

            # Trier les lignes par leur position verticale
            sorted_lines = sorted(lines.items(), key=lambda x: x[0])

            for _, line_chars in sorted_lines:
                # Trier les caractères horizontalement
                line_chars.sort(key=lambda c: c['x0'])
                
                # Construire le texte avec espaces corrects
                line_text = ""
                prev_char = None

                for char in line_chars:
                    if prev_char:
                        # Vérifier la distance entre les caractères pour insérer un espace
                        space_threshold = char['size'] * 0.25  # Seuil basé sur la taille de la police
                        if char['x0'] - prev_char['x1'] > space_threshold:
                            line_text += " "  # Ajouter un espace si l'écart est significatif
                    
                    line_text += char['text']
                    prev_char = char  # Mettre à jour le précédent caractère
                
                avg_size = sum(char['size'] for char in line_chars) / len(line_chars)
                font_names = [char['fontname'] for char in line_chars]
                most_common_font = max(set(font_names), key=font_names.count)
                
                lines_with_attrs.append({
                    'text': line_text.strip(),
                    'size': avg_size,
                    'fontname': most_common_font
                })
    
    return lines_with_attrs

def extract_sections(pdf_path):
    """
    Extrait les sections et leur contenu sous forme de liste d'objets JSON.
    """
    lines = extract_lines_with_attributes(pdf_path)
    sections = []
    current_section = None

    for i in range(len(lines) - 1):
        current_line = lines[i]
        next_line = lines[i + 1]

        if is_section(current_line, next_line):
            # Sauvegarder la section précédente avant d'en créer une nouvelle
            if current_section:
                current_section["contenu"] = current_section["contenu"].strip()  # Nettoyage des espaces inutiles
                sections.append(current_section)
            
            # Créer une nouvelle section
            current_section = {"title": current_line['text'], "contenu": ""}
        else:
            if current_section:
                current_section["contenu"] += " " + current_line['text']

    # Ajouter la dernière section
    if current_section:
        current_section["contenu"] = current_section["contenu"].strip()
        sections.append(current_section)

    return sections

def save_sections_to_json(pdf_path, output_json):
    """
    Extrait les sections d'un PDF et les sauvegarde dans un fichier JSON.
    """
    sections = extract_sections(pdf_path)
    with open(output_json, "w", encoding="utf-8") as json_file:
        json.dump(sections, json_file, indent=4, ensure_ascii=False)
    print(f"Les sections ont été enregistrées dans {output_json}")

# Exemple d'utilisation
pdf_path = "cours-python.pdf"
output_json = "sections.json"
save_sections_to_json(pdf_path, output_json)
