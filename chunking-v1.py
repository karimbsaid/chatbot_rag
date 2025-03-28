import pdfplumber
import json
import re
def extract_sections_by_font(pdf_path, heading_font_threshold=14, space_threshold=1.5):
    """
    Extrait des sections en se basant sur la taille de police des lignes, tout en
    préservant les espaces entre les mots en se basant sur la position des caractères.
    
    Args:
        pdf_path (str): Chemin vers le document PDF.
        heading_font_threshold (float): Seuil de taille de police pour considérer une ligne comme titre.
        space_threshold (float): Seuil de distance entre caractères pour insérer un espace.
    
    Returns:
        List[Dict]: Liste de sections avec leur titre et leur contenu.
    """
    sections = []
    current_section = None
    current_text = ""
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            # Extraction des caractères avec leurs métadonnées
            chars = page.chars
            if not chars:
                continue
            
            # Grouper les caractères par ligne en se basant sur leur position verticale (top)
            lines = {}
            for ch in chars:
                # On arrondit la position pour regrouper les caractères de la même ligne
                top = round(ch['top'], 0)
                lines.setdefault(top, []).append(ch)
            
            # Trier les lignes par leur position verticale
            sorted_lines = [lines[k] for k in sorted(lines)]
            
            for line_chars in sorted_lines:
                # Reconstituer la ligne en triant les caractères par position horizontale et en insérant un espace si besoin
                sorted_chars = sorted(line_chars, key=lambda x: x["x0"])
                line_text = ""
                prev_ch = None
                for ch in sorted_chars:
                    if prev_ch:
                        # Calculer l'écart entre la fin du caractère précédent et le début du suivant
                        gap = ch["x0"] - prev_ch["x1"]
                        if gap > space_threshold:
                            line_text += " "
                    line_text += ch["text"]
                    prev_ch = ch
                
                # Calculer la taille de police moyenne pour la ligne
                avg_font_size = sum(ch["size"] for ch in line_chars) / len(line_chars)
                
                # Si la ligne a une taille de police supérieure au seuil, c'est potentiellement un titre
                if avg_font_size >= heading_font_threshold:
                    # Sauvegarder le contenu accumulé pour la section précédente (si existant)
                    if current_text.strip():
                        sections.append({
                            "heading": current_section,
                            "content": current_text.strip()
                        })
                        current_text = ""
                    # Le titre devient la nouvelle section
                    current_section =      re.sub(r'^\d+(\.\d+)?\s*', '', line_text.strip())  # Supprimer les chiffres et le point au début du texte

                else:
                    # Conserver les retours à la ligne pour le contenu
                    current_text += line_text + "\n"
    
    # Ajouter la dernière section
    if current_text.strip():
        sections.append({
            "heading": current_section,
            "content": current_text.strip()
        })
    
    return sections

pdf_path = "Course notes - Convolutional Neural Networks.pdf"
sections = extract_sections_by_font(pdf_path, heading_font_threshold=14, space_threshold=1.5)

with open("sections_by_font.json", "w", encoding="utf-8") as f:
    json.dump(sections, f, ensure_ascii=False, indent=4)

print("Extraction par mise en forme terminée. Résultats enregistrés dans sections_by_font.json")
