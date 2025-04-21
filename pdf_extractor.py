import pdfplumber
import re

def extract_sections_by_font(pdf_path, heading_font_threshold=12, space_threshold=1.5 , debut_document = 5):
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
        for i , page in enumerate(pdf.pages):
            # Commencer à parser uniquement à partir de la page 'debut_document'

            if i<debut_document :
                continue
            chars = page.chars
            if not chars:
                continue
            
            lines = {}
            for ch in chars:
                top = round(ch['top'], 0)
                lines.setdefault(top, []).append(ch)
            
            sorted_lines = [lines[k] for k in sorted(lines)]
            
            for line_chars in sorted_lines:
                sorted_chars = sorted(line_chars, key=lambda x: x["x0"])
                line_text = ""
                prev_ch = None
                for ch in sorted_chars:
                    if prev_ch:
                        gap = ch["x0"] - prev_ch["x1"]
                        if gap > space_threshold:
                            line_text += " "
                    line_text += ch["text"]
                    prev_ch = ch
                
                avg_font_size = sum(ch["size"] for ch in line_chars) / len(line_chars)
                
                if avg_font_size >= heading_font_threshold:
                    if current_text.strip():
                        sections.append({
                            "title": current_section,
                            "content": current_text.strip()
                        })
                        current_text = ""
                    current_section = re.sub(r'^\d+(\.\d+)?\s*', '', line_text.strip())  # Supprimer les chiffres et le point au début du texte
                else:
                    current_text += line_text + "\n"
    
    if current_text.strip():
        sections.append({
            "title": current_section,
            "content": current_text.strip()
        })
    
    return sections
