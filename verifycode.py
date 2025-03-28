import re

def detect_code(text):
    # Rechercher des blocs de code en utilisant une expression régulière
    # Par exemple, du texte entre des backticks ou des indentations
    code_pattern = r'(```.*?```|`.*?`| {4}.*)'  # Recherche du code entre backticks ou indenté
    matches = re.findall(code_pattern, text, re.DOTALL)
    return matches



detect_code()