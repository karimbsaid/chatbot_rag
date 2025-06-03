from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,
    chunk_overlap=128,
    separators=["\n\n##", "\n#", "\n•", ". ", "\n", " "]
)

def split_text(content: str, title: str) -> list[dict]:
    chunks = splitter.split_text(content.strip())
    return [{"title": title, "content": chunk.strip(".") if chunk.startswith(".") else chunk} for chunk in chunks]

