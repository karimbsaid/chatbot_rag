# from langchain.prompts import PromptTemplate
# from langchain_ollama import ChatOllama
# from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferMemory
# from app_config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME

# prompt = PromptTemplate(
#     template="""You are an assistant for question-answering tasks.\nUse the following documents to answer the question.\nIf you don't know the answer, say so.\nAnswer in 5 sentences max:\nQuestion: {question}\nDocuments: {documents}\nAnswer:""",
#     input_variables=["question", "documents"],
# )

# llm = ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL_NAME)
# memory = ConversationBufferMemory(return_messages=True)
# conversation = ConversationChain(llm=llm, memory=memory)

# def get_llm_answer(question: str, documents: str) -> str:
#     formatted = prompt.format(question=question, documents=documents)
#     response = conversation.predict(input=formatted)
#     return response



from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from app_config import OLLAMA_BASE_URL, OLLAMA_MODEL_NAME

prompt = PromptTemplate(
    template="""You are EduSphere's virtual assistant, here to help students understand their course material.

Carefully read the following documents and answer the student's question as clearly and concisely as possible. 
If the answer cannot be found in the documents, say you don't know — do not guess.

Keep your response friendly, focused, and limited to 5 sentences.

Question from student: {question}
Course documents: {documents}

Helpful answer:""",
    input_variables=["question", "documents"],
)

llm = ChatOllama(
    base_url=OLLAMA_BASE_URL,
    model=OLLAMA_MODEL_NAME,
    streaming=True
)

parser = StrOutputParser()

# ⛓️ This is the chain you export
chain = prompt | llm | parser
__all__ = ["chain"]
