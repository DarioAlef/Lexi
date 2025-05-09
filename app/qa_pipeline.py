import os
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from .config import CHROMA_PATH, OPENROUTER_API_KEY

# Configura chave de API para OpenRouter
os.environ["OPENAI_API_KEY"] = OPENROUTER_API_KEY

# Inicializa embeddings e vetor store
embedding = OpenAIEmbeddings()
vectordb = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embedding
)
retriever = vectordb.as_retriever()

# Cria o LLM usando o endpoint OpenRouter
llm = ChatOpenAI(
    model_name="openrouter/mistralai/mixtral-8x7b",
    temperature=0
)
# Pipeline de RetrievalQA: busca + geração
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=False
)

def ask_question(query: str) -> str:
    """
    Recebe uma string de pergunta, executa busca semântica
    e retorna a resposta gerada pelo modelo.
    """
    return qa.run(query)