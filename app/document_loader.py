import os
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from .config import CHROMA_PATH, PDF_FOLDER

# Inicializa o objeto de embeddings
embedding = OpenAIEmbeddings()

# Função para extrair, processar e indexar todos os PDFs
def index_documents():
    # Carrega todos os documentos da pasta
    documents = []
    for filename in os.listdir(PDF_FOLDER):
        if filename.lower().endswith(".pdf"):
            path = os.path.join(PDF_FOLDER, filename)
            loader = PyMuPDFLoader(path)
            docs = loader.load()
            documents.extend(docs)

    # Divide o texto em chunks para melhor recuperação
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(documents)

    # Indexa no ChromaDB
    vectordb = Chroma.from_documents(
        split_docs,
        embedding,
        persist_directory=CHROMA_PATH
    )
    vectordb.persist()
    return vectordb