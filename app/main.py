import os
import shutil
from fastapi import FastAPI, UploadFile, File, HTTPException
from .qa_pipeline import ask_question
from .document_loader import index_documents
from .config import PDF_FOLDER

# Garante que a pasta de PDFs exista
os.makedirs(PDF_FOLDER, exist_ok=True)

app = FastAPI(
    title="Lexi Chatbot API",
    description="API para upload de PDFs e consulta via LLM com OpenRouter",
    version="1.0.0"
)

@app.post("/upload", summary="Faz upload e indexa um PDF")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Apenas arquivos PDF são suportados.")

    file_path = os.path.join(PDF_FOLDER, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    # Reindexa todos os documentos após novo upload
    index_documents()
    return {"message": "Arquivo processado e indexado com sucesso."}

@app.post("/ask", summary="Faz uma pergunta ao modelo")
async def ask(query: str):
    if not query:
        raise HTTPException(status_code=400, detail="A pergunta não pode ficar vazia.")
    resposta = ask_question(query)
    return {"resposta": resposta}