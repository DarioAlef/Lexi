import os
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
CHROMA_PATH = "./app/vector_store"
PDF_FOLDER = "./data/pdfs"
