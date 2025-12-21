import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Warning: GROQ_API_KEY not found in environment variables.")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
PDF_PATH = os.path.join(ASSETS_DIR, "constitution.pdf")
VECTOR_DB_DIR = os.path.join(BASE_DIR, "vector_db")

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "llama-3.1-8b-instant"

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 300

RETRIEVER_K = 6

PROMPT_TEMPLATE = """You are a strict legal assistant specializing in the 1999 Constitution of Nigeria.
Answer the following question based ONLY on the provided context.
If the answer is not contained in the context, explicitly state: "I cannot find the answer in the provided legal context."
Do not make up answers.
Wherever possible, cite the specific Section or Article number available in the context.

Context:
{context}

Question:
{input}

Answer:"""
