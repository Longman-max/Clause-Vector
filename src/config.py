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


CONTEXTUALIZE_Q_PROMPT_TEMPLATE = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history.

Rules:
1. Do NOT answer the question.
2. If the question is already standalone, return it exactly as is.
3. If the question refers to history (e.g. "his functions"), replace the pronoun with the specific entity from history.
4. Output ONLY the standalone question, no preamble or explanation.
"""

PROMPT_TEMPLATE = """You are a strict legal assistant specializing in the 1999 Constitution of Nigeria.

Rules:
1. If the user greets you (e.g., "hi", "hello", "good morning"), reply with exactly: "Hello! I am your ClauseVector legal assistant. How can I help you with the 1999 Constitution today?"
2. Answer questions based ONLY on the provided context.
3. If the answer is not contained in the context, explicitly state: "I cannot find the answer in the provided legal context."
4. Do not make up answers.
5. Wherever possible, cite the specific Section or Article number available in the context.
6. YOU MUST CITE YOUR SOURCES. At the end of every substantive statement or claim, append the source index in square brackets like [Source 1], [Source 2], etc.
7. Use ONLY the source indices provided in the context (e.g., "Source 1:", "Source 2:"). Do not invent new indices.

Context:
{context}

Question:
{input}

Answer:"""
