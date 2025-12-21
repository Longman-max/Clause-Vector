import os
import sys

# Add project root to sys.path to allow direct execution
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from src.config import (
    PDF_PATH,
    VECTOR_DB_DIR,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    EMBEDDING_MODEL_NAME
)
from src.utils import clean_text

def load_and_process_pdf():
    """
    Loads the PDF, splits it into chunks, and returns the splits.
    """
    print(f"Loading PDF from: {PDF_PATH}")
    if not os.path.exists(PDF_PATH):
        raise FileNotFoundError(f"PDF not found at {PDF_PATH}")

    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()
    print(f"Loaded {len(pages)} pages.")

    for page in pages:
        page.page_content = clean_text(page.page_content)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        is_separator_regex=False,
    )
    
    splits = text_splitter.split_documents(pages)
    print(f"Split into {len(splits)} chunks.")
    return splits

def ingest_data():
    """
    Main ingestion function.
    Checks if vector DB exists. If not, creates it.
    """
    if os.path.exists(VECTOR_DB_DIR) and os.listdir(VECTOR_DB_DIR):
        print(f"Vector Database found at {VECTOR_DB_DIR}. Skipping ingestion.")
        return

    print("Initializing embedding model (running on CPU)...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cpu'}
    )

    splits = load_and_process_pdf()

    print("Creating VectorStore...")
    vectorstore = Chroma.from_documents(
        documents=splits,
        embedding=embeddings,
        persist_directory=VECTOR_DB_DIR
    )
    print(f"VectorStore created and saved to {VECTOR_DB_DIR}")

if __name__ == "__main__":
    ingest_data()
