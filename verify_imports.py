
import sys
import importlib

def check_import(module_name, install_hint=""):
    try:
        importlib.import_module(module_name)
        print(f"✅ {module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import {module_name}: {e}")
        if install_hint:
             print(f"   Hint: Try installing {install_hint}")
        return False

print("--- Verifying Dependencies ---")
required_modules = [
    ("langchain", "langchain"),
    ("langchain.chains", "langchain"),
    ("langchain_community", "langchain-community"),
    ("langchain_huggingface", "langchain-huggingface"),
    ("langchain_chroma", "langchain-chroma"),
    ("langchain_groq", "langchain-groq"),
    ("chromadb", "chromadb"),
    ("streamlit", "streamlit"),
    ("dotenv", "python-dotenv"),
    ("sentence_transformers", "sentence-transformers"),
    ("src.ingest", "local source"),
    ("src.rag_engine", "local source"),
    ("src.config", "local source"),
    ("src.utils", "local source"),
]

all_passed = True
for module, package in required_modules:
    if not check_import(module, package):
        all_passed = False

if all_passed:
    print("\n✅ All configurations look good!")
    sys.exit(0)
else:
    print("\n❌ Some dependencies are missing.")
    sys.exit(1)
