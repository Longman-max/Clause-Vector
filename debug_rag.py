
from src.rag_engine import ClauseVectorEngine

def debug_retrieval(query):
    print(f"Query: {query}")
    engine = ClauseVectorEngine()
    retriever = engine.get_retriever()
    
    docs = retriever.invoke(query)
    
    print(f"\nRetrieved {len(docs)} documents:")
    for i, doc in enumerate(docs):
        print(f"\n--- Document {i+1} ---")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content Preview: {doc.page_content[:200]}...") 
        print(f"Full Content:\n{doc.page_content}")

if __name__ == "__main__":
    debug_retrieval("What are the requirements for becoming President?")
