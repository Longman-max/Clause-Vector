from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from src.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    PROMPT_TEMPLATE,
    RETRIEVER_K
)

class ClauseVectorEngine:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL_NAME,
            model_kwargs={'device': 'cpu'}
        )
        self.vector_db = Chroma(
            persist_directory=VECTOR_DB_DIR,
            embedding_function=self.embeddings
        )

    def get_retriever(self):
        return self.vector_db.as_retriever(
            search_kwargs={"k": RETRIEVER_K}
        )

    def get_chain(self):
        if not GROQ_API_KEY:
            raise ValueError("GROQ_API_KEY is not set. Please check your .env file.")

        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            model_name=LLM_MODEL_NAME,
            temperature=0
        )

        retriever = self.get_retriever()

        prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        return rag_chain
