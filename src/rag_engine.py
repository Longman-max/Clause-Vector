from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.config import (
    VECTOR_DB_DIR,
    EMBEDDING_MODEL_NAME,
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    PROMPT_TEMPLATE,
    CONTEXTUALIZE_Q_PROMPT_TEMPLATE,
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

        # 1. Contextualize question based on history
        contextualize_q_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", CONTEXTUALIZE_Q_PROMPT_TEMPLATE),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]
        )
        history_aware_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        # 2. Answer question with indexed sources
        def format_docs_with_sources(docs):
            formatted_docs = []
            for i, doc in enumerate(docs):
                formatted_docs.append(f"Source {i+1}:\n{doc.page_content}")
            return "\n\n".join(formatted_docs)

        qa_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_TEMPLATE),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}"),
            ]create_stuff_documents_chain
        )
        
        # We need to manually feed the formatted docs into the context
        from langchain_core.runnables import RunnablePassthrough
        from langchain_core.output_parsers import StrOutputParser

        # This chain will:
        # 1. Take the retrieved docs from history_aware_retriever
        # 2. Format them with indices
        # 3. Pass to the LLM with the prompt
        
        rag_chain = (
            history_aware_retriever
            | {
                "context": lambda docs: format_docs_with_sources(docs),
                "chat_history": lambda x: x["chat_history"],create_stuff_documents_chain
                "input": lambda x: x["input"]
            }
            | qa_prompt
            | llm
            | StrOutputParser()
        )
        
        # We also need to return the source documents to the UI, so we wrap it
        # to return both answer and context. 
        # However, the standard create_retrieval_chain does this automatically but doesn't let us easy format docs with indices in the middle.
        
        # Le'ts try to stick to create_stuff_documents_chain but pass a custom document separator / formatter if possible? 
        # No, create_stuff_documents_chain expects a list of docs.
        
        # Alternative: Use the previous structure but ensure the prompt template handles it?
        # The issue is the standard chain joins docs with just newlines. we need "Source N:" labels.
        
        # Let's build a custom Runnable that returns the dict expected by app.py: {"answer": ..., "context": ...}
        
        from langchain_core.runnables import RunnableParallel

        rag_chain_with_source = (
            RunnableParallel(
                {"context": history_aware_retriever, "input": lambda x: x["input"], "chat_history": lambda x: x["chat_history"]}
            )
            .assign(answer=(
                RunnablePassthrough.assign(
                    context=lambda x: format_docs_with_sources(x["context"])
                )
                | qa_prompt
                | llm
                | StrOutputParser()
            ))
        )

        return rag_chain_with_source
