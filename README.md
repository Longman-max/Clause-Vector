# ClauseVector: Nigerian Legal Intelligence

ClauseVector is a local-first Retrieval Augmented Generation (RAG) system designed to answer questions about the **1999 Constitution of the Federal Republic of Nigeria**.

It uses vector embeddings to search the constitution and a Large Language Model (LLM) to generate precise, legally grounded answers with citations.

## Features

- **Local-First Processing**: Uses local CPU-based embeddings (`sentence-transformers`) and a local vector database (`ChromaDB`) for privacy and speed.
- **Fast Inference**: Powered by the Groq API (`llama-3.1-8b-instant`) for near-instant responses.
- **Precise Citations**: Every answer includes an expandable "View Legal Sources" section showing the exact articles/sections used.
- **Strict Persona**: The AI acts as a formal legal assistant, admitting ignorance rather than fabricating information.

## Architecture

- **Language**: Python 3.10+
- **Orchestration**: LangChain
- **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- **Vector DB**: ChromaDB (Persistent)
- **Frontend**: Streamlit

<!-- ## Directory Structure

```text
clause-vector/
├── assets/
│   └── constitution.pdf       # Source Document
├── vector_db/                 # Created automatically during ingestion
├── src/
│   ├── config.py              # Settings & Prompts
│   ├── ingest.py              # ETL Script
│   ├── rag_engine.py          # RAG Logic
│   └── utils.py               # Helpers
├── app.py                     # Main Interface
└── requirements.txt
``` -->

## Setup & Installation

1.  **Clone the repository** (if applicable).

2.  **Install Dependencies**:

    ```bash
    pip install -r requirements.txt
    ```

3.  **Environment Configuration**:
    Create a `.env` file in the root directory and add your Groq API Key:

    ```bash
    GROQ_API_KEY=your_groq_api_key_here
    ```

4.  **Add Source Document**:
    Ensure the Nigerian Constitution PDF is placed at `assets/constitution.pdf`. You can download it from: https://nigeriarights.gov.ng/files/constitution.pdf

5.  **Ingest Data**:
    Run the ingestion script to parse the PDF and build the vector database. This only needs to be done once.

    ```bash
    python src/ingest.py
    ```

6.  **Run the Application**:
    Launch the Streamlit interface.
    ```bash
    streamlit run app.py
    ```

## Usage

1.  Enter a legal question in the chat bar (e.g., _"What are the requirements for becoming President?"_).
2.  The system will analyze the question, search the vector database for relevant sections, and generate an answer.
3.  Expand the **View Legal Sources** dropdown to verify the text against the actual constitution.

## License

Copyright (c) 2025 Obasi Agbai. See [LICENSE](LICENSE) for details.
