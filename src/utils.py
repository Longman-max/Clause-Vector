import re

def clean_text(text: str) -> str:
    """
    Cleans the input text by removing excessive whitespace and common PDF parsing artifacts.
    """
    # Replace multiple newlines with a single newline
    text = re.sub(r'\n+', '\n', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    # Strip leading/trailing whitespace
    text = text.strip()
    return text

def format_docs(docs):
    """
    Formats a list of documents into a single string for the Prompt.
    """
    return "\n\n".join(doc.page_content for doc in docs)
