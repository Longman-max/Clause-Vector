import streamlit as st
import time
from src.rag_engine import ClauseVectorEngine

st.set_page_config(
    page_title="ClauseVector",
    page_icon=None,
    layout="centered"
)

st.title("ClauseVector: Nigerian Legal Intelligence")
st.markdown("ask questions about the *1999 Constitution of the Federal Republic of Nigeria*.")

if "messages" not in st.session_state:
    st.session_state.messages = []

if "engine" not in st.session_state:
    try:
        st.session_state.engine = ClauseVectorEngine()
        st.session_state.chain = st.session_state.engine.get_chain()
        st.success("System Ready. Knowledge Base Loaded.")
    except Exception as e:
        st.error(f"Failed to initialize RAG engine: {e}")
        st.stop()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a legal question..."):
    # Add user message to history
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Consulting the Constitution..."):
            try:
                # Prepare chat history
                chat_history = []
                for msg in st.session_state.messages[:-1]: # Exclude the current user message being processed
                    if msg["role"] == "user":
                        from langchain_core.messages import HumanMessage
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        from langchain_core.messages import AIMessage
                        chat_history.append(AIMessage(content=msg["content"]))

                # Run the chain
                response = st.session_state.chain.invoke({
                    "input": prompt,
                    "chat_history": chat_history
                })
                answer = response["answer"]
                context = response["context"]

                message_placeholder.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

                # Extract cited sources
                import re
                source_indices = set()
                matches = re.findall(r'\[Source (\d+)\]', answer)
                for match in matches:
                    source_indices.add(int(match) - 1) # Convert to 0-based index

                # Only show sources if we found a relevant answer and it's not just a greeting
                if context and "I cannot find the answer" not in answer and "Hello! I am your ClauseVector legal assistant" not in answer:
                    valid_indices = [i for i in sorted(list(source_indices)) if 0 <= i < len(context)]
                    
                    if valid_indices:
                        with st.expander("View Legal Sources"):
                            for idx in valid_indices:
                                doc = context[idx]
                                st.markdown(f"**Source {idx+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                                st.caption(doc.page_content)
                                st.divider()
                            
            except Exception as e:
                st.error(f"An error occurred: {e}")

