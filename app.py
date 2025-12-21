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
                # Run the chain
                response = st.session_state.chain.invoke({"input": prompt})
                answer = response["answer"]
                context = response["context"]

                message_placeholder.markdown(answer)
                
                st.session_state.messages.append({"role": "assistant", "content": answer})

                if context:
                    with st.expander("View Legal Sources"):
                        for i, doc in enumerate(context):
                            st.markdown(f"**Source {i+1}** (Page {doc.metadata.get('page', 'N/A')}):")
                            st.caption(doc.page_content)
                            st.divider()
                            
            except Exception as e:
                st.error(f"An error occurred: {e}")

