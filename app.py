# app.py
import streamlit as st
import tempfile
import os
from rag_engine import RAGEngine

# Page config
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

st.title("📄 RAG Document Q&A System")
st.markdown("Upload PDF documents and ask questions!")

# Initialize RAG engine in session state
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()
    st.session_state.chain_ready = False
    st.session_state.messages = []

# Sidebar: Document Upload
with st.sidebar:
    st.header("📁 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            # Save to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                tmp_path = tmp.name
            
            # Load into RAG engine
            with st.spinner(f"Processing {uploaded_file.name}..."):
                chunks = st.session_state.rag_engine.load_pdf(tmp_path)
                st.success(f"Loaded {uploaded_file.name} ({chunks} chunks)")
            
            os.unlink(tmp_path)
    
    # Setup chain button
    st.markdown("**LLM:** Google Gemini 2.0 Flash")
    
    if st.button("Initialize Q&A System"):
        if st.session_state.rag_engine.get_doc_count() == 0:
            st.error("Please upload at least one document first!")
        else:
            with st.spinner("Setting up RAG pipeline with Gemini..."):
                st.session_state.rag_engine.setup_chain()
                st.session_state.chain_ready = True
            st.success("System ready! Ask your questions.")
    
    # Stats
    doc_count = st.session_state.rag_engine.get_doc_count()
    st.metric("Chunks in Database", doc_count)
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.rag_engine.clear_memory()
        st.rerun()

# Main chat area
if not st.session_state.chain_ready:
    st.info("Please upload documents and click 'Initialize Q&A System' to start.")
else:
    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("sources"):
                with st.expander("View Sources"):
                    for src in msg["sources"]:
                        st.markdown(
                            f"**Page {src['page']}** ({src['source']}): "
                            f"_{src['content']}_"
                        )
    
    # Chat input
    if question := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.write(question)
        
        # Get RAG answer
        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                result = st.session_state.rag_engine.ask(question)
            
            st.write(result["answer"])
            
            if result["sources"]:
                with st.expander("View Sources"):
                    for src in result["sources"]:
                        st.markdown(
                            f"**Page {src['page']}** ({src['source']}): "
                            f"_{src['content']}_"
                        )
        
        # Save to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": result["sources"]
        })