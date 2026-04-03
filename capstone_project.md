# Day 41: Capstone Project — RAG-Powered Document Q&A System

## 🎯 Project Goal
Build a production-ready RAG system that can answer questions from uploaded PDF documents with source citations, using everything learned in Days 34-40.

---

## Project Overview

### What We Are Building
```
┌──────────────────────────────────────────────────┐
│       RAG Document Q&A System                     │
│                                                    │
│  Features:                                        │
│  1. Upload PDF documents                          │
│  2. Automatic chunking and embedding              │
│  3. Ask questions in natural language              │
│  4. Get accurate answers with source citations    │
│  5. Conversational follow-up questions            │
│  6. Streamlit web interface                       │
│                                                    │
│  Tech Stack:                                      │
│  - LangChain (RAG pipeline)                       │
│  - ChromaDB (vector store)                        │
│  - Sentence Transformers (embeddings)             │
│  - OpenAI GPT-4 or Ollama (LLM)                  │
│  - Streamlit (web UI)                             │
└──────────────────────────────────────────────────┘
```

### Architecture
```
User uploads PDF
       |
       v
  [PDF Loader] -> Extract text from PDF
       |
       v
  [Text Splitter] -> Split into 500-token chunks with overlap
       |
       v
  [Embedding Model] -> Convert chunks to vectors
       |
       v
  [ChromaDB] -> Store vectors + metadata
       |
       v
  User asks question
       |
       v
  [Retriever] -> Find top-5 relevant chunks
       |
       v
  [Re-Ranker] -> Re-rank for accuracy (optional)
       |
       v
  [LLM + Prompt] -> Generate answer with citations
       |
       v
  Display answer + source chunks + page numbers
```

---

## Step 1: Project Setup

```python
# requirements.txt
# langchain==0.1.20
# langchain-community==0.0.38
# langchain-openai==0.1.6
# chromadb==0.4.24
# sentence-transformers==2.7.0
# pypdf==4.2.0
# streamlit==1.33.0
# python-dotenv==1.0.1

# Install:
# pip install langchain langchain-community langchain-openai chromadb
# pip install sentence-transformers pypdf streamlit python-dotenv
```

---

## Step 2: RAG Backend (rag_engine.py)

```python
# rag_engine.py
import os
from typing import List, Dict
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory

load_dotenv()

class RAGEngine:
    def __init__(self, persist_dir="./chroma_db"):
        self.persist_dir = persist_dir
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.vectorstore = None
        self.chain = None
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer",
            k=5  # Remember last 5 exchanges
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def load_pdf(self, pdf_path: str) -> int:
        """Load a PDF and add to vector store. Returns number of chunks."""
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        
        # Add page number to metadata
        for i, page in enumerate(pages):
            page.metadata["page"] = i + 1
            page.metadata["source"] = os.path.basename(pdf_path)
        
        # Split into chunks
        chunks = self.text_splitter.split_documents(pages)
        
        # Store in ChromaDB
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding_model,
                persist_directory=self.persist_dir
            )
        else:
            self.vectorstore.add_documents(chunks)
        
        print(f"Loaded {len(chunks)} chunks from {pdf_path}")
        return len(chunks)
    
    def setup_chain(self, use_openai=True):
        """Set up the RAG chain with LLM."""
        if self.vectorstore is None:
            raise ValueError("No documents loaded! Call load_pdf first.")
        
        # Choose LLM
        if use_openai:
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4",
                temperature=0,
                max_tokens=1000
            )
        else:
            from langchain_community.llms import Ollama
            llm = Ollama(model="llama3.1", temperature=0)
        
        # Custom prompt
        prompt_template = """You are a helpful assistant that answers questions based on the provided documents.
Use ONLY the following context to answer. If the answer is not in the context, say "I could not find this information in the uploaded documents."

Always cite the source document and page number when possible.

Context:
{context}

Chat History:
{chat_history}

Question: {question}

Answer (with citations):"""
        
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )
        
        # Create retriever with MMR for diversity
        retriever = self.vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k": 5, "fetch_k": 20}
        )
        
        # Create conversational chain
        self.chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": PROMPT}
        )
        
        print("RAG chain ready!")
    
    def ask(self, question: str) -> Dict:
        """Ask a question and get an answer with sources."""
        if self.chain is None:
            raise ValueError("Chain not set up! Call setup_chain first.")
        
        result = self.chain.invoke({"question": question})
        
        # Format sources
        sources = []
        for doc in result.get("source_documents", []):
            sources.append({
                "content": doc.page_content[:200] + "...",
                "page": doc.metadata.get("page", "N/A"),
                "source": doc.metadata.get("source", "Unknown")
            })
        
        return {
            "answer": result["answer"],
            "sources": sources
        }
    
    def clear_memory(self):
        """Clear conversation history."""
        self.memory.clear()
    
    def get_doc_count(self) -> int:
        """Get number of documents in vector store."""
        if self.vectorstore is None:
            return 0
        return self.vectorstore._collection.count()


# ===========================
# Test the RAG Engine
# ===========================
if __name__ == "__main__":
    engine = RAGEngine()
    
    # Load a sample PDF
    engine.load_pdf("sample_document.pdf")
    
    # Setup chain (set use_openai=False for Ollama)
    engine.setup_chain(use_openai=True)
    
    # Ask questions
    result = engine.ask("What is the main topic of this document?")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for src in result['sources']:
        print(f"  - Page {src['page']} ({src['source']}): {src['content']}")
    
    # Follow-up question
    result = engine.ask("Can you elaborate on that?")
    print(f"\nFollow-up Answer: {result['answer']}")
```

---

## Step 3: Streamlit Web Interface (app.py)

```python
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
    use_openai = st.checkbox("Use OpenAI GPT-4", value=True)
    
    if st.button("Initialize Q&A System"):
        if st.session_state.rag_engine.get_doc_count() == 0:
            st.error("Please upload at least one document first!")
        else:
            with st.spinner("Setting up RAG pipeline..."):
                st.session_state.rag_engine.setup_chain(use_openai=use_openai)
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
```

---

## Step 4: Run the Application

```bash
# Set up environment variables
echo "OPENAI_API_KEY=sk-your-key-here" > .env

# Run the Streamlit app
streamlit run app.py
```

---

## Step 5: Evaluation Script (evaluate.py)

```python
# evaluate.py
from rag_engine import RAGEngine

def evaluate_rag():
    engine = RAGEngine()
    engine.load_pdf("test_document.pdf")
    engine.setup_chain(use_openai=True)
    
    # Test questions with expected answers
    test_cases = [
        {
            "question": "What year was the company founded?",
            "expected_keywords": ["2020"]
        },
        {
            "question": "How many vacation days do employees get?",
            "expected_keywords": ["21", "days"]
        },
        {
            "question": "What are the work hours?",
            "expected_keywords": ["9", "6", "Monday", "Friday"]
        }
    ]
    
    correct = 0
    total = len(test_cases)
    
    for tc in test_cases:
        result = engine.ask(tc["question"])
        answer = result["answer"].lower()
        
        # Check if expected keywords are in the answer
        found = all(kw.lower() in answer for kw in tc["expected_keywords"])
        
        status = "PASS" if found else "FAIL"
        if found:
            correct += 1
        
        print(f"[{status}] Q: {tc['question']}")
        print(f"       A: {result['answer'][:200]}")
        print()
    
    print(f"\nResults: {correct}/{total} ({correct/total*100:.1f}%)")

if __name__ == "__main__":
    evaluate_rag()
```

---

## Project Extensions (Bonus)

### 1. Add Re-Ranking
```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

def rerank_documents(query, documents, top_k=3):
    pairs = [[query, doc.page_content] for doc in documents]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, documents), reverse=True)
    return [doc for _, doc in ranked[:top_k]]
```

### 2. Add Multiple File Type Support
```python
from langchain_community.document_loaders import (
    PyPDFLoader, Docx2txtLoader, TextLoader, CSVLoader
)

def load_document(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf": PyPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader,
        ".csv": CSVLoader,
    }
    loader_cls = loaders.get(ext)
    if loader_cls is None:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader_cls(file_path).load()
```

### 3. Add Web URL Loading
```python
from langchain_community.document_loaders import WebBaseLoader

def load_url(url):
    loader = WebBaseLoader(url)
    return loader.load()
```

---

## Project Folder Structure
```
rag_document_qa/
  |-- app.py                 # Streamlit web interface
  |-- rag_engine.py          # RAG backend logic
  |-- evaluate.py            # Evaluation script
  |-- requirements.txt       # Dependencies
  |-- .env                   # API keys (not committed)
  |-- chroma_db/             # Vector database (auto-created)
  |-- sample_documents/      # Test PDFs
  |-- README.md              # Project documentation
```

---

## 📝 Quick Revision Points

1. **RAG Pipeline**: Load -> Chunk -> Embed -> Store -> Retrieve -> Generate
2. **ChromaDB** for storage, **Sentence Transformers** for embeddings
3. **MMR retrieval** for diverse results
4. **ConversationalRetrievalChain** for follow-up questions with memory
5. **Source citations** = always show where the answer came from
6. **Streamlit** for quick web UI
7. **Evaluation** = test with known Q&A pairs, check keyword presence
8. Production: Add re-ranking, hybrid search, caching, error handling

---

## 🗣️ Discussion Points
1. How would you deploy this to production? Docker + cloud hosting
2. How to handle documents that change frequently? Re-index pipeline
3. What if the answer spans multiple documents? Stuff chain handles this
4. How to add authentication? Streamlit has built-in auth options

---

## 📚 Homework
- Complete the full project with at least 3 PDF documents
- Add at least one advanced technique (re-ranking OR hybrid search)
- Write 15 test questions and evaluate the system
- Record a short demo video showing the system in action
- Prepare to present this as a portfolio project
