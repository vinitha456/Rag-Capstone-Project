# 🔍 PDF RAG Engine -Conversational Document Q&A with Gemini & LangChain

> Ask natural language questions against your own PDF documents. Answers are grounded strictly in your content — no hallucinations, full source citations.

---

## 📑 Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Pipeline](#pipeline)
- [Steps Involved](#steps-involved)
- [Modules Breakdown](#modules-breakdown)
- [Results](#results)
- [Limitations](#limitations)

---

## 🧭 Overview

This project implements a **Retrieval-Augmented Generation (RAG)** engine that allows users to have multi-turn conversations with their PDF documents. Rather than relying on a general-purpose LLM, the system constrains responses strictly to the content of uploaded files — making it well-suited for document research, policy review, technical documentation Q&A, and knowledge extraction tasks.

**Key capabilities:**
- Ingests one or more PDF files and indexes them into a persistent local vector store
- Retrieves the most contextually relevant chunks per question using MMR-based semantic search
- Generates grounded answers via **Google Gemini 2.0 Flash**, citing source documents and page numbers
- Maintains a sliding conversation window for coherent multi-turn follow-up questions

---

## 🛠️ Tech Stack

| Layer               | Technology                              | Purpose                                      |
|---------------------|-----------------------------------------|----------------------------------------------|
| LLM                 | Google Gemini 2.0 Flash (via API)       | Answer generation                            |
| Embeddings          | `all-MiniLM-L6-v2` (HuggingFace)       | Semantic vector encoding of text chunks      |
| Vector Store        | ChromaDB (local persistent)             | Similarity search over document embeddings   |
| Document Loading    | LangChain `PyPDFLoader`                 | PDF parsing and page extraction              |
| Text Splitting      | `RecursiveCharacterTextSplitter`        | Chunking documents for retrieval             |
| Retrieval Chain     | `ConversationalRetrievalChain`          | Orchestrates retrieval + LLM + memory        |
| Memory              | `ConversationBufferWindowMemory`        | Sliding window of last 5 Q&A exchanges       |
| Environment Config  | `python-dotenv`                         | Secure API key management via `.env`         |
| Language            | Python 3.9+                             | Core implementation                          |

---

## 🔄 Pipeline

```
┌────────────────────────────────────────────────────────────────────────┐
│                        INGESTION PIPELINE                              │
│                                                                        │
│   PDF File(s)                                                          │
│       │                                                                │
│       ▼                                                                │
│   PyPDFLoader  ──►  Raw Pages (with metadata: page #, filename)       │
│       │                                                                │
│       ▼                                                                │
│   RecursiveCharacterTextSplitter                                       │
│       │  chunk_size=500, chunk_overlap=50                              │
│       ▼                                                                │
│   Text Chunks  ──►  HuggingFace Embeddings (all-MiniLM-L6-v2)        │
│                           │                                            │
│                           ▼                                            │
│                     ChromaDB Vector Store  (persisted to ./chroma_db) │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                         QUERY PIPELINE                                 │
│                                                                        │
│   User Question                                                        │
│       │                                                                │
│       ▼                                                                │
│   HuggingFace Embeddings  (same model as ingestion)                   │
│       │                                                                │
│       ▼                                                                │
│   ChromaDB MMR Retriever                                               │
│       │  fetch_k=20 candidates → select k=5 most diverse              │
│       ▼                                                                │
│   Retrieved Chunks + Chat History (last 5 turns)                      │
│       │                                                                │
│       ▼                                                                │
│   Custom PromptTemplate                                                │
│       │  context + chat_history + question                             │
│       ▼                                                                │
│   Google Gemini 2.0 Flash                                              │
│       │                                                                │
│       ▼                                                                │
│   Answer  +  Source Citations (document name, page number, snippet)   │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 🪜 Steps Involved

### Step 1 - Environment Setup
Configure your `.env` file with a valid Google Gemini API key:
```
GOOGLE_API_KEY=your_gemini_api_key_here
```
Install dependencies:
```bash
pip install langchain langchain-community langchain-google-genai \
            chromadb sentence-transformers pypdf python-dotenv
```

### Step 2 - Initialize the RAG Engine
```python
from rag_engine import RAGEngine

engine = RAGEngine(persist_dir="./chroma_db")
```
This initializes the embedding model, text splitter, and conversation memory.

### Step 3 - Load PDF Documents
```python
engine.load_pdf("your_document.pdf")
```
Each PDF is parsed into pages, chunked, embedded, and upserted into ChromaDB. Multiple PDFs can be loaded sequentially into the same vector store.

### Step 4 - Set Up the RAG Chain
```python
engine.setup_chain()
```
This connects ChromaDB's MMR retriever to Gemini via a `ConversationalRetrievalChain` with a custom grounding prompt.

### Step 5 - Ask Questions
```python
result = engine.ask("What are the key findings in Section 3?")
print(result["answer"])
for src in result["sources"]:
    print(f"Page {src['page']} — {src['source']}: {src['content']}")
```
Follow-up questions work naturally thanks to the conversation memory window.

### Step 6 - (Optional) Reset Memory
```python
engine.clear_memory()   # Clears conversation history, keeps vector store intact
```

---

## 🧩 Modules Breakdown

### `RAGEngine.__init__`
| Component | Detail |
|-----------|--------|
| `embedding_model` | `all-MiniLM-L6-v2` - lightweight 384-dim model, fast CPU inference |
| `text_splitter` | chunk_size=500, overlap=50, recursive separators for clean boundaries |
| `memory` | `ConversationBufferWindowMemory` with k=5; `output_key="answer"` set explicitly to handle multi-output chains |
| `persist_dir` | ChromaDB persisted locally - survives restarts without re-ingestion |

---

### `load_pdf(pdf_path)`
- Uses `PyPDFLoader` to extract text per page
- Injects `page` and `source` (filename) into each chunk's metadata
- First call creates a new ChromaDB collection; subsequent calls append via `add_documents`
- Returns total chunk count for transparency

---

### `setup_chain()`
- Validates `GOOGLE_API_KEY` from environment before initializing the LLM
- Instantiates `ChatGoogleGenerativeAI` with `gemini-2.0-flash`, `temperature=0` for deterministic answers
- `convert_system_message_to_human=True` - required workaround as Gemini does not natively support system-role messages
- Builds a `PromptTemplate` that instructs the model to answer **only from context**, and to say so explicitly if the answer isn't present
- MMR retriever: fetches 20 candidates from ChromaDB, selects 5 with maximum marginal relevance to reduce redundancy

---

### `ask(question)`
- Invokes `chain.invoke({"question": question})`
- Extracts `source_documents` from result to build a citation list (page, filename, 200-char snippet)
- Returns a clean dict: `{ "answer": str, "sources": List[Dict] }`

---

### `clear_memory()` / `get_doc_count()`
- `clear_memory()` - resets conversation history without touching the vector store
- `get_doc_count()` - returns total number of indexed chunks via ChromaDB collection count

---

## 📊 Results

| Metric                        | Observed Behavior                                          |
|-------------------------------|------------------------------------------------------------|
| Answer grounding              | Responses strictly scoped to retrieved context             |
| Source citation               | Page number and document filename returned per answer      |
| Multi-turn coherence          | Follow-up questions resolve correctly across 5-turn window |
| Retrieval diversity (MMR)     | Avoids redundant chunks from same page/section             |
| Unanswerable question handling| Model responds with explicit "not found in documents" message |
| Ingestion speed               | ~1-3 seconds per PDF page on CPU (embedding bottleneck)    |

> **Note:** Accuracy is bounded by chunk quality and retrieval relevance. Results are best on structured, text-dense PDFs. Scanned PDFs without OCR preprocessing will yield poor results.

---

## ⚠️ Limitations

| Limitation | Description |
|------------|-------------|
| **No OCR support** | Scanned or image-based PDFs are not readable — text extraction requires prior OCR processing |
| **Chunk boundary loss** | Answers spanning multiple chunks may be incomplete if the split breaks a key passage |
| **Memory window is local** | Conversation memory is in-process only; it is not persisted across Python sessions |
| **Gemini API dependency** | Requires a valid `GOOGLE_API_KEY`; network errors or quota limits will halt query execution |
| **Legacy LangChain chain** | `ConversationalRetrievalChain` is a deprecated LangChain construct; migration to LCEL (`RunnablePassthrough`) is recommended for production use |
| **No reranking** | MMR provides diversity but not semantic reranking (e.g., Cohere Rerank) - relevance may degrade on large corpora |


---

## 📁 Project Structure

```
rag-engine/
│
├── rag_engine.py          # Core RAG engine class
├── .env                   # API keys (not committed)
├── .env.example           # Template for environment setup
├── requirements.txt       # Python dependencies
├── chroma_db/             # Persisted ChromaDB vector store (auto-created)
└── sample_document.pdf    # Example PDF for testing
```

---

## 🔐 Environment Variables

```bash
# .env
GOOGLE_API_KEY=your_gemini_api_key_here
```

> Never commit your `.env` file. Add it to `.gitignore`.

---

*Built with LangChain · ChromaDB · Google Gemini · HuggingFace Transformers*
