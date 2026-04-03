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

    def setup_chain(self):
        """Set up the RAG chain with Google Gemini LLM."""
        if self.vectorstore is None:
            raise ValueError("No documents loaded! Call load_pdf first.")

        # Verify API key is set
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            raise ValueError(
                "GOOGLE_API_KEY not set! Please add your Gemini API key to the .env file."
            )

        # Use Google Gemini
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=api_key,
            temperature=0,
            max_output_tokens=1024,
            convert_system_message_to_human=True,
        )

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

        print("RAG chain ready with Google Gemini!")

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

    # Setup chain with Gemini
    engine.setup_chain()

    # Ask questions
    result = engine.ask("What is the main topic of this document?")
    print(f"\nAnswer: {result['answer']}")
    print(f"\nSources:")
    for src in result['sources']:
        print(f"  - Page {src['page']} ({src['source']}): {src['content']}")

    # Follow-up question
    result = engine.ask("Can you elaborate on that?")
    print(f"\nFollow-up Answer: {result['answer']}")
