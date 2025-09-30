from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import os
import shutil     
from dotenv import load_dotenv
from langchain_chroma import Chroma
from rag_core import get_embedding_function
from pathlib import Path
import time

DATA_PATH = "."
CHROMA_PATH = "chroma"

def main():
    load_dotenv()  # Load environment variables from .env file
    # Disable Chroma/telemetry noise
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")
    generate_data_store()

def generate_data_store():
    documents = load_documents()
    chunks = split_text(documents)
    # CLI build: rebuild vector store from scratch
    save_to_chroma(chunks, reset=True)

def load_documents():
    print("Loading PDFs from:", os.path.abspath(DATA_PATH))
    pdf_paths = list(Path(DATA_PATH).rglob("*.pdf"))
    if not pdf_paths:
        print("No PDF files found.")
        return []

    documents: list[Document] = []
    for pdf_path in pdf_paths:
        try:
            loader = PyPDFLoader(str(pdf_path))
            documents.extend(loader.load())
        except Exception as e:
            print(f"Error loading file {pdf_path.name}: {e}")
    print(f"Loaded {len(documents)} documents from {len(pdf_paths)} PDF files.")
    return documents

def split_text(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 500,
        length_function = len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(chunks)} chunks.")

    if chunks:
        sample = chunks[min(10, len(chunks) - 1)]
        print(sample.page_content[:300] + ("..." if len(sample.page_content) > 300 else ""))
        print(sample.metadata)

    return chunks

# get_embedding_function now comes from rag_core


def save_to_chroma(chunks: list[Document], reset: bool = False):
    if not chunks:
        print("No chunks to save. Skipping vector store creation.")
        return
    if reset and os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)
    # Ensure persist directory exists before Chroma tries to open sqlite
    os.makedirs(CHROMA_PATH, exist_ok=True)

    # Select embedding backend (LOCAL by default)
    embeddings = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embeddings)

    # Add documents in batches, with basic retry and friendly messaging on quota issues.
    batch_size = 64
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        for attempt in range(3):
            try:
                db.add_documents(batch)
                break
            except Exception as e:
                msg = str(e)
                if "insufficient_quota" in msg or "RateLimit" in msg or "429" in msg:
                    print("Embedding failed due to API quota/rate limit. Check billing, wait, or set EMBEDDING_PROVIDER=LOCAL to use a local embedding model.")
                    return
                if attempt < 2:
                    sleep_s = 2 ** attempt
                    print(f"Transient error while adding batch {i//batch_size+1}: {e}. Retrying in {sleep_s}s...")
                    time.sleep(sleep_s)
                else:
                    raise

    # Data is persisted automatically when using a persist_directory.
    print(f"Saved {len(chunks)} chunks to {CHROMA_PATH}.")


if __name__ == "__main__":
    main()


