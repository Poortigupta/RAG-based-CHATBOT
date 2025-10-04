from fastapi import FastAPI, UploadFile, File
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

# Reuse existing pipeline pieces
from loader import load_documents, split_text, save_to_chroma, CHROMA_PATH, DATA_PATH
from langchain_chroma import Chroma
from rag_core import query_rag


"""API uses rag_core.query_rag for retrieval and answering."""


class QueryRequest(BaseModel):
    question: str
    k: int = 8
    threshold: float = 0.5
    source: Optional[str] = None  # restrict retrieval to a specific source file


class QueryResponse(BaseModel):
    response: str
    sources: List[str]
    hits: List[Dict[str, Any]]


class MinimalQueryRequest(BaseModel):
    question: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")
    # Ensure vector store directory exists early
    os.makedirs(os.path.abspath(CHROMA_PATH), exist_ok=True)
    print("[lifespan] Environment initialized. Provider:", os.getenv("EMBEDDING_PROVIDER", "GOOGLE"))
    # Auto-ingest bundled PDFs (e.g., Samadhan.pdf) if vector store is empty.
    try:
        from rag_core import get_embedding_function
        embedding_fn = get_embedding_function()
        db = Chroma(persist_directory=os.path.abspath(CHROMA_PATH), embedding_function=embedding_fn)
        collection = getattr(db, "_collection", None)
        current_count = collection.count() if collection else 0
        if current_count == 0:
            print("[lifespan] Vector store empty. Attempting auto-ingest of bundled PDFs...")
            docs = load_documents()
            if docs:
                chunks = split_text(docs)
                # Full rebuild since store is empty
                from loader import save_to_chroma
                save_to_chroma(chunks, reset=True)
                print(f"[lifespan] Auto-ingest complete. Chunks saved: {len(chunks)}")
            else:
                print("[lifespan] No PDFs found to auto-ingest. Place PDFs in the project root or mount a volume.")
        else:
            print(f"[lifespan] Existing vector store detected with {current_count} vectors; skipping auto-ingest.")
    except Exception as e:
        print("[lifespan][warn] Auto-ingest skipped due to error:", e)
    yield
    # (Optional) cleanup logic could go here

app = FastAPI(title="RAG Chatbot API", version="1.0.0", lifespan=lifespan)

# CORS for easy testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


## Deprecated startup event removed (migrated to lifespan)


@app.get("/")
@app.get("/health")
def health() -> Dict[str, Any]:
    """Basic health plus quick vector store stats (safe / best-effort)."""
    provider = os.getenv("EMBEDDING_PROVIDER", "GOOGLE")
    stats: Dict[str, Any] = {}
    try:
        # Attempt lightweight status (do not rebuild embeddings)
        db = Chroma(persist_directory=os.path.abspath(CHROMA_PATH), embedding_function=lambda x: [[0.0]])  # type: ignore
        # Access underlying collection count if available
        collection = getattr(db, "_collection", None)
        if collection is not None:
            stats["vector_count"] = collection.count()
            stats["collection_name"] = getattr(collection, "name", None)
    except Exception as e:
        stats["vector_store_error"] = str(e)
    return {
        "status": "ok",
        "provider": provider,
        "persist_directory": os.path.abspath(CHROMA_PATH),
        "data_path": os.path.abspath(DATA_PATH),
        "vectors": stats.get("vector_count"),
        "collection": stats.get("collection_name"),
        "vector_store_error": stats.get("vector_store_error"),
        "ingest_endpoint": "/ingest/upload",
        "query_endpoint": "/query",
        "docs": "/docs",
    }


@app.get("/debug/store")
def debug_store() -> Dict[str, Any]:
    """Return detailed store diagnostics to help troubleshoot empty results."""
    info: Dict[str, Any] = {"persist_directory": os.path.abspath(CHROMA_PATH)}
    try:
        # Use a dummy embedding function (Chroma won't embed new texts here)
        db = Chroma(persist_directory=os.path.abspath(CHROMA_PATH), embedding_function=lambda x: [[0.0]])  # type: ignore
        collection = getattr(db, "_collection", None)
        if collection:
            info["collection_name"] = getattr(collection, "name", None)
            info["vector_count"] = collection.count()
            # Fetch up to 3 metadata samples
            try:
                raw = collection.get(limit=3)
                info["sample_ids"] = raw.get("ids")
                info["sample_metadatas"] = raw.get("metadatas")
            except Exception as sub_e:
                info["sample_error"] = str(sub_e)
    except Exception as e:
        info["error"] = str(e)
    # Tell user next step if empty
    if info.get("vector_count", 0) == 0:
        info["hint"] = "Vector store empty. Upload a PDF via /upload or /ingest/upload before querying."
    return info


@app.post("/ingest")
async def ingest(file: UploadFile | None = File(default=None)) -> Dict[str, Any]:
    # Optionally accept a PDF upload and save it into DATA_PATH before indexing
    saved_path = None
    if file is not None and file.filename:
        filename = file.filename
        if not filename.lower().endswith(".pdf"):
            return {"status": "error", "message": "Only PDF files are supported."}
        os.makedirs(DATA_PATH, exist_ok=True)
        saved_path = os.path.join(DATA_PATH, filename)
        content = await file.read()
        with open(saved_path, "wb") as f:
            f.write(content)

    docs = load_documents()
    chunks = split_text(docs)
    # In API, append to existing Chroma store; do not delete on each upload
    save_to_chroma(chunks, reset=False)
    return {
        "status": "ok",
        "saved_file": saved_path,
        "documents": len(docs),
        "chunks": len(chunks),
        "persist_directory": CHROMA_PATH,
    }


@app.post("/ingest/upload")
async def ingest_upload(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Strict file upload endpoint that always shows a file picker in Swagger UI."""
    if not file.filename:
        return {"status": "error", "message": "No file provided."}
    if not file.filename.lower().endswith(".pdf"):
        return {"status": "error", "message": "Only PDF files are supported."}
    os.makedirs(DATA_PATH, exist_ok=True)
    saved_path = os.path.join(DATA_PATH, file.filename)
    content = await file.read()
    with open(saved_path, "wb") as f:
        f.write(content)

    docs = load_documents()
    chunks = split_text(docs)
    # Append to existing Chroma store
    save_to_chroma(chunks, reset=False)
    return {
        "status": "ok",
        "saved_file": saved_path,
        "documents": len(docs),
        "chunks": len(chunks),
        "persist_directory": CHROMA_PATH,
    }


@app.get("/upload", response_class=HTMLResponse)
def upload_form() -> HTMLResponse:
        """Simple HTML form for uploading a PDF via the browser."""
        html = f"""
        <!doctype html>
        <html>
            <head>
                <meta charset='utf-8'/>
                <title>Upload PDF for Ingestion</title>
                <style>
                    body {{ font-family: sans-serif; margin: 2rem; }}
                    .card {{ max-width: 480px; padding: 1.5rem; border: 1px solid #ddd; border-radius: 8px; }}
                    button {{ padding: 0.5rem 1rem; }}
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>Upload PDF for Ingestion</h2>
                    <form action="/ingest/upload" method="post" enctype="multipart/form-data">
                        <input type="file" name="file" accept="application/pdf" required />
                        <div style="margin-top: 1rem;">
                            <button type="submit">Upload & Ingest</button>
                        </div>
                    </form>
                    <p style="margin-top: 1rem;">
                        Or use the <a href="/docs">API docs</a> to test endpoints programmatically.
                    </p>
                </div>
            </body>
        </html>
        """
        return HTMLResponse(content=html)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    response_text, sources, hits = query_rag(
        question=req.question,
        chroma_path=CHROMA_PATH,
        k=req.k,
        threshold=req.threshold,
        source=req.source,
    )
    return QueryResponse(response=response_text, sources=sources, hits=hits)


@app.post("/answer", response_model=QueryResponse)
def answer(req: MinimalQueryRequest) -> QueryResponse:
    """Minimal endpoint: only question and source are required.

    Uses default k and threshold (configurable via env if needed later).
    """
    default_source = os.getenv("DEFAULT_SOURCE")  # if set, constrain to one doc globally
    response_text, sources, hits = query_rag(
        question=req.question,
        chroma_path=CHROMA_PATH,
        k=int(os.getenv("RETRIEVAL_K", "8")),
        threshold=float(os.getenv("RELEVANCE_THRESHOLD", "0.5")),
        source=default_source,
    )
    return QueryResponse(response=response_text, sources=sources, hits=hits)
