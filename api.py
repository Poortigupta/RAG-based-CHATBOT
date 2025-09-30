from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from dotenv import load_dotenv
from fastapi.responses import HTMLResponse

# Reuse existing pipeline pieces
from loader import load_documents, split_text, save_to_chroma, CHROMA_PATH, DATA_PATH
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


app = FastAPI(title="RAG Chatbot API", version="1.0.0")

# CORS for easy testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_event():
    load_dotenv()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")


@app.get("/")
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "provider": os.getenv("EMBEDDING_PROVIDER", "GOOGLE"),
        "persist_directory": CHROMA_PATH,
        "data_path": DATA_PATH,
        "docs": "/docs",
    }


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
