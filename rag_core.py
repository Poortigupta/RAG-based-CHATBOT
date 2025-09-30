import os
from typing import Any, Dict, List, Optional, Tuple

from langchain_chroma import Chroma
import os
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

try:
    from langchain_google_genai import (
        GoogleGenerativeAIEmbeddings,
        ChatGoogleGenerativeAI,
    )
except Exception:
    GoogleGenerativeAIEmbeddings = None  # type: ignore
    ChatGoogleGenerativeAI = None  # type: ignore


PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def get_embedding_function():
    provider = os.getenv("EMBEDDING_PROVIDER", "GOOGLE").upper()
    if provider == "OPENAI":
        from importlib import import_module

        OpenAIEmbeddings = getattr(
            import_module("langchain_openai.embeddings"), "OpenAIEmbeddings"
        )
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    if provider == "GOOGLE":
        if GoogleGenerativeAIEmbeddings is None:
            raise RuntimeError(
                "langchain-google-genai is not installed. Add it to requirements or set EMBEDDING_PROVIDER=LOCAL."
            )
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set. Add it to .env or set EMBEDDING_PROVIDER=LOCAL.")
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
        # Normalize to expected name format for Google APIs
        if not model.startswith("models/"):
            model = f"models/{model}"
        return GoogleGenerativeAIEmbeddings(model=model)
    # LOCAL
    hf_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=hf_model)


def get_chat_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "GOOGLE").upper()
    if provider == "GOOGLE":
        if ChatGoogleGenerativeAI is None:
            raise RuntimeError(
                "langchain-google-genai is not installed. Add it to requirements or set EMBEDDING_PROVIDER=LOCAL."
            )
        chat_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
        if not chat_model.startswith("models/"):
            chat_model = f"models/{chat_model}"
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set; add it to your .env or set EMBEDDING_PROVIDER=LOCAL.")
        return ChatGoogleGenerativeAI(model=chat_model)
    if provider == "OPENAI":
        from importlib import import_module

        ChatOpenAI = getattr(import_module("langchain_openai"), "ChatOpenAI")
        return ChatOpenAI()
    return None


def build_db(chroma_path: str):
    embedding_function = get_embedding_function()
    # Ensure absolute path and existence for sqlite persistence
    abs_path = os.path.abspath(chroma_path)
    os.makedirs(abs_path, exist_ok=True)
    return Chroma(persist_directory=abs_path, embedding_function=embedding_function)


def retrieve(db: Chroma, query_text: str, k: int = 8):
    return db.similarity_search_with_relevance_scores(query_text, k=k)


def format_hits(results) -> List[Dict[str, Any]]:
    hits = []
    for (doc, score) in results[: min(10, len(results))]:
        hits.append(
            {
                "score": float(score),
                "source": doc.metadata.get("source"),
                "page": doc.metadata.get("page"),
            }
        )
    return hits


def answer_with_model(context_text: str, question: str) -> str:
    model = get_chat_model()
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=question)
    if model is None:
        return context_text[:1000]
    try:
        result = model.invoke(prompt)
        content = result.content if hasattr(result, "content") else result
        if isinstance(content, str):
            return content
        return str(content)
    except Exception as e:
        return f"Chat model invocation failed: {e}"


def query_rag(
    question: str,
    chroma_path: str,
    k: int = 8,
    threshold: float = 0.5,
    source: Optional[str] = None,
) -> Tuple[str, List[str], List[Dict[str, Any]]]:
    db = build_db(chroma_path)
    results = retrieve(db, question, k=k)
    if source:
        # Normalize to case-insensitive basename match (works with relative or absolute paths)
        src_norm = os.path.basename(source).casefold()
        def _norm(val):
            return os.path.basename(str(val)).casefold() if val is not None else ""
        results = [(d, s) for (d, s) in results if _norm(d.metadata.get("source")) == src_norm]
    hits = format_hits(results)

    if not results:
        return ("No results found.", [], hits)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _ in results])
    response_text = answer_with_model(context_text, question)
    sources = [str(doc.metadata.get("source")) for doc, _ in results if doc.metadata.get("source") is not None]
    return (response_text, sources, hits)
