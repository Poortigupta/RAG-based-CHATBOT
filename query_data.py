import argparse
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI

CHROMA_PATH = "chroma"

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
        OpenAIEmbeddings = getattr(import_module("langchain_openai.embeddings"), "OpenAIEmbeddings")
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        return OpenAIEmbeddings(model=model)
    if provider == "GOOGLE":
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set; add it to your .env or set EMBEDDING_PROVIDER=LOCAL.")
        model = os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
        return GoogleGenerativeAIEmbeddings(model=model)
    # LOCAL
    hf_model = os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    return HuggingFaceEmbeddings(model_name=hf_model)


def get_chat_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "GOOGLE").upper()
    # Use same provider selection for chat model
    if provider == "GOOGLE":
        # Default Gemini chat model
        chat_model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-2.5-flash")
        if not os.getenv("GOOGLE_API_KEY"):
            raise RuntimeError("GOOGLE_API_KEY not set; add it to your .env or set EMBEDDING_PROVIDER=LOCAL.")
        return ChatGoogleGenerativeAI(model=chat_model)
    if provider == "OPENAI":
        # Lazy import to avoid importing unless needed
        from importlib import import_module
        ChatOpenAI = getattr(import_module("langchain_openai"), "ChatOpenAI")
        return ChatOpenAI()
    # LOCAL provider has no chat model; return None
    return None


def main():
    load_dotenv()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB with more results and a lower threshold to improve recall.
    k = int(os.getenv("RETRIEVAL_K", "8"))
    threshold = float(os.getenv("RELEVANCE_THRESHOLD", "0.5"))
    results = db.similarity_search_with_relevance_scores(query_text, k=k)

    # Debug: show top hits, scores, and sources to verify indexing
    if not results:
        print("No results returned from vector store. Did you run loader.py after adding your PDFs?")
        return
    print("Top hits (score, source, page):")
    for (doc, score) in results[:min(5, len(results))]:
        src = doc.metadata.get("source")
        page = doc.metadata.get("page")
        print(f"  {score:.3f} | {src} | page {page}")

    if results[0][1] < threshold:
        print(f"Best result score {results[0][1]:.3f} is below threshold {threshold:.2f}. Returning top results anyway for context.")

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    model = get_chat_model()
    if model is None:
        # LOCAL provider: Return context-only answer
        print("LOCAL provider selected; returning context snippet only (no chat model).")
        response_text = "\n\n" + context_text[:1000]
    else:
        try:
            result = model.invoke(prompt)
            response_text = result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            print(f"Chat model invocation failed: {e}")
            response_text = ""

    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)


if __name__ == "__main__":
    main()