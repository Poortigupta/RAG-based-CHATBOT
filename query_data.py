import argparse
import os
from dotenv import load_dotenv
from rag_core import build_db, get_chat_model, PROMPT_TEMPLATE
from langchain.prompts import ChatPromptTemplate

CHROMA_PATH = "chroma"


"""All embedding/chat selection and prompt text come from rag_core now."""


def main():
    load_dotenv()
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    os.environ.setdefault("CHROMA_TELEMETRY", "False")
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Prepare the DB (embedding provider comes from rag_core)
    db = build_db(CHROMA_PATH)

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