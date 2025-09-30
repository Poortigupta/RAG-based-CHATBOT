from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables. Assumes that project contains .env file with API keys
load_dotenv()

def main():
    # Get embedding for a word.
    provider = os.getenv("EMBEDDING_PROVIDER", "GOOGLE").upper()
    if provider == "OPENAI":
        from importlib import import_module
        OpenAIEmbeddings = getattr(import_module("langchain_openai.embeddings"), "OpenAIEmbeddings")
        embedding_function = OpenAIEmbeddings()
    elif provider == "GOOGLE":
        embedding_function = GoogleGenerativeAIEmbeddings(model=os.getenv("GOOGLE_EMBEDDING_MODEL", "text-embedding-004"))
    else:
        embedding_function = HuggingFaceEmbeddings(model_name=os.getenv("HF_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"))
    vector = embedding_function.embed_query("apple")
    print(f"Vector for 'apple': {vector}")
    print(f"Vector length: {len(vector)}")

    # Compare vectors of two words using cosine similarity
    def cos_sim(a, b):
        a = np.array(a)
        b = np.array(b)
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    words = ("apple", "iphone")
    v1 = embedding_function.embed_query(words[0])
    v2 = embedding_function.embed_query(words[1])
    print(f"Cosine similarity ({words[0]}, {words[1]}): {cos_sim(v1, v2):.4f}")


if __name__ == "__main__":
    main()