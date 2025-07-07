import requests
from qdrant_client import QdrantClient

QDRANT_URL = "http://10.1.60.67:6333"
OLLAMA_URL = "http://10.50.1.82:11434"
COLLECTION_NAME = "pechanga_faq"

def embed_query(text):
    prefixed_text = f"search_query: {text}"
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": prefixed_text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def search_qdrant(embedding, top_k=3):
    client = QdrantClient(url=QDRANT_URL)
    search_result = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=embedding,
        limit=top_k,
    )
    return search_result

def run_completion(context, question):
    prompt = (
        f"Use the following context to answer the question.\n\n"
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        f"Answer:"
    )
    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={"model": "llama3.1:latest", "prompt": prompt, "stream": False}
    )
    response.raise_for_status()
    return response.json()["response"]

if __name__ == "__main__":
    question = input("Enter your question: ")
    embedding = embed_query(question)
    results = search_qdrant(embedding, top_k=5)

    if not results:
        print("No relevant documents found.")
    else:
        top_context = "\n\n".join(r.payload["text"] for r in results)
        print(f"\nTop contexts:\n{top_context}\n")
        answer = run_completion(top_context, question)
        print(f"Answer:\n{answer}")
