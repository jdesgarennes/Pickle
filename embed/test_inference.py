import requests

OLLAMA_URL = "http://10.50.1.82:11434"

def embed_query(text):
    prefixed_text = f"search_query: {text}"
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": prefixed_text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

if __name__ == "__main__":
    query = input("Enter your test query: ")
    embedding = embed_query(query)
    print(f"\nEmbedding length: {len(embedding)}\n")
    print(f"First 10 dimensions: {embedding[:10]}")
