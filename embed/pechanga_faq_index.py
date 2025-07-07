import re
import uuid
import requests
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance

QDRANT_URL = "http://10.1.60.67:6333"
OLLAMA_URL = "http://10.50.1.82:11434"
COLLECTION_NAME = "pechanga_faq"

def read_and_chunk_faq(filepath):
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            text = f.read()
    except UnicodeDecodeError:
        with open(filepath, "r", encoding="latin-1") as f:
            text = f.read()

    # Split once to isolate intro (before first ### or **Q:)
    intro_split = re.split(r"(?=^### )|(?=\*\*Q:)", text, maxsplit=1, flags=re.MULTILINE)

    chunks = []
    if intro_split:
        # Add intro chunk if exists
        if intro_split[0].strip():
            chunks.append(intro_split[0].strip())
        # Split remaining text on every section or FAQ marker
        if len(intro_split) > 1:
            rest_chunks = re.split(r"(?=^### )|(?=\*\*Q:)", intro_split[1], flags=re.MULTILINE)
            chunks.extend([c.strip() for c in rest_chunks if c.strip()])

    return chunks

def embed_text(text):
    prefixed_text = f"search_document: {text}"
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={"model": "nomic-embed-text:latest", "prompt": prefixed_text}
    )
    response.raise_for_status()
    return response.json()["embedding"]

def main():
    client = QdrantClient(url=QDRANT_URL)

    # Create or recreate collection
    existing_collections = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME not in existing_collections:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        )
        print(f"Collection '{COLLECTION_NAME}' created.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists.")

    chunks = read_and_chunk_faq("PechangaIt.txt")
    points = []

    for idx, chunk in enumerate(chunks):
        print(f"Embedding chunk {idx + 1}/{len(chunks)}...")
        embedding = embed_text(chunk)
        point_id = str(uuid.uuid4())
        points.append(PointStruct(
            id=point_id,
            vector=embedding,
            payload={"text": chunk},
        ))

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    print(f"Inserted {len(points)} chunks into collection '{COLLECTION_NAME}'.")

if __name__ == "__main__":
    main()
