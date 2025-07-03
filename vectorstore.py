from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

client = QdrantClient(path="qdrant_data")

def setup_collection():
    collections_info = client.get_collections()
    existing = [c.name for c in collections_info.collections]
    if "docs" not in existing:
        client.create_collection(
            collection_name="docs",
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )

def add_document(doc_id: int, text: str, embedding: list[float]):
    point = PointStruct(id=doc_id, vector=embedding, payload={"text": text})
    client.upsert(collection_name="docs", points=[point])

def search_similar(query_vec: list[float], top_k=3) -> list[str]:
    results = client.search(
        collection_name="docs",
        query_vector=query_vec,
        limit=top_k
    )
    if not results:
        return []
    return [r.payload["text"] for r in results if r and r.payload and "text" in r.payload]