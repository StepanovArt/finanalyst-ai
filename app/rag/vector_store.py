"""
Qdrant collection setup and management for SEC filings.

Collection schema:
- Named vector  "dense":  1024-dimensional, Cosine distance (bge-m3 dense)
- Sparse vector "sparse": variable-length, dot product (bge-m3 lexical weights)

Payload fields indexed for fast metadata filtering:
- ticker, filing_type, section, quarter, currency  → KEYWORD (exact match)
- year, chunk_index                                → INTEGER (range queries)

Why named vectors (not default)?
Qdrant supports multiple vector types per point — required for hybrid search.
Default vector slot only allows one; named vectors allow dense + sparse together.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PayloadSchemaType,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
    VectorsConfig,
)

COLLECTION_NAME = "sec_filings"
DENSE_DIM = 1024
QDRANT_URL = "http://localhost:6333"


def get_client(url: str = QDRANT_URL) -> QdrantClient:
    return QdrantClient(url=url)


def create_collection(client: QdrantClient, recreate: bool = False) -> None:
    """Create the sec_filings collection with hybrid search schema.

    Args:
        client: Qdrant client instance
        recreate: if True, delete existing collection first
    """
    existing = [c.name for c in client.get_collections().collections]

    if COLLECTION_NAME in existing:
        if recreate:
            client.delete_collection(COLLECTION_NAME)
            print(f"Deleted existing collection '{COLLECTION_NAME}'")
        else:
            print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation")
            return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            # on_disk=False keeps sparse index in RAM for fast retrieval
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    print(f"Created collection '{COLLECTION_NAME}'")

    # Payload indexes speed up metadata filtering in hybrid search
    keyword_fields = ["ticker", "filing_type", "section", "quarter", "currency", "company"]
    integer_fields = ["year", "chunk_index"]

    for field in keyword_fields:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD,
        )

    for field in integer_fields:
        client.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field,
            field_schema=PayloadSchemaType.INTEGER,
        )

    print(f"Created payload indexes: {keyword_fields + integer_fields}")


def collection_info(client: QdrantClient) -> dict:
    """Return collection stats as a dict."""
    info = client.get_collection(COLLECTION_NAME)
    return {
        "status": info.status,
        "points_count": info.points_count,
        "vectors_count": info.vectors_count,
    }
