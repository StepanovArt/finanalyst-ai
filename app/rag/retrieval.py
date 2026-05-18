"""
Hybrid search over SEC filings using Qdrant.

Strategy: Reciprocal Rank Fusion (RRF) over dense + sparse results.
- Dense search  → semantic similarity (finds paraphrases, synonyms)
- Sparse search → keyword match (finds exact tickers, metrics, GAAP terms)
- RRF           → merges ranked lists: score = Σ 1/(k + rank_i)

Optional second pass: cross-encoder reranker (bge-reranker-v2-m3).
- Processes (query, document) jointly — sees both at once, much more accurate
- Slower than bi-encoder but only runs on the small RRF candidate set
- Enable with rerank=True in search()

Why RRF over weighted sum?
Rank-based fusion is more robust than score-based because dense and sparse
scores live on different scales — a weight that works for one query breaks
for another. RRF only uses rank positions, which are stable across queries.
"""

from dataclasses import dataclass
from functools import lru_cache

from FlagEmbedding import BGEM3FlagModel, FlagReranker
from qdrant_client import QdrantClient
from qdrant_client.models import (
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchValue,
    Prefetch,
    SparseVector,
)

from app.rag.vector_store import COLLECTION_NAME, QDRANT_URL

PREFETCH_LIMIT = 20   # candidates per vector type before RRF
RERANK_POOL = 20      # RRF candidates to pass to cross-encoder when reranking
MAX_LENGTH = 512


@lru_cache(maxsize=1)
def _get_reranker() -> FlagReranker:
    return FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)


@dataclass
class SearchResult:
    chunk_id: str
    score: float
    ticker: str
    company: str
    filing_type: str
    year: int
    quarter: str
    section: str
    text: str
    context_prefix: str


def _build_filter(filters: dict) -> Filter | None:
    """Build Qdrant filter from a plain dict of {field: value}."""
    if not filters:
        return None
    conditions = [
        FieldCondition(key=k, match=MatchValue(value=v)) for k, v in filters.items()
    ]
    return Filter(must=conditions)


class HybridRetriever:
    """Retrieves SEC filing chunks via hybrid dense+sparse search with RRF."""

    def __init__(
        self,
        qdrant_url: str = QDRANT_URL,
        model: BGEM3FlagModel | None = None,
    ) -> None:
        self._client = QdrantClient(url=qdrant_url)
        # Accept an injected model (avoids reloading in scripts that already have it)
        self._model = model

    def _get_model(self) -> BGEM3FlagModel:
        if self._model is None:
            self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        return self._model

    def _embed_query(self, query: str) -> tuple[list[float], SparseVector]:
        model = self._get_model()
        output = model.encode(
            [query],
            max_length=MAX_LENGTH,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        dense = output["dense_vecs"][0].tolist()
        sparse_raw = output["lexical_weights"][0]
        sparse = SparseVector(
            indices=[int(k) for k in sparse_raw],
            values=[float(sparse_raw[k]) for k in sparse_raw],
        )
        return dense, sparse

    def search(
        self,
        query: str,
        limit: int = 5,
        filters: dict | None = None,
        rerank: bool = False,
    ) -> list[SearchResult]:
        """Run hybrid search with RRF fusion and optional cross-encoder reranking.

        Args:
            query: natural language question
            limit: number of results to return
            filters: optional metadata filter e.g. {"ticker": "AAPL", "year": 2024}
            rerank: if True, re-score top candidates with bge-reranker-v2-m3

        Returns:
            list of SearchResult ordered by score (best first)
        """
        dense_vec, sparse_vec = self._embed_query(query)
        qdrant_filter = _build_filter(filters or {})

        # Fetch more candidates when reranking so the cross-encoder has a richer pool
        fetch_limit = RERANK_POOL if rerank else limit

        results = self._client.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=[
                Prefetch(
                    query=dense_vec,
                    using="dense",
                    limit=PREFETCH_LIMIT,
                    filter=qdrant_filter,
                ),
                Prefetch(
                    query=sparse_vec,
                    using="sparse",
                    limit=PREFETCH_LIMIT,
                    filter=qdrant_filter,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=fetch_limit,
            with_payload=True,
        )

        hits = []
        for point in results.points:
            p = point.payload or {}
            hits.append(
                SearchResult(
                    chunk_id=str(point.id),
                    score=point.score,
                    ticker=p.get("ticker", ""),
                    company=p.get("company", ""),
                    filing_type=p.get("filing_type", ""),
                    year=p.get("year", 0),
                    quarter=p.get("quarter", ""),
                    section=p.get("section", ""),
                    text=p.get("text", ""),
                    context_prefix=p.get("context_prefix", ""),
                )
            )

        if not rerank:
            return hits

        # Cross-encoder reranking: process (query, doc) pairs jointly
        reranker = _get_reranker()
        pairs = [[query, h.text] for h in hits]
        scores: list[float] = reranker.compute_score(pairs, normalize=True)

        for hit, score in zip(hits, scores):
            hit.score = float(score)

        hits.sort(key=lambda h: h.score, reverse=True)
        return hits[:limit]
