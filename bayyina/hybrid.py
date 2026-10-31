"""Reciprocal rank fusion and the hybrid retriever."""
from __future__ import annotations

import numpy as np

from bayyina.bm25 import BM25Index
from bayyina.types import Chunk, Embedder, Hit, Reranker


def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """Ranks start at 1; score = sum of 1 / (k + rank); by score descending, then ID ascending."""
    scores: dict[str, float] = {}
    for rank_list in rankings:
        seen_in_list: set[str] = set()
        for rank, chunk_id in enumerate(rank_list, start=1):
            if chunk_id in seen_in_list:
                continue
            seen_in_list.add(chunk_id)
            scores[chunk_id] = scores.get(chunk_id, 0.0) + (1.0 / (k + rank))

    results = list(scores.items())
    results.sort(key=lambda item: (-item[1], item[0]))
    return results


class HybridRetriever:
    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        *,
        k1: float = 1.5,
        b: float = 0.75,
        reranker: Reranker | None = None,
    ):
        self.chunks = list(chunks)
        self.chunk_by_id: dict[str, Chunk] = {c.chunk_id: c for c in chunks}
        self.embedder = embedder
        self.reranker = reranker

        # Build BM25 index
        self.bm25 = BM25Index(k1=k1, b=b)
        self.bm25.add(chunks)

        # Build dense representations
        if self.chunks:
            self.chunk_texts = [c.text for c in self.chunks]
            self.chunk_ids = [c.chunk_id for c in self.chunks]
            self.chunk_embeddings = self.embedder.embed(self.chunk_texts)
        else:
            self.chunk_texts = []
            self.chunk_ids = []
            self.chunk_embeddings = np.zeros((0, 512))

    def search(self, query: str, top_k: int = 5, candidates: int = 50) -> list[Hit]:
        """Hybrid search combining BM25 and dense retrieval with RRF and optional reranking."""
        if not query or not query.strip() or not self.chunks or top_k <= 0:
            return []

        # 1. BM25 Candidates
        bm25_res = self.bm25.search(query, top_k=candidates)
        bm25_candidate_ids = [cid for cid, _ in bm25_res]

        # 2. Dense Candidates
        query_vec = self.embedder.embed([query])
        if len(query_vec) > 0 and len(self.chunk_embeddings) > 0:
            q_emb = query_vec[0]
            # Dot product with pre-computed chunk embeddings (both L2 normalized)
            dense_scores = np.dot(self.chunk_embeddings, q_emb)
            # Sort dense candidates: dot product descending, then chunk_id ascending
            # Combine chunk_id with dense score
            dense_items = list(zip(self.chunk_ids, dense_scores, strict=False))
            dense_items.sort(key=lambda item: (-item[1], item[0]))
            dense_candidate_ids = [cid for cid, _ in dense_items[:candidates]]
        else:
            dense_candidate_ids = []

        # 3. Fuse with RRF
        fused = rrf([bm25_candidate_ids, dense_candidate_ids], k=60)
        fused_candidates = fused[:candidates]

        # 4. Build Hits from fused candidates
        hits = [
            Hit(
                chunk_id=cid,
                doc_id=self.chunk_by_id[cid].doc_id,
                start=self.chunk_by_id[cid].start,
                end=self.chunk_by_id[cid].end,
                text=self.chunk_by_id[cid].text,
                score=rrf_score,
            )
            for cid, rrf_score in fused_candidates
        ]

        # 5. Optional Reranker
        if self.reranker is not None and hits:
            reranker_scores = self.reranker(query, hits)
            reranked_hits = [
                Hit(
                    chunk_id=h.chunk_id,
                    doc_id=h.doc_id,
                    start=h.start,
                    end=h.end,
                    text=h.text,
                    score=r_score,
                )
                for h, r_score in zip(hits, reranker_scores, strict=False)
            ]
            reranked_hits.sort(key=lambda item: (-item.score, item.chunk_id))
            hits = reranked_hits

        return hits[:top_k]
