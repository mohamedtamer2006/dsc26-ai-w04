"""BM25, written by hand (no search libraries)."""
from __future__ import annotations

import math
from collections import Counter

from bayyina.normalize import tokenize
from bayyina.types import Chunk


class BM25Index:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.chunks: list[Chunk] = []
        self.chunk_ids: list[str] = []
        self.doc_lens: list[int] = []
        self.avgdl: float = 0.0
        self.n_docs: int = 0
        # term -> list of (doc_index, tf)
        self.inverted_index: dict[str, list[tuple[int, int]]] = {}
        # term -> idf
        self.idf: dict[str, float] = {}

    def add(self, chunks: list[Chunk]) -> None:
        """Index a list of chunks."""
        self.chunks = list(chunks)
        self.n_docs = len(chunks)
        if self.n_docs == 0:
            self.avgdl = 0.0
            return

        self.chunk_ids = [c.chunk_id for c in chunks]
        self.doc_lens = []
        doc_term_freqs: list[dict[str, int]] = []
        df_counts: Counter[str] = Counter()

        for c in chunks:
            tokens = tokenize(c.text)
            self.doc_lens.append(len(tokens))
            tf_counter = Counter(tokens)
            doc_term_freqs.append(tf_counter)
            for term in tf_counter:
                df_counts[term] += 1

        self.avgdl = sum(self.doc_lens) / self.n_docs if self.n_docs > 0 else 0.0

        # Calculate IDF: ln(1 + (N - df + 0.5) / (df + 0.5))
        self.idf = {}
        for term, df in df_counts.items():
            self.idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

        # Build inverted index
        self.inverted_index = {}
        for idx, tf_counter in enumerate(doc_term_freqs):
            for term, tf in tf_counter.items():
                if term not in self.inverted_index:
                    self.inverted_index[term] = []
                self.inverted_index[term].append((idx, tf))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        """[(chunk_id, score), ...] with score > 0, by score descending then chunk ID ascending."""
        if not query or self.n_docs == 0 or top_k <= 0:
            return []

        query_terms = set(tokenize(query))
        if not query_terms:
            return []

        scores: dict[int, float] = {}

        for term in query_terms:
            if term not in self.inverted_index:
                continue
            term_idf = self.idf[term]
            postings = self.inverted_index[term]
            for doc_idx, tf in postings:
                dl = self.doc_lens[doc_idx]
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (dl / self.avgdl))
                term_score = term_idf * (tf * (self.k1 + 1.0)) / denominator
                scores[doc_idx] = scores.get(doc_idx, 0.0) + term_score

        # Filter score > 0 and sort: score descending, then chunk_id ascending
        valid_results = [
            (self.chunk_ids[doc_idx], score)
            for doc_idx, score in scores.items()
            if score > 0.0
        ]

        valid_results.sort(key=lambda item: (-item[1], item[0]))
        return valid_results[:top_k]
