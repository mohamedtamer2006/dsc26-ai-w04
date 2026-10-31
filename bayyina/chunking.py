"""Offset-preserving chunker adhering to the seven specification rules."""
from __future__ import annotations

from bayyina.types import Chunk

_SENTENCE_ENDS = {".", "!", "?", "؟", "۔", "\n"}


def chunk_text(doc_id: str, text: str, max_chars: int = 500, overlap: int = 50) -> list[Chunk]:
    """Split `text` into chunks whose offsets point into the ORIGINAL text.

    ValueError unless max_chars > 0 and 0 <= overlap < max_chars.
    """
    if max_chars <= 0 or not (0 <= overlap < max_chars):
        raise ValueError(f"Invalid max_chars={max_chars} or overlap={overlap}")

    if not text or not text.strip():
        return []

    n = len(text)
    chunks: list[Chunk] = []

    # First chunk starts at first non-whitespace character
    s = 0
    while s < n and text[s].isspace():
        s += 1
    if s >= n:
        return []

    while s < n:
        if n - s <= max_chars:
            chunk_slice = text[s:n]
            stripped_len = len(chunk_slice.rstrip())
            if stripped_len > 0:
                end_pos = s + stripped_len
                chunks.append(
                    Chunk(
                        doc_id=doc_id,
                        chunk_id=f"{doc_id}#{len(chunks)}",
                        start=s,
                        end=end_pos,
                        text=text[s:end_pos],
                    )
                )
            break

        window = text[s : s + max_chars]
        min_c = max_chars // 2

        best_c: int | None = None

        # Priority 1: paragraph break \n\n (positions c-2, c-1 in window)
        for c in range(max_chars, min_c - 1, -1):
            if c >= 2 and window[c - 2 : c] == "\n\n":
                best_c = c
                break

        # Priority 2: sentence end (. ! ? ؟ ۔ or \n at c-1)
        if best_c is None:
            for c in range(max_chars, min_c - 1, -1):
                if window[c - 1] in _SENTENCE_ENDS:
                    best_c = c
                    break

        # Priority 3: whitespace character at c-1
        if best_c is None:
            for c in range(max_chars, min_c - 1, -1):
                if window[c - 1].isspace():
                    best_c = c
                    break

        # Fallback: hard cut at window end
        if best_c is None:
            best_c = max_chars

        raw_cut = s + best_c
        chunk_slice = text[s:raw_cut]
        stripped_len = len(chunk_slice.rstrip())
        end_pos = s + stripped_len

        # A chunk ended with a mid-word cut when the characters on both sides of raw_cut are not whitespace
        is_midword_cut = (
            raw_cut < n and not text[raw_cut - 1].isspace() and not text[raw_cut].isspace()
        )

        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{len(chunks)}",
                start=s,
                end=end_pos,
                text=text[s:end_pos],
            )
        )

        # Find the next non-whitespace character at or after end_pos to ensure complete coverage
        p = end_pos
        while p < n and text[p].isspace():
            p += 1
        if p >= n:
            break

        if is_midword_cut:
            cand_s = max(s + 1, end_pos - overlap)
            while cand_s < n and text[cand_s].isspace():
                cand_s += 1
            next_s = cand_s
        else:
            min_cand = max(s + 1, end_pos - overlap)
            # Find earliest word start between min_cand and p
            word_starts = [
                i
                for i in range(min_cand, p + 1)
                if not text[i].isspace() and (i == 0 or text[i - 1].isspace())
            ]
            if word_starts:
                next_s = word_starts[0]
            else:
                next_s = p

        s = next_s

    return chunks
