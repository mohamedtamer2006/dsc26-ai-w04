"""Citation verification and formatting."""
from __future__ import annotations

from bayyina.types import Hit


def verify_citation(quote: str, document: str, start: int, end: int) -> bool:
    """True only when 0 <= start < end <= len(document), quote is not blank,
    and quote (whitespace collapsed) occurs inside document[start:end] (whitespace collapsed).
    """
    if not (0 <= start < end <= len(document)):
        return False

    if not quote or not quote.strip():
        return False

    norm_quote = " ".join(quote.split())
    norm_span = " ".join(document[start:end].split())

    return norm_quote in norm_span


def format_citation(hit: Hit, max_quote: int = 160) -> str:
    """Return [doc_id:start-end] "quote" where quote is hit.text with whitespace collapsed;
    when longer than max_quote, take first max_quote characters, cut back to the last space
    inside them (dropping the space), and append ….
    """
    norm_text = " ".join(hit.text.split())

    if len(norm_text) <= max_quote:
        quote = norm_text
    else:
        prefix = norm_text[:max_quote]
        last_space_idx = prefix.rfind(" ")
        if last_space_idx != -1:
            quote = prefix[:last_space_idx] + "…"
        else:
            quote = prefix + "…"

    return f'[{hit.doc_id}:{hit.start}-{hit.end}] "{quote}"'
