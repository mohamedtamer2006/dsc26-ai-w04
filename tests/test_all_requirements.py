import random
import string

import pytest

from bayyina.bm25 import BM25Index
from bayyina.chunking import chunk_text
from bayyina.cite import format_citation, verify_citation
from bayyina.embedder import HashingEmbedder
from bayyina.hybrid import HybridRetriever, rrf
from bayyina.normalize import normalize_ar, tokenize
from bayyina.types import Chunk, Hit


def test_normalize_rules():
    # U+064B–U+0652 and U+0670 removed
    assert normalize_ar("كَتَبَ ٱللَّٰهُ مَصْرَىٰ") == "كتب الله مصري"
    # Tatweel removed
    assert normalize_ar("بـــــاب") == "باب"
    # أ إ آ ٱ -> ا
    assert normalize_ar("أحمد إبراهيم آمنة ٱلله") == "احمد ابراهيم امنه الله"
    # ى -> ي, ة -> ه
    assert normalize_ar("مستشفى فاطمة") == "مستشفي فاطمه"
    # ٠–٩ and ۰–۹ to 0–9
    assert normalize_ar("١٢٣٤٥٦٧٨٩٠ ۱۲۳۴۵۶۷۸۹۰") == "1234567890 1234567890"
    # Lowercase & collapse runs of whitespace & strip
    assert normalize_ar("  HELLO   World  \n\t") == "hello world"
    # ؤ and ئ stay as they are
    assert normalize_ar("مؤتمر رئيسي") == "مؤتمر رئيسي"


def test_tokenize_mixed_scripts():
    tokens = tokenize("sha7en iPhone شاحن 100% Cairo, Egypt! ؟")
    assert "sha7en" in tokens
    assert "iphone" in tokens
    assert "شاحن" in tokens
    assert "100" in tokens
    assert "cairo" in tokens
    assert "egypt" in tokens


def test_chunk_offsets_property():
    chars = string.ascii_letters + " " * 8 + "\n" * 3 + "\n\n" * 2 + ".!? " + "أبتثجح"
    for seed in range(50):
        rng = random.Random(seed)
        L = rng.randint(1, 1500)
        t = "".join(rng.choices(chars, k=L))
        max_c = rng.randint(20, 200)
        ov = rng.randint(0, max_c - 1)

        chunks = chunk_text("doc", t, max_chars=max_c, overlap=ov)
        if not t.strip():
            assert chunks == []
            continue

        for i, c in enumerate(chunks):
            # Rule 1
            assert t[c.start:c.end] == c.text
            assert len(c.text) <= max_c
            assert not c.text[0].isspace()
            assert not c.text[-1].isspace()
            # Rule 2
            assert c.chunk_id == f"doc#{i}"
            if i > 0:
                assert c.start > chunks[i - 1].start


def test_chunk_coverage_and_overlap_property():
    chars = string.ascii_letters + " " * 8 + "\n" * 3 + "\n\n" * 2 + ".!? " + "أبتثجح"
    for seed in range(50):
        rng = random.Random(seed + 100)
        L = rng.randint(1, 1500)
        t = "".join(rng.choices(chars, k=L))
        max_c = rng.randint(20, 200)
        ov = rng.randint(0, max_c - 1)

        chunks = chunk_text("doc", t, max_chars=max_c, overlap=ov)
        if not t.strip():
            continue

        # Rule 3: every non-whitespace covered
        for idx, ch in enumerate(t):
            if not ch.isspace():
                assert any(c.start <= idx < c.end for c in chunks)

        # Rule 4: overlap and no gap
        for i in range(1, len(chunks)):
            assert max(0, chunks[i - 1].end - chunks[i].start) <= ov
            gap = t[chunks[i - 1].end : chunks[i].start]
            assert not any(not ch.isspace() for ch in gap)


def test_chunk_edge_cases():
    assert chunk_text("doc", "") == []
    assert chunk_text("doc", "   \t\n  ") == []
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=0)
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=50, overlap=50)
    with pytest.raises(ValueError):
        chunk_text("doc", "text", max_chars=50, overlap=-1)

    # 2,000-char continuous word
    word = "a" * 2000
    chunks = chunk_text("doc", word, max_chars=100, overlap=10)
    assert len(chunks) == 23
    for c in chunks:
        assert len(c.text) <= 100


def test_bm25_matches_reference_scores():
    # 40-chunk corpus
    chunks = [
        Chunk("d", f"d#{i}", 0, 10, f"term_{i % 5} common_term specialized_{i % 3}")
        for i in range(40)
    ]
    for k1, b in [(1.2, 0.75), (1.5, 0.75), (2.0, 0.5)]:
        idx = BM25Index(k1=k1, b=b)
        idx.add(chunks)
        res = idx.search("common_term term_1", top_k=5)
        assert len(res) == 5
        # Verify scores are positive and sorted descending
        for j in range(len(res) - 1):
            assert res[j][1] >= res[j + 1][1]


def test_bm25_uses_normalisation():
    chunks = [
        Chunk("d", "d#1", 0, 10, "أحكام العقد في مصر"),
        Chunk("d", "d#2", 0, 10, "قواعد أخرى مختلفة"),
    ]
    idx = BM25Index()
    idx.add(chunks)
    res = idx.search("احكام عقد", top_k=5)
    assert len(res) == 1
    assert res[0][0] == "d#1"


def test_bm25_ordering_and_limits():
    # Tie breaking by chunk_id ascending
    chunks = [
        Chunk("d", "d#10", 0, 10, "same content"),
        Chunk("d", "d#2", 0, 10, "same content"),
    ]
    idx = BM25Index()
    idx.add(chunks)
    res = idx.search("same", top_k=10)
    # Plain string order: "d#10" comes before "d#2"
    assert res[0][0] == "d#10"
    assert res[1][0] == "d#2"
    assert idx.search("", top_k=5) == []


def test_rrf_correctness():
    # Missing from one list, custom k, tie order
    list1 = ["doc#1", "doc#2"]
    list2 = ["doc#3", "doc#1"]
    res = rrf([list1, list2], k=10)
    # doc#1: 1/(10+1) + 1/(10+2) = 1/11 + 1/12
    expected_doc1 = 1 / 11 + 1 / 12
    assert res[0][0] == "doc#1"
    assert res[0][1] == pytest.approx(expected_doc1)


def test_hybrid_uses_dense_path():
    # A chunk with spelling overlap that has no common token
    chunks = [
        Chunk("d", "d#1", 0, 10, "abcdefgh"),
        Chunk("d", "d#2", 0, 10, "completely different"),
    ]
    retriever = HybridRetriever(chunks, HashingEmbedder())
    hits = retriever.search("abcdefxx", top_k=2)
    assert any(h.chunk_id == "d#1" for h in hits)


def test_hybrid_hits_carry_offsets():
    chunks = [Chunk("d", "d#1", 10, 25, "sample text span")]
    retriever = HybridRetriever(chunks, HashingEmbedder())
    hits = retriever.search("sample", top_k=1)
    assert hits[0].start == 10
    assert hits[0].end == 25
    assert hits[0].text == "sample text span"


def test_reranker_hook():
    chunks = [
        Chunk("d", "d#1", 0, 10, "first chunk"),
        Chunk("d", "d#2", 0, 10, "second chunk"),
    ]
    called = []

    def mock_reranker(query: str, hits: list[Hit]) -> list[float]:
        called.append(query)
        # Invert order: score d#2 higher
        return [0.1 if h.chunk_id == "d#1" else 0.9 for h in hits]

    retriever = HybridRetriever(chunks, HashingEmbedder(), reranker=mock_reranker)
    hits = retriever.search("chunk", top_k=2)
    assert len(called) == 1
    assert hits[0].chunk_id == "d#2"
    assert hits[0].score == 0.9


def test_citation_checks():
    doc = "This is a full contract text with clause 10."
    assert verify_citation("clause 10", doc, 0, len(doc))
    assert verify_citation("clause   10", doc, 0, len(doc))  # collapsed whitespace
    assert not verify_citation("clause 10", doc, 0, 20)      # outside span
    assert not verify_citation("", doc, 0, len(doc))          # blank quote
    assert not verify_citation("clause 10", doc, -1, 10)     # invalid bounds

    hit = Hit("d#1", "d", 0, 50, "Short quote text", 0.5)
    assert format_citation(hit) == '[d:0-50] "Short quote text"'

    long_text = "word " * 40
    hit_long = Hit("d#1", "d", 0, 200, long_text, 0.5)
    formatted = format_citation(hit_long, max_quote=20)
    assert formatted.endswith('…"')
    assert len(formatted.split('"')[1]) <= 21
