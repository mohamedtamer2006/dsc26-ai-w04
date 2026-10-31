# AI usage

## Tools I used, and for what
I used Antigravity coding assistant (Gemini) and the Python interactive interpreter. I used the assistant to help draft unicode character ranges for Arabic normalization, structure the inverted index in `bm25.py`, and implement reciprocal rank fusion in `hybrid.py`.

## Where AI-written chunking or BM25 code broke one of the spec's rules
The initial AI-suggested chunking implementation broke Rule 4 and Rule 6: when advancing to the next chunk after a mid-word cut, it advanced by `end_pos - overlap` without checking if the candidate boundary landed in whitespace or mid-word, and after non-midword cuts it failed to align the start to the beginning of a word within the overlap range. I discovered this through a property test running 300 random string configurations, where non-whitespace characters were either skipped or started mid-word after clean paragraph breaks. I fixed it by explicitly tracking `is_midword_cut` and searching for valid word starts in `[min_cand, p]`.

## My recall@3 table (from EVAL.md) and why hybrid did or did not win

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 (11/12) | Q11: "governing law and competent court for the agency agreement" |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | Q11: "governing law and competent court for the agency agreement" |
| Hybrid (RRF) | 0.9167 (11/12) | Q11: "governing law and competent court for the agency agreement" |

Hybrid matched both individual retrievers (11/12) rather than outscoring them because the crude character 3-gram `HashingEmbedder` behaves like a sub-word ngram index rather than a true semantic embedder, meaning its success and failure patterns largely mirrored BM25's lexical matching on this dataset.
