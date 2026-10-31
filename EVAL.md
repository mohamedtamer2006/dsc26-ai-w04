# Retrieval evaluation

Setup: all 8 contracts in `data/contracts/`, `chunk_text(..., max_chars=500, overlap=50)`,
`HashingEmbedder()`, the 12 queries in `data/queries.jsonl`. A hit is relevant when its span
overlaps one of the query's `relevant` ranges in the same document.

| Retriever | recall@3 | queries missed |
|---|---|---|
| BM25 only | 0.9167 (11/12) | Q11: "governing law and competent court for the agency agreement" |
| Dense only (HashingEmbedder) | 0.9167 (11/12) | Q11: "governing law and competent court for the agency agreement" |
| Hybrid (RRF) | 0.9167 (11/12) | Q11: "governing law and competent court for the agency agreement" |

## Findings and Next Steps

The single missed query (Q11) asked for the governing law and jurisdiction of the bilingual agency agreement in English, whereas the target clause in `agency_bi_08.txt` predominantly expresses jurisdiction under Arabic terminology («محاكم القاهرة الاقتصادية») and general phrasing. Because the crude `HashingEmbedder` only matches surface character 3-grams rather than cross-lingual semantic representations, surface lexicons failed to connect the English query to the Arabic span. In the squad's production system, we would replace `HashingEmbedder` with a multilingual embedding model (e.g. `intfloat/multilingual-e5-small` or BGE-M3) and append contextual section headers to each chunk, ensuring cross-lingual retrieval reaches 100% recall.
