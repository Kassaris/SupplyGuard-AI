# Retrieval benchmark

31 questions, user `u-pm-01` (scopes: internal, procurement, public), metrics on the top 5 documents.

| Mode | Recall@5 | MRR | Scope leaks | Latency (ms) |
|---|---|---|---|---|
| pgvector | 0.967 | 0.967 | 0 | 90 |
| chroma | 0.967 | 0.967 | 0 | 122 |
| keyword | 0.95 | 0.899 | 0 | 39 |
| hybrid | 1.0 | 0.958 | 0 | 171 |

## Recall@5 by question type

| Mode | identifier | mc17 | paraphrase |
|---|---|---|---|
| pgvector | 0.9 | 1.0 | 1.0 |
| chroma | 0.9 | 1.0 | 1.0 |
| keyword | 1.0 | 0.938 | 0.917 |
| hybrid | 1.0 | 1.0 | 1.0 |

## Misses (relevant document not in the top 5)

- **pgvector**: Q23
- **chroma**: Q23
- **keyword**: Q10, Q13
- **hybrid**: none

## Notes

- Latency is search time only: query embeddings are computed once before timing,
  so every mode is measured on the same footing.
- `keyword` is Postgres full-text search; `hybrid` fuses ChromaDB and keyword results (RRF).
- Scope leaks must be 0 for every mode.
