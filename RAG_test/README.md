# RAG Test Workflow

Recommendation: start with `1` most recent 10-Q filing per company.

Why:
- It keeps the benchmark easier to label and debug.
- It reduces retrieval ambiguity while you are still building oracle answers and oracle chunk ids.
- It makes answer failures easier to attribute to retrieval vs generation instead of cross-filing confusion.

Once the single-filing benchmark is stable, scale to `4` filings per company to test temporal reasoning and harder retrieval settings.

Suggested workflow:

1. Cache raw filings locally:
   `PYTHONPATH=. python RAG_test/cache_raw_filings.py --num-filings 1`

2. Chunk the cached filings locally:
   `PYTHONPATH=. python RAG_test/cache_chunked_filings.py`

3. Fill in `RAG_test/benchmark_dataset.json` with:
   - `filing_date`
   - `oracle_answer`
   - `oracle_chunk_id`

4. Run local-cache RAG evaluation:
   `PYTHONPATH=. python RAG_test/run_rag_evaluation.py`

Useful evaluation flags:

- Enable oracle-context generation:
  `PYTHONPATH=. python RAG_test/run_rag_evaluation.py --oracle-generator-test`
- Disable cross-encoder reranking:
  `PYTHONPATH=. python RAG_test/run_rag_evaluation.py --disable-reranker`
- Enable LLM-as-judge scoring:
  `PYTHONPATH=. python RAG_test/run_rag_evaluation.py --llm-judge`

Notes:

- Hybrid retrieval is now built in: dense embeddings + BM25 + reciprocal-rank fusion.
- Cross-encoder reranking turns on automatically when `sentence-transformers` is installed, unless disabled.
- BERTScore fields are added to the report automatically when `bert-score` is installed.
