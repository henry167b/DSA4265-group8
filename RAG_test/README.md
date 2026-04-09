# RAG Test Workflow

The benchmark workflow now supports the shared five-company set:
- `AAPL`
- `GOOG`
- `META`
- `NVDA`
- `TSLA`

Recommendation:
- Use the default scripts to cache and chunk all five benchmark companies.
- Use `--tickers` when you want a smaller debug pass on one or more names.

Suggested workflow:

1. Cache the raw filings locally:
   `PYTHONPATH=. python RAG_test/cache_raw_filings.py --num-filings 1`

   For a smaller pass:
   `PYTHONPATH=. python RAG_test/cache_raw_filings.py --num-filings 1 --tickers AAPL`

2. Chunk the cached filings locally:
   `PYTHONPATH=. python RAG_test/cache_chunked_filings.py`

   For a smaller pass:
   `PYTHONPATH=. python RAG_test/cache_chunked_filings.py --tickers AAPL`

3. Fill in `RAG_test/benchmark_dataset.json` with:
   - `filing_date`
   - `oracle_answer`

4. Run local-cache RAG evaluation:
   `PYTHONPATH=. python RAG_test/run_rag_evaluation.py`

   It also keeps cross-encoder reranking off by default.
   To turn reranking on later:
   `PYTHONPATH=. python RAG_test/run_rag_evaluation.py --enable-reranker`

Useful evaluation flags:

- Enable cross-encoder reranking:
  `PYTHONPATH=. python RAG_test/run_rag_evaluation.py --enable-reranker`
- Enable LLM-as-judge scoring:
  `PYTHONPATH=. python RAG_test/run_rag_evaluation.py --llm-judge`

Notes:

- Hybrid retrieval is now built in: dense embeddings + BM25 + reciprocal-rank fusion.
- Cross-encoder reranking turns on automatically when `sentence-transformers` is installed, unless disabled.
- BERTScore fields are added to the report automatically when `bert-score` is installed.
- `run_rag_evaluation.py` will evaluate every benchmark example in `RAG_test/benchmark_dataset.json`, so once the chunk cache exists for all five benchmark tickers it will score the full multi-company set.
