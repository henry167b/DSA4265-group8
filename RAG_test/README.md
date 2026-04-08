# RAG Test Workflow

Recommendation: start with the `1` most recent Apple 10-Q filing.

Why:
- It keeps the benchmark easier to debug.
- It reduces retrieval ambiguity while you are iterating on prose preprocessing and chunking.
- It makes answer failures easier to attribute to retrieval vs generation.

Once the single-filing benchmark is stable, you can scale back out later if needed.

Suggested workflow:

1. Cache the raw filing locally:
   `PYTHONPATH=. python RAG_test/cache_raw_filings.py --num-filings 1`

2. Chunk the cached filing locally:
   `PYTHONPATH=. python RAG_test/cache_chunked_filings.py`

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
