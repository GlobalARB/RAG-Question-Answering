# Retrieval-Augmented Generation for Question Answering

RAG pipeline on SQuAD 2.0 comparing 9 approaches: vanilla, oracle, word overlap, TF-IDF, BM25, dense BERT, hybrid retrieval, cross-encoder re-ranking, and few-shot prompting. Includes Recall@k curves, error decomposition, and confidence calibration.

## Pipeline

```
Question → Retriever (BM25 / BERT / Hybrid) → Re-ranker (optional) → LLM + Context → Answer
```

## Approaches

| Method | Context Source | Type |
|--------|--------------|------|
| Vanilla QA | None (LLM memory) | Lower bound |
| Oracle QA | Gold paragraph | Upper bound |
| Word Overlap | Token overlap ranking | Lexical |
| TF-IDF | TF-IDF cosine similarity | Lexical |
| BM25 | Okapi BM25 | Lexical |
| Dense (BERT) | BERT embedding cosine | Semantic |
| Hybrid | Dense + BM25 interpolation | Combined |
| Cross-Encoder | Dense → MS MARCO re-rank | Two-stage |
| Few-Shot | 3-shot prompt + dense | Prompt engineering |

## Sections

| # | Section | Description |
|---|---------|-------------|
| 1–3 | Setup, Data, Metrics | SQuAD 2.0 benchmark, EM/F1/ROUGE-2 |
| 4–5 | Vanilla & Oracle QA | Performance bounds |
| 6–7 | Word Overlap & Dense | Core retrieval methods |
| 8 | k Experiments | Performance vs number of contexts |
| 9 | TF-IDF | Improved lexical baseline |
| 10 | Summary | Comparison of first 5 methods |
| 11 | BM25 | Standard IR baseline |
| 12 | Hybrid | Dense+BM25 with alpha grid search |
| 13 | Cross-Encoder | Two-stage re-ranking |
| 14 | Few-Shot | In-context learning |
| 15 | Recall@k | Retrieval curves for all methods |
| 16 | Error Analysis | Retrieval vs extraction failures |
| 17 | Confidence Calibration | Accuracy by retriever confidence |
| 18 | Final Comparison | All 9 approaches side by side |

## Data

SQuAD 2.0 — auto-downloads on first run. 250 answerable questions, 19,035 candidate contexts.

## Setup

```bash
pip install -r requirements.txt
```

Requires GPU for OLMo-2 1B, BERT encoding, and cross-encoder inference.

## License

MIT
