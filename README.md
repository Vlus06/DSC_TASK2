# LegalQA — reproducible 0.5780 pipeline

Clean Python package frozen to the **full-corpus hierarchical run that reported METEOR = 0.5780** on the 1,000-question validation split (`seed=42`).

This package intentionally removes notebook history, ablations, losing configs, debug experiments, and hard-coded Kaggle paths. It keeps only the code/data needed to rebuild caches, validate the reported result, and run public inference.

## Frozen configuration

- validation: first 1,000 shuffled train qids, `seed=42`
- BM25: top 400, Vietnamese stopwords, number boost `0.4`
- dense parent retrieval: top 250
- fusion weights: BM25 `0.5`, dense `1.0`
- final documents: `3`
- hierarchical retrieval: parent Điều + child Khoản
- cross-encoder candidates: `10`
- CE margin: `0.5`
- max kept chunks: `2`
- answer max chars: `5800`
- models:
  - `AITeamVN/Vietnamese_Embedding`
  - `AITeamVN/Vietnamese_Reranker`

## Reproducibility guards

The project fails early if rebuilt caches do not match the experiment structure:

- corpus documents: **8,532**
- dense parent chunks: **738,506**
- dense parent cache name: `dense_chunk_index_AITeamVN_Vietnamese_Embedding_48ba5ddb4b.pkl`
- hierarchical child chunks: **595,597** across **7,078** documents
- validation target: **0.57x METEOR**; the notebook result is **0.5780**

The parent builder is the supplied v4.1 source, including the corrected `join_broken_lines` rule that does not merge list markers such as `a) ...\nb) ...`.

## Layout

```text
.
├── data/
│   ├── TASK2/
│   │   ├── selected-contexts/
│   │   ├── train.json
│   │   └── public-official.json
│   ├── stopwords.txt
│   └── cache/                 # generated .pkl files
├── scripts/
│   ├── build_bm25.py
│   ├── build_dense_parent.py
│   └── run.py
├── src/legalqa/
└── requirements.txt
```

## Run from scratch

Python 3.10+ and a CUDA GPU are recommended.

```bash
pip install -r requirements.txt
python scripts/run.py --mode validate --build-missing
```

The first run builds BM25, the 738,506-chunk parent dense cache, and the 595,597-chunk child cache. After validation, `outputs/val_debug.json` is written and the script checks that METEOR is in the 0.57x regression band.

Run public inference:

```bash
python scripts/run.py --mode infer --build-missing
```

or validation + inference:

```bash
python scripts/run.py --mode full --build-missing
```

Submission is written to `outputs/submission.json`.

## Important environment note

The notebooks did not record immutable Hugging Face model commit hashes or an exact `pip freeze`. The code, data split, chunk builders, cache structure, retrieval/reranking parameters, and scoring path are frozen here, but exact floating-point equality across future model revisions/CUDA/library versions cannot be guaranteed. For archival reproduction, keep the generated `.pkl` caches and the installed environment after the successful 0.5780 run.
