# LegalQA Pipeline (OOP refactor)

Class-based rewrite of the two original notebooks (hard-negative-mining +
fine-tuning, and the v9 RAG QA pipeline) into a proper Python package, so
the whole thing is reproducible end-to-end and runnable on
[Modal](https://modal.com).

## Layout

```
legalqa/
  config.py                 # all hyperparameters, as dataclasses
  utils.py                  # seeding, memory, device, logging
  corpus.py                 # Corpus loader (selected-contexts/*.json)
  legal_metadata.py         # VBQPPL / Điều / Khoản / Điểm regex extraction
  tokenization.py           # Vietnamese tokenizer + overlap scoring
  retrieval/
    bm25.py                 # BM25Retriever (+ number-boosting)
    dense.py                # DenseChunkCache, DenseChunkCacheBuilder, DenseRetriever
    fusion.py                # ScoreFusion, RecencyBooster
    reranker.py              # CrossEncoderReranker
  training/
    positive_miner.py        # PositiveMiner (gold answer -> doc_id/chunk)
    negative_miner.py         # HardNegativeMiner (BM25 + Dense + intra-doc)
    bi_encoder_trainer.py      # BiEncoderTrainer (CachedMultipleNegativesRankingLoss)
    cross_encoder_trainer.py   # CrossEncoderTrainer
    finetune_orchestrator.py   # RetrieverFinetuner (ties Stage 1 together)
  answer/
    answer_builder.py          # header/body/conclusion assembly + post-process
    chunk_selector.py           # threshold + lexical tie-break selection
  eval/
    meteor.py                   # MeteorScorer (official competition metric)
  pipeline/
    builder.py                   # factory helpers (load corpus/bm25/dense/model)
    qa_pipeline.py                # QAPipeline (single predict() call)
    tuner.py                       # PipelineTuner (staged grid-search)
    inference_runner.py             # batch inference -> submission.json

scripts/
  01_finetune_retriever_reranker.py   # Stage 1
  02_build_dense_cache.py             # Stage 2 (run twice, see below)
  03_run_pipeline.py                  # Stage 3

modal_app.py                # Modal.com GPU functions for all 3 stages
requirements.txt
setup.py
```

## The 3 stages (must run in this order)

The chicken-and-egg dependency is: **hard-negative mining needs a dense
chunk cache to look up candidates in, but the cache is built by a
bi-encoder** — and the bi-encoder that pipeline inference should use is
the *fine-tuned* one. So the cache gets built **twice**:

```
        base model                         fine-tuned model
             |                                     |
             v                                     v
2a) build_dense_cache (BASE)      1) finetune (mines negs using          2b) build_dense_cache (FINE-TUNED)
    -> cache_base.pkl                 cache_base.pkl for Dense negs)         -> cache_finetuned.pkl
                                       -> finetuned_bi_encoder/
                                       -> finetuned_cross_encoder/
                                                                                        |
                                                                                        v
                                                                          3) run_pipeline (tunes + infers,
                                                                             uses cache_finetuned.pkl +
                                                                             finetuned_bi_encoder +
                                                                             finetuned_cross_encoder)
```

### Local reproduction (CLI)

```bash
pip install -r requirements.txt
pip install -e .

# unzip TASK2.zip somewhere, e.g. /data/TASK2/TASK2
# put stopwords.txt at /data/stopwords.txt

# 2a) dense cache with the BASE model (needed for mining)
python scripts/02_build_dense_cache.py \
    --task2-data-dir /data/TASK2/TASK2 \
    --output-dir /data/outputs \
    --bi-encoder-name AITeamVN/Vietnamese_Embedding \
    --cache-out /data/outputs/cache/dense_chunk_index_base.pkl

# 1) mine hard negatives + fine-tune both models
python scripts/01_finetune_retriever_reranker.py \
    --task2-data-dir /data/TASK2/TASK2 \
    --output-dir /data/outputs \
    --dense-cache-file /data/outputs/cache/dense_chunk_index_base.pkl \
    --stopwords-file /data/stopwords.txt

# 2b) rebuild dense cache with the FINE-TUNED model
python scripts/02_build_dense_cache.py \
    --task2-data-dir /data/TASK2/TASK2 \
    --output-dir /data/outputs \
    --bi-encoder-name /data/outputs/finetuned_bi_encoder \
    --cache-out /data/outputs/cache/dense_chunk_index_finetuned.pkl

# 3) tune + run the full pipeline -> submission.json
python scripts/03_run_pipeline.py \
    --task2-data-dir /data/TASK2/TASK2 \
    --output-dir /data/outputs \
    --dense-cache-file /data/outputs/cache/dense_chunk_index_finetuned.pkl \
    --bi-encoder-name /data/outputs/finetuned_bi_encoder \
    --cross-encoder-name /data/outputs/finetuned_cross_encoder \
    --stopwords-file /data/stopwords.txt \
    --val-size 1000
```

BM25 cache (`--bm25-cache-file`) is optional in every script — if omitted
or missing, `BM25Retriever.build()` builds the index from scratch (a few
minutes for ~8.5k documents) instead of loading a pickle.

### On Modal.com

```bash
pip install modal
modal setup                      # one-time auth

modal volume create legalqa-data
modal volume put legalqa-data ./TASK2 /TASK2
modal volume put legalqa-data ./stopwords.txt /stopwords.txt
# (optional) modal volume put legalqa-data ./bm25_index_stopword.pkl /bm25_index_stopword.pkl

modal run modal_app.py::build_dense_cache_base
modal run modal_app.py::finetune
modal run modal_app.py::build_dense_cache_finetuned
modal run modal_app.py::run_pipeline
```

Or run the whole chain in one shot: `modal run modal_app.py`.

All outputs (fine-tuned model dirs, dense caches, mined pairs,
`submission_final.json`, tuning/inference logs) land under
`/data/outputs/` on the `legalqa-data` volume — pull them down with
`modal volume get legalqa-data /outputs ./outputs`.

## Config

Everything tunable lives in `legalqa/config.py` as dataclasses
(`PathConfig`, `ModelConfig`, `MiningConfig`, `BiEncoderTrainConfig`,
`CrossEncoderTrainConfig`, `ChunkCacheConfig`, `RetrievalConfig`,
`AnswerConfig`, `CrossEncoderRerankConfig`, `TuningConfig`), bundled into
a single `PipelineConfig`. Scripts build one `PipelineConfig()` and
override fields from CLI args; `PipelineTuner.run()` further mutates the
`retrieval` / `rerank` / `answer` sub-configs in place with the
grid-search winners, exactly mirroring the original notebook's staged
sweeps (W_BM25/W_DENSE -> TOP_N_DOCS -> cross-encoder strategy -> alpha
blend -> max_chars -> dedupe on/off).

## Notes on fidelity

- All regex-based legal-metadata extraction (`legal_metadata.py`),
  header/conclusion templates, post-processing, and the mining/fine-tuning
  hyperparameters are ported verbatim from the two source notebooks.
- `DenseChunkCacheBuilder` (chunking logic) is a faithful reconstruction
  consistent with the `chunk_logic_version` string recorded in the
  original cache's metadata (Điều-boundary splitting, doc-name +
  Điều-title prefixing, boilerplate stripping that explicitly preserves
  "Giải thích từ ngữ", line-join fixes, char-budget packing) — the
  original chunk-builder notebook itself wasn't provided, so exact
  byte-for-byte chunk boundaries may differ slightly; sweep
  `POST_PROCESS_MAX_CHARS`/`TOP_N_DOCS` again after building your cache if
  you're chasing the exact 0.56 METEOR checkpoint.
- `CrossEncoderTrainer` tries the newer
  `CrossEncoderClassificationEvaluator` API first and falls back to the
  deprecated `CEBinaryClassificationEvaluator` (this sidesteps the
  `TypeError` at the bottom of the original fine-tuning notebook's log).