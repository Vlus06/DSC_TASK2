"""
Modal.com deployment for the LegalQA pipeline.

Data layout on the Modal Volume ("legalqa-data"), mounted at /data:
    /data/TASK2/TASK2/...              <- unzip TASK2.zip here
    /data/stopwords.txt
    /data/bm25_index_stopword.pkl      <- optional, built if missing
    /data/outputs/...                  <- all pipeline outputs land here
    /data/outputs/cache/dense_chunk_index_base.pkl
    /data/outputs/cache/dense_chunk_index_finetuned.pkl
    /data/outputs/finetuned_bi_encoder/
    /data/outputs/finetuned_cross_encoder/

Usage (from your machine, with `modal` CLI installed & authenticated):

    # 0) one-time: upload your data into the volume
    modal volume create legalqa-data
    modal volume put legalqa-data ./TASK2 /TASK2
    modal volume put legalqa-data ./stopwords.txt /stopwords.txt

    # 1) build the dense cache with the BASE model (needed for mining)
    modal run modal_app.py::build_dense_cache_base

    # 2) fine-tune retriever + reranker (mines hard negatives, trains both)
    modal run modal_app.py::finetune

    # 3) rebuild the dense cache with the FINE-TUNED bi-encoder
    modal run modal_app.py::build_dense_cache_finetuned

    # 4) tune + run the full QA pipeline, produce submission.json
    modal run modal_app.py::run_pipeline

Each function is also runnable as a plain `modal run` entrypoint; edit the
default kwargs below (or pass `--kwarg value`) to change paths.
"""
from __future__ import annotations

import modal

APP_NAME = "legalqa-pipeline"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git")
    .pip_install_from_requirements("requirements.txt")
    .add_local_dir("legalqa", remote_path="/root/legalqa")
    .add_local_dir("scripts", remote_path="/root/scripts")
)

volume = modal.Volume.from_name("legalqa-data", create_if_missing=True)
VOLUME_MOUNT = "/data"

app = modal.App(APP_NAME, image=image)

GPU = "A100"  # bump down to "L4" / "T4" for cheaper (slower) runs if needed
TIMEOUT_SECONDS = 60 * 60 * 12  # 12h ceiling; adjust per stage if needed


@app.function(gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=TIMEOUT_SECONDS)
def build_dense_cache_base(
    task2_data_dir: str = "/data/TASK2/TASK2",
    output_dir: str = "/data/outputs",
    bi_encoder_name: str = "AITeamVN/Vietnamese_Embedding",
    cache_out: str = "/data/outputs/cache/dense_chunk_index_base.pkl",
    encode_max_seq_length: int = 2048,
):
    """Stage 2a: chunk + embed the corpus with the BASE bi-encoder.

    Needed before Stage 1 (fine-tuning), since hard-negative mining looks
    up candidate chunks via a pre-built dense cache.
    """
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/02_build_dense_cache.py",
            "--task2-data-dir", task2_data_dir,
            "--output-dir", output_dir,
            "--bi-encoder-name", bi_encoder_name,
            "--cache-out", cache_out,
            "--encode-max-seq-length", str(encode_max_seq_length),
        ],
        check=True,
        cwd="/root",
    )
    volume.commit()


@app.function(gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=TIMEOUT_SECONDS)
def finetune(
    task2_data_dir: str = "/data/TASK2/TASK2",
    output_dir: str = "/data/outputs",
    dense_cache_file: str = "/data/outputs/cache/dense_chunk_index_base.pkl",
    stopwords_file: str = "/data/stopwords.txt",
    bm25_cache_file: str = "/data/bm25_index_stopword.pkl",
):
    """Stage 1: mine hard negatives + fine-tune bi-encoder & cross-encoder."""
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/01_finetune_retriever_reranker.py",
        "--task2-data-dir", task2_data_dir,
        "--output-dir", output_dir,
        "--dense-cache-file", dense_cache_file,
    ]
    if stopwords_file and os.path.exists(stopwords_file):
        cmd += ["--stopwords-file", stopwords_file]
    if bm25_cache_file and os.path.exists(bm25_cache_file):
        cmd += ["--bm25-cache-file", bm25_cache_file]

    subprocess.run(cmd, check=True, cwd="/root")
    volume.commit()


@app.function(gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=TIMEOUT_SECONDS)
def build_dense_cache_finetuned(
    task2_data_dir: str = "/data/TASK2/TASK2",
    output_dir: str = "/data/outputs",
    bi_encoder_name: str = "/data/outputs/finetuned_bi_encoder",
    cache_out: str = "/data/outputs/cache/dense_chunk_index_finetuned.pkl",
    encode_max_seq_length: int = 2048,
):
    """Stage 2b: rebuild the dense cache with the FINE-TUNED bi-encoder --
    this is the cache the QA pipeline actually retrieves against.
    """
    import subprocess
    import sys

    subprocess.run(
        [
            sys.executable,
            "scripts/02_build_dense_cache.py",
            "--task2-data-dir", task2_data_dir,
            "--output-dir", output_dir,
            "--bi-encoder-name", bi_encoder_name,
            "--cache-out", cache_out,
            "--encode-max-seq-length", str(encode_max_seq_length),
        ],
        check=True,
        cwd="/root",
    )
    volume.commit()


@app.function(gpu=GPU, volumes={VOLUME_MOUNT: volume}, timeout=TIMEOUT_SECONDS)
def run_pipeline(
    task2_data_dir: str = "/data/TASK2/TASK2",
    output_dir: str = "/data/outputs",
    dense_cache_file: str = "/data/outputs/cache/dense_chunk_index_finetuned.pkl",
    bi_encoder_name: str = "/data/outputs/finetuned_bi_encoder",
    cross_encoder_name: str = "/data/outputs/finetuned_cross_encoder",
    stopwords_file: str = "/data/stopwords.txt",
    bm25_cache_file: str = "/data/bm25_index_stopword.pkl",
    val_size: int = 1000,
    run_tag: str = "final",
):
    """Stage 3: grid-search tune, then run inference -> submission.json."""
    import os
    import subprocess
    import sys

    cmd = [
        sys.executable,
        "scripts/03_run_pipeline.py",
        "--task2-data-dir", task2_data_dir,
        "--output-dir", output_dir,
        "--dense-cache-file", dense_cache_file,
        "--bi-encoder-name", bi_encoder_name,
        "--val-size", str(val_size),
        "--run-tag", run_tag,
    ]
    if os.path.isdir(cross_encoder_name):
        cmd += ["--cross-encoder-name", cross_encoder_name]
    if stopwords_file and os.path.exists(stopwords_file):
        cmd += ["--stopwords-file", stopwords_file]
    if bm25_cache_file and os.path.exists(bm25_cache_file):
        cmd += ["--bm25-cache-file", bm25_cache_file]

    subprocess.run(cmd, check=True, cwd="/root")
    volume.commit()


@app.local_entrypoint()
def main():
    """`modal run modal_app.py` with no sub-function runs the full chain
    end-to-end (base cache -> finetune -> finetuned cache -> pipeline).
    Prefer calling the individual stages directly while iterating.
    """
    build_dense_cache_base.remote()
    finetune.remote()
    build_dense_cache_finetuned.remote()
    run_pipeline.remote()