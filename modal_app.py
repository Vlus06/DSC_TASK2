"""Cost-aware Modal workflow for the LegalQA pipeline."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parent
APP_DATA = Path("/app/data")
REMOTE_CACHE = APP_DATA / "cache"
REMOTE_RESULTS = Path("/results")
REMOTE_MODELS = Path("/models/legalqa")

app = modal.App("legalqa-pipeline")
data_volume = modal.Volume.from_name("legalqa-data", create_if_missing=True)
results_volume = modal.Volume.from_name("legalqa-results", create_if_missing=True)
models_volume = modal.Volume.from_name("legalqa-models", create_if_missing=True)

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "numpy>=1.26,<3", "pandas>=2.2,<3", "xgboost==3.2.0",
        "rank-bm25==0.2.2", "underthesea>=6.8,<9", "nltk>=3.9,<4",
    )
    .env({"PYTHONUNBUFFERED": "1", "NLTK_DATA": "/models/nltk"})
    .add_local_dir(str(ROOT / "src"), "/app/src", ignore=["**/__pycache__/**"])
    .add_local_dir(str(ROOT / "scripts"), "/app/scripts", ignore=["**/__pycache__/**"])
)

gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install_from_requirements(str(ROOT / "requirements.txt"))
    .env({
        "HF_HOME": "/models/huggingface",
        "NLTK_DATA": "/models/nltk",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    .add_local_dir(str(ROOT / "src"), "/app/src", ignore=["**/__pycache__/**"])
    .add_local_dir(str(ROOT / "scripts"), "/app/scripts", ignore=["**/__pycache__/**"])
)

DATA_VOLUMES = {"/app/data": data_volume, "/models": models_volume}
ALL_VOLUMES = {**DATA_VOLUMES, "/results": results_volume}

BASE_CACHE_NAMES = {
    "bm25": "bm25_index.pkl",
    "dense": "parent_embeddings.pkl",
    "child": "child_embeddings.pkl",
}
FEATURE_CACHE_NAMES = {
    "top_documents": "top_documents.pkl",
    "candidate_audit": "candidate_audit.pkl",
    "pair_features": "pair_features.pkl",
    "singleton_features": "singleton_features.pkl",
}
PACKED_CORPUS_NAME = "corpus_metadata.pkl"
FEATURE_MARKER_NAME = "feature_cache_complete.json"
MODEL_NAMES = [f"answer_ranker_seed_{seed}.json" for seed in (42, 10042, 20042, 30042, 40042)]
MODEL_MANIFEST_NAME = "training_manifest.json"


def _has_file(directory: Path, names: tuple[str, ...]) -> bool:
    return any(
        (directory / name).is_file() and (directory / name).stat().st_size > 0
        for name in names
    )


def _stream(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = env.get("LEGALQA_CPU_THREADS", "16")
    print("Command:", " ".join(command), flush=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="", flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code != 0:
        raise RuntimeError(f"Stage failed with exit code {code}; log={log_path}")


@app.function(image=cpu_image, volumes=ALL_VOLUMES, cpu=2, memory=8192, timeout=30 * 60)
def inspect_inputs() -> dict:
    for volume in (data_volume, results_volume, models_volume):
        volume.reload()
    task2 = APP_DATA / "TASK2"
    train = task2 / "train.json"
    public = task2 / "public-official.json"
    context_dirs = [task2 / "selected-contexts", task2 / "selected-contexts/selected-contexts"]
    context_dir = next((p for p in context_dirs if p.is_dir()), None)
    dataset_ready = train.is_file() and public.is_file() and context_dir is not None
    stopwords_ready = (APP_DATA / "stopwords.txt").is_file()
    base = {
        key: (REMOTE_CACHE / name).is_file() and (REMOTE_CACHE / name).stat().st_size > 0
        for key, name in BASE_CACHE_NAMES.items()
    }
    feature_caches = {
        key: _has_file(REMOTE_CACHE, (name,))
        for key, name in FEATURE_CACHE_NAMES.items()
    }
    models = {
        name: (REMOTE_MODELS / name).is_file() and (REMOTE_MODELS / name).stat().st_size > 0
        for name in MODEL_NAMES
    }
    model_manifest_ready = False
    model_manifest_path = REMOTE_MODELS / MODEL_MANIFEST_NAME
    if model_manifest_path.is_file() and model_manifest_path.stat().st_size > 0:
        try:
            manifest = json.loads(model_manifest_path.read_text(encoding="utf-8"))
            model_manifest_ready = (
                manifest.get("version") == "legalqa_ranker_ensemble_v1"
                and manifest.get("rows") == 105000
                and manifest.get("qids") == 7000
                and manifest.get("seeds") == [42, 10042, 20042, 30042, 40042]
            )
        except (OSError, TypeError, ValueError):
            model_manifest_ready = False
    state = {
        "dataset_ready": dataset_ready,
        "stopwords_ready": stopwords_ready,
        "packed_corpus": _has_file(REMOTE_CACHE, (PACKED_CORPUS_NAME,)),
        "base": base,
        "feature_caches": feature_caches,
        "feature_marker": _has_file(REMOTE_CACHE, (FEATURE_MARKER_NAME,)),
        "models": models,
        "model_manifest": model_manifest_ready,
    }
    print(json.dumps(state, ensure_ascii=False, indent=2), flush=True)
    return state


@app.function(image=cpu_image, volumes=DATA_VOLUMES, cpu=8, memory=16384, timeout=3 * 3600)
def pack_corpus() -> None:
    data_volume.reload()
    models_volume.reload()
    try:
        _stream(
            [sys.executable, "-u", "/app/scripts/modal_stages.py", "pack-corpus"],
            Path("/tmp/pack_corpus.log"),
        )
    finally:
        data_volume.commit()


@app.function(image=cpu_image, volumes=DATA_VOLUMES, cpu=16, memory=32768, timeout=6 * 3600)
def build_bm25() -> None:
    data_volume.reload()
    models_volume.reload()
    try:
        _stream(
            [sys.executable, "-u", "/app/scripts/modal_stages.py", "build-bm25"],
            Path("/tmp/build_bm25.log"),
        )
    finally:
        data_volume.commit()


@app.function(
    image=cpu_image, volumes=ALL_VOLUMES, cpu=16, memory=65536,
    timeout=3 * 3600, scaledown_window=30,
)
def check_base(run_id: str = "base-check") -> None:
    for volume in (data_volume, results_volume, models_volume):
        volume.reload()
    out = REMOTE_RESULTS / run_id
    try:
        _stream(
            [
                sys.executable, "-u", "/app/scripts/run.py",
                "--mode", "check",
                "--task2-dir", "/app/data/TASK2",
                "--cache-dir", "/app/data/cache",
                "--stopwords", "/app/data/stopwords.txt",
                "--output-dir", str(out),
                "--device", "cpu",
            ],
            out / "00_check_base.log",
        )
    finally:
        results_volume.commit()
        models_volume.commit()


@app.function(
    image=gpu_image, volumes=DATA_VOLUMES, gpu="H100", cpu=16,
    memory=131072, timeout=24 * 3600, scaledown_window=30,
)
def build_dense_or_child(kind: str, batch_size: int) -> None:
    data_volume.reload()
    models_volume.reload()
    try:
        _stream(
            [
                sys.executable, "-u", "/app/scripts/modal_stages.py",
                f"build-{kind}", "--batch-size", str(batch_size),
            ],
            Path(f"/tmp/build_{kind}.log"),
        )
    finally:
        data_volume.commit()
        models_volume.commit()


@app.function(
    image=gpu_image, volumes=ALL_VOLUMES, gpu="H100", cpu=16,
    memory=131072, timeout=24 * 3600, scaledown_window=30,
)
def prepare_features(run_id: str, ce_batch_size: int) -> None:
    for volume in (data_volume, results_volume, models_volume):
        volume.reload()
    out = REMOTE_RESULTS / run_id
    try:
        _stream(
            [
                sys.executable, "-u", "/app/scripts/run.py",
                "--mode", "prepare",
                "--task2-dir", "/app/data/TASK2",
                "--cache-dir", "/app/data/cache",
                "--stopwords", "/app/data/stopwords.txt",
                "--output-dir", str(out),
                "--device", "cuda",
                "--ce-batch-size", str(ce_batch_size),
            ],
            out / "01_prepare_features.log",
        )
        files = {}
        for key, name in FEATURE_CACHE_NAMES.items():
            path = REMOTE_CACHE / name
            files[key] = {"name": path.name, "size_bytes": path.stat().st_size}
        marker = {
            "version": "legalqa_feature_cache_complete_v1",
            "files": files,
        }
        (REMOTE_CACHE / FEATURE_MARKER_NAME).write_text(
            json.dumps(marker, indent=2), encoding="utf-8",
        )
    finally:
        data_volume.commit()
        results_volume.commit()
        models_volume.commit()


@app.function(
    image=cpu_image, volumes=ALL_VOLUMES, cpu=32, memory=65536,
    timeout=24 * 3600, scaledown_window=30,
)
def train_rankers(run_id: str) -> None:
    for volume in (data_volume, results_volume, models_volume):
        volume.reload()
    out = REMOTE_RESULTS / run_id
    os.environ["LEGALQA_CPU_THREADS"] = "32"
    try:
        _stream(
            [
                sys.executable, "-u", "/app/scripts/modal_stages.py", "train",
                "--run-dir", str(out),
            ],
            out / "02_train_rankers.log",
        )
    finally:
        results_volume.commit()
        models_volume.commit()


@app.function(
    image=gpu_image, volumes=ALL_VOLUMES, gpu="H100", cpu=16,
    memory=131072, timeout=24 * 3600, scaledown_window=30,
)
def infer_public(run_id: str, public_limit: int, checkpoint_every: int, ce_batch_size: int) -> None:
    for volume in (data_volume, results_volume, models_volume):
        volume.reload()
    out = REMOTE_RESULTS / run_id
    try:
        _stream(
            [
                sys.executable, "-u", "/app/scripts/modal_stages.py", "infer",
                "--run-dir", str(out),
                "--public-limit", str(public_limit),
                "--checkpoint-every", str(checkpoint_every),
                "--ce-batch-size", str(ce_batch_size),
            ],
            out / "03_infer_public.log",
        )
    finally:
        results_volume.commit()
        models_volume.commit()


@app.function(image=cpu_image, volumes=ALL_VOLUMES, cpu=0.25, memory=512, timeout=24 * 3600)
def workflow(run_id: str, ce_batch_size: int, public_limit: int, checkpoint_every: int) -> str:
    print(f"Run ID: {run_id}", flush=True)
    state = inspect_inputs.remote()
    if not state["dataset_ready"] or not state["stopwords_ready"]:
        raise RuntimeError("TASK2 dataset/stopwords is missing on legalqa-data")
    if not state["packed_corpus"]:
        print("[workflow] packed corpus missing -> CPU pack once", flush=True)
        pack_corpus.remote()
    if not state["base"]["bm25"]:
        print("[workflow] BM25 missing -> CPU build", flush=True)
        build_bm25.remote()
    if not state["base"]["dense"]:
        print("[workflow] dense parent missing -> H100 build", flush=True)
        build_dense_or_child.remote("dense", 64)
    if not state["base"]["child"]:
        print("[workflow] child missing -> H100 build", flush=True)
        build_dense_or_child.remote("child", 64)

    state = inspect_inputs.remote()
    if not state["packed_corpus"]:
        raise RuntimeError("Packed corpus build incomplete")
    if not all(state["base"].values()):
        raise RuntimeError(f"Base cache build incomplete: {state['base']}")
    features_rebuilt = not (all(state["feature_caches"].values()) and state["feature_marker"])
    if features_rebuilt:
        print("[workflow] feature cache missing/incomplete -> H100 prepare", flush=True)
        prepare_features.remote(run_id, ce_batch_size)
    else:
        print("[workflow] four feature caches found -> reuse", flush=True)

    models_ready = all(state["models"].values()) and state["model_manifest"]
    if models_ready and not features_rebuilt:
        print("[workflow] five trained rankers found -> reuse", flush=True)
    else:
        print("[workflow] trained rankers missing/stale -> CPU train", flush=True)
        train_rankers.remote(run_id)
    print("[workflow] public inference -> H100", flush=True)
    infer_public.remote(run_id, public_limit, checkpoint_every, ce_batch_size)
    print(f"[workflow] DONE: /results/{run_id}", flush=True)
    return run_id


def _sync_local_inputs(state: dict) -> bool:
    """Upload only files absent remotely; returns whether anything changed."""
    uploads: list[tuple[str, Path, str]] = []
    if not state["dataset_ready"]:
        uploads.append(("dir", ROOT / "data/TASK2", "/TASK2"))
    if not state["stopwords_ready"]:
        uploads.append(("file", ROOT / "data/stopwords.txt", "/stopwords.txt"))
    for key, ready in state["base"].items():
        local = ROOT / "data/cache" / BASE_CACHE_NAMES[key]
        if not ready and local.is_file():
            uploads.append(("file", local, f"/cache/{local.name}"))
    if not uploads:
        return False
    print("Uploading only missing local inputs to legalqa-data:", flush=True)
    with data_volume.batch_upload(force=True) as batch:
        for kind, local, remote in uploads:
            if not local.exists():
                raise FileNotFoundError(f"Missing local input: {local}")
            print(f"  {local} -> {remote}", flush=True)
            if kind == "dir":
                batch.put_directory(local, remote)
            else:
                batch.put_file(local, remote)
    return True


def _download_outputs(run_id: str, public_limit: int) -> Path:
    destination = ROOT / "outputs/modal" / run_id
    destination.mkdir(parents=True, exist_ok=True)
    names = [
        "01_prepare_features.log", "02_train_rankers.log", "03_infer_public.log",
        "training_manifest.json", "run_manifest.json",
        (f"submission_smoke_{public_limit}.json" if public_limit else "submission.json"),
        (f"inference_log_smoke_{public_limit}.json" if public_limit else "inference_log.json"),
    ]
    for name in names:
        try:
            content = b"".join(results_volume.read_file(f"/{run_id}/{name}"))
        except Exception:
            continue
        (destination / name).write_bytes(content)
    return destination


@app.local_entrypoint()
def main(public_limit: int = 0, ce_batch_size: int = 32, checkpoint_every: int = 25):
    if not 0 <= public_limit <= 1000:
        raise ValueError("public_limit must be in 0..1000")
    run_id = uuid.uuid4().hex
    state = inspect_inputs.remote()
    if _sync_local_inputs(state):
        state = inspect_inputs.remote()
    print(f"Starting cost-aware workflow; Run ID: {run_id}", flush=True)
    workflow.remote(run_id, ce_batch_size, public_limit, checkpoint_every)
    destination = _download_outputs(run_id, public_limit)
    print(f"Downloaded outputs to: {destination}", flush=True)
