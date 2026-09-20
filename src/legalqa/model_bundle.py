"""Native persistence and validation for the frozen selector bundle."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path
from typing import Mapping, Sequence

from .meta_selector import GATE_THRESHOLD, META_FEATURES, V21_FEATURES, FrozenSelectorV21
from .selector_models import (
    MODEL_NAMES,
    RETARGET_SEEDS,
    make_heterogeneous_model,
    make_retarget_xgb,
)


BUNDLE_VERSION = "legalqa_selector_v21_gate055_06299_v1"
TRAIN_SPLIT = "scale5000_seed42_positions_2000_6999"
MANIFEST_NAME = "selector_manifest.json"


def model_filename(kind: str, name: str) -> str:
    if kind == "retarget":
        return f"retarget_{name}.json"
    if kind == "heterogeneous":
        if name.startswith("xgb"):
            return f"heterogeneous_{name}.json"
        if name.startswith("lgb"):
            return f"heterogeneous_{name}.txt"
        return f"heterogeneous_{name}.cbm"
    if kind == "meta":
        return "xgb_rank_meta.json"
    if kind == "pairwise":
        return "pairwise_regret_v2.json"
    raise KeyError(kind)


def save_native_model(model, path: Path, family: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the native suffix: CatBoost and XGBoost use it to select the model format.
    temporary = path.with_name(f"{path.stem}.tmp{path.suffix}")
    if family == "lightgbm":
        booster = getattr(model, "booster_", model)
        booster.save_model(str(temporary))
    else:
        model.save_model(str(temporary))
    temporary.replace(path)


def load_native_model(path: Path, kind: str, name: str):
    path = Path(path)
    if kind == "retarget":
        model = make_retarget_xgb(int(name))
        model.load_model(str(path))
        return model
    if kind == "heterogeneous":
        if name.startswith("lgb"):
            import lightgbm as lgb

            return lgb.Booster(model_file=str(path))
        model = make_heterogeneous_model(name)
        model.load_model(str(path))
        return model
    if kind == "meta":
        from .meta_selector import make_meta_model

        model = make_meta_model()
        model.load_model(str(path))
        return model
    if kind == "pairwise":
        from .meta_selector import make_pairwise_model

        model = make_pairwise_model()
        model.load_model(str(path))
        return model
    raise KeyError(kind)


def _family(name: str) -> str:
    if name.startswith("lgb"):
        return "lightgbm"
    if name.startswith("cat"):
        return "catboost"
    return "xgboost"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def expected_model_files() -> tuple[str, ...]:
    return tuple(
        [model_filename("retarget", str(seed)) for seed in RETARGET_SEEDS]
        + [model_filename("heterogeneous", name) for name in MODEL_NAMES]
        + [model_filename("meta", "xgb_rank_meta")]
        + [model_filename("pairwise", "pairwise_regret_v2")]
    )


def build_manifest(
    bundle_dir: Path,
    *,
    oof_rows: int,
    training_seconds: float,
    training_signature: str,
    target_cache: Mapping,
    raw_reproduction: Mapping,
) -> dict:
    bundle_dir = Path(bundle_dir)
    files = {
        name: {
            "size_bytes": (bundle_dir / name).stat().st_size,
            "sha256": sha256_file(bundle_dir / name),
        }
        for name in expected_model_files()
    }
    return {
        "version": BUNDLE_VERSION,
        "train_split": TRAIN_SPLIT,
        "action_feature_count": 40,
        "meta_feature_count": len(META_FEATURES),
        "v21_feature_count": len(V21_FEATURES),
        "retarget_seeds": list(RETARGET_SEEDS),
        "model_names": list(MODEL_NAMES),
        "pairwise_seed": 2026,
        "gate_threshold": GATE_THRESHOLD,
        "postprocess": "best_06252",
        "training_qid_count": 5000,
        "final_target_rows": 75000,
        "oof_qid_count": 5000,
        "oof_rows": int(oof_rows),
        "model_count": len(files),
        "files": files,
        "library_versions": {
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
            "xgboost": _version("xgboost"),
            "lightgbm": _version("lightgbm"),
            "catboost": _version("catboost"),
        },
        "training_seconds": float(training_seconds),
        "training_signature": training_signature,
        "target_cache": dict(target_cache),
        "raw_reproduction": dict(raw_reproduction),
    }


def validate_manifest(manifest: Mapping, bundle_dir: Path, verify_hashes: bool = False) -> None:
    expected = {
        "version": BUNDLE_VERSION,
        "train_split": TRAIN_SPLIT,
        "action_feature_count": 40,
        "meta_feature_count": 36,
        "v21_feature_count": 109,
        "retarget_seeds": list(RETARGET_SEEDS),
        "model_names": list(MODEL_NAMES),
        "pairwise_seed": 2026,
        "gate_threshold": GATE_THRESHOLD,
        "postprocess": "best_06252",
        "training_qid_count": 5000,
        "final_target_rows": 75000,
        "oof_qid_count": 5000,
        "oof_rows": 75000,
        "model_count": 17,
    }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Invalid selector V2.1 manifest: {mismatches}")
    signature = manifest.get("training_signature")
    if not isinstance(signature, str) or len(signature) != 64:
        raise RuntimeError("Selector manifest has no valid training signature")
    target_cache = manifest.get("target_cache")
    if (
        not isinstance(target_cache, Mapping)
        or target_cache.get("signature") != signature
        or target_cache.get("rows") != 75000
    ):
        raise RuntimeError("Selector manifest target-cache provenance mismatch")
    reproduction = manifest.get("raw_reproduction")
    if (
        not isinstance(reproduction, Mapping)
        or reproduction.get("sample_actions", 0) <= 0
        or reproduction.get("max_abs_difference") != 0.0
    ):
        raise RuntimeError("Selector manifest raw-target reproduction guard failed")
    files = manifest.get("files")
    if not isinstance(files, Mapping) or tuple(files) != expected_model_files():
        raise RuntimeError("Selector manifest model file order/schema mismatch")
    for name, metadata in files.items():
        path = Path(bundle_dir) / name
        if not path.is_file() or path.stat().st_size != metadata.get("size_bytes"):
            raise RuntimeError(f"Missing or truncated selector artifact: {path}")
        if verify_hashes and sha256_file(path) != metadata.get("sha256"):
            raise RuntimeError(f"Selector artifact hash mismatch: {path}")


def save_bundle(
    bundle_dir: Path,
    retarget_models: Sequence,
    heterogeneous_models: Mapping[str, object],
    meta_model,
    pairwise_model,
    *,
    oof_rows: int,
    training_seconds: float,
    training_signature: str,
    target_cache: Mapping,
    raw_reproduction: Mapping,
) -> dict:
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)
    if len(retarget_models) != 5 or tuple(heterogeneous_models) != MODEL_NAMES:
        raise RuntimeError("Incomplete selector models cannot be published")
    for seed, model in zip(RETARGET_SEEDS, retarget_models):
        save_native_model(
            model, bundle_dir / model_filename("retarget", str(seed)), "xgboost",
        )
    for name in MODEL_NAMES:
        save_native_model(
            heterogeneous_models[name],
            bundle_dir / model_filename("heterogeneous", name),
            _family(name),
        )
    save_native_model(meta_model, bundle_dir / model_filename("meta", "xgb_rank_meta"), "xgboost")
    save_native_model(
        pairwise_model,
        bundle_dir / model_filename("pairwise", "pairwise_regret_v2"),
        "xgboost",
    )
    manifest = build_manifest(
        bundle_dir,
        oof_rows=oof_rows,
        training_seconds=training_seconds,
        training_signature=training_signature,
        target_cache=target_cache,
        raw_reproduction=raw_reproduction,
    )
    temporary = bundle_dir / (MANIFEST_NAME + ".tmp")
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(bundle_dir / MANIFEST_NAME)
    return manifest


def load_bundle(bundle_dir: Path, verify_hashes: bool = False) -> tuple[FrozenSelectorV21, dict]:
    bundle_dir = Path(bundle_dir)
    manifest_path = bundle_dir / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Missing selector V2.1 manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_manifest(manifest, bundle_dir, verify_hashes=verify_hashes)
    retarget_models = [
        load_native_model(
            bundle_dir / model_filename("retarget", str(seed)), "retarget", str(seed),
        )
        for seed in RETARGET_SEEDS
    ]
    heterogeneous_models = {
        name: load_native_model(
            bundle_dir / model_filename("heterogeneous", name), "heterogeneous", name,
        )
        for name in MODEL_NAMES
    }
    meta_model = load_native_model(
        bundle_dir / model_filename("meta", "xgb_rank_meta"), "meta", "xgb_rank_meta",
    )
    pairwise_model = load_native_model(
        bundle_dir / model_filename("pairwise", "pairwise_regret_v2"),
        "pairwise",
        "pairwise_regret_v2",
    )
    selector = FrozenSelectorV21(
        retarget_models, heterogeneous_models, meta_model, pairwise_model,
    )
    selector.bundle_signature = hashlib.sha256(
        json.dumps(manifest["files"], sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return selector, manifest
