from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from typing import Dict

from ..config import PipelineConfig
from ..corpus import load_public_official_json
from ..pipeline.qa_pipeline import QAPipeline
from ..utils import free_memory, logger


class InferenceRunner:
    """Runs `QAPipeline.predict()` over every question in
    `public-official.json` and writes `submission.json` + a per-question
    log to `config.paths.output_dir`.
    """

    def __init__(self, config: PipelineConfig, pipeline: QAPipeline, run_tag: str = "final"):
        self.config = config
        self.pipeline = pipeline
        self.run_tag = run_tag

    def run(self) -> Dict[str, dict]:
        infer_raw = load_public_official_json(self.config.paths.public_official_path)
        infer_data = {qid: v["question"] for qid, v in infer_raw.items()}
        infer_qids = list(infer_data.keys())
        n_infer = len(infer_qids)
        logger.info(f"Loaded {n_infer} questions from public-official.json for inference.")

        predictions: Dict[str, dict] = {}
        infer_log = []
        t_start = time.time()

        for i, qid in enumerate(infer_qids):
            question = infer_data[qid]
            log_tmp: list = []
            t0 = time.time()
            try:
                pred_answer = self.pipeline.predict(question, log_list=log_tmp, qid=qid)
                err = None
            except Exception as e:
                pred_answer = ""
                err = str(e)
                free_memory()
            t_total = time.time() - t0

            n_docs, t_r, t_g, n_kept = (
                (log_tmp[0][1], log_tmp[0][2], log_tmp[0][3], log_tmp[0][4]) if log_tmp else (0, 0.0, 0.0, 0)
            )

            predictions[qid] = {"answer": pred_answer}
            infer_log.append(
                {
                    "qid": qid,
                    "question": question,
                    "pred": pred_answer,
                    "t_retrieve": t_r,
                    "t_gen": t_g,
                    "t_total": t_total,
                    "n_docs": n_docs,
                    "n_kept_chunks": n_kept,
                    "error": err,
                }
            )

            n_done = i + 1
            if n_done % 20 == 0 or n_done == n_infer:
                elapsed = time.time() - t_start
                avg = elapsed / n_done
                eta = avg * (n_infer - n_done)
                logger.info(f"[{n_done:4d}/{n_infer}] elapsed={elapsed:.0f}s ETA={eta:.0f}s")
                free_memory()

        total_elapsed = time.time() - t_start
        n_errors = sum(1 for r in infer_log if r["error"])
        n_empty = sum(1 for r in infer_log if not r["pred"])
        logger.info("=" * 70)
        logger.info(f"INFERENCE done: {n_infer} questions in {total_elapsed:.0f}s")
        logger.info(f"Errors: {n_errors}/{n_infer} | Empty predictions: {n_empty}/{n_infer}")
        logger.info("=" * 70)

        self.config.paths.ensure_dirs()
        submission_path = os.path.join(self.config.paths.output_dir, f"submission_{self.run_tag}.json")
        with open(submission_path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved submission to: {submission_path}")

        log_path = os.path.join(self.config.paths.output_dir, f"inference_log_{self.run_tag}.json")
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(infer_log, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved inference log to: {log_path}")

        return predictions