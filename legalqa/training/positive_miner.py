from __future__ import annotations

import re
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional

from tqdm.auto import tqdm

from ..legal_metadata import LegalMetadataExtractor, TitleIndex
from ..retrieval.dense import DenseChunkCache
from ..tokenization import VietnameseTokenizer
from ..utils import logger


@dataclass
class MinedPositive:
    qid: str
    question: str
    gold: str
    doc_id: str
    positive_chunk_pos: int
    match_type: str


class PositiveMiner:
    """Links each training question's gold answer text back to a concrete
    (doc_id, chunk_position) positive, by:

      1. extracting a VBQPPL reference + Điều number from the gold text,
      2. resolving the VBQPPL string to a doc_id via `TitleIndex`,
      3. finding the chunk within that doc whose "Điều N" header matches
         exactly (falling back to lexical-overlap best-match).
    """

    def __init__(
        self,
        chunk_cache: DenseChunkCache,
        tokenizer: VietnameseTokenizer,
        title_index: TitleIndex,
        overlap_fallback_threshold: float = 0.15,
    ):
        self.chunk_cache = chunk_cache
        self.tokenizer = tokenizer
        self.title_index = title_index
        self.overlap_fallback_threshold = overlap_fallback_threshold

    def mine_positive_for_item(self, gold: str) -> Optional[dict]:
        vbqppl = LegalMetadataExtractor.extract_vbqppl(gold)
        dieu_info = LegalMetadataExtractor.extract_dieu_info(gold)
        dieu_nums = dieu_info.dieu_numbers

        doc_id, match_type = self.title_index.find_doc_id_by_vbqppl(vbqppl)
        if doc_id is None:
            return None

        positions = self.chunk_cache.doc_to_chunk_positions.get(doc_id, [])
        if not positions:
            return None

        gold_tokens_set = set(self.tokenizer.tokenize_clean(gold, max_chars=3000))

        if dieu_nums:
            target = dieu_nums[0]
            dieu_re = re.compile(r"Điều\s+" + re.escape(target) + r"[a-zA-Z]?\s*[\.:]")
            exact_matches = [pos for pos in positions if dieu_re.search(self.chunk_cache.chunk_texts_all[pos])]
            if exact_matches:
                if len(exact_matches) == 1:
                    return {"doc_id": doc_id, "chunk_pos": exact_matches[0], "match_type": f"{match_type}+dieu_exact"}
                best = max(
                    exact_matches,
                    key=lambda pos: self.tokenizer.token_overlap_score(
                        gold_tokens_set, self.chunk_cache.chunk_texts_all[pos]
                    ),
                )
                return {"doc_id": doc_id, "chunk_pos": best, "match_type": f"{match_type}+dieu_exact_tiebreak"}

        best_pos, best_score = None, -1.0
        for pos in positions:
            s = self.tokenizer.token_overlap_score(gold_tokens_set, self.chunk_cache.chunk_texts_all[pos])
            if s > best_score:
                best_score, best_pos = s, pos
        if best_pos is not None and best_score >= self.overlap_fallback_threshold:
            return {
                "doc_id": doc_id,
                "chunk_pos": best_pos,
                "match_type": f"{match_type}+overlap_fallback({best_score:.2f})",
            }
        return None

    def mine_all(self, train_raw: Dict[str, dict], show_progress: bool = True) -> List[MinedPositive]:
        mined: List[MinedPositive] = []
        skip_reasons = defaultdict(int)
        t0 = time.time()

        items = train_raw.items()
        iterator = tqdm(items, desc="Mining positives from gold answers") if show_progress else items
        for qid, item in iterator:
            question = item["question"]
            gold = item.get("answer", "") or item.get("gold_answer", "") or ""
            if not gold.strip():
                skip_reasons["empty_gold"] += 1
                continue
            result = self.mine_positive_for_item(gold)
            if result is None:
                skip_reasons["could_not_mine"] += 1
                continue
            mined.append(
                MinedPositive(
                    qid=qid,
                    question=question,
                    gold=gold,
                    doc_id=result["doc_id"],
                    positive_chunk_pos=result["chunk_pos"],
                    match_type=result["match_type"],
                )
            )

        logger.info(f"Positive mining done in {time.time() - t0:.1f}s.")
        logger.info(f"Mined positives: {len(mined)}/{len(train_raw)} ({len(mined) / max(1, len(train_raw)) * 100:.1f}%)")
        logger.info(f"Skip reasons: {dict(skip_reasons)}")
        if len(mined) < 0.3 * len(train_raw):
            logger.warning("Positive-mining yield is LOW (<30%) -- inspect before proceeding to training.")
        return mined