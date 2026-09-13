import math
import random
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from legalqa.engine import ACTION_FEATURES, PAIR_FEATURES, pair_features, singleton_action_features
from legalqa.artifacts import split_training_data


class RankingFeatureTests(unittest.TestCase):
    def setUp(self):
        self.q = 'Điều kiện cấp giấy phép năm 2020 là gì?'
        self.c1 = {
            'doc_id': 'A', 'text': 'Văn bản A.\nĐiều 1 Điều kiện cấp giấy phép năm 2020.',
            'emb': 0.8, 'ce': 1.2, 'kind': 'parent', 'ce_rank': 1,
        }
        self.c2 = {
            'doc_id': 'B', 'text': 'Văn bản B.\nĐiều 2 Hồ sơ gồm các giấy tờ cần thiết.',
            'emb': 0.7, 'ce': 0.9, 'kind': 'child', 'ce_rank': 3,
        }

    def test_feature_counts(self):
        self.assertEqual(len(PAIR_FEATURES), 39)
        self.assertEqual(len(ACTION_FEATURES), 40)
        self.assertEqual(len(set(PAIR_FEATURES)), 39)

    def test_pair_feature_schema(self):
        f = pair_features(self.q, self.c1, self.c2)
        self.assertEqual(set(f), set(PAIR_FEATURES))
        self.assertEqual(f['contains_rank1'], 1)
        self.assertEqual(f['contains_rank2'], 0)
        self.assertEqual(f['same_doc'], 0)
        self.assertEqual(f['same_kind'], 0)
        self.assertEqual(f['rank_sum'], 4)
        self.assertEqual(f['rank_gap'], 2)

    def test_singleton_exact_nan_policy(self):
        f = singleton_action_features(self.q, self.c1)
        self.assertEqual(set(f), set(PAIR_FEATURES))
        self.assertTrue(math.isnan(f['c2_ce']))
        self.assertTrue(math.isnan(f['same_doc']))
        self.assertTrue(math.isnan(f['len_ratio']))
        self.assertTrue(math.isnan(f['rank_gap']))
        self.assertEqual(f['ce_sum'], self.c1['ce'])
        self.assertEqual(f['ce_mean'], self.c1['ce'])
        self.assertEqual(f['ce_gap'], 0.0)
        self.assertEqual(f['pair_query_coverage'], f['c1_q_overlap'])
        self.assertEqual(f['incremental_q_cov_c2'], 0.0)
        self.assertEqual(f['incremental_q_cov_c1'], f['c1_q_overlap'])
        self.assertEqual(f['action_size'] if 'action_size' in f else 1, 1)

    def test_split_training_data_is_exact_seed42_order(self):
        train = {str(i): {} for i in range(7000)}
        got = split_training_data(train, 42)
        qids = list(train.keys())
        random.Random(42).shuffle(qids)
        self.assertEqual(got[0], qids[:1000])
        self.assertEqual(got[1], qids[1000:2000])
        self.assertEqual(got[2], qids[2000:7000])


if __name__ == '__main__':
    unittest.main()
