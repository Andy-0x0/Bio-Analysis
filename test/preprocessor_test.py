import time
from pathlib import Path

import random
import pandas as pd
import numpy as np

from src.util import ABlockLogger, fetch_tokens, load_paths
from src.preprocess import Preprocessor, BlockPreprocessor, EventPreprocessor, TARGETS

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory


# global test report logger
logger = ABlockLogger(level="DEBUG")


def test_preprocessor_determinism():

    def generate_random_config(token, seed=42):
        random.seed(seed)
        np.random.seed(seed)

        tvt_ratio = np.random.rand(3)
        tvt_ratio = tvt_ratio / sum(tvt_ratio)

        return {
            "beh_path": PATH_DATA / Path(f'{token}_B.csv'),
            "sig_path": PATH_DATA / Path(f'{token}_S.csv'),
            "tvt_ratio": tuple(tvt_ratio.tolist()),
            "shuffle": random.choice([True, False]),
            "normalizer": random.choice(['z-score', 'min-max', None]),
            "agg": random.choice([True, False]),
            "smote": random.choice([random.random(), None]),
            "resize": random.choice([True, False]),
            "target": random.choice(['cont', 'cont&inv', 'cont_inv', 'both', 'both_inv', 'both&inv']),
            "log_level": "ERROR"
        }

    tokens = fetch_tokens(folder=PATH_DATA, shard="all")
    count = 0
    for token in tokens:
        config = generate_random_config(token, int(time.time()))
        try:
            prep1 = Preprocessor(**config)
            prep2 = Preprocessor(**config)
        except Exception as e:
            continue

        try:
            pd.testing.assert_frame_equal(prep1.get_origin_total(shard="all"), prep2.get_origin_total(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_after_total(shard="all"), prep2.get_after_total(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_train(shard="all"), prep2.get_train(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_valid(shard="all"), prep2.get_valid(shard="all"))
            pd.testing.assert_frame_equal(prep1.get_test(shard="all"), prep2.get_test(shard="all"))
            count += 1
        except AssertionError as e:
            logger.debug(role="test_preprocessor_determinism", message=f"Failed: {e}")
            return

    logger.debug(role="test_preprocessor_determinism", message=f"Success [{count}/{len(tokens)}] tested")


def test_targets_consistency():
    """
    every target yields the same number of events on the frame grid (BlockPreprocessor) and on the native rate
    (EventPreprocessor) up to the runs merged by adjacent annotations, and the derived targets are subsets of
    their parents
    """

    token = fetch_tokens(folder=PATH_DATA, shard="all")[0]
    ev = EventPreprocessor(beh_path=PATH_DATA / Path(f"{token}_B.csv"), sig_path=PATH_DATA / Path(f"{token}_S.csv"), log_level="ERROR")

    for target in TARGETS:
        prep = BlockPreprocessor(beh_path=PATH_DATA / Path(f"{token}_B.csv"), sig_path=PATH_DATA / Path(f"{token}_S.csv"), target=target, log_level="ERROR")
        n_frame, n_event = len(prep.onsets(prep.y)), len(ev.select(target))
        assert n_frame <= n_event, (target, n_frame, n_event)
        logger.debug(role="test_targets_consistency", message=f"{token} | {target:20s} | frame-grid onsets {n_frame:4d} | native events {n_event:4d}")

    assert set(ev.select("cue_none").index) <= set(ev.select("cue").index)
    assert set(ev.select("cont_inv_near").index) <= set(ev.select("cont_inv").index)
    assert set(ev.select("cont_inv_isolated").index) <= set(ev.select("cont_inv").index)
    assert not (set(ev.select("cont_inv_near").index) & set(ev.select("cont_inv_isolated").index))

    logger.debug(role="test_targets_consistency", message="Success")


if __name__ == '__main__':
    test_preprocessor_determinism()
    test_targets_consistency()
