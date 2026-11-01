import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from src.util import ABlockLogger, load_paths
from src.clarity import ClarityEngine, ContrastEngine
from src.prediction import ExtraTreesEngine

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory


# global test report logger
logger = ABlockLogger(level="DEBUG")


def test_clarity_cache_reproduction(token="K227_A_D7", beh="cont"):
    """
    the clarity engine recomputes the committed cache of a token to floating-point precision
    """

    tmp = Path(tempfile.mkdtemp())
    ce = ClarityEngine(sig_folder=PATH_DATA, beh_folder=PATH_DATA, log_level="ERROR")

    new = ce.compute(token=token, beh=beh, rerun=True, tgt_folder=tmp)
    old = pd.read_csv(PATH_CACHE / Path(f"clarity/init/{beh}/{token}-{beh}_Z.csv"))

    assert new["n_events"].iloc[0] == old["n_events"].iloc[0]
    assert np.nanmax(np.abs(new["z"].to_numpy() - old["z"].to_numpy())) < 1e-5
    assert np.nanmax(np.abs(new["coherence"].to_numpy() - old["coherence"].to_numpy())) < 1e-7

    # the single-cell route gives the same score as the whole-recording route (up to float32 accumulation order)
    cell = new["cell"].iloc[0]
    row = ce.compute_cell(token=token, cell=cell, beh=beh)
    assert abs(row["z"] - new["z"].iloc[0]) < 1e-2

    logger.debug(role="test_clarity_cache_reproduction", message=f"Success [{token}-{beh}] reproduced")


def test_extratrees_cache_reproduction(token="K227_A_D7", beh="cont"):
    """
    the ExtraTrees engine recomputes the committed cache of a token exactly (the forest is seeded, the folds and
    the normalization are deterministic)
    """

    tmp = Path(tempfile.mkdtemp())
    et = ExtraTreesEngine(sig_folder=PATH_DATA, beh_folder=PATH_DATA, log_level="ERROR")

    new = et.predict(token=token, beh=beh, subsets=("all",), rerun=True, tgt_folder=tmp)
    old = pd.read_csv(PATH_CACHE / Path(f"extratrees/init/{beh}/{token}-{beh}_R.csv"))

    assert np.allclose(new["auc_fold"].to_numpy(), old["auc_fold"].to_numpy(), atol=1e-12)
    assert np.allclose(new["auc"].to_numpy(), old["auc"].to_numpy(), atol=1e-12)

    logger.debug(role="test_extratrees_cache_reproduction", message=f"Success [{token}-{beh}] reproduced")


def test_contrast_determinism(token="K227_A_D7", pair=("cont", "cont_missed")):
    """
    the contrast engine is deterministic and its overlap record is self-consistent
    """

    ct = ContrastEngine(sig_folder=PATH_DATA, beh_folder=PATH_DATA, log_level="ERROR")
    clarity_folder = lambda beh: PATH_CACHE / Path(f"clarity/init/{beh}")

    first, _ = ct.compute(token=token, pair=pair, clarity_folder=clarity_folder, tgt_folder=None)
    second, _ = ct.compute(token=token, pair=pair, clarity_folder=clarity_folder, tgt_folder=None)
    pd.testing.assert_frame_equal(first, second)

    row = first.iloc[0]
    assert row["n_both"] <= min(row["n_first"], row["n_second"])
    assert 0.0 <= row["jaccard"] <= 1.0 and 0.0 <= row["containment"] <= 1.0

    logger.debug(role="test_contrast_determinism", message=f"Success [{token}-{pair[0]}-{pair[1]}]")


if __name__ == '__main__':
    test_clarity_cache_reproduction()
    test_extratrees_cache_reproduction()
    test_contrast_determinism()
