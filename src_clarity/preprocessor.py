import os
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass

from dotenv import load_dotenv

import numpy as np

from tools import ABlockLogger
from src_predict import Preprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


@dataclass
class Fold:
    """
    One contiguous-block fold: the frame indexes of the training / validation / testing subsets
    """
    index:  int
    train:  np.ndarray
    valid:  np.ndarray
    test:   np.ndarray

    def __repr__(self):
        return f"Fold(index={self.index}, len4train={len(self.train)}, len4valid={len(self.valid)}, len4test={len(self.test)})"


class BlockPreprocessor:
    def __init__(
        self,
        beh_path: str | os.PathLike | Path,
        sig_path: str | os.PathLike | Path,
        target: str     = "cont",
        resize: bool    = True,
        noise: bool     = False,
        seed: int       = 42,
        log_level: str  = "INFO",
    ) -> None:
        """
        The time-ordered wrapper of src_predict.Preprocessor: the frames keep their original order, nothing is
        shuffled, resampled or normalized here, so that every split / scaling decision is made later on contiguous
        blocks only

        :param beh_path:    The path to the behavior file
        :param sig_path:    The path to the signal file
        :param target:      The mode for labeling interested behaviors
        :param resize:      The toggle for resizing when doing x-y alignment
        :param noise:       The toggle for the control group: the cell signals are replaced by random walks of the
                            same shape, keeping the labels (like `skew` of rfecv_predict, but structure-preserving)
        :param seed:        The random seed of the control group
        :param log_level:   The print level for logging
        """

        self.target = target
        self.noise = noise
        self.seed = seed

        self.logger = ABlockLogger(level=log_level)

        # reuse the alignment logic of src_predict, but keep the frames in their original order
        prep = Preprocessor(
            beh_path=beh_path,
            sig_path=sig_path,
            target=target,
            tvt_ratio=(1.0, 0.0, 0.0),
            shuffle=False,
            normalizer=None,
            agg=False,
            smote=None,
            resize=resize,
            seed=seed,
            log_level="WARNING"
        )
        total = prep.get_origin_total(shard="all")

        self.time = total.index.to_numpy(dtype=np.float32)
        self.cells = total.columns[:-1].to_list()
        self.x = total.iloc[:, :-1].to_numpy(dtype=np.float32)
        self.y = total.iloc[:, -1].to_numpy().astype(int)

        # generate the control group: random walks replace the cells, the labels stay untouched
        if noise:
            rng = np.random.default_rng(seed)
            self.x = np.cumsum(rng.standard_normal(self.x.shape, dtype=np.float32), axis=0)

        self.logger.info(
            role="Clarity.BlockPreprocessor",
            message=f"Config: frames[{len(self.y)}] | cells[{len(self.cells)}] | positive[{self.y.mean():.3f}] | "
                    f"control[{'noise' if noise else 'real'}]"
        )

    @staticmethod
    def runs_of(indices: np.ndarray) -> List[np.ndarray]:
        """
        Split a sorted frame-index array into its maximal contiguous runs

        :param indices: The sorted frame indexes

        :return: The list of contiguous runs
        """

        if len(indices) == 0:
            return []

        cuts = np.flatnonzero(np.diff(indices) != 1) + 1

        return np.split(indices, cuts)

    @staticmethod
    def within_runs(indices: np.ndarray, left: int, right: int) -> np.ndarray:
        """
        Keep the frames whose window [i - left, i + right] lies entirely inside one contiguous run of the indexes

        :param indices: The sorted frame indexes of one subset
        :param left:    The number of frames the window extends to the past
        :param right:   The number of frames the window extends to the future

        :return: The frame indexes that own a complete window
        """

        kept = []
        for run in BlockPreprocessor.runs_of(indices):
            lo, hi = run[0] + left, run[-1] - right
            if hi >= lo:
                kept.append(np.arange(lo, hi + 1))

        return np.concatenate(kept) if kept else np.array([], dtype=int)

    @staticmethod
    def onsets(y: np.ndarray) -> np.ndarray:
        """
        Locate the first frame of every positive run of the labels (the event onsets)

        :param y:   The labels of the whole recording

        :return: The onset frame indexes
        """

        positives = np.flatnonzero(y == 1)

        return np.array([run[0] for run in BlockPreprocessor.runs_of(positives)], dtype=int)

    def block_folds(
        self,
        n_blocks: int       = 5,
        purge: int          = 50,
        valid_ratio: float  = 0.15,
    ) -> List[Fold]:
        """
        Cut the recording into contiguous blocks; each block is tested once, the frames within `purge` of the block
        are dropped, and a contiguous tail of the remaining frames is reserved for validation

        :param n_blocks:    The number of contiguous time blocks
        :param purge:       The number of frames dropped on both sides of the testing / validation blocks
        :param valid_ratio: The ratio of the remaining frames reserved for validation

        :return: The list of folds
        """

        frames = np.arange(len(self.y))
        blocks = np.array_split(frames, n_blocks)

        folds = []
        for k, test in enumerate(blocks):
            # every frame farther than `purge` from the testing block is eligible for training
            pool = frames[(frames < test[0] - purge) | (frames > test[-1] + purge)]

            # the validation block is the tail of the longest eligible run, purged from the frames before it
            longest = sorted(self.runs_of(pool), key=len, reverse=True)[0]
            valid_len = min(int(round(valid_ratio * len(pool))), max(len(longest) - purge - 1, 0))

            if valid_len > 0:
                valid = longest[len(longest) - valid_len:]
                train = pool[(pool < valid[0] - purge) | (pool > valid[-1])]
            else:
                valid = np.array([], dtype=int)
                train = pool

            # make sure the subsets are disjoint and the training frames keep the purge distance
            if len(np.intersect1d(train, test)) or len(np.intersect1d(train, valid)) or len(np.intersect1d(valid, test)):
                self.logger.error(role="Clarity.BlockPreprocessor", message="the fold subsets overlap")
                raise RuntimeError("the fold subsets overlap")

            folds.append(Fold(index=k, train=train, valid=valid, test=test))

        self.logger.debug(role="Clarity.BlockPreprocessor", message=f"folds: {folds}")

        return folds
