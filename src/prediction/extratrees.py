import os
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.ensemble import ExtraTreesClassifier

from ..util import ABlockLogger, check_folders, fetch_tokens, load_paths, parallel_map, share_threads
from ..preprocess import BlockPreprocessor, Fold
from ..clarity import ClarityEngine


# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory


class ExtraTreesEngine:
    @check_folders("sig_folder", "beh_folder")
    def __init__(
        self,
        sig_folder: str | os.PathLike | Path,
        beh_folder: str | os.PathLike | Path,
        seed: int       = 42,
        n_jobs: int     = -1,
        log_level: str  = "INFO",
    ) -> None:
        """
        The leak-free ExtraTrees predictor: contiguous-block folds with purge gaps, normalization fitted on the
        training block only, a causal window of past frames as features, and the within-block AUC as the score;
        the forest runs on CPU threads (scikit-learn has no GPU path for trees), the tokens on worker processes

        :param sig_folder:  The folder of the signal files
        :param beh_folder:  The folder of the behavior files
        :param seed:        The random seed of the model, the folds and the random cell subsets
        :param n_jobs:      The number of CPU threads of the forest
        :param log_level:   The print level for logging
        """

        self.sig_folder = Path(sig_folder)
        self.beh_folder = Path(beh_folder)
        self.seed = seed
        self.n_jobs = n_jobs
        self.log_level = log_level

        self.logger = ABlockLogger(level=log_level)

    def __getstate__(self) -> Dict:
        # the logger owns a worker thread, so it is rebuilt on the receiving process instead of being pickled
        return {key: value for key, value in self.__dict__.items() if key != "logger"}

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        self.logger = ABlockLogger(level=self.log_level)

    @staticmethod
    def _offsets(
        window: int,
        causal: bool
    ) -> Tuple[int, int, np.ndarray]:
        """
        Describe the window geometry: a causal window ends at the labelled frame, a centered one straddles it

        :param window:  The number of frames in the window
        :param causal:  The toggle for the causal window

        :return: The (left, right, offsets) triple, offsets being the frame shifts relative to the labelled frame
        """

        left = window - 1 if causal else window // 2
        right = 0 if causal else window - 1 - window // 2

        return left, right, np.arange(-left, right + 1)

    def _features(
        self,
        x: np.ndarray,
        frames: np.ndarray,
        window: int,
        causal: bool,
    ) -> np.ndarray:
        """
        Flatten the sub-sampled window of every frame into a feature row (one frame every `window // 10` frames)

        :param x:       The normalized features of the whole recording
        :param frames:  The frames owning a complete window
        :param window:  The number of frames in the window
        :param causal:  The toggle for the causal window

        :return: The (frames, positions * cells) feature matrix
        """

        left, right, offsets = self._offsets(window=window, causal=causal)

        stride = max(1, window // 10)
        picks = np.unique(np.concatenate([np.arange(0, -left - 1, -stride), np.arange(0, right + 1, stride)]))
        positions = np.searchsorted(offsets, picks)

        return x[frames[:, None] + offsets[None, :]][:, positions, :].reshape(len(frames), -1)

    def _fit(
        self,
        x: np.ndarray,
        y: np.ndarray,
        fold: Fold,
        window: int,
        causal: bool,
        n_jobs: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit the forest on the training frames of the fold and score the testing frames

        :param x:       The normalized features of the whole recording (the selected cells only)
        :param y:       The labels of the whole recording
        :param fold:    The fold describing the training / validation / testing frames
        :param window:  The number of frames in the window
        :param causal:  The toggle for the causal window
        :param n_jobs:  The number of CPU threads of the forest

        :return: The (testing frames, testing probabilities) pair
        """

        left, right, _ = self._offsets(window=window, causal=causal)
        tr = BlockPreprocessor.within_runs(fold.train, left, right)
        te = BlockPreprocessor.within_runs(fold.test, left, right)

        model = ExtraTreesClassifier(
            n_estimators=400,
            min_samples_leaf=5,
            max_features="sqrt",
            class_weight="balanced_subsample",
            n_jobs=n_jobs,
            random_state=self.seed,
        )
        model.fit(self._features(x=x, frames=tr, window=window, causal=causal), y[tr])

        return te, model.predict_proba(self._features(x=x, frames=te, window=window, causal=causal))[:, 1]

    @check_folders("tgt_folder")
    def predict(
        self,
        token: str,
        beh: str                                    = "cont",
        subsets: Tuple[str, ...]                    = ("all",),
        fraction: float                             = 0.3,
        noise: bool                                 = False,
        n_blocks: int                               = 5,
        purge: int                                  = 50,
        valid_ratio: float                          = 0.15,
        window: int                                 = 50,
        causal: bool                                = True,
        inner_width: Tuple[int, int]                = (-50, 50),
        n_shift: int                                = 1000,
        n_jobs: None | int                          = None,
        rerun: bool                                 = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> pd.DataFrame:
        """
        Score the forest on cell subsets of a token over contiguous blocks: "all" cells, the "top" / "bottom"
        `fraction` of the cells by the training-only clarity of every fold, and a size-matched "random" subset

        :param token:       The token of the subject
        :param beh:         The interested behavior (the target)
        :param subsets:     The cell subsets to score ("all", "top", "bottom", "random")
        :param fraction:    The fraction of the cells kept in the "top" / "bottom" / "random" subsets
        :param noise:       The toggle for the random-walk control group
        :param n_blocks:    The number of contiguous time blocks
        :param purge:       The number of frames dropped on both sides of the testing / validation blocks
        :param valid_ratio: The ratio of the remaining frames reserved for validation
        :param window:      The number of frames in the window
        :param causal:      The toggle for the causal window
        :param inner_width: The window around the onset used by the clarity ranking
        :param n_shift:     The number of circular label shifts used by the clarity ranking
        :param n_jobs:      The number of CPU threads of the forest (None for the engine default)
        :param rerun:       The toggle for re-running even if the cached metrics exist
        :param tgt_folder:  The folder storing the cached metrics (None computes without caching)

        :return: The metrics table with one row per subset: the within-block AUC (`auc_fold`, averaged over the
                 blocks holding both classes) and the pooled AUC (`auc`)
        """

        config = {
            "token": token,
            "beh": beh,
            "subsets": list(subsets),
            "fraction": fraction,
            "noise": noise,
            "n_blocks": n_blocks,
            "purge": purge,
            "valid_ratio": valid_ratio,
            "window": window,
            "causal": causal,
            "inner_width": list(inner_width),
            "n_shift": n_shift,
            "seed": self.seed,
        }
        stem = f"{token}-{beh}{'_noise' if noise else ''}"
        n_jobs = self.n_jobs if n_jobs is None else n_jobs

        # fetch the cached metrics if the configuration is unchanged
        try:
            if rerun or tgt_folder is None:
                raise Exception("rerun hack")

            with open(Path(tgt_folder) / Path(f"{stem}_C.json"), 'r') as fp:
                config_ref = json.load(fp)

            if config_ref != config:
                raise Exception("rerun hack")

            metrics = pd.read_csv(Path(tgt_folder) / Path(f"{stem}_R.csv"))

        # compute the metrics if not exists
        except Exception as e:
            prep = BlockPreprocessor(
                beh_path=self.beh_folder / Path(f"{token}_B.csv"),
                sig_path=self.sig_folder / Path(f"{token}_S.csv"),
                target=beh,
                noise=noise,
                seed=self.seed,
                log_level="WARNING"
            )
            folds = prep.block_folds(n_blocks=n_blocks, purge=purge, valid_ratio=valid_ratio)
            n_keep = max(int(round(fraction * len(prep.cells))), 2)

            # the per-fold normalized features and cell subsets (the ranking only sees the training frames)
            prepared = []
            for fold in folds:
                x = prep.normalized(frames=fold.train)

                lookup: Dict[str, np.ndarray] = {"all": np.arange(len(prep.cells))}
                if any(subset != "all" for subset in subsets):
                    order = ClarityEngine.rank(x=x, y=prep.y, frames=fold.train, inner_width=inner_width, n_shift=n_shift, seed=self.seed + fold.index)
                    rng = np.random.default_rng(self.seed + fold.index)
                    lookup["top"] = np.sort(order[:n_keep])
                    lookup["bottom"] = np.sort(order[-n_keep:])
                    lookup["random"] = np.sort(rng.choice(len(prep.cells), size=n_keep, replace=False))

                prepared.append((fold, x, lookup))

            records = []
            for subset in subsets:
                frames, probs, aucs = [], [], []
                for fold, x, lookup in prepared:
                    te, p_te = self._fit(x=x[:, lookup[subset]], y=prep.y, fold=fold, window=window, causal=causal, n_jobs=n_jobs)

                    frames.append(te)
                    probs.append(p_te)

                    # the within-block score exists only when the testing block holds both classes
                    if prep.y[te].min() != prep.y[te].max():
                        aucs.append(roc_auc_score(prep.y[te], p_te))

                frames = np.concatenate(frames)
                probs = np.concatenate(probs)
                label = prep.y[frames]

                record = {
                    "token": token,
                    "beh": beh,
                    "control": "noise" if noise else "real",
                    "subset": subset,
                    "n_cells": len(prepared[0][2][subset]),
                    "n_events": len(BlockPreprocessor.onsets(prep.y)),
                    "n_folds": len(aucs),
                    "auc_fold": float(np.mean(aucs)) if aucs else np.nan,
                    "auc": roc_auc_score(label, probs) if label.min() != label.max() else np.nan,
                }
                records.append(record)

                self.logger.info(
                    role="ExtraTreesEngine",
                    message=f"{token}-{beh} | {record['control']:5s} | {subset:6s} | cells {record['n_cells']:3d} | "
                            f"auc_fold {record['auc_fold']:.3f} ({record['n_folds']}) | auc {record['auc']:.3f}"
                )

            # save the metrics & config into cache
            metrics = pd.DataFrame(records)
            if tgt_folder is not None:
                metrics.to_csv(Path(tgt_folder) / Path(f"{stem}_R.csv"), index=False)
                with open(Path(tgt_folder) / Path(f"{stem}_C.json"), 'w') as fp:
                    json.dump(config, fp, indent=4)

        else:
            self.logger.info(role="ExtraTreesEngine", message=f"{stem} metrics already computed")

        return metrics

    @check_folders("tgt_folder")
    def collect(
        self,
        beh: str                                    = "cont",
        shard: str                                  = "all",
        token_filter: Callable[[str], bool]         = lambda x: True,
        subsets: Tuple[str, ...]                    = ("all",),
        fraction: float                             = 0.3,
        noise: bool                                 = False,
        n_blocks: int                               = 5,
        purge: int                                  = 50,
        valid_ratio: float                          = 0.15,
        window: int                                 = 50,
        causal: bool                                = True,
        inner_width: Tuple[int, int]                = (-50, 50),
        n_shift: int                                = 1000,
        n_workers: int                              = 1,
        rerun: bool                                 = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> pd.DataFrame:
        """
        Score (or fetch) every token of the signal folder and stack the metrics; with several workers the tokens
        run on separate processes and the forest threads are shared out among them

        :param beh:             The interested behavior (the target)
        :param shard:           Which shard of the tokens to process
        :param token_filter:    The filter deciding which tokens are scored
        :param subsets:         The cell subsets to score ("all", "top", "bottom", "random")
        :param fraction:        The fraction of the cells kept in the "top" / "bottom" / "random" subsets
        :param noise:           The toggle for the random-walk control group
        :param n_blocks:        The number of contiguous time blocks
        :param purge:           The number of frames dropped on both sides of the testing / validation blocks
        :param valid_ratio:     The ratio of the remaining frames reserved for validation
        :param window:          The number of frames in the window
        :param causal:          The toggle for the causal window
        :param inner_width:     The window around the onset used by the clarity ranking
        :param n_shift:         The number of circular label shifts used by the clarity ranking
        :param n_workers:       The number of worker processes over the tokens
        :param rerun:           The toggle for re-running even if the cached metrics exist
        :param tgt_folder:      The folder storing the cached metrics

        :return: The stacked metrics table
        """

        n_jobs = self.n_jobs if n_workers <= 1 else share_threads(n_workers)

        def process(token: str) -> pd.DataFrame:
            return self.predict(
                token=token,
                beh=beh,
                subsets=subsets,
                fraction=fraction,
                noise=noise,
                n_blocks=n_blocks,
                purge=purge,
                valid_ratio=valid_ratio,
                window=window,
                causal=causal,
                inner_width=inner_width,
                n_shift=n_shift,
                n_jobs=n_jobs,
                rerun=rerun,
                tgt_folder=tgt_folder,
            )

        tokens = list(filter(token_filter, fetch_tokens(folder=self.sig_folder, shard=shard)))
        tables = [table for table in parallel_map(process, tokens, n_workers=n_workers, role="ExtraTreesEngine", log_level=self.log_level) if table is not None]

        return pd.concat(tables, axis=0, ignore_index=True) if tables else pd.DataFrame()
