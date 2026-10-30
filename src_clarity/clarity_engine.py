import os
import json
from pathlib import Path
from typing import Callable, List, Tuple

from dotenv import load_dotenv

import numpy as np
import pandas as pd
from scipy.stats import norm

from tools import ABlockLogger, check_folders, fetch_tokens
from .preprocessor import BlockPreprocessor


# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory


class ClarityEngine:
    @check_folders("sig_folder", "beh_folder")
    def __init__(
        self,
        sig_folder: str | os.PathLike | Path,
        beh_folder: str | os.PathLike | Path,
        seed: int       = 42,
        log_level: str  = "INFO",
    ) -> None:
        """
        The engine scoring how clearly every cell locks onto the behavior events: the event windows of a cell (the
        rows of the sps_select event heatmap) are compared with each other, and the consistency is calibrated against
        the recording's own null obtained by circularly shifting the labels

        :param sig_folder:  The folder of the signal files
        :param beh_folder:  The folder of the behavior files
        :param seed:        The random seed of the label shifts
        :param log_level:   The print level for logging
        """

        self.sig_folder = Path(sig_folder)
        self.beh_folder = Path(beh_folder)
        self.seed = seed

        self.logger = ABlockLogger(level=log_level)

    @staticmethod
    def event_windows(
        x: np.ndarray,
        y: np.ndarray,
        inner_width: Tuple[int, int]    = (-50, 50),
        frames: None | np.ndarray       = None,
    ) -> np.ndarray:
        """
        Stack, per cell, the signal windows around every event onset

        :param x:           The features of the whole recording
        :param y:           The labels of the whole recording
        :param inner_width: The window around the onset in frames
        :param frames:      The frame subset the windows must lie in entirely (None for the whole recording), so that
                            a training-only ranking never sees a frame outside the training block

        :return: The stacked windows shaped (cells, events, width)
        """

        allowed = np.zeros(len(y), dtype=bool)
        allowed[np.arange(len(y)) if frames is None else frames] = True

        offsets = np.arange(inner_width[0], inner_width[1] + 1)

        # keep the onsets whose whole window lies inside the allowed frames
        kept = []
        for start in BlockPreprocessor.onsets(y):
            span = start + offsets
            if span[0] >= 0 and span[-1] < len(y) and allowed[span].all():
                kept.append(start)
        kept = np.array(kept, dtype=int)

        if len(kept) == 0:
            return np.zeros((x.shape[1], 0, len(offsets)), dtype=np.float32)

        return np.transpose(x[kept[:, None] + offsets[None, :]], (2, 0, 1))

    @staticmethod
    def coherence(windows: np.ndarray) -> np.ndarray:
        """
        The event-to-event consistency of every cell: the mean pairwise Pearson correlation between its event windows

        :param windows: The stacked windows shaped (cells, events, width)

        :return: The per-cell coherence (nan when fewer than 3 events)
        """

        n_cells, n_events, _ = windows.shape
        if n_events < 3:
            return np.full(n_cells, np.nan)

        centred = windows - windows.mean(axis=2, keepdims=True)
        centred = centred / (np.linalg.norm(centred, axis=2, keepdims=True) + 1e-6)
        gram = np.einsum("cet,cft->cef", centred, centred)

        return (gram.sum(axis=(1, 2)) - np.trace(gram, axis1=1, axis2=2)) / (n_events * (n_events - 1))

    @staticmethod
    def clarity(
        x: np.ndarray,
        y: np.ndarray,
        cells: List[str],
        inner_width: Tuple[int, int]    = (-50, 50),
        frames: None | np.ndarray       = None,
        n_shift: int                    = 1000,
        seed: int                       = 42,
    ) -> pd.DataFrame:
        """
        Score the clarity of every cell: the observed coherence, its z-score against `n_shift` circular shifts of the
        labels (restricted to the allowed frames) and the one-sided Gaussian p-value of that z-score

        :param x:           The normalized features of the whole recording
        :param y:           The labels of the whole recording
        :param cells:       The cell names
        :param inner_width: The window around the onset in frames
        :param frames:      The frame subset the windows must lie in entirely, None for the whole recording
        :param n_shift:     The number of circular label shifts
        :param seed:        The random seed of the shifts

        :return: The per-cell table with columns `cell`, `n_events`, `coherence`, `z`, `p`
        """

        windows = ClarityEngine.event_windows(x=x, y=y, inner_width=inner_width, frames=frames)
        observed = ClarityEngine.coherence(windows)

        # the labels outside the allowed frames are blanked, so that the shifted onsets only ever come from them
        y_allowed = y.copy()
        if frames is not None:
            mask = np.zeros(len(y), dtype=bool)
            mask[frames] = True
            y_allowed[~mask] = 0

        rng = np.random.default_rng(seed)
        margin = max(int(0.1 * len(y)), inner_width[1] - inner_width[0] + 1)

        nulls = []
        for _ in range(n_shift):
            shifted = ClarityEngine.event_windows(
                x=x, y=np.roll(y_allowed, int(rng.integers(margin, len(y) - margin))), inner_width=inner_width, frames=frames
            )
            if shifted.shape[1] >= 3:
                nulls.append(ClarityEngine.coherence(shifted))

        # the calibration needs enough events and enough valid shifts
        if windows.shape[1] < 3 or len(nulls) < 10:
            z = np.full(len(cells), np.nan)
        else:
            nulls = np.stack(nulls, axis=0)
            z = (observed - nulls.mean(axis=0)) / (nulls.std(axis=0) + 1e-9)

        return pd.DataFrame({
            "cell": cells,
            "n_events": windows.shape[1],
            "coherence": observed,
            "z": z,
            "p": norm.sf(z),
        })

    @staticmethod
    def rank(
        x: np.ndarray,
        y: np.ndarray,
        frames: np.ndarray,
        inner_width: Tuple[int, int]    = (-50, 50),
        n_shift: int                    = 1000,
        seed: int                       = 42,
    ) -> np.ndarray:
        """
        Rank the cells by their training-only clarity (clearest first), for the subset experiments of the predictors

        :param x:           The normalized features of the whole recording
        :param y:           The labels of the whole recording
        :param frames:      The training frames the ranking may use
        :param inner_width: The window around the onset in frames
        :param n_shift:     The number of circular label shifts
        :param seed:        The random seed of the shifts

        :return: The cell indexes sorted from the clearest to the least clear
        """

        table = ClarityEngine.clarity(x=x, y=y, cells=list(map(str, range(x.shape[1]))), inner_width=inner_width, frames=frames, n_shift=n_shift, seed=seed)

        return np.argsort(-table["z"].fillna(-np.inf).to_numpy(), kind="stable")

    @check_folders("tgt_folder")
    def compute(
        self,
        token: str,
        beh: str                                    = "cont",
        inner_width: Tuple[int, int]                = (-50, 50),
        n_shift: int                                = 1000,
        rerun: bool                                 = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> pd.DataFrame:
        """
        Score the clarity of every cell of a token on the whole recording and cache the table

        :param token:       The token of the subject
        :param beh:         The interested behavior
        :param inner_width: The window around the onset in frames
        :param n_shift:     The number of circular label shifts
        :param rerun:       The toggle for re-running even if the cached table exists
        :param tgt_folder:  The folder storing the cached tables

        :return: The per-cell clarity table
        """

        config = {
            "token": token,
            "beh": beh,
            "inner_width": list(inner_width),
            "n_shift": n_shift,
            "seed": self.seed,
        }

        # fetch the cached table if the configuration is unchanged
        try:
            if rerun:
                raise Exception("rerun hack")

            with open(Path(tgt_folder) / Path(f"{token}-{beh}_C.json"), 'r') as fp:
                config_ref = json.load(fp)

            if config_ref != config:
                raise Exception("rerun hack")

            table = pd.read_csv(Path(tgt_folder) / Path(f"{token}-{beh}_Z.csv"))

        # compute the table if not exists
        except Exception as e:
            prep = BlockPreprocessor(
                beh_path=self.beh_folder / Path(f"{token}_B.csv"),
                sig_path=self.sig_folder / Path(f"{token}_S.csv"),
                target=beh,
                seed=self.seed,
                log_level="WARNING"
            )
            x = ((prep.x - prep.x.mean(axis=0)) / (prep.x.std(axis=0) + 1e-6)).astype(np.float32)

            table = self.clarity(x=x, y=prep.y, cells=prep.cells, inner_width=inner_width, n_shift=n_shift, seed=self.seed)
            table.insert(0, "token", token)

            # save the table & config into cache
            table.to_csv(Path(tgt_folder) / Path(f"{token}-{beh}_Z.csv"), index=False)
            with open(Path(tgt_folder) / Path(f"{token}-{beh}_C.json"), 'w') as fp:
                json.dump(config, fp, indent=4)

            self.logger.info(
                role="ClarityEngine",
                message=f"{token}-{beh} | events {int(table['n_events'].iloc[0])} | "
                        f"significant cells {int((table['z'] >= 2.0).sum())}/{len(table)}"
            )

        else:
            self.logger.info(role="ClarityEngine", message=f"{token}-{beh} clarity already computed")

        return table

    @staticmethod
    def describe(
        table: pd.DataFrame,
        threshold: float = 2.0,
    ) -> float:
        """
        Compress the per-cell table of one token into its clarity descriptor: the fraction of significant cells

        :param table:       The per-cell clarity table
        :param threshold:   The minimum z-score of a significant cell

        :return: The fraction of cells whose z-score reaches the threshold
        """

        return float((table["z"] >= threshold).mean())

    @check_folders("tgt_folder")
    def collect(
        self,
        beh: str                                    = "cont",
        shard: str                                  = "all",
        on_compute: Callable                        = lambda x: True,
        inner_width: Tuple[int, int]                = (-50, 50),
        n_shift: int                                = 1000,
        threshold: float                            = 2.0,
        rerun: bool                                 = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> pd.DataFrame:
        """
        Compute (or fetch) the clarity of every token of the signal folder and summarize it per token

        :param beh:         The interested behavior
        :param shard:       Which shard the folder belongs to
        :param on_compute:  The filter deciding which tokens are computed
        :param inner_width: The window around the onset in frames
        :param n_shift:     The number of circular label shifts
        :param threshold:   The minimum z-score of a significant cell
        :param rerun:       The toggle for re-running even if the cached tables exist
        :param tgt_folder:  The folder storing the cached tables

        :return: The per-token table with columns `token`, `n_cells`, `n_events`, `fraction`
        """

        records = []
        for token in filter(on_compute, fetch_tokens(folder=self.sig_folder, shard=shard)):
            table = self.compute(token=token, beh=beh, inner_width=inner_width, n_shift=n_shift, rerun=rerun, tgt_folder=tgt_folder)

            records.append({
                "token": token,
                "n_cells": len(table),
                "n_events": int(table["n_events"].iloc[0]),
                "fraction": self.describe(table=table, threshold=threshold),
            })

        return pd.DataFrame(records)
