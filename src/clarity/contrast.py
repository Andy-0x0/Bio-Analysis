import os
import json
from pathlib import Path
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from ..util import ABlockLogger, check_folders, fetch_tokens, load_paths, parallel_map
from ..preprocess import EventPreprocessor
from .engine import ClarityEngine


# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory


class ContrastEngine:
    @check_folders("sig_folder", "beh_folder")
    def __init__(
        self,
        sig_folder: str | os.PathLike | Path,
        beh_folder: str | os.PathLike | Path,
        seed: int       = 42,
        log_level: str  = "INFO",
    ) -> None:
        """
        The engine contrasting two targets of the same recording (e.g. successful vs missed reaches): do the same
        cells lock onto both (overlap of the significant sets), and do the cells respond the same way (correlation
        of the per-cell response profiles across cells, against a relabelling null), plus the population time
        courses of the reference cells around both kinds of events

        :param sig_folder:  The folder of the signal files
        :param beh_folder:  The folder of the behavior files
        :param seed:        The random seed of the relabelling null
        :param log_level:   The print level for logging
        """

        self.sig_folder = Path(sig_folder)
        self.beh_folder = Path(beh_folder)
        self.seed = seed
        self.log_level = log_level

        self.logger = ABlockLogger(level=log_level)
        self.clarity_engine = ClarityEngine(sig_folder=sig_folder, beh_folder=beh_folder, seed=seed, log_level="WARNING")

    def __getstate__(self) -> Dict:
        # the logger owns a worker thread, so it is rebuilt on the receiving process instead of being pickled
        return {key: value for key, value in self.__dict__.items() if key != "logger"}

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        self.logger = ABlockLogger(level=self.log_level)

    @staticmethod
    def overlap(
        first: pd.DataFrame,
        second: pd.DataFrame,
        threshold: float = 2.0,
    ) -> Dict[str, float]:
        """
        Compare the significant cells of two clarity tables of the same recording

        :param first:       The per-cell clarity table of the first target
        :param second:      The per-cell clarity table of the second target
        :param threshold:   The minimum z-score of a significant cell

        :return: The overlap record: the set sizes, the Jaccard index, the containment (shared / smaller set), both
                 as expected if the two sets were drawn independently (the chance containment is the larger set over
                 all cells), and the Spearman correlation of the two z-scores across cells
        """

        set_first = ClarityEngine.significant(first, threshold=threshold)
        set_second = ClarityEngine.significant(second, threshold=threshold)
        n_cells, n_first, n_second, n_both = len(first), len(set_first), len(set_second), len(set_first & set_second)

        # the chance level: the expected intersection of two independent draws of the same sizes
        expected = n_first * n_second / n_cells if n_cells else np.nan
        union = n_first + n_second - n_both

        z_first, z_second = first["z"].to_numpy(), second["z"].to_numpy()
        valid = np.isfinite(z_first) & np.isfinite(z_second)

        return {
            "n_cells": n_cells,
            "n_first": n_first,
            "n_second": n_second,
            "n_both": n_both,
            "jaccard": n_both / union if union else np.nan,
            "containment": n_both / min(n_first, n_second) if min(n_first, n_second) else np.nan,
            "jaccard_chance": expected / (n_first + n_second - expected) if (n_first + n_second - expected) > 0 else np.nan,
            "containment_chance": max(n_first, n_second) / n_cells if n_cells and min(n_first, n_second) else np.nan,
            "rho": spearmanr(z_first[valid], z_second[valid])[0] if valid.sum() >= 3 else np.nan,
        }

    @staticmethod
    def profile(
        windows: np.ndarray,
        offsets: np.ndarray,
        pre: Tuple[float, float]    = (-2.0, -1.0),
        post: Tuple[float, float]   = (0.0, 1.0),
    ) -> np.ndarray:
        """
        The per-cell response to one kind of event: the window mean over the post offsets minus the mean over the
        pre offsets, averaged over the events

        :param windows: The windows shaped (events, cells, offsets)
        :param offsets: The offsets of the windows in seconds
        :param pre:     The inclusive offset range of the baseline
        :param post:    The inclusive offset range of the response

        :return: The response of every cell
        """

        pre_mask = (offsets >= pre[0]) & (offsets <= pre[1])
        post_mask = (offsets >= post[0]) & (offsets <= post[1])

        return (windows[:, :, post_mask].mean(axis=2) - windows[:, :, pre_mask].mean(axis=2)).mean(axis=0)

    @staticmethod
    def similarity(
        windows_first: np.ndarray,
        windows_second: np.ndarray,
        offsets: np.ndarray,
        pre: Tuple[float, float]    = (-2.0, -1.0),
        post: Tuple[float, float]   = (0.0, 1.0),
        n_perm: int                 = 300,
        seed: int                   = 42,
    ) -> Dict[str, float]:
        """
        Compare the per-cell response profiles of two kinds of events: Pearson correlation across cells, the
        regression slope, and the z-score of the correlation against the null where the events of the two kinds are
        relabelled at random (the correlation two samples of the same kind would show)

        :param windows_first:   The windows of the first kind shaped (events, cells, offsets)
        :param windows_second:  The windows of the second kind shaped (events, cells, offsets)
        :param offsets:         The offsets of the windows in seconds
        :param pre:             The inclusive offset range of the baseline
        :param post:            The inclusive offset range of the response
        :param n_perm:          The number of relabellings of the null
        :param seed:            The random seed of the relabellings

        :return: The similarity record (nan when either kind has fewer than 3 events)
        """

        n_first, n_second = len(windows_first), len(windows_second)
        if n_first < 3 or n_second < 3:
            return {"n_events_first": n_first, "n_events_second": n_second, "r": np.nan, "slope": np.nan, "r_null": np.nan, "z_r": np.nan, "amp_first": np.nan, "amp_second": np.nan}

        p_first = ContrastEngine.profile(windows_first, offsets, pre=pre, post=post)
        p_second = ContrastEngine.profile(windows_second, offsets, pre=pre, post=post)

        # the null: relabel the pooled events at random, keeping the two counts
        pooled = np.concatenate([windows_first, windows_second], axis=0)
        rng = np.random.default_rng(seed)
        nulls = []
        for _ in range(n_perm):
            order = rng.permutation(len(pooled))
            nulls.append(np.corrcoef(
                ContrastEngine.profile(pooled[order[:n_first]], offsets, pre=pre, post=post),
                ContrastEngine.profile(pooled[order[n_first:]], offsets, pre=pre, post=post),
            )[0, 1])
        nulls = np.array(nulls)

        r = np.corrcoef(p_first, p_second)[0, 1]

        return {
            "n_events_first": n_first,
            "n_events_second": n_second,
            "r": r,
            "slope": np.polyfit(p_first, p_second, 1)[0],
            "r_null": nulls.mean(),
            "z_r": (r - nulls.mean()) / (nulls.std() + 1e-9),
            "amp_first": np.abs(p_first).mean(),
            "amp_second": np.abs(p_second).mean(),
        }

    @check_folders("tgt_folder")
    def compute(
        self,
        token: str,
        pair: Tuple[str, str]                       = ("cont", "cont_missed"),
        clarity_folder: Callable[[str], Path]       = lambda beh: None,
        threshold: float                            = 2.0,
        inner_width: Tuple[int, int]                = (-50, 50),
        n_shift: int                                = 1000,
        offsets: np.ndarray                         = np.arange(-5.0, 5.0 + 1e-6, 1.0),
        pre: Tuple[float, float]                    = (-2.0, -1.0),
        post: Tuple[float, float]                   = (0.0, 1.0),
        n_perm: int                                 = 300,
        curve_step: float                           = 0.25,
        rerun: bool                                 = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Contrast two targets on one token and cache the result: the overlap of the significant cells, the similarity
        of the per-cell response profiles, and the population time courses of the first target's significant cells
        around both kinds of events

        :param token:           The token of the subject
        :param pair:            The (first, second) targets, the first one providing the reference cells
        :param clarity_folder:  The function mapping a target to the folder caching its clarity tables
        :param threshold:       The minimum z-score of a significant cell
        :param inner_width:     The window of the clarity computation in frames
        :param n_shift:         The number of circular label shifts of the clarity computation
        :param offsets:         The native-rate offsets of the event windows in seconds
        :param pre:             The inclusive offset range of the baseline
        :param post:            The inclusive offset range of the response
        :param n_perm:          The number of relabellings of the similarity null
        :param curve_step:      The sampling step of the population time courses in seconds (akima-interpolated
                                between the native samples; the statistics keep the `offsets` sampling)
        :param rerun:           The toggle for re-running even if the cached result exists
        :param tgt_folder:      The folder storing the cached results (None computes without caching)

        :return: The (record, curves) pair: the one-row record table and the time-course table indexed by offset
                 with one column per target
        """

        first, second = pair
        config = {
            "token": token,
            "pair": list(pair),
            "threshold": threshold,
            "inner_width": list(inner_width),
            "n_shift": n_shift,
            "offsets": [float(o) for o in offsets],
            "pre": list(pre),
            "post": list(post),
            "n_perm": n_perm,
            "curve_step": curve_step,
            "seed": self.seed,
        }
        stem = f"{token}-{first}-{second}"

        # fetch the cached result if the configuration is unchanged
        try:
            if rerun or tgt_folder is None:
                raise Exception("rerun hack")

            with open(Path(tgt_folder) / Path(f"{stem}_C.json"), 'r') as fp:
                config_ref = json.load(fp)

            if config_ref != config:
                raise Exception("rerun hack")

            record = pd.read_csv(Path(tgt_folder) / Path(f"{stem}_R.csv"))
            curves = pd.read_csv(Path(tgt_folder) / Path(f"{stem}_T.csv"), index_col=0)

        # compute the result if not exists
        except Exception as e:
            tables = {
                beh: self.clarity_engine.compute(token=token, beh=beh, inner_width=inner_width, n_shift=n_shift, tgt_folder=clarity_folder(beh))
                for beh in pair
            }

            prep = EventPreprocessor(
                beh_path=self.beh_folder / Path(f"{token}_B.csv"),
                sig_path=self.sig_folder / Path(f"{token}_S.csv"),
                log_level="WARNING"
            )
            windows = {beh: prep.windows(target=beh, offsets=offsets)[1] for beh in pair}

            record = {"token": token, "first": first, "second": second}
            record.update(self.overlap(first=tables[first], second=tables[second], threshold=threshold))
            record.update(self.similarity(windows_first=windows[first], windows_second=windows[second], offsets=offsets, pre=pre, post=post, n_perm=n_perm, seed=self.seed))
            record = pd.DataFrame([record])

            # the population time course of the reference cells (significant for the first target) per kind, on a
            # dense akima-interpolated grid so that the curves are as smooth as the frame-level signal
            reference = [prep.cells.index(cell) for cell in ClarityEngine.significant(tables[first], threshold=threshold) if cell in prep.cells]
            dense = np.round(np.arange(offsets[0], offsets[-1] + 1e-6, curve_step), 6)
            smooth = {beh: prep.windows(target=beh, offsets=dense, method="akima")[1] for beh in pair}
            curves = pd.DataFrame(
                {beh: smooth[beh][:, reference, :].mean(axis=(0, 1)) if len(reference) and len(smooth[beh]) else np.full(len(dense), np.nan) for beh in pair},
                index=pd.Index(dense, name="offset"),
            )

            # save the result & config into cache
            if tgt_folder is not None:
                record.to_csv(Path(tgt_folder) / Path(f"{stem}_R.csv"), index=False)
                curves.to_csv(Path(tgt_folder) / Path(f"{stem}_T.csv"))
                with open(Path(tgt_folder) / Path(f"{stem}_C.json"), 'w') as fp:
                    json.dump(config, fp, indent=4)

            row = record.iloc[0]
            self.logger.info(
                role="ContrastEngine",
                message=f"{stem} | cells {int(row['n_first'])} / {int(row['n_second'])} shared {int(row['n_both'])} | "
                        f"containment {row['containment']:.2f} (chance {row['containment_chance']:.2f}) | jaccard {row['jaccard']:.2f} | rho {row['rho']:.2f} | "
                        f"events {int(row['n_events_first'])} / {int(row['n_events_second'])} | r {row['r']:.2f} (z {row['z_r']:+.1f})"
            )

        else:
            self.logger.info(role="ContrastEngine", message=f"{stem} contrast already computed")

        return record, curves

    @check_folders("tgt_folder")
    def collect(
        self,
        pair: Tuple[str, str]                       = ("cont", "cont_missed"),
        shard: str                                  = "all",
        token_filter: Callable[[str], bool]         = lambda x: True,
        clarity_folder: Callable[[str], Path]       = lambda beh: None,
        threshold: float                            = 2.0,
        inner_width: Tuple[int, int]                = (-50, 50),
        n_shift: int                                = 1000,
        offsets: np.ndarray                         = np.arange(-5.0, 5.0 + 1e-6, 1.0),
        pre: Tuple[float, float]                    = (-2.0, -1.0),
        post: Tuple[float, float]                   = (0.0, 1.0),
        n_perm: int                                 = 300,
        curve_step: float                           = 0.25,
        n_workers: int                              = 1,
        rerun: bool                                 = False,
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Contrast two targets on every token of the signal folder and stack the results

        :param pair:            The (first, second) targets, the first one providing the reference cells
        :param shard:           Which shard of the tokens to process
        :param token_filter:    The filter deciding which tokens are processed
        :param clarity_folder:  The function mapping a target to the folder caching its clarity tables
        :param threshold:       The minimum z-score of a significant cell
        :param inner_width:     The window of the clarity computation in frames
        :param n_shift:         The number of circular label shifts of the clarity computation
        :param offsets:         The native-rate offsets of the event windows in seconds
        :param pre:             The inclusive offset range of the baseline
        :param post:            The inclusive offset range of the response
        :param n_perm:          The number of relabellings of the similarity null
        :param curve_step:      The sampling step of the population time courses in seconds (akima-interpolated
                                between the native samples; the statistics keep the `offsets` sampling)
        :param n_workers:       The number of worker processes over the tokens
        :param rerun:           The toggle for re-running even if the cached results exist
        :param tgt_folder:      The folder storing the cached results

        :return: The (records, curves) pair: the stacked one-row records, and the stacked time courses in long format
                 with columns `token`, `first`, `second`, `beh`, `offset`, `value`
        """

        def process(token: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
            record, curves = self.compute(
                token=token, pair=pair, clarity_folder=clarity_folder, threshold=threshold, inner_width=inner_width, n_shift=n_shift,
                offsets=offsets, pre=pre, post=post, n_perm=n_perm, curve_step=curve_step, rerun=rerun, tgt_folder=tgt_folder,
            )
            long = curves.reset_index().melt(id_vars="offset", var_name="beh", value_name="value")
            long.insert(0, "second", pair[1])
            long.insert(0, "first", pair[0])
            long.insert(0, "token", token)

            return record, long

        tokens = list(filter(token_filter, fetch_tokens(folder=self.sig_folder, shard=shard)))
        results = [result for result in parallel_map(process, tokens, n_workers=n_workers, role="ContrastEngine", log_level=self.log_level) if result is not None]

        records = pd.concat([record for record, _ in results if len(record)], axis=0, ignore_index=True) if results else pd.DataFrame()
        curves = pd.concat([long.dropna(subset=["value"]) for _, long in results if len(long)], axis=0, ignore_index=True) if results else pd.DataFrame()

        return records, curves
