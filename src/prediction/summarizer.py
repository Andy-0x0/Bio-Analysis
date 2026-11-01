import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import spearmanr, wilcoxon

from ..util import ABlockLogger, check_folders, load_paths

matplotlib.use("Agg")

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory


class PredictionSummarizer:
    def __init__(
        self,
        log_level: str = "INFO",
    ) -> None:
        """
        The plotting helper of the predictors: the tokens ranked by AUC against their clarity fraction, and the AUC
        of the clarity-ranked cell subsets

        :param log_level:   The print level for logging
        """

        self.logger = ABlockLogger(level=log_level)

        # the dark tone draws the AUC, the light tone of the same hue draws the clarity fraction
        self.color_lookup = {
            "init":     {"dark": "#4e7abd", "light": "#9dbbe0"},
            "memo":     {"dark": "#c43639", "light": "#e39b9c"},
            "noise":    {"dark": "#d1d1d1", "light": "#e8e8e8"},
        }
        self.subset_lookup = {
            "all":      {"name": "ALL",     "color": "#4e7abd"},
            "top":      {"name": "Top",     "color": "#c43639"},
            "random":   {"name": "Random",  "color": "#56a4d9"},
            "bottom":   {"name": "Bottom",  "color": "#d1d1d1"},
        }

    @check_folders("tgt_folder")
    def display_rank(
        self,
        table: pd.DataFrame,
        tgt_folder: str | os.PathLike | Path,
        beh: str        = "cont",
        noise: bool     = False,
    ) -> None:
        """
        Plot the tokens ranked by the ExtraTrees within-block AUC above their clarity fraction on the same order

        :param table:       The per-token table with columns `token`, `shard`, `n_events`, `fraction`, `auc_fold` and,
                            when `noise` is on, `auc_fold_noise`
        :param tgt_folder:  The folder where the plot will be saved
        :param beh:         The interested behavior
        :param noise:       The toggle for drawing the random-walk control next to the real AUC

        :return: None
        """

        table = table.dropna(subset=["auc_fold", "fraction"]).sort_values("auc_fold", ascending=False).reset_index(drop=True)
        rho, pval = spearmanr(table["fraction"], table["auc_fold"])

        x = np.arange(len(table))
        dark = table["shard"].map(lambda s: self.color_lookup[s]["dark"]).to_numpy()
        light = table["shard"].map(lambda s: self.color_lookup[s]["light"]).to_numpy()
        labels = table["token"].to_list()

        fig, (ax_auc, ax_frac) = plt.subplots(2, 1, figsize=(20, 10), sharex=True)

        # the AUC bars (and the control bars beside them when requested); the clarity bars below share the width and
        # the position of the real-AUC bars, so that the two panels line up token by token
        width = 0.4 if noise else 0.8
        shift = width / 2 if noise else 0.0
        ax_auc.bar(x - shift, table["auc_fold"].to_numpy(), width, color=dark)
        if noise:
            ax_auc.bar(x + shift, table["auc_fold_noise"].to_numpy(), width, color=self.color_lookup["noise"]["dark"])
        ax_auc.axhline(y=0.5, color="black", linestyle="--", linewidth=1)

        ax_frac.bar(x - shift, table["fraction"].to_numpy(), width, color=light)

        plt.suptitle(f"Clarity vs. AUC (spearman {rho:.2f}, p {pval:.1e})")
        ax_auc.set_ylabel("ExtraTrees within-block AUC")
        ax_frac.set_ylabel("Fraction of significant cells")
        ax_frac.set_xlabel("Tokens")

        for ax in (ax_auc, ax_frac):
            ax.set_ylim((0.0, 1.0))
        ax_frac.set_xticks(x)
        ax_frac.set_xticklabels(labels, rotation=90, fontsize=9)

        handles = [
            Patch(color=self.color_lookup["init"]["dark"], label=f"Init ({int((table['shard'] == 'init').sum())} tokens)"),
            Patch(color=self.color_lookup["memo"]["dark"], label=f"Memo ({int((table['shard'] == 'memo').sum())} tokens)"),
        ]
        if noise:
            handles.append(Patch(color=self.color_lookup["noise"]["dark"], label="Random-walk control"))
        ax_auc.legend(handles=handles, loc="upper right")

        plt.tight_layout(pad=1.0)

        # save the ranking plot
        plt.savefig(Path(tgt_folder) / Path(f"clarity_auc-{beh}{'_noise' if noise else ''}.svg"))
        plt.close()

        # report the statistics behind the plot
        for shard in ("init", "memo"):
            part = table.loc[table["shard"] == shard]
            if len(part) >= 3:
                rho_s, pval_s = spearmanr(part["fraction"], part["auc_fold"])
                self.logger.info(role="PredictionSummarizer", message=f"{beh} | {shard} | tokens {len(part)} | spearman {rho_s:.3f} | p {pval_s:.1e}")
        self.logger.info(role="PredictionSummarizer", message=f"{beh} | all | tokens {len(table)} | spearman {rho:.3f} | p {pval:.1e}")
        if noise:
            self.logger.info(
                role="PredictionSummarizer",
                message=f"{beh} | control | real auc_fold {table['auc_fold'].mean():.3f} | noise auc_fold {table['auc_fold_noise'].mean():.3f}"
            )
        self.logger.info(role="PredictionSummarizer", message=f"{beh} clarity vs. AUC plotted")

    @check_folders("tgt_folder")
    def display_subset(
        self,
        table: pd.DataFrame,
        tgt_folder: str | os.PathLike | Path,
        beh: str                    = "cont",
        subsets: Tuple[str, ...]    = ("all", "top", "random", "bottom"),
        fraction: float             = 0.3,
        noise: bool                 = False,
    ) -> None:
        """
        Plot the mean within-block AUC (with the across-token standard error) of every cell subset per shard

        :param table:       The stacked metrics table of the ExtraTrees engine with a `shard` column
        :param tgt_folder:  The folder where the plot will be saved
        :param beh:         The interested behavior
        :param subsets:     The cell subsets to plot, in order
        :param fraction:    The fraction of the cells kept in the "top" / "bottom" / "random" subsets
        :param noise:       The toggle marking the table as the random-walk control

        :return: None
        """

        wide = table.pivot_table(index=["shard", "token"], columns="subset", values="auc_fold").dropna()
        groups = [("Init", wide.loc["init"]), ("Memo", wide.loc["memo"]), ("Pooled", wide)]

        # the paired statistics behind the plot
        diff = wide["top"] - wide["bottom"]
        pval = wilcoxon(diff)[1] if len(diff) >= 5 else np.nan

        fig, ax = plt.subplots(figsize=(16, 9))

        width = 0.8 / len(subsets)
        x = np.arange(len(groups))

        for idx, subset in enumerate(subsets):
            means = [group[subset].mean() for _, group in groups]
            errs = [group[subset].std() / np.sqrt(len(group)) for _, group in groups]

            ax.bar(
                x + (idx - (len(subsets) - 1) / 2) * width,
                means,
                width,
                yerr=errs,
                capsize=3,
                label=self.subset_lookup[subset]["name"] + ("" if subset == "all" else f" {int(fraction * 100)}%"),
                color=self.subset_lookup[subset]["color"],
            )

        ax.axhline(y=0.5, color="black", linestyle="--", linewidth=1)

        plt.title(f"Clearest vs. Least-clear cells{' (random-walk control)' if noise else ''} (top - bottom {diff.mean():+.3f}, p {pval:.1e})")
        plt.xlabel("Shards")
        plt.ylabel("ExtraTrees within-block AUC")

        ax.set_xticks(x)
        ax.set_ylim((0.0, 1.0))
        ax.set_xticklabels([f"{label} ({len(group)} tokens)" for label, group in groups])

        plt.legend(loc="upper right")
        plt.tight_layout(pad=1.0)

        # save the subset plot
        plt.savefig(Path(tgt_folder) / Path(f"clarity_subset-{beh}{'_noise' if noise else ''}.svg"))
        plt.close()

        # report the statistics behind the plot
        for label, group in groups:
            self.logger.info(
                role="PredictionSummarizer",
                message=f"{beh} | {label.lower():6s} | tokens {len(group)} | " + " | ".join(f"{subset} {group[subset].mean():.3f}" for subset in subsets)
            )
        for first, second in (("top", "bottom"), ("top", "all"), ("top", "random")):
            d = wide[first] - wide[second]
            self.logger.info(
                role="PredictionSummarizer",
                message=f"{beh} | {first} - {second:6s} | mean {d.mean():+.3f} | wins {int((d > 0).sum())}/{len(d)} | p {wilcoxon(d)[1] if len(d) >= 5 else float('nan'):.1e}"
            )
        self.logger.info(role="PredictionSummarizer", message=f"{beh} clarity subsets plotted")
