import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import wilcoxon

from ..util import ABlockLogger, check_folders, load_paths
from ..preprocess import BlockPreprocessor, LABELS
from .engine import ClarityEngine

matplotlib.use("Agg")

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory


class ClarityVisualizer:
    @check_folders("sig_folder", "beh_folder")
    def __init__(
        self,
        sig_folder: str | os.PathLike | Path,
        beh_folder: str | os.PathLike | Path,
        seed: int       = 42,
        log_level: str  = "INFO",
    ) -> None:
        """
        The plotting helper of the clarity engines: the per-cell event heatmaps / curves of one recording with the
        significant cells first, the overlap bars and the response-profile panels of the target contrasts

        :param sig_folder:  The folder of the signal files
        :param beh_folder:  The folder of the behavior files
        :param seed:        The random seed handed to the preprocessors
        :param log_level:   The print level for logging
        """

        self.sig_folder = Path(sig_folder)
        self.beh_folder = Path(beh_folder)
        self.seed = seed
        self.log_level = log_level

        self.logger = ABlockLogger(level=log_level)

        # the significant cells are drawn warm, the others cool; the mask hides the frames the clarity never saw
        self.color_lookup = {
            "selected": {"cmap": "afmhot",  "line": "#c43639", "fill": "#e39b9c"},
            "other":    {"cmap": "viridis", "line": "#4e7abd", "fill": "#9dbbe0"},
            "mask":     "#3a3a3a",
            "observed": "#4e7abd",
            "chance":   "#d1d1d1",
            "clear":    "#4e7abd",
            "unclear":  "#9dbbe0",
        }
        self.target_lookup = {
            "cont":                 "#4e7abd",
            "cont_missed":          "#2e9c6b",
            "cont_inv":             "#c43639",
            "cont_inv_near":        "#e58a8c",
            "cont_inv_isolated":    "#7a1c1e",
            "cue":                  "#e3922b",
            "cue_none":             "#e3922b",
            "walk":                 "#8a63b3",
        }

    def __getstate__(self) -> Dict:
        # the logger owns a worker thread, so it is rebuilt on the receiving process instead of being pickled
        return {key: value for key, value in self.__dict__.items() if key != "logger"}

    def __setstate__(self, state: Dict) -> None:
        self.__dict__.update(state)
        self.logger = ABlockLogger(level=self.log_level)

    @staticmethod
    def _minmax(array: np.ndarray) -> np.ndarray:
        """
        Compress an array into the 0-1 range (a flat array stays flat at 0)

        :param array: The array

        :return: The normalized array
        """

        span = array.max() - array.min()

        return (array - array.min()) / span if span > 0 else np.zeros_like(array)

    @staticmethod
    def _label(first: str, second: str) -> str:
        """
        The display label of a target pair

        :param first:   The first target
        :param second:  The second target

        :return: The label
        """

        return f"{LABELS.get(first, first)} vs {LABELS.get(second, second)}"

    @check_folders("tgt_folder")
    def display_map(
        self,
        token: str,
        table: pd.DataFrame,
        beh: str                                    = "cont",
        threshold: float                            = 2.0,
        inner_width: Tuple[int, int]                = (-50, 50),
        view_width: Tuple[int, int]                 = (-50, 50),
        map_width: int                              = 8,
        kind: str                                   = "heatmap",
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> None:
        """
        Draw the event-locked view of every cell of a token: one panel per cell, the significant cells first (warm),
        the others after (cool); a heatmap stacks the events as rows, a curve averages them; when the view is wider
        than the clarity window, the frames the clarity never saw are dimmed by a grey mask

        :param token:       The token of the subject
        :param table:       The per-cell clarity table of the token for the behavior
        :param beh:         The interested behavior (the target)
        :param threshold:   The minimum z-score of a significant cell
        :param inner_width: The window of the clarity computation in frames
        :param view_width:  The window drawn in frames (must contain the clarity window)
        :param map_width:   The number of panels per row
        :param kind:        "heatmap" or "curve"
        :param tgt_folder:  The folder where the plot will be saved

        :return: None
        """

        if view_width[0] > inner_width[0] or view_width[1] < inner_width[1]:
            self.logger.error(role="ClarityVisualizer", message=f"the view {view_width} must contain the clarity window {inner_width}")
            raise RuntimeError(f"the view {view_width} must contain the clarity window {inner_width}")

        prep = BlockPreprocessor(
            beh_path=self.beh_folder / Path(f"{token}_B.csv"),
            sig_path=self.sig_folder / Path(f"{token}_S.csv"),
            target=beh,
            seed=self.seed,
            log_level="WARNING"
        )
        windows = ClarityEngine.event_windows(x=prep.normalized(), y=prep.y, inner_width=view_width)
        offsets = np.arange(view_width[0], view_width[1] + 1)

        if windows.shape[1] == 0:
            self.logger.warning(role="ClarityVisualizer", message=f"{token}-{beh} has no complete event window, {kind} skipped")
            return

        # the significant cells sorted by their z-score, then the remaining cells in their file order
        z = table.set_index("cell")["z"].reindex(prep.cells)
        selected = z.loc[z >= threshold].sort_values(ascending=False).index.to_list()
        others = [cell for cell in prep.cells if cell not in set(selected)]

        rows_selected = int(np.ceil(len(selected) / map_width))
        rows_others = int(np.ceil(len(others) / map_width))

        fig, axes = plt.subplots(rows_selected + rows_others, map_width, figsize=(3 * map_width, 2.6 * (rows_selected + rows_others)), squeeze=False)
        for ax in axes.ravel():
            ax.axis("off")

        # draw the two groups block by block, the second group starting on a fresh row
        for group, cells, row_start in (("selected", selected, 0), ("other", others, rows_selected)):
            style = self.color_lookup[group]

            for idx, cell in enumerate(cells):
                ax = axes[row_start + idx // map_width, idx % map_width]
                ax.axis("on")

                matrix = windows[prep.cells.index(cell)]

                if kind == "heatmap":
                    ax.imshow(
                        self._minmax(matrix),
                        aspect="auto",
                        cmap=style["cmap"],
                        origin="upper",
                        interpolation="nearest",
                        extent=(offsets[0], offsets[-1], matrix.shape[0], 0),
                    )
                    ax.axvline(x=0, color="white", linestyle="--", linewidth=0.8)
                    ax.set_yticks([])
                else:
                    curve = self._minmax(matrix.mean(axis=0))
                    ax.plot(offsets, curve, color=style["line"], linewidth=1.2)
                    ax.fill_between(offsets, curve, 0, color=style["fill"], alpha=0.6, linewidth=0)
                    ax.axvline(x=0, color="black", linestyle="--", linewidth=0.8)
                    ax.set_ylim((0.0, 1.0))
                    ax.set_yticks([0, 1])

                # dim the frames outside the clarity window
                if view_width != inner_width:
                    ax.axvspan(offsets[0], inner_width[0], color=self.color_lookup["mask"], alpha=0.45, linewidth=0)
                    ax.axvspan(inner_width[1], offsets[-1], color=self.color_lookup["mask"], alpha=0.45, linewidth=0)

                ax.set_xlim((offsets[0], offsets[-1]))
                ax.set_xticks(np.arange(int(np.ceil(offsets[0] / 25)) * 25, offsets[-1] + 1, 25))
                ax.tick_params(labelsize=8)
                ax.set_title(f"{cell} (z {z.loc[cell]:+.1f})" if np.isfinite(z.loc[cell]) else f"{cell} (z n/a)", fontsize=9)

        fig.suptitle(f"{token}-{beh} event {kind} ({len(selected)}/{len(prep.cells)} significant cells, z >= {threshold:g})")
        fig.supxlabel("frames around the onset (0.1 s)")
        plt.tight_layout(pad=1.0, rect=(0.0, 0.0, 1.0, 0.975))

        # save the map
        plt.savefig(Path(tgt_folder) / Path(f"{token}-{beh}_{kind}.svg"))
        plt.close(fig)

        self.logger.info(role="ClarityVisualizer", message=f"{token}-{beh} {kind} plotted ({len(selected)} significant, {windows.shape[1]} events)")

    def map_callback(
        self,
        threshold: float                            = 2.0,
        inner_width: Tuple[int, int]                = (-50, 50),
        view_width: Tuple[int, int]                 = (-50, 50),
        map_width: int                              = 8,
        kinds: Tuple[str, ...]                      = ("heatmap", "curve"),
        tgt_folder: None | str | os.PathLike | Path = None,
    ) -> Callable[[str, str, pd.DataFrame], None]:
        """
        Build the callback mounted on `ClarityEngine.collect`, drawing the requested maps of every token as soon as
        its clarity table is ready

        :param threshold:   The minimum z-score of a significant cell
        :param inner_width: The window of the clarity computation in frames
        :param view_width:  The window drawn in frames
        :param map_width:   The number of panels per row
        :param kinds:       The maps to draw ("heatmap", "curve")
        :param tgt_folder:  The folder where the plots will be saved

        :return: The callback `callback(token, beh, table)`
        """

        def callback(token: str, beh: str, table: pd.DataFrame) -> None:
            for kind in kinds:
                self.display_map(
                    token=token, table=table, beh=beh, threshold=threshold, inner_width=inner_width, view_width=view_width,
                    map_width=map_width, kind=kind, tgt_folder=tgt_folder,
                )

        return callback

    @check_folders("tgt_folder")
    def display_overlap(
        self,
        table: pd.DataFrame,
        tgt_folder: str | os.PathLike | Path,
        tag: str                        = "cont",
        threshold: float                = 2.0,
        min_cells: int                  = 3,
        edges: Tuple[float, ...]        = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0),
        clear_fraction: float           = 0.4,
    ) -> None:
        """
        Plot, per target pair, the overlap of the significant cells as a histogram over the clarity of the tokens:
        the tokens are bucketed by the fraction of reference-significant cells, every bucket shows the mean
        containment (shared cells over the smaller set) of its tokens, the clear buckets in the dark tone, the
        unclear ones in the light tone, with the chance containment of two independent draws marked per bucket

        :param table:           The stacked records of the contrast engine (one row per token and pair)
        :param tgt_folder:      The folder where the plot will be saved
        :param tag:             The tag of the plot file (the reference target of the pairs)
        :param threshold:       The minimum z-score of a significant cell (title only)
        :param min_cells:       The minimum size of both significant sets for a token to count
        :param edges:           The bucket edges of the clarity fraction (reference-significant cells / all cells)
        :param clear_fraction:  The clarity fraction from which a bucket counts as clear

        :return: None
        """

        pairs = list(table[["first", "second"]].drop_duplicates().itertuples(index=False))
        edges = np.asarray(edges, dtype=float)
        centers = (edges[:-1] + edges[1:]) / 2
        clear = centers >= clear_fraction

        n_cols = min(3, len(pairs))
        n_rows = int(np.ceil(len(pairs) / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.5 * n_cols, 4.5 * n_rows), squeeze=False, sharey=True)
        for ax in axes.ravel():
            ax.axis("off")

        for idx, (first, second) in enumerate(pairs):
            ax = axes[idx // n_cols, idx % n_cols]
            ax.axis("on")

            part = table.loc[(table["first"] == first) & (table["second"] == second) & (table["n_first"] >= min_cells) & (table["n_second"] >= min_cells)]
            part = part.dropna(subset=["containment", "containment_chance"])
            fraction = (part["n_first"] / part["n_cells"]).to_numpy()
            bucket = np.clip(np.digitize(fraction, edges[1:-1]), 0, len(centers) - 1)

            means = np.array([part["containment"].to_numpy()[bucket == b].mean() if (bucket == b).any() else np.nan for b in range(len(centers))])
            errs = np.array([part["containment"].to_numpy()[bucket == b].std() / np.sqrt((bucket == b).sum()) if (bucket == b).sum() > 1 else 0.0 for b in range(len(centers))])
            chances = np.array([part["containment_chance"].to_numpy()[bucket == b].mean() if (bucket == b).any() else np.nan for b in range(len(centers))])
            counts = np.array([(bucket == b).sum() for b in range(len(centers))])

            # the bars: dark for the clear buckets, light for the unclear ones; the chance level as a black tick
            width = (edges[1:] - edges[:-1]) * 0.8
            colors = np.where(clear, self.color_lookup["clear"], self.color_lookup["unclear"])
            ax.bar(centers, np.nan_to_num(means), width, yerr=errs, capsize=3, color=colors, linewidth=0)
            ax.hlines(chances, centers - width / 2, centers + width / 2, color="black", linestyle="--", linewidth=1.2)

            ax.set_xlim((edges[0], edges[-1]))
            ax.set_ylim((0.0, 1.0))
            ax.set_xticks(centers)
            ax.set_xticklabels([f"{lo:.1f}-{hi:.1f}\n({count} tokens)" for lo, hi, count in zip(edges[:-1], edges[1:], counts)], fontsize=8)
            ax.set_title(f"{LABELS.get(second, second)} ({len(part)} tokens)")
            if idx % n_cols == 0:
                ax.set_ylabel("Shared significant cells / smaller set")
            if idx // n_cols == n_rows - 1:
                ax.set_xlabel(f"Clarity of the token (fraction of {LABELS.get(first, first)}-significant cells)")

            # report the statistics behind the panel
            for side, mask in (("clear", clear[bucket]), ("unclear", ~clear[bucket])):
                obs, chance = part["containment"].to_numpy()[mask], part["containment_chance"].to_numpy()[mask]
                diff = obs - chance
                self.logger.info(
                    role="ClarityVisualizer",
                    message=f"{self._label(first, second):32s} | {side:7s} | tokens {int(mask.sum()):2d} | containment {obs.mean() if mask.any() else float('nan'):.3f} "
                            f"(chance {chance.mean() if mask.any() else float('nan'):.3f}) | pooled {int(part['n_both'].to_numpy()[mask].sum())}/{int(np.minimum(part['n_first'], part['n_second']).to_numpy()[mask].sum())} | "
                            f"above chance {int((diff > 0).sum())}/{int(mask.sum())} | p {wilcoxon(diff)[1] if len(diff) >= 5 and (diff != 0).any() else float('nan'):.1e}"
                )

        handles = [
            Patch(color=self.color_lookup["clear"], label=f"Clear tokens (fraction >= {clear_fraction:g})"),
            Patch(color=self.color_lookup["unclear"], label=f"Unclear tokens (fraction < {clear_fraction:g})"),
            matplotlib.lines.Line2D([], [], color="black", linestyle="--", label="Chance"),
        ]
        axes[0, 0].legend(handles=handles, loc="upper left", fontsize=8)

        fig.suptitle(f"Overlap with the {LABELS.get(tag, tag)} cells by clarity (z >= {threshold:g}, both sets >= {min_cells} cells)")
        plt.tight_layout(pad=1.0, rect=(0.0, 0.0, 1.0, 0.96))

        # save the overlap plot
        plt.savefig(Path(tgt_folder) / Path(f"clarity_overlap-{tag}.svg"))
        plt.close(fig)

        self.logger.info(role="ClarityVisualizer", message=f"{tag} overlap plotted")

    @check_folders("tgt_folder")
    def display_profile(
        self,
        table: pd.DataFrame,
        curves: pd.DataFrame,
        tgt_folder: str | os.PathLike | Path,
        tag: str            = "cont",
        min_cells: int      = 3,
        min_events: int     = 3,
    ) -> None:
        """
        Plot, per target pair, the similarity of the per-cell response profiles against its relabelling null, and the
        population time courses of the reference cells around every kind of event

        :param table:       The stacked records of the contrast engine (one row per token and pair)
        :param curves:      The stacked time courses of the contrast engine in long format
        :param tgt_folder:  The folder where the plot will be saved
        :param tag:         The tag of the plot file (the reference target of the pairs)
        :param min_cells:   The minimum number of reference cells for a token to count
        :param min_events:  The minimum number of events of both kinds for a token to count

        :return: None
        """

        pairs = list(table[["first", "second"]].drop_duplicates().itertuples(index=False))
        groups = []
        for first, second in pairs:
            part = table.loc[
                (table["first"] == first) & (table["second"] == second) & (table["n_first"] >= min_cells) &
                (table["n_events_first"] >= min_events) & (table["n_events_second"] >= min_events)
            ]
            groups.append((first, second, part.dropna(subset=["r", "r_null"])))

        fig, (ax_r, ax_curve) = plt.subplots(1, 2, figsize=(18, 7), gridspec_kw={"width_ratios": [1, 1.3]})

        # the left panel: the observed correlation of the profiles next to the relabelling null
        x = np.arange(len(groups))
        width = 0.4
        means = [group["r"].mean() if len(group) else np.nan for _, _, group in groups]
        errs = [group["r"].std() / np.sqrt(len(group)) if len(group) > 1 else 0.0 for _, _, group in groups]
        nulls = [group["r_null"].mean() if len(group) else np.nan for _, _, group in groups]
        null_errs = [group["r_null"].std() / np.sqrt(len(group)) if len(group) > 1 else 0.0 for _, _, group in groups]

        ax_r.bar(x - width / 2, means, width, yerr=errs, capsize=3, color=self.color_lookup["observed"])
        ax_r.bar(x + width / 2, nulls, width, yerr=null_errs, capsize=3, color=self.color_lookup["chance"])
        ax_r.axhline(y=0, color="black", linestyle="--", linewidth=1)

        ax_r.set_ylim((-1.0, 1.0))
        ax_r.set_xticks(x)
        ax_r.set_xticklabels([f"{LABELS.get(second, second)} ({len(group)} tokens)" for _, second, group in groups], rotation=45, ha="right")
        ax_r.set_ylabel("Pearson r of the per-cell responses")
        ax_r.set_title(f"Response profile similarity with {LABELS.get(tag, tag)}")
        ax_r.legend(handles=[Patch(color=self.color_lookup["observed"], label="Observed"), Patch(color=self.color_lookup["chance"], label="Relabelling null")], loc="upper right")

        # the right panel: the population time course of the reference cells per kind of event; the reference
        # curve pools the tokens of every pair, every other kind keeps the tokens of its own pair
        series = []
        for first, second, group in groups:
            if not any(beh == first for beh, _ in series):
                reference = curves.loc[(curves["first"] == first) & (curves["beh"] == first) & (curves["token"].isin(table.loc[table["n_first"] >= min_cells, "token"]))]
                series.append((first, reference.drop_duplicates(subset=["token", "offset"])))
            series.append((second, curves.loc[(curves["first"] == first) & (curves["second"] == second) & (curves["beh"] == second) & (curves["token"].isin(group["token"]))]))

        for beh, part in series:
            wide = part.pivot_table(index="token", columns="offset", values="value").dropna() if not part.empty else part
            if wide.empty:
                continue
            mean, err = wide.mean(axis=0), wide.std(axis=0) / np.sqrt(len(wide))
            color = self.target_lookup.get(beh, self.color_lookup["chance"])
            ax_curve.plot(mean.index, mean.to_numpy(), color=color, linewidth=2, label=f"{LABELS.get(beh, beh)} ({len(wide)} tokens)")
            ax_curve.fill_between(mean.index, (mean - err).to_numpy(), (mean + err).to_numpy(), color=color, alpha=0.1, linewidth=0)

        ax_curve.axvline(x=0, color="black", linestyle="--", linewidth=1)
        ax_curve.axhline(y=0, color="black", linewidth=0.8)
        ax_curve.set_xticks(np.unique(curves["offset"]))
        plt.setp(ax_curve.get_xticklabels(), rotation=45, ha="right")
        ax_curve.set_xlabel("seconds around the onset")
        ax_curve.set_ylabel("Population mean z of the reference cells")
        ax_curve.set_title(f"Population response of the {LABELS.get(pairs[0][0], pairs[0][0]) if pairs else tag} cells")
        ax_curve.legend(loc="upper left")

        plt.tight_layout(pad=1.0)

        # save the profile plot
        plt.savefig(Path(tgt_folder) / Path(f"clarity_profile-{tag}.svg"))
        plt.close(fig)

        # report the statistics behind the plot
        for first, second, group in groups:
            if len(group) == 0:
                self.logger.warning(role="ClarityVisualizer", message=f"{self._label(first, second)} | no token with enough cells / events")
                continue
            self.logger.info(
                role="ClarityVisualizer",
                message=f"{self._label(first, second):32s} | tokens {len(group):2d} | r {group['r'].mean():.3f} (null {group['r_null'].mean():.3f}) | "
                        f"slope {group['slope'].mean():.2f} | z {group['z_r'].mean():+.2f} | tokens z <= -2: {int((group['z_r'] <= -2).sum())}, z >= 2: {int((group['z_r'] >= 2).sum())} | "
                        f"amplitude {group['amp_first'].mean():.3f} / {group['amp_second'].mean():.3f}"
            )
        self.logger.info(role="ClarityVisualizer", message=f"{tag} profile plotted")
