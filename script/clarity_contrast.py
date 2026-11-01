import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.util import ABlockLogger, load_paths
from src.clarity import ContrastEngine, ClarityVisualizer

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory

# config the script parameters
parser = argparse.ArgumentParser(description="contrast the cells locked onto a reference behavior with the cells locked onto other behaviors: overlap of the significant sets, similarity of the per-cell responses, population time courses")
parser.add_argument("--first",          type=str,   default="cont",                                                     help="the reference behavior providing the reference cells")
parser.add_argument("--seconds",        type=str,   default=("cont_missed", "cont_inv", "cont_inv_near", "cont_inv_isolated", "cue_none", "walk"), nargs="+", help="the behaviors contrasted with the reference")
parser.add_argument("--n-shift",        type=int,   default=1000,                                                       help="the number of circular label shifts of the clarity null")
parser.add_argument("--threshold",      type=float, default=2.0,                                                        help="the minimum z-score of a significant cell")
parser.add_argument("--inner-width",    type=int,   default=(-50, 50), nargs=2,                                         help="the clarity window around the onset in frames")
parser.add_argument("--offsets",        type=float, default=(-5.0, 5.0, 1.0), nargs=3,                                  help="the native-rate window around the onset in seconds: start, stop (inclusive), step")
parser.add_argument("--pre",            type=float, default=(-2.0, -1.0), nargs=2,                                      help="the inclusive baseline range of the per-cell response in seconds")
parser.add_argument("--post",           type=float, default=(0.0, 1.0), nargs=2,                                        help="the inclusive response range of the per-cell response in seconds")
parser.add_argument("--n-perm",         type=int,   default=300,                                                        help="the number of relabellings of the similarity null")
parser.add_argument("--min-cells",      type=int,   default=3,                                                          help="the minimum number of significant cells for a token to be plotted")
parser.add_argument("--min-events",     type=int,   default=3,                                                          help="the minimum number of events of both kinds for a token to be plotted")
parser.add_argument("--n-workers",      type=int,   default=4,                                                          help="the number of worker processes over the tokens")
parser.add_argument("--rerun",          action="store_true",                                                            help="re-run even if the contrast caches exist")
args = parser.parse_args()

seed        = 42
first       = args.first
seconds     = tuple(args.seconds)
n_shift     = args.n_shift
threshold   = args.threshold
inner_width = tuple(args.inner_width)
offsets     = np.arange(args.offsets[0], args.offsets[1] + 1e-6, args.offsets[2])
pre         = tuple(args.pre)
post        = tuple(args.post)
n_perm      = args.n_perm
min_cells   = args.min_cells
min_events  = args.min_events
n_workers   = args.n_workers
rerun       = args.rerun
shards      = {"init": PATH_DATA, "memo": PATH_MEMO}
cla_folder  = lambda shard: (lambda beh: PATH_CACHE / Path(f"clarity/{shard}/{beh}"))
con_folder  = lambda shard: PATH_CACHE / Path(f"contrast/{shard}/{first}")
tgt_folder  = PATH_PLOT / Path(f"clarity/contrast/{first}")

# main execution logics
if __name__ == '__main__':
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.clarity_contrast", message="start execute clarity_contrast.py")

    records, curves = [], []
    for shard, src_folder in shards.items():
        ct = ContrastEngine(
            sig_folder=src_folder,
            beh_folder=src_folder,
            seed=seed,
        )

        # every pair shares the reference cells of the first behavior; the clarity tables come from (or go to) the
        # per-behavior clarity caches of the shard
        for second in seconds:
            record, curve = ct.collect(
                pair=(first, second),
                clarity_folder=cla_folder(shard),
                threshold=threshold,
                inner_width=inner_width,
                n_shift=n_shift,
                offsets=offsets,
                pre=pre,
                post=post,
                n_perm=n_perm,
                n_workers=n_workers,
                rerun=rerun,
                tgt_folder=con_folder(shard),
            )
            record.insert(0, "shard", shard)
            curve.insert(0, "shard", shard)
            records.append(record)
            curves.append(curve)

    table = pd.concat(records, axis=0, ignore_index=True)
    curves = pd.concat(curves, axis=0, ignore_index=True)

    viz = ClarityVisualizer(
        sig_folder=PATH_DATA,
        beh_folder=PATH_DATA,
        seed=seed,
    )
    viz.display_overlap(
        table=table,
        tgt_folder=tgt_folder,
        tag=first,
        threshold=threshold,
        min_cells=min_cells,
    )
    viz.display_profile(
        table=table,
        curves=curves,
        tgt_folder=tgt_folder,
        tag=first,
        min_cells=min_cells,
        min_events=min_events,
    )

    logger.warning(role="script.clarity_contrast", message="complete execute clarity_contrast.py")
