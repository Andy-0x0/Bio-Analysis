import argparse
from pathlib import Path

import pandas as pd

from src.util import ABlockLogger, load_paths
from src.clarity import ClarityEngine
from src.prediction import ExtraTreesEngine, PredictionSummarizer

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory

# config the script parameters
parser = argparse.ArgumentParser(description="rank the tokens by the ExtraTrees AUC of a behavior against their clarity fraction")
parser.add_argument("--beh",        type=str,   default="cont",     help="the interested behavior (a key of src.preprocess.TARGETS)")
parser.add_argument("--n-shift",    type=int,   default=1000,       help="the number of circular label shifts of the clarity null")
parser.add_argument("--threshold",  type=float, default=2.0,        help="the minimum z-score of a significant cell")
parser.add_argument("--window",     type=int,   default=50,         help="the number of frames in the causal window")
parser.add_argument("--min-events", type=int,   default=0,          help="the minimum number of events a token needs to be plotted")
parser.add_argument("--n-workers",  type=int,   default=4,          help="the number of worker processes over the tokens")
parser.add_argument("--noise",      action="store_true",            help="also score the random-walk control and draw it beside the real AUC")
parser.add_argument("--rerun",      action="store_true",            help="re-run even if the caches exist")
args = parser.parse_args()

seed        = 42
beh         = args.beh
n_shift     = args.n_shift
threshold   = args.threshold
window      = args.window
min_events  = args.min_events
n_workers   = args.n_workers
noise       = args.noise
rerun       = args.rerun
shards      = {"init": PATH_DATA, "memo": PATH_MEMO}
cla_folder  = lambda shard: PATH_CACHE / Path(f"clarity/{shard}/{beh}")
et_folder   = lambda shard: PATH_CACHE / Path(f"extratrees/{shard}/{beh}")
tgt_folder  = PATH_PLOT / Path(f"clarity/auc/{beh}")

# main execution logics
if __name__ == '__main__':
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.clarity_auc", message="start execute clarity_auc.py")

    tables = []
    for shard, src_folder in shards.items():
        # the clarity fraction of every token (whole recording, model-free)
        ce = ClarityEngine(
            sig_folder=src_folder,
            beh_folder=src_folder,
            seed=seed,
        )
        clarity = ce.collect(
            beh=beh,
            n_shift=n_shift,
            threshold=threshold,
            n_workers=n_workers,
            rerun=rerun,
            tgt_folder=cla_folder(shard),
        ).set_index("token")

        # the ExtraTrees within-block AUC of every token on all cells (and on the control group when requested)
        et = ExtraTreesEngine(
            sig_folder=src_folder,
            beh_folder=src_folder,
            seed=seed,
        )
        metrics = et.collect(
            beh=beh,
            subsets=("all",),
            window=window,
            n_shift=n_shift,
            n_workers=n_workers,
            rerun=rerun,
            tgt_folder=et_folder(shard),
        ).set_index("token")
        clarity["auc_fold"] = metrics["auc_fold"]

        if noise:
            control = et.collect(
                beh=beh,
                subsets=("all",),
                noise=True,
                window=window,
                n_shift=n_shift,
                n_workers=n_workers,
                rerun=rerun,
                tgt_folder=et_folder(shard),
            ).set_index("token")
            clarity["auc_fold_noise"] = control["auc_fold"]

        clarity.insert(0, "shard", shard)
        tables.append(clarity.reset_index())

    # drop the tokens too short on events to be scored, then plot
    table = pd.concat(tables, axis=0, ignore_index=True)
    table = table.loc[table["n_events"] >= min_events]

    sm = PredictionSummarizer()
    sm.display_rank(
        table=table,
        tgt_folder=tgt_folder,
        beh=beh,
        noise=noise,
    )

    logger.warning(role="script.clarity_auc", message="complete execute clarity_auc.py")
