import os
import argparse
from pathlib import Path
from dotenv import load_dotenv

import pandas as pd

from tools import ABlockLogger
from src_clarity import ExtraTreesEngine, ClaritySummarizer

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory

# config the script parameters
parser = argparse.ArgumentParser(description="score ExtraTrees on the clearest / least-clear / random / all cells of every token")
parser.add_argument("--beh",        type=str,   default="cont",     help="the interested behavior")
parser.add_argument("--fraction",   type=float, default=0.3,        help="the fraction of the cells kept in the top / bottom / random subsets")
parser.add_argument("--n-shift",    type=int,   default=1000,       help="the number of circular label shifts of the clarity ranking")
parser.add_argument("--window",     type=int,   default=50,         help="the number of frames in the causal window")
parser.add_argument("--min-events", type=int,   default=0,          help="the minimum number of events a token needs to be plotted")
parser.add_argument("--noise",      action="store_true",            help="score the random-walk control group instead of the recordings")
parser.add_argument("--rerun",      action="store_true",            help="re-run even if the caches exist")
args = parser.parse_args()

seed        = 42
beh         = args.beh
fraction    = args.fraction
n_shift     = args.n_shift
window      = args.window
min_events  = args.min_events
noise       = args.noise
rerun       = args.rerun
subsets     = ("all", "top", "random", "bottom")
shards      = {"init": PATH_DATA, "memo": PATH_MEMO}
et_folder   = lambda shard: PATH_CACHE / Path(f"extratrees_{shard}/{beh}/subset_{int(fraction * 100)}")
tgt_folder  = PATH_PLOT / Path(f"clarity_subset/{beh}")

# main execution logics
if __name__ == '__main__':
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.clarity_subset", message="start execute clarity_subset.py")

    tables = []
    for shard, src_folder in shards.items():
        et = ExtraTreesEngine(
            sig_folder=src_folder,
            beh_folder=src_folder,
            seed=seed,
        )
        metrics = et.collect(
            beh=beh,
            subsets=subsets,
            fraction=fraction,
            noise=noise,
            window=window,
            n_shift=n_shift,
            rerun=rerun,
            tgt_folder=et_folder(shard),
        )
        metrics.insert(0, "shard", shard)
        tables.append(metrics)

    # drop the tokens too short on events to be scored, then plot
    table = pd.concat(tables, axis=0, ignore_index=True)
    table = table.loc[table["n_events"] >= min_events]

    sm = ClaritySummarizer()
    sm.display_subset(
        table=table,
        tgt_folder=tgt_folder,
        beh=beh,
        subsets=subsets,
        fraction=fraction,
        noise=noise,
    )

    logger.warning(role="script.clarity_subset", message="complete execute clarity_subset.py")
