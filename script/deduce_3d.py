import argparse
from pathlib import Path

from src.util import ABlockLogger, fetch_tokens, load_paths
from src.deduction import Locator, PenaltyComputer, Visualizer

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory

# config the script parameters
parser = argparse.ArgumentParser(description="expand the spike windows of every token and behavior, then compare the penalties of the expansion directions")
parser.add_argument("--behs",           type=str,   default=("cont", "cont_inv", "both", "both_inv", "walk"), nargs="+", help="the behaviors to deduce")
parser.add_argument("--shard",          type=str,   default="all",      help="which shard of the tokens to process")
parser.add_argument("--smooth",         type=int,   default=10,         help="the half width of the moving average smoothing the signals")
parser.add_argument("--threshold",      type=float, default=0.55,       help="the relative height a plateau needs to count as a spike")
parser.add_argument("--method",         type=str,   default="max",      help="the method locating the spike inside a plateau (max / center)")
parser.add_argument("--step",           type=int,   default=1,          help="the number of frames the windows expand per iteration")
parser.add_argument("--label-penalty",  type=float, default=100.0,      help="the weight of the missing-label penalty")
parser.add_argument("--spike-penalty",  type=float, default=0.1,        help="the weight of the expansion-width penalty")
parser.add_argument("--rerun",          action="store_true",            help="re-run even if the window caches exist")
args = parser.parse_args()

behs            = tuple(args.behs)
shard           = args.shard
smooth          = args.smooth
threshold       = args.threshold
method          = args.method
step            = args.step
label_penalty   = args.label_penalty
spike_penalty   = args.spike_penalty
rerun           = args.rerun
log_level       = "INFO"
sig_folder      = PATH_DATA
beh_folder      = PATH_DATA
plt_folder      = PATH_PLOT  / Path("deduce")
tgt_root        = PATH_CACHE / Path("deduce")

# main execution logics
if __name__ == '__main__':
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.deduce_3d", message="start execute deduce_3d.py")

    loc = Locator(
        sig_folder=sig_folder,
        beh_folder=beh_folder,
        log_level=log_level
    )

    window_configs = []
    for token in fetch_tokens(folder=sig_folder, shard=shard):
        for beh in behs:
            window_configs.append(loc.get_windows(
                token=token,
                beh=beh,
                smooth=smooth,
                threshold=threshold,
                method=method,
                step=step,
                rerun=rerun,
                tgt_folder=tgt_root / Path(beh)
            ))

    penalty_table = PenaltyComputer.compute_penalties(
        window_configs=tuple(window_configs),
        label_penalty=label_penalty,
        spike_penalty=spike_penalty
    )
    Visualizer.display_3d_penalty(
        table=penalty_table,
        dx=0.15,
        dy=0.15,
        left_till=0,
        rotation=0,
        tgt_folder=plt_folder,
    )

    for beh, row in penalty_table.iterrows():
        logger.info(role="script.deduce_3d", message=f"{beh:10s} | " + " | ".join(f"{direction} {value:.3f}" for direction, value in row.items()))

    logger.warning(role="script.deduce_3d", message="complete execute deduce_3d.py")
