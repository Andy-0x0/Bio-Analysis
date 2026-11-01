import argparse
from pathlib import Path

from src.util import ABlockLogger, load_paths
from src.deduction import Locator, Visualizer

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory

# config the script parameters
parser = argparse.ArgumentParser(description="draw the expanded spike windows of one token and behavior in every expansion direction")
parser.add_argument("--token",      type=str,   default="K215_A_D1",    help="the token of the subject")
parser.add_argument("--beh",        type=str,   default="cont",         help="the interested behavior")
parser.add_argument("--smooth",     type=int,   default=10,             help="the half width of the moving average smoothing the signals")
parser.add_argument("--threshold",  type=float, default=0.55,           help="the relative height a plateau needs to count as a spike")
parser.add_argument("--method",     type=str,   default="max",          help="the method locating the spike inside a plateau (max / center)")
parser.add_argument("--step",       type=int,   default=1,              help="the number of frames the windows expand per iteration")
parser.add_argument("--viz",        action="store_true",                help="show the plots besides saving them")
parser.add_argument("--rerun",      action="store_true",                help="re-run even if the window caches exist")
args = parser.parse_args()

token       = args.token
beh         = args.beh
smooth      = args.smooth
threshold   = args.threshold
method      = args.method
step        = args.step
viz         = args.viz
rerun       = args.rerun
log_level   = "INFO"
sig_folder  = PATH_DATA
beh_folder  = PATH_DATA
tgt_folder  = PATH_CACHE / Path(f"deduce/{beh}")
plt_folder  = PATH_PLOT  / Path(f"deduce/{beh}")

# main execution logics
if __name__ == "__main__":
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.deduce_viz", message="start execute deduce_viz.py")

    loc = Locator(
        sig_folder=sig_folder,
        beh_folder=beh_folder,
        log_level=log_level
    )

    window_config = loc.get_windows(
        token=token,
        beh=beh,
        smooth=smooth,
        threshold=threshold,
        method=method,
        step=step,
        rerun=rerun,
        tgt_folder=tgt_folder
    )

    for direction in ("none", "both", "left", "right"):
        Visualizer.display_window(
            window=window_config[direction],
            direction=direction,
            viz=viz,
            tgt_folder=plt_folder,
        )

    logger.warning(role="script.deduce_viz", message="complete execute deduce_viz.py")
