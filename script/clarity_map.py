import argparse
from pathlib import Path

from src.util import ABlockLogger, load_paths
from src.clarity import ClarityEngine, ClarityVisualizer

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory
PATH_PLOT = PATHS["PLOT_PATH"]          # path of the plot directory
PATH_CACHE = PATHS["CACHE_PATH"]        # path of the cache directory

# config the script parameters
parser = argparse.ArgumentParser(description="score the clarity of every cell and draw the event heatmap / curve of every token")
parser.add_argument("--beh",            type=str,   default="cont",                     help="the interested behavior (a key of src.preprocess.TARGETS)")
parser.add_argument("--shards",         type=str,   default=("init", "memo"), nargs="+", choices=("init", "memo"), help="the data shards to process")
parser.add_argument("--tokens",         type=str,   default=None, nargs="*",            help="the tokens to process (default: every token of the shards)")
parser.add_argument("--n-shift",        type=int,   default=1000,                       help="the number of circular label shifts of the clarity null")
parser.add_argument("--threshold",      type=float, default=2.0,                        help="the minimum z-score of a significant cell")
parser.add_argument("--inner-width",    type=int,   default=(-50, 50), nargs=2,         help="the clarity window around the onset in frames")
parser.add_argument("--view-width",     type=int,   default=(-80, 80), nargs=2,         help="the drawn window around the onset in frames (contains the clarity window)")
parser.add_argument("--map-width",      type=int,   default=8,                          help="the number of panels per row")
parser.add_argument("--kinds",          type=str,   default=("heatmap", "curve"), nargs="+", choices=("heatmap", "curve"), help="the maps to draw")
parser.add_argument("--n-workers",      type=int,   default=4,                          help="the number of worker processes over the tokens")
parser.add_argument("--rerun",          action="store_true",                            help="re-run even if the clarity caches exist")
args = parser.parse_args()

seed        = 42
beh         = args.beh
shards      = {shard: {"init": PATH_DATA, "memo": PATH_MEMO}[shard] for shard in args.shards}
tokens      = None if args.tokens is None else set(args.tokens)
n_shift     = args.n_shift
threshold   = args.threshold
inner_width = tuple(args.inner_width)
view_width  = tuple(args.view_width)
map_width   = args.map_width
kinds       = tuple(args.kinds)
n_workers   = args.n_workers
rerun       = args.rerun
cla_folder  = lambda shard: PATH_CACHE / Path(f"clarity/{shard}/{beh}")
tgt_folder  = PATH_PLOT / Path(f"clarity/map/{beh}")

# main execution logics
if __name__ == '__main__':
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.clarity_map", message="start execute clarity_map.py")

    for shard, src_folder in shards.items():
        ce = ClarityEngine(
            sig_folder=src_folder,
            beh_folder=src_folder,
            seed=seed,
        )
        viz = ClarityVisualizer(
            sig_folder=src_folder,
            beh_folder=src_folder,
            seed=seed,
        )

        # the maps are drawn by the callback as soon as the clarity table of a token is ready
        summary = ce.collect(
            beh=beh,
            token_filter=lambda token: tokens is None or token in tokens,
            callbacks=(viz.map_callback(
                threshold=threshold,
                inner_width=inner_width,
                view_width=view_width,
                map_width=map_width,
                kinds=kinds,
                tgt_folder=tgt_folder,
            ),),
            inner_width=inner_width,
            n_shift=n_shift,
            threshold=threshold,
            n_workers=n_workers,
            rerun=rerun,
            tgt_folder=cla_folder(shard),
        )

        for record in summary.to_dict("records"):
            logger.info(
                role="script.clarity_map",
                message=f"{shard} | {record['token']} | events {record['n_events']:3d} | significant {record['n_significant']:2d}/{record['n_cells']:2d} ({record['fraction']:.2f})"
            )

    logger.warning(role="script.clarity_map", message="complete execute clarity_map.py")
