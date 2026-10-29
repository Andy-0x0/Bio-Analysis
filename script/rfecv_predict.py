import os
from pathlib import Path
from dotenv import load_dotenv

from tools import ABlockLogger
from src_predict import rfecv_predict

# import path consts
load_dotenv("../config/.env.path")
PATH_ROOT = Path(os.getenv("ROOT_PATH"))        # path of this project directory
PATH_DATA = Path(os.getenv("INIT_DATA_PATH"))   # path of the initialization data directory
PATH_MEMO = Path(os.getenv("MEMO_DATA_PATH"))   # path of the memory data directory
PATH_PLOT = Path(os.getenv("PLOT_PATH"))        # path of the plot directory
PATH_CACHE = Path(os.getenv("CACHE_PATH"))      # path of the cache directory

# config the script parameters
seed            = 42
beh             = "cont"
rerun           = True
standard        = 3
skew            = True
axis            = False
colormap        = "reds"
on_predict      = lambda x: True
sel_folder      = PATH_CACHE / Path(f"rfecv_memo_vote/{beh}")
src_folder      = PATH_MEMO
tgt_int_folder  = PATH_PLOT / Path(f"rfecv_predict/memo/interval_{'skew' if skew else 'regular'}")
tgt_mat_folder  = PATH_PLOT / Path(f"rfecv_predict/memo/matrix_{'skew' if skew else 'regular'}")
tgt_raw_folder  = PATH_PLOT / Path(f"rfecv_predict/memo/raw_{'skew' if skew else 'regular'}")

# main execution logics
if __name__ == '__main__':
    logger = ABlockLogger(level="INFO")
    logger.warning(role="script.rfecv_predict", message="start execute rfecv_predict.py")

    rfecv_predict(
        sel_folder=sel_folder,
        src_folder=src_folder,
        tgt_int_folder=tgt_int_folder,
        tgt_mat_folder=tgt_mat_folder,
        tgt_raw_folder=tgt_raw_folder,
        on_predict=on_predict,
        beh=beh,
        standard=standard,
        seed=seed,
        skew=skew,
        axis=axis,
        colormap=colormap,
        rerun=rerun
    )

    logger.warning(role="script.rfecv_predict", message="complete execute rfecv_predict.py")