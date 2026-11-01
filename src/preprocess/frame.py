from typing import Tuple

import os
import json
from pathlib import Path

import random
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from imblearn.over_sampling import SMOTE

from ..util import ABlockLogger, load_paths
from .target import TARGETS, normalize_name

# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory


class Preprocessor:
    def __init__(
            self,
            beh_path: str | os.PathLike,
            sig_path: str | os.PathLike,
            tvt_ratio: Tuple[float, float, float] = (0.5, 0, 0.5),
            shuffle: bool = True,
            normalizer: str | None = 'z-score',
            agg: bool = False,
            smote: None | float | str = 'auto',
            resize: bool = False,
            target: str = 'cont',
            seed: int = 42,
            log_level: str = 'INFO'
    ) -> None:
        """
        The generic frame-level preprocessor: the ~1 Hz signal is interpolated onto the 0.1 s behavior grid, every
        annotation of the target is painted onto the grid as a run of positive labels, and the aligned dataset is
        split / resampled / normalized as configured

        :param beh_path:    the path to the behavior file
        :param sig_path:    the path to the signal file
        :param tvt_ratio:   the ratio of (train_size : valid_size : test_size)
        :param shuffle:     The toggle for shuffling the dataset (leaky for prediction: neighbouring frames are near
                            copies of each other, so use the time-ordered BlockPreprocessor for any score)
        :param normalizer:  the type of normalizer to use
        :param agg:         the toggle for generate the SUM column (sum(all-signal-columns))
        :param smote:       the config for SMOTE
        :param resize:      the toggle for resizing when doing x-y alignment
        :param target:      the binary target, one key of `src.preprocess.TARGETS`
        :param seed:        the random seed of the shuffling / resampling
        :param log_level:   the print level for logging
        """

        self.target_rule = None
        self.behavior_df = pd.read_csv(beh_path)
        self.signal_df = pd.read_csv(sig_path)
        self.dataset_df = None
        self.core = {}

        self.beh_path = beh_path
        self.sig_path = sig_path
        self.shuffle = shuffle
        self.tvt_ratio = tvt_ratio
        self.normalizer = normalizer
        self.smote = smote
        self.agg = agg
        self.resize = resize
        self.target = target
        self.seed = seed

        self.logger = ABlockLogger(level=log_level)
        self.sig_fps = 1 / (1.000045 if Path(sig_path).stem.split('_')[0] == 'K215' else 1.001286)
        self.beh_fps = 7.06
        self.beh_frame_to_second = lambda x: x / self.beh_fps
        self.beh_second_to_frame = lambda x: x * self.beh_fps
        self.sig_frame_to_second = lambda x: x / self.sig_fps
        self.sig_second_to_frame = lambda x: x * self.sig_fps

        # alignment
        self._config_target(mode=self.target)
        self._config_signal(deci=1)
        self._config_behavior(deci=1)
        self._align(deci=1)

        # transformation
        self._transform(
            tvt_ratio=tvt_ratio,
            shuffle=shuffle,
            smote_strategy=smote,
            normalizer=normalizer
        )

        # report the config
        self.logger.info(
            role="Preprocessor",
            message=f"Config: "
                    f"origin[{len(self.core['total_df_ori']) if self.core['total_df_ori'] is not None else '0'}] | "
                    f"train[{len(self.core['train_df_aft']) if self.core['train_df_aft'] is not None else '0'}] | "
                    f"valid[{len(self.core['valid_df_aft']) if self.core['valid_df_aft'] is not None else '0'}] | "
                    f"test[{len(self.core['test_df_aft']) if self.core['test_df_aft'] is not None else '0'}]"
        )

    def _config_random(self, seed: int = 42) -> None:
        self.logger.info(role="Preprocessor", message=f"Random seed set: {seed}")

        random.seed(seed)
        np.random.seed(seed)

    def _config_target(self, mode: str = "cont") -> None:
        """
        Pick the target rule deciding which annotations are labelled positive

        :param mode:  The key of the target in `src.preprocess.TARGETS`

        :return: None
        """

        if mode not in TARGETS:
            self.logger.error(role="Preprocessor", message=f"unknown target when configuring the labels: {mode}")
            raise RuntimeError(f"unknown target when configuring the labels: {mode}")

        self.target_rule = TARGETS[mode]

    def _config_signal(self, deci: int = 1) -> None:
        """
        populate the signal pd.DataFrame as self.signal_df on the 10 ** (-deci) seconds grid

        :param deci: the offset of the floating point that decides to what extent the rounded up should be for the time indexes

        :return: None
        """

        # Set the Index column for signal CSV
        if 'x0000' in self.signal_df.columns:
            self.signal_df = self.signal_df.set_index('x0000')
            self.signal_df = self.signal_df.rename_axis('index', axis='index')

        if self.agg:
            agg_sr = pd.Series(self.signal_df.sum(axis=1), name='SUM')
            self.signal_df = pd.concat([self.signal_df, agg_sr], axis=1)

        # Fill the gap at a certain frequency for Signal CSV (akima: smooth, but it reads the next native sample,
        # i.e. up to one native step of the future leaks into every interpolated frame)
        core_index = np.array(self.signal_df.index.to_list(), dtype=np.float32)
        core_index = self.sig_frame_to_second(core_index)
        core_index = np.round(core_index, decimals=deci)

        fill_index = np.arange(core_index[0], core_index[-1] + 10 ** (-deci), 10 ** (-deci), dtype=np.float32)
        fill_index = np.round(fill_index, decimals=deci)

        self.signal_df.index = core_index
        self.signal_df = self.signal_df.reindex(fill_index)
        self.signal_df = self.signal_df.interpolate(axis=0, method='akima').ffill().bfill()

    def _config_behavior(self, deci: int = 1) -> None:
        """
        populate the behavior pd.Series as self.behavior_df: the labels on the 10 ** (-deci) seconds grid

        :param deci: the offset of the floating point that decides to what extent the rounded up should be for the time indexes

        :return: None
        """

        self.behavior_df = self.behavior_df.loc[:, ['Behavior', 'Start', 'End']]

        # Extra 0.5 second offset (0.5 + 1.5[already added in the CSVs] = 2.0)
        self.behavior_df.loc[:, ['Start', 'End']] += self.beh_second_to_frame(0.5)

        # transform the frame numbers into the corresponding sec timestamps
        self.behavior_df.loc[:, ['Start', 'End']] = (
            self.behavior_df.loc[:, ['Start', 'End']]
            .map(self.beh_frame_to_second)
            .map(lambda x: np.round(x, decimals=deci))
        )

        # create the init labels pd.Series for populating the interested behaviors later
        start = self.behavior_df.loc[:, "Start"].iloc[0]
        end = self.behavior_df.loc[:, 'End'].iloc[-1]

        fill_index = np.arange(start, end + 10 ** (-deci), 10 ** (-deci), dtype=np.float32)
        fill_index = np.round(fill_index, decimals=deci)

        labels = pd.Series(np.zeros(len(fill_index)), index=fill_index, name='Label')

        # decide which annotations are positive (the rule may look at the neighbouring annotations, e.g. a cue
        # followed by nothing), then paint them onto the grid
        table = pd.DataFrame({
            "name": self.behavior_df.loc[:, 'Behavior'].map(normalize_name),
            "start": self.behavior_df.loc[:, 'Start'].astype(float),
            "end": self.behavior_df.loc[:, 'End'].astype(float),
        })
        positives = self.target_rule.mask(table)

        for (idx, row), positive in zip(self.behavior_df.iterrows(), positives):
            if positive:
                labels.loc[row.loc['Start']: row.loc['End']] = 1

        self.behavior_df = labels

    def _align(self, deci: int = 1) -> None:
        """
        Align the previously independent signal pd.DataFrame and behavior pd.Series and
        do proper starting point alignment and cutting / shrinking / expanding if necessary.
        Populate self.dataset_df using the result

        :param deci: the offset of the floating point that decides to what extent the rounded up should be for the time indexes

        :return: None
        """

        # cut the extra part of the signal file which happens before the camera USB is triggered
        self.signal_df = self.signal_df.loc[self.behavior_df.index[0]:, :]

        # do the shrinking / expansion alignment
        if self.resize:
            # shrink the signal to match behavior
            if len(self.signal_df) > len(self.behavior_df):
                long_length = len(self.signal_df)
                short_length = len(self.behavior_df)

                # pick {short_length} rows from the original signal dataframe evenly
                idx = np.linspace(0, long_length - 1, short_length, dtype=int)
                self.signal_df = pd.DataFrame(
                    self.signal_df.iloc[idx, :].values,
                    index=self.behavior_df.index,
                    columns=self.signal_df.columns
                )

                self.behavior_df.index = self.behavior_df.index.map(lambda x: np.round(x, decimals=deci))
                self.signal_df.index = self.signal_df.index.map(lambda x: np.round(x, decimals=deci))

            # expand the signal to match behavior
            else:
                long_length = len(self.behavior_df)
                short_length = len(self.signal_df)
                ori_short = self.signal_df.values

                # create {long_length - short_length} extra rows from the original signal dataframe evenly
                idx_old = np.arange(short_length)
                idx_new = np.linspace(0, short_length - 1, long_length)
                new_short = np.apply_along_axis(
                    lambda col: np.interp(idx_new, idx_old, col),
                    axis=0,
                    arr=ori_short
                )
                self.signal_df = pd.DataFrame(new_short, index=self.behavior_df.index, columns=self.signal_df.columns)

                self.behavior_df.index = self.behavior_df.index.map(lambda x: np.round(x, decimals=deci))
                self.signal_df.index = self.signal_df.index.map(lambda x: np.round(x, decimals=deci))

        # do the cutting alignment
        else:
            # cut the extra part at the end of the signal to match behavior
            if len(self.signal_df) > len(self.behavior_df):
                self.signal_df = self.signal_df.iloc[:len(self.behavior_df), :]

            # cut the extra part at the end of the behavior to match signal
            else:
                self.behavior_df = self.behavior_df.iloc[:len(self.signal_df)]

        # assert equal length after alignment
        if len(self.signal_df) != len(self.behavior_df):
            self.logger.error(role="Preprocessor", message="signal and behavior DataFrames do not match after alignment")
            raise RuntimeError("signal and behavior DataFrames do not match after alignment")

        # combine the aligned signal pd.Dataframe with the aligned behavior pd.Dataframe into the total dataset
        self.dataset_df = self.signal_df.copy()
        self.dataset_df.loc[:, self.behavior_df.name] = self.behavior_df.values

    def _transform(
            self,
            tvt_ratio: Tuple[float, float, float] = (0.5, 0.0, 0.5),
            shuffle: bool = True,
            smote_strategy: float | str | None = 'auto',
            normalizer: str = 'z-score',
    ) -> None:
        """
        Transform the dataset

        :param tvt_ratio:       The ratio of (train_size : valid_size : test_size)
        :param shuffle:         The toggle for shuffling the dataset (will happen before the train-validate-test-split)
        :param smote_strategy:  The smote config (None for disable, other please refer to https://imbalanced-learn.org/stable/references/generated/imblearn.over_sampling.SMOTE.html)
        :param normalizer:      The normalization method (will perform appropriately if tt_ratio != (1, 0, 0))

        :return: None
        """

        dataset = self.dataset_df.copy()

        # shuffle the dataset in the first place
        if shuffle:
            dataset = dataset.sample(frac=1, random_state=self.seed)

        # split the train, valid, test dataset immediately, preventing information leak
        if sum(tvt_ratio) > 1.0:
            self.logger.error(role="Preprocessor", message=f"the tvt_ratio should be less or equal to 1.0, {sum(tvt_ratio)} instead")

        if tvt_ratio[0] <= 0:
            self.logger.error(role="Preprocessor", message=f"train dataset size should be greater than 0, {tvt_ratio[0]} instead")

        if tvt_ratio[1] < 0 or tvt_ratio[2] < 0:
            self.logger.error(role="Preprocessor", message=f"tvt_ratio should be non-negative")

        train_len, valid_len, test_len = list(map(lambda x: int(x * len(dataset)), tvt_ratio))
        train_df = dataset.iloc[:train_len, :]
        valid_df = dataset.iloc[train_len:(train_len + valid_len), :] if valid_len else None
        test_df = dataset.iloc[(train_len + valid_len):, :] if test_len else None

        # SMOTE for train, valid, test separately therefore preventing information leak and shuffle immediately
        if smote_strategy is not None:
            smote = SMOTE(sampling_strategy=smote_strategy, random_state=self.seed)
            train_x, train_y = smote.fit_resample(train_df.iloc[:, :-1], train_df.iloc[:, -1])
            train_df = pd.concat((train_x, train_y), axis=1)
            if shuffle:
                train_df = train_df.sample(frac=1, random_state=self.seed)

        total_df = pd.concat(list(filter(lambda x: x is not None, [train_df, valid_df, test_df])), axis=0)

        # normalization for all signal columns without information leak
        if normalizer in ("z-score", "min-max"):
            ct = ColumnTransformer(
                transformers=[
                    ("scaler", StandardScaler() if normalizer == "z-score" else MinMaxScaler(), train_df.columns[:-1])
                ],
                remainder="passthrough"
            )
            ct.fit(train_df)

            total_df = pd.DataFrame(ct.transform(total_df), index=total_df.index, columns=total_df.columns)
            train_df = pd.DataFrame(ct.transform(train_df), index=train_df.index, columns=train_df.columns)
            valid_df = pd.DataFrame(
                ct.transform(valid_df),
                index=valid_df.index,
                columns=valid_df.columns
            ) if valid_df is not None else None
            test_df = pd.DataFrame(
                ct.transform(test_df),
                index=test_df.index,
                columns=test_df.columns
            ) if test_df is not None else None
            self.dataset_df = pd.DataFrame(
                ct.transform(self.dataset_df),
                index=self.dataset_df.index,
                columns=self.dataset_df.columns
            )

        self.core["total_df_ori"] = self.dataset_df.copy()
        self.core["total_df_aft"] = total_df
        self.core["train_df_aft"] = train_df
        self.core["valid_df_aft"] = valid_df
        self.core["test_df_aft"] = test_df

    def _get(self, key: str, shard: str = "all"):
        """
        Fetch one stored frame of the core, or its features / labels

        :param key:     The core key of the frame
        :param shard:   "all" for the whole frame, "x" for the signal columns, "y" for the label column

        :return: The requested frame / series (None when the frame does not exist)
        """

        frame = self.core[key]

        if frame is None:
            return None
        elif shard == "all":
            return frame
        elif shard == "x":
            return frame.iloc[:, :-1]
        elif shard == "y":
            return frame.iloc[:, -1]
        else:
            self.logger.error(role="Preprocessor", message=f"Unknown shard: {shard}")
            return None

    def get_origin_total(self, shard="all"):
        return self._get("total_df_ori", shard=shard)

    def get_after_total(self, shard="all"):
        return self._get("total_df_aft", shard=shard)

    def get_train(self, shard="all"):
        return self._get("train_df_aft", shard=shard)

    def get_valid(self, shard="all"):
        return self._get("valid_df_aft", shard=shard)

    def get_test(self, shard="all"):
        return self._get("test_df_aft", shard=shard)

    def to_dict(self):
        config = {
            "beh_path": str(self.beh_path),
            "sig_path": str(self.sig_path),
            "tvt_ratio": self.tvt_ratio,
            "shuffle": self.shuffle,
            "normalizer": self.normalizer,
            "agg": self.agg,
            "smote": self.smote,
            "resize": self.resize,
            "target": self.target,
            "seed": self.seed,
        }

        return config

    def json(self, path: str | os.PathLike | Path) -> None:
        with Path(path).open("w") as fp:
            json.dump(self.to_dict(), fp)

        self.logger.debug(role="Preprocessor", message=f"preprocessor object saved to {path}")
