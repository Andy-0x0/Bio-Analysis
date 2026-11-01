import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd

from ..util import ABlockLogger, load_paths
from .target import TARGETS, NAMES_SUCCESS, NAMES_CUE, normalize_name


# import path consts
PATHS = load_paths()
PATH_ROOT = PATHS["ROOT_PATH"]          # path of this project directory
PATH_DATA = PATHS["INIT_DATA_PATH"]     # path of the initialization data directory
PATH_MEMO = PATHS["MEMO_DATA_PATH"]     # path of the memory data directory


class EventPreprocessor:
    def __init__(
        self,
        beh_path: str | os.PathLike | Path,
        sig_path: str | os.PathLike | Path,
        log_level: str = "INFO",
    ) -> None:
        """
        The event-level preprocessor: the signal stays at its native ~1 Hz rate (z-scored per cell over the whole
        recording, label-free) and every annotation becomes one event with its context (latency to the previous cue,
        gap since the previous successful reach), so that the windows around the events of any target can be cut
        without the 10x interpolation of the frame-level preprocessor

        :param beh_path:    The path to the behavior file
        :param sig_path:    The path to the signal file
        :param log_level:   The print level for logging
        """

        self.logger = ABlockLogger(level=log_level)

        self.sig_fps = 1 / (1.000045 if Path(sig_path).stem.split('_')[0] == 'K215' else 1.001286)
        self.beh_fps = 7.06

        # the native signal: one row per two-photon frame, z-scored per cell
        signal_df = pd.read_csv(sig_path)
        if 'x0000' in signal_df.columns:
            signal_df = signal_df.set_index('x0000')

        self.cells = signal_df.columns.to_list()
        self.time = signal_df.index.to_numpy(dtype=float) / self.sig_fps
        values = signal_df.to_numpy(dtype=float)
        self.x = ((values - values.mean(axis=0)) / (values.std(axis=0) + 1e-6)).astype(np.float32)

        # the event table: every annotation in seconds (same 0.5 s offset as the frame-level preprocessor)
        behavior_df = pd.read_csv(beh_path).loc[:, ['Behavior', 'Start', 'End']].dropna()
        self.events = pd.DataFrame({
            "name": behavior_df['Behavior'].map(normalize_name),
            "start": (behavior_df['Start'].to_numpy(dtype=float) + 0.5 * self.beh_fps) / self.beh_fps,
            "end": (behavior_df['End'].to_numpy(dtype=float) + 0.5 * self.beh_fps) / self.beh_fps,
        }).sort_values("start", kind="stable").reset_index(drop=True)

        # the context of every event: seconds since the previous cue, seconds since the previous success ended
        self.events["latency"] = self.events["start"] - self._previous(NAMES_CUE, "start")
        self.events["gap"] = self.events["start"] - self._previous(NAMES_SUCCESS, "end")

        self.logger.info(
            role="EventPreprocessor",
            message=f"Config: samples[{len(self.time)}] | cells[{len(self.cells)}] | events[{len(self.events)}]"
        )

    def _previous(self, names, column: str) -> np.ndarray:
        """
        For every event, the value of `column` of the latest event of the given names starting strictly before it

        :param names:   The annotation names of the reference events
        :param column:  The column whose value is looked up

        :return: The looked-up values (nan when no reference event precedes)
        """

        reference = self.events.loc[self.events["name"].isin(names)]
        position = np.searchsorted(reference["start"].to_numpy(), self.events["start"].to_numpy(), side="left") - 1

        values = np.full(len(self.events), np.nan)
        values[position >= 0] = reference[column].to_numpy()[position[position >= 0]]

        return values

    def select(self, target: str) -> pd.DataFrame:
        """
        The events positive for a target

        :param target: The binary target, one key of `src.preprocess.TARGETS`

        :return: The sub-table of the positive events
        """

        if target not in TARGETS:
            self.logger.error(role="EventPreprocessor", message=f"unknown target when selecting the events: {target}")
            raise RuntimeError(f"unknown target when selecting the events: {target}")

        return self.events.loc[TARGETS[target].mask(self.events)]

    def windows(
        self,
        target: str,
        offsets: np.ndarray = np.arange(-2.0, 4.0 + 1e-6, 1.0),
    ) -> Tuple[pd.DataFrame, np.ndarray]:
        """
        Cut the native-rate window around the onset of every event of a target

        :param target:  The binary target, one key of `src.preprocess.TARGETS`
        :param offsets: The sampling offsets of the window relative to the onset, in seconds

        :return: The (events, windows) pair, windows shaped (events, cells, offsets); the events whose window exceeds
                 the recording are dropped from both
        """

        events = self.select(target)
        starts = events["start"].to_numpy()

        # only the windows fully inside the recording are kept
        inside = (starts + offsets[0] >= self.time[0]) & (starts + offsets[-1] <= self.time[-1])
        events, starts = events.loc[inside], starts[inside]

        if len(starts) == 0:
            return events, np.zeros((0, len(self.cells), len(offsets)), dtype=np.float32)

        # linear interpolation of every cell at the requested seconds
        sample_times = starts[:, None] + offsets[None, :]
        windows = np.stack([np.interp(sample_times, self.time, self.x[:, c]) for c in range(len(self.cells))], axis=1)

        return events, windows.astype(np.float32)
