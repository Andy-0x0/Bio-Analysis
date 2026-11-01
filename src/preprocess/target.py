from dataclasses import dataclass
from typing import Callable, Dict, FrozenSet

import numpy as np
import pandas as pd


# the annotation names grouped by what the mouse (or the rig) did
NAMES_SUCCESS   = frozenset({"cont reach", "cont reaching"})
NAMES_MISSED    = frozenset({"cont missed", "cont miss"})
NAMES_INVAIN    = frozenset({"cont in vain"})
NAMES_WALK      = frozenset({"walk"})
NAMES_BOTH      = frozenset({"bothreach", "both reach", "both reaching"})
NAMES_BOTH_INV  = frozenset({"both in vain", "both invain", "both paw in vain"})
NAMES_CUE       = frozenset({"valve opened", "spout movement", "spout moved after whisker detection", "spout movement after whisker detection"})
NAMES_ATTEMPT   = NAMES_SUCCESS | NAMES_MISSED | NAMES_INVAIN

# the gaps (in seconds) that split the derived targets off their parent annotation
CUE_FOLLOW      = 5.0   # a cue counts as "none" when no contralateral attempt starts within this many seconds
INVAIN_NEAR     = 3.0   # an in-vain reach is "near" when a successful reach ended at most this many seconds before
INVAIN_ISOLATED = 8.0   # an in-vain reach is "isolated" when no successful reach ended within this many seconds before


def normalize_name(name: str) -> str:
    """
    Normalize an annotation name: lower case, single spaces, and the trailing repetition counter "(n)" removed

    :param name: The raw annotation text

    :return: The normalized name
    """

    key = " ".join(str(name).strip().lower().split())

    return key[:key.index("(")].strip() if "(" in key else key


@dataclass(frozen=True)
class Target:
    """
    One binary target on the behavior table: the annotation names labelled positive, plus an optional rule picking
    which of those rows actually count (None keeps them all)

    The rule receives the whole behavior table (columns `name`, `start`, `end`, in seconds, time-sorted) and
    returns the boolean mask of the rows to keep
    """
    names:  FrozenSet[str]
    rule:   None | Callable[[pd.DataFrame], np.ndarray] = None

    def mask(self, table: pd.DataFrame) -> np.ndarray:
        """
        Mark the rows of the behavior table that are positive for this target

        :param table: The behavior table with columns `name`, `start`, `end`

        :return: The boolean mask of the positive rows
        """

        hit = table["name"].isin(self.names).to_numpy()

        return hit if self.rule is None else hit & self.rule(table)


def _previous(table: pd.DataFrame, names: FrozenSet[str], column: str) -> np.ndarray:
    """
    For every row, the value of `column` of the latest row of the given names that started strictly before the row

    :param table:   The time-sorted behavior table
    :param names:   The annotation names of the reference rows
    :param column:  The column whose value is looked up ("start" or "end")

    :return: The looked-up values (nan when no reference row precedes)
    """

    reference = table.loc[table["name"].isin(names)].sort_values("start")
    position = np.searchsorted(reference["start"].to_numpy(), table["start"].to_numpy(), side="left") - 1

    values = np.full(len(table), np.nan)
    values[position >= 0] = reference[column].to_numpy()[position[position >= 0]]

    return values


def _cue_followed_by_nothing(table: pd.DataFrame) -> np.ndarray:
    """
    The cues after which no contralateral attempt (success / missed / in vain) starts within CUE_FOLLOW seconds
    """

    attempts = np.sort(table.loc[table["name"].isin(NAMES_ATTEMPT), "start"].to_numpy())
    starts = table["start"].to_numpy()

    following = np.searchsorted(attempts, starts, side="left")
    gap = np.where(following < len(attempts), attempts[np.minimum(following, len(attempts) - 1)] - starts, np.inf)

    return gap > CUE_FOLLOW


def _invain_near(table: pd.DataFrame) -> np.ndarray:
    """
    The in-vain reaches starting at most INVAIN_NEAR seconds after the end of a successful reach
    """

    gap = table["start"].to_numpy() - _previous(table, NAMES_SUCCESS, "end")

    return np.nan_to_num(gap, nan=np.inf) <= INVAIN_NEAR


def _invain_isolated(table: pd.DataFrame) -> np.ndarray:
    """
    The in-vain reaches with no successful reach ending within INVAIN_ISOLATED seconds before
    """

    gap = table["start"].to_numpy() - _previous(table, NAMES_SUCCESS, "end")

    return np.nan_to_num(gap, nan=np.inf) >= INVAIN_ISOLATED


# the registry of every binary target the preprocessors understand (the key is the `beh` / `target` argument)
TARGETS: Dict[str, Target] = {
    # the repository's historical targets
    "cont":                 Target(names=NAMES_SUCCESS),
    "cont_inv":             Target(names=NAMES_INVAIN),
    "cont&inv":             Target(names=NAMES_SUCCESS | NAMES_INVAIN),
    "both":                 Target(names=NAMES_BOTH),
    "both_inv":             Target(names=NAMES_BOTH_INV),
    "both&inv":             Target(names=NAMES_BOTH | NAMES_BOTH_INV),
    "walk":                 Target(names=NAMES_WALK),

    # the targets of the success / missed / in vain / cue comparison
    "cont_missed":          Target(names=NAMES_MISSED),
    "cue":                  Target(names=NAMES_CUE),
    "cue_none":             Target(names=NAMES_CUE, rule=_cue_followed_by_nothing),
    "cont_inv_near":        Target(names=NAMES_INVAIN, rule=_invain_near),
    "cont_inv_isolated":    Target(names=NAMES_INVAIN, rule=_invain_isolated),
}


# the display names of the targets on the plots
LABELS: Dict[str, str] = {
    "cont":                 "success",
    "cont_inv":             "in vain",
    "cont&inv":             "success + in vain",
    "both":                 "both-paw reach",
    "both_inv":             "both-paw in vain",
    "both&inv":             "both-paw reach + in vain",
    "walk":                 "walk",
    "cont_missed":          "missed",
    "cue":                  "cue",
    "cue_none":             "cue (no attempt)",
    "cont_inv_near":        "in vain (near)",
    "cont_inv_isolated":    "in vain (isolated)",
}
