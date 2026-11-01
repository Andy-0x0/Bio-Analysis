import os
from typing import Any, Callable, Iterable, List

from joblib import Parallel, delayed

from .logger import ABlockLogger


def parallel_map(
    funct: Callable,
    items: Iterable[Any],
    n_workers: int  = 1,
    role: str       = "parallel_map",
    log_level: str  = "INFO",
) -> List[Any]:
    """
    Apply a function to every item, on worker processes when more than one worker is requested; the results keep the
    order of the items, and a failing item is logged and reported as None instead of aborting the whole batch

    :param funct:       The function applied to every item (must be picklable when `n_workers` > 1)
    :param items:       The items
    :param n_workers:   The number of worker processes (<= 0 for every CPU)
    :param role:        The logging role of the batch
    :param log_level:   The print level for logging

    :return: The list of results (None where the item failed)
    """

    logger = ABlockLogger(level=log_level)
    items = list(items)
    n_workers = os.cpu_count() if n_workers <= 0 else min(n_workers, max(len(items), 1))

    def guarded(item: Any) -> Any:
        try:
            return funct(item)
        except Exception as e:
            ABlockLogger(level=log_level).warning(role=role, message=f"{item} failed ({str(e).strip()})")
            return None

    # the sequential route keeps everything in the caller's process (easier debugging, identical results)
    if n_workers <= 1:
        return [guarded(item) for item in items]

    logger.info(role=role, message=f"dispatching {len(items)} items onto {n_workers} workers")

    return Parallel(n_jobs=n_workers, backend="loky")(delayed(guarded)(item) for item in items)


def share_threads(n_workers: int = 1) -> int:
    """
    Decide how many threads every worker may use, so that the workers together do not oversubscribe the CPUs

    :param n_workers: The number of worker processes (<= 0 for every CPU)

    :return: The number of threads per worker
    """

    n_cpu = os.cpu_count() or 1
    n_workers = n_cpu if n_workers <= 0 else n_workers

    return max(1, n_cpu // n_workers)
