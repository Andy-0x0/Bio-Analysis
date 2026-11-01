import os
import re
import random
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from dotenv import dotenv_values

from .logger import ABlockLogger


def check_folders(*path_args: str) -> Callable:
    """
    The decorator generator that makes sure all dependent folders exist before the decorated function is called

    :param path_args: All dependent folders (the keyword names at function calling)

    :return: The decorated function
    """

    logger = ABlockLogger(level="INFO")

    def decorator(funct: Callable) -> Callable:

        def wrapper(*args, **kwargs) -> Any:
            # check and create the dependencies if needed
            for path_arg in path_args:
                path = kwargs.get(path_arg)

                if path is None:
                    continue

                try:
                    path = Path(path)
                except Exception as e:
                    logger.warning(role=f"f{funct.__name__}", message=f"invalid path at function calling: {path_arg}")
                else:
                    if not path.exists():
                        logger.info(role=f"f{funct.__name__}", message=f"Automatic created {path}")
                        path.mkdir(parents=True, exist_ok=True)

            # core function execution
            return funct(*args, **kwargs)

        wrapper.__name__ = funct.__name__
        wrapper.__doc__ = funct.__doc__

        return wrapper

    return decorator


def load_paths(env_path: None | str | os.PathLike | Path = None) -> Dict[str, Path]:
    """
    Load the project path constants from the dotenv file, so that every module and script resolves the same folders
    no matter where it is launched from

    :param env_path: The dotenv file; None looks for `config/.env.path` next to the `src` package, then `../config/.env.path`

    :return: The dict mapping every key of the dotenv file (ROOT_PATH, INIT_DATA_PATH, ...) to a Path
    """

    candidates = [Path(env_path)] if env_path is not None else [
        Path(__file__).resolve().parents[2] / Path("config/.env.path"),
        Path("../config/.env.path"),
        Path("config/.env.path"),
    ]

    # the first readable dotenv file wins
    for candidate in candidates:
        if candidate.is_file():
            return {key: Path(value) for key, value in dotenv_values(candidate).items() if value is not None}

    raise FileNotFoundError(f"no dotenv path file found among {[str(c) for c in candidates]}")


def fetch_tokens(
    folder: str | os.PathLike | Path,
    shard: str = "all"
) -> Tuple[str, ...]:
    """
    Fetch all valid tokens from a given folder (non-recursive)

    :param folder: The path of the folder
    :param shard: Which shard of the tokens to fetch ("all", "good", "other", "test{n}")

    :return: The sorted tuple of the tokens
    """

    logger = ABlockLogger(level="INFO")

    pattern = re.compile(r"^(K\d{3}_[AB]_[A-Za-z]\d)[_-][A-Za-z_]+$")

    matched_tokens = []

    for file in Path(folder).iterdir():
        if file.is_file():
            m = pattern.match(file.stem)
            if m:
                token = m.group(1)
                if token not in matched_tokens:
                    matched_tokens.append(token)

    good_tokens = [
        "K215_B_D2",
        "K215_B_D4",
        "K215_B_D8",

        "K223_A_D1",
        "K223_A_D4",
        "K223_A_D7",

        "K232_A_D1",  # Sparce cont&inv exclude at fitting cont&inv
        "K232_A_D4",
        "K232_A_D7",
        "K232_B_D2",
        "K232_B_D5",
        "K232_B_D8",

        "K228_B_D2",  # Sparce cont_inv exclude at fitting cont_inv
        "K228_B_D5",
        "K228_B_D8"
    ]
    good_tokens = set(good_tokens)

    if shard == "all":
        logger.debug(role="fetch_tokens", message="fetching all tokens")
        return tuple(sorted(matched_tokens))
    elif shard == "good":
        logger.debug(role="fetch_tokens", message="fetching good tokens only")
        return tuple(sorted(filter(lambda x: x in good_tokens, matched_tokens)))
    elif shard == "other":
        logger.debug(role="fetch_tokens", message="fetching other tokens only")
        return tuple(sorted(filter(lambda x: x not in good_tokens, matched_tokens)))
    elif shard.startswith("test"):
        if shard[4:].isnumeric():
            logger.debug(role="fetch_tokens", message="fetching test tokens only")
            return tuple(sorted(random.sample(matched_tokens, k=int(shard[4:]))))
        else:
            logger.error(role="fetch_tokens", message="invalid shard name when fetching tokens")
            raise RuntimeError(f"invalid shard name when fetching tokens")
    else:
        logger.error(role="fetch_tokens", message="invalid shard name when fetching tokens")
        raise RuntimeError(f"invalid shard name when fetching tokens")
