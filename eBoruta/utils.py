import logging
import sys
import typing as t
from itertools import tee
from pathlib import Path

import numpy as np

A = t.TypeVar("A")
B = t.TypeVar("B")


def zip_partition(
    pred: t.Callable[[B], bool], a: t.Iterable[A], b: t.Iterable[B]
) -> t.Tuple[t.Iterator[A], t.Iterator[A]]:
    t1, t2 = tee((pred(y), x) for x, y in zip(a, b))
    return (
        (x for (cond, x) in t1 if not cond),
        (x for (cond, x) in t2 if cond),
    )


def convert_to_array(a):
    try:
        return np.array(a)
    except Exception as e:
        raise TypeError(
            f"Input type is not supported: failed to convert type {type(a)} into an array due to {e}"
        )


def get_duplicates(it: t.Iterable[A]) -> t.Iterator[A]:
    seen = []
    for x in it:
        if x in seen:
            seen.append(x)
            yield x


def setup_logger(
    log_path: t.Optional[t.Union[str, Path]] = None,
    file_level: t.Optional[int] = None,
    stdout_level: t.Optional[int] = None,
    stderr_level: t.Optional[int] = None,
    logger: t.Optional[logging.Logger] = None,
) -> logging.Logger:
    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s [%(module)s--%(funcName)s]: %(message)s"
    )
    if logger is None:
        logger = logging.getLogger(__name__)

    if log_path is not None:
        level = file_level or logging.DEBUG
        handler = logging.FileHandler(log_path, "w")
        handler.setFormatter(formatter)
        handler.setLevel(level)
        logger.addHandler(handler)
    if stderr_level is not None:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(formatter)
        handler.setLevel(stderr_level)
        logger.addHandler(handler)
    if stdout_level is not None:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(formatter)
        handler.setLevel(stdout_level)
        logger.addHandler(handler)

    return logger


if __name__ == "__main__":
    raise RuntimeError
