import typing as t
from itertools import tee

import numpy as np

A = t.TypeVar('A')
B = t.TypeVar('B')


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
        raise TypeError(f'Input type is not supported: failed to convert type {type(a)} into an array due to {e}')


def get_duplicates(it: t.Iterable[A]) -> t.Iterator[A]:
    seen = []
    for x in it:
        if x in seen:
            seen.append(x)
            yield x




if __name__ == '__main__':
    raise RuntimeError
