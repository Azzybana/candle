from typing import TypeVar, Union, Sequence

_T = TypeVar("_T")

_ArrayLike = Union[
    _T,
    Sequence[_T],
    Sequence[Sequence[_T]],
    Sequence[Sequence[Sequence[_T]]],
    Sequence[Sequence[Sequence[Sequence[_T]]]],
]

CPU: str = "cpu"

Device = TypeVar("Device", CPU)

Scalar = Union[int, float]

Index = Union[int, slice, None, "Ellipsis"]

Shape = Union[int, Sequence[int]]
