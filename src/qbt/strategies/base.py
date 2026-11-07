"""Strategy interface and registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields
from typing import Any, ClassVar, TypeVar

import pandas as pd

S = TypeVar("S", bound="type[Strategy]")

REGISTRY: dict[str, type[Strategy]] = {}


def register(key: str):
    """Class decorator that makes a strategy available to the CLI by ``key``."""

    def wrap(cls: S) -> S:
        if key in REGISTRY:
            raise ValueError(f"strategy {key!r} already registered")
        cls.key = key
        REGISTRY[key] = cls
        return cls

    return wrap


def get_strategy(key: str) -> type[Strategy]:
    try:
        return REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(REGISTRY))
        raise KeyError(f"unknown strategy {key!r}; available: {known}") from None


@dataclass
class Strategy(ABC):
    """Base class for strategies.

    A strategy is a dataclass of parameters plus a pure function from a wide
    price frame to a frame of target weights with the same shape. Row ``t``
    may only depend on prices up to and including bar ``t``; the simulator
    takes care of execution timing. NaN weights mean "no position yet".

    Subclasses should validate parameters in ``__post_init__`` and keep
    ``generate_weights`` free of side effects so grid searches can reuse it.
    """

    key: ClassVar[str] = ""
    title: ClassVar[str] = ""
    description: ClassVar[str] = ""

    @property
    def name(self) -> str:
        return self.key or type(self).__name__

    @property
    def params(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def param_names(cls) -> list[str]:
        return [f.name for f in fields(cls)]

    @property
    def warmup(self) -> int:
        """Bars of history needed before the first valid signal."""
        return 0

    @abstractmethod
    def generate_weights(self, prices: pd.DataFrame) -> pd.DataFrame:
        """Return target weights indexed like ``prices``."""

    def __str__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({args})"
