"""Deterministic, named random-number streams."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from numpy.random import Generator, SeedSequence


def _namespace_words(namespace: str) -> list[int]:
    if not namespace:
        raise ValueError("namespace must not be empty")
    digest = hashlib.sha256(namespace.encode("utf-8")).digest()
    return [int.from_bytes(digest[i : i + 4], "little") for i in range(0, 16, 4)]


@dataclass(frozen=True, slots=True)
class RngFactory:
    """Create stable streams without relying on global NumPy state.

    A stream is derived from ``root_seed`` and its name.  Consequently adding
    another stream or asking for streams in a different order does not alter
    existing sequences.
    """

    root_seed: int

    def __post_init__(self) -> None:
        if self.root_seed < 0:
            raise ValueError("root_seed must be >= 0")

    def stream(self, namespace: str) -> Generator:
        seed_sequence = SeedSequence([self.root_seed, *_namespace_words(namespace)])
        return np.random.default_rng(seed_sequence)
