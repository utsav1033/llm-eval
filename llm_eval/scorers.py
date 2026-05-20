from abc import ABC, abstractmethod
from typing import Optional
# import numpy as np


class Scorer(ABC):
    """A scorer takes an output and (optionally) an expected answer
    and returns a float between 0.0 and 1.0."""

    name: str

    @abstractmethod
    async def score(self, output: str, expected: Optional[str]) -> float:
        ...


class ExactMatch(Scorer):
    name = "exact_match"

    async def score(self, output: str, expected: Optional[str]) -> float:
        if expected is None:
            raise ValueError("ExactMatch requires an expected answer")
        return 1.0 if output.strip() == expected.strip() else 0.0


