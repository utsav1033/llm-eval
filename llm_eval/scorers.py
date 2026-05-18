from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from openai import OpenAI
from anthropic import Anthropic


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


class EmbeddingSimilarity(Scorer):
    name = "embedding_similarity"

    def __init__(self, model: str = "text-embedding-3-small"):
        self.model = model
        self.client = OpenAI()

    async def score(self, output: str, expected: Optional[str]) -> float:
        if expected is None:
            raise ValueError("EmbeddingSimilarity requires an expected answer")
        # NOTE: openai sync SDK; wrap in to_thread for true async
        import asyncio
        resp = await asyncio.to_thread(
            self.client.embeddings.create,
            model=self.model,
            input=[output, expected],
        )
        v1 = np.array(resp.data[0].embedding)
        v2 = np.array(resp.data[1].embedding)
        return float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))


class LLMJudge(Scorer):
    name = "llm_judge"

    def __init__(
        self,
        rubric: str,
        model: str = "claude-sonnet-4-5",
        scale: int = 5,
    ):
        self.rubric = rubric
        self.model = model
        self.scale = scale
        self.client = Anthropic()

    async def score(self, output: str, expected: Optional[str]) -> float:
        import asyncio
        prompt = self._build_prompt(output, expected)
        resp = await asyncio.to_thread(
            self.client.messages.create,
            model=self.model,
            max_tokens=100,
            messages=[{"role": "user", "content": prompt}],
        )
        text = resp.content[0].text.strip()
        score = self._parse_score(text)
        return score / self.scale  # normalize to 0..1

    def _build_prompt(self, output: str, expected: Optional[str]) -> str:
        expected_section = f"\nExpected answer:\n{expected}\n" if expected else ""
        return (
            f"You are a grader. Rubric:\n{self.rubric}\n\n"
            f"Output to grade:\n{output}\n"
            f"{expected_section}"
            f"Respond with ONLY a single integer from 1 to {self.scale}. "
            f"No explanation, no other text."
        )

    def _parse_score(self, text: str) -> float:
        # Pull the first integer out of the response, clamp to scale
        import re
        match = re.search(r"\d+", text)
        if not match:
            raise ValueError(f"Could not parse score from: {text!r}")
        n = int(match.group())
        return max(1, min(self.scale, n))