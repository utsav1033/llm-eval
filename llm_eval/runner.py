import asyncio
import inspect
from dataclasses import dataclass
from typing import Callable, Awaitable
from .dataset import Dataset, TestCase
from .scorers import Scorer
from .report import Results, CaseResult


PipelineFn = Callable[[str], str] | Callable[[str], Awaitable[str]]


@dataclass
class Runner:
    llm_fn: PipelineFn
    scorers: list[Scorer]
    concurrency: int = 10

    async def _run_case(
        self,
        case: TestCase,
        sem: asyncio.Semaphore,
    ) -> CaseResult:
        async with sem:
            try:
                if inspect.iscoroutinefunction(self.llm_fn):
                    output = await self.llm_fn(case.input)
                else:
                    output = await asyncio.to_thread(self.llm_fn, case.input)
            except Exception:
                return CaseResult(
                    input=case.input,
                    output="",
                    expected=case.expected,
                    scores={s.name: None for s in self.scorers},
                )

            scores = {}
            for scorer in self.scorers:
                try:
                    scores[scorer.name] = await scorer.score(output, case.expected)
                except Exception as e:
                    scores[scorer.name] = None  # mark as failed, don't crash whole run
            return CaseResult(
                input=case.input,
                output=output,
                expected=case.expected,
                scores=scores,
            )

    async def run_async(self, dataset: Dataset) -> Results:
        sem = asyncio.Semaphore(self.concurrency)
        tasks = [self._run_case(c, sem) for c in dataset]
        case_results = await asyncio.gather(*tasks)
        return Results(
            cases=case_results,
            scorer_names=[s.name for s in self.scorers],
        )

    def run(self, dataset: Dataset) -> Results:
        """Sync wrapper around run_async for convenience."""
        return asyncio.run(self.run_async(dataset))

