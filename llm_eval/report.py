from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import statistics
from pathlib import Path
from rich.console import Console
from rich.table import Table


@dataclass
class CaseResult:
    input: str
    output: str
    expected: Optional[str]
    scores: dict[str, Optional[float]]


@dataclass
class Results:
    cases: list[CaseResult]
    scorer_names: list[str]
    metadata: dict = field(default_factory=dict)

    def aggregate(self, scorer: str) -> dict:
        values = [c.scores[scorer] for c in self.cases if c.scores.get(scorer) is not None]
        if not values:
            return {"mean": None, "p50": None, "p95": None, "n": 0}
        return {
            "mean": statistics.mean(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 95),
            "n": len(values),
        }

    def print(self) -> None:
        console = Console()
        table = Table(show_lines=False)
        table.add_column("#", style="dim", width=3)
        table.add_column("Input", max_width=40, overflow="ellipsis")
        table.add_column("Output", max_width=40, overflow="ellipsis")
        for name in self.scorer_names:
            table.add_column(name)

        for i, c in enumerate(self.cases, start=1):
            row = [str(i), c.input, c.output]
            for name in self.scorer_names:
                s = c.scores.get(name)
                row.append(f"{s:.2f}" if s is not None else "—")
            table.add_row(*row)

        console.print(table)

        # Aggregates
        console.print("\n[bold]Aggregates[/bold]")
        for name in self.scorer_names:
            agg = self.aggregate(name)
            if agg["n"] == 0:
                console.print(f"  {name}: no valid scores")
                continue
            console.print(
                f"  {name}: mean={agg['mean']:.3f} "
                f"p50={agg['p50']:.3f} p95={agg['p95']:.3f} "
                f"n={agg['n']}"
            )

    def save(self, path: str | Path) -> None:
        data = {
            "scorer_names": self.scorer_names,
            "metadata": self.metadata,
            "cases": [asdict(c) for c in self.cases],
        }
        Path(path).write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Results":
        data = json.loads(Path(path).read_text())
        cases = [CaseResult(**c) for c in data["cases"]]
        return cls(
            cases=cases,
            scorer_names=data["scorer_names"],
            metadata=data.get("metadata", {}),
        )


def _percentile(values: list[float], p: float) -> float:
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)