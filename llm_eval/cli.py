import importlib.util
import sys
from pathlib import Path
import typer
from rich.console import Console
from rich.table import Table
from .dataset import Dataset
from .runner import Runner
from .report import Results

app = typer.Typer(add_completion=False)
console = Console()


def _load_user_config(config_path: Path):
    spec = importlib.util.spec_from_file_location("user_eval_config", config_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load config file at {config_path}")
    module = importlib.util.module_from_spec(spec)
    # Add the config's parent dir to sys.path so the user's imports work
    sys.path.insert(0, str(config_path.parent.absolute()))
    spec.loader.exec_module(module)
    return module


@app.command()
def run(
    config: Path = typer.Argument(
        Path("eval_config.py"),
        help="Path to the eval config Python file.",
    ),
    output: Path = typer.Option(
        None,
        "--output", "-o",
        help="Optional path to save results as JSON.",
    ),
):
    """Run an eval using the given config file."""
    if not config.exists():
        console.print(f"[red]Config file not found:[/red] {config}")
        raise typer.Exit(1)

    module = _load_user_config(config)

    # Required attributes on the config module
    pipeline = getattr(module, "pipeline", None)
    scorers = getattr(module, "scorers", None)
    dataset_path = getattr(module, "dataset_path", None)

    if pipeline is None or scorers is None or dataset_path is None:
        console.print(
            "[red]Config must define:[/red] pipeline, scorers, dataset_path"
        )
        raise typer.Exit(1)

    dataset = Dataset.from_jsonl(dataset_path)
    runner = Runner(llm_fn=pipeline, scorers=scorers)

    console.print(f"Running {len(dataset)} cases with {len(scorers)} scorer(s)...")
    results = runner.run(dataset)
    results.print()

    if output:
        results.save(output)
        console.print(f"\nResults saved to {output}")


@app.command()
def compare(
    run_a: Path = typer.Argument(..., help="Earlier run JSON."),
    run_b: Path = typer.Argument(..., help="Later run JSON."),
):
    """Compare two saved runs, surfacing regressions."""
    a = Results.load(run_a)
    b = Results.load(run_b)

    if len(a.cases) != len(b.cases):
        console.print(
            f"[yellow]Warning:[/yellow] case counts differ "
            f"({len(a.cases)} vs {len(b.cases)}). "
            f"Comparing by index up to min."
        )

    n = min(len(a.cases), len(b.cases))
    common_scorers = [s for s in a.scorer_names if s in b.scorer_names]
    if not common_scorers:
        console.print("[red]No common scorers between the two runs.[/red]")
        raise typer.Exit(1)

    for scorer in common_scorers:
        console.print(f"\n[bold]{scorer}[/bold]")
        table = Table()
        table.add_column("#", style="dim", width=3)
        table.add_column("Input", max_width=40, overflow="ellipsis")
        table.add_column("Run A")
        table.add_column("Run B")
        table.add_column("Δ")

        regressions = 0
        improvements = 0
        for i in range(n):
            sa = a.cases[i].scores.get(scorer)
            sb = b.cases[i].scores.get(scorer)
            if sa is None or sb is None:
                delta_str = "—"
            else:
                delta = sb - sa
                if delta > 0.01:
                    delta_str = f"[green]+{delta:.2f}[/green]"
                    improvements += 1
                elif delta < -0.01:
                    delta_str = f"[red]{delta:.2f}[/red]"
                    regressions += 1
                else:
                    delta_str = f"{delta:+.2f}"
            table.add_row(
                str(i + 1),
                a.cases[i].input,
                f"{sa:.2f}" if sa is not None else "—",
                f"{sb:.2f}" if sb is not None else "—",
                delta_str,
            )
        console.print(table)

        agg_a = a.aggregate(scorer)
        agg_b = b.aggregate(scorer)
        console.print(
            f"  Mean: {agg_a['mean']:.3f} → {agg_b['mean']:.3f} "
            f"({agg_b['mean'] - agg_a['mean']:+.3f})"
        )
        console.print(f"  Regressions: {regressions} | Improvements: {improvements}")