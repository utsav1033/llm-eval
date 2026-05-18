import typer
from typing import Optional

app = typer.Typer(
    name="llm-eval",
    help="Local-first eval framework for LLM applications.",
    add_completion=False,
)


def version_callback(value: bool):
    if value:
        typer.echo("llm-eval 0.1.0")
        raise typer.Exit()


@app.callback()
def main(
    version: Optional[bool] = typer.Option(
        None, "--version", "-v",
        callback=version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
):
    """llm-eval — run and score LLM evaluations locally."""


if __name__ == "__main__":
    app()
