from llm_eval import Results
from llm_eval.report import CaseResult


def _make_results(score, scorer="exact_match"):
    return Results(
        cases=[CaseResult(input="hi", output="x", expected="y", scores={scorer: score})],
        scorer_names=[scorer],
    )


def test_compare_improvement(tmp_path):
    path_a = tmp_path / "run_a.json"
    path_b = tmp_path / "run_b.json"
    _make_results(0.0).save(path_a)
    _make_results(1.0).save(path_b)

    a = Results.load(path_a)
    b = Results.load(path_b)
    delta = b.cases[0].scores["exact_match"] - a.cases[0].scores["exact_match"]
    assert delta == 1.0


def test_compare_regression(tmp_path):
    path_a = tmp_path / "run_a.json"
    path_b = tmp_path / "run_b.json"
    _make_results(1.0).save(path_a)
    _make_results(0.0).save(path_b)

    a = Results.load(path_a)
    b = Results.load(path_b)
    delta = b.cases[0].scores["exact_match"] - a.cases[0].scores["exact_match"]
    assert delta == -1.0


def test_compare_no_change(tmp_path):
    path_a = tmp_path / "run_a.json"
    path_b = tmp_path / "run_b.json"
    _make_results(1.0).save(path_a)
    _make_results(1.0).save(path_b)

    a = Results.load(path_a)
    b = Results.load(path_b)
    delta = b.cases[0].scores["exact_match"] - a.cases[0].scores["exact_match"]
    assert delta == 0.0
