from llm_eval import Dataset


def test_dataset_length(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text(
        '{"input": "hello", "expected": "world"}\n'
        '{"input": "foo", "expected": "bar"}\n'
        '{"input": "baz", "expected": "qux"}\n'
    )
    dataset = Dataset.from_jsonl(f)
    assert len(dataset) == 3
    assert dataset.cases[0].input == "hello"
    assert dataset.cases[0].expected == "world"


def test_dataset_missing_expected(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"input": "hello"}\n')
    dataset = Dataset.from_jsonl(f)
    assert dataset.cases[0].expected is None


def test_dataset_skips_blank_lines(tmp_path):
    f = tmp_path / "test.jsonl"
    f.write_text('{"input": "hi"}\n\n{"input": "bye"}\n')
    dataset = Dataset.from_jsonl(f)
    assert len(dataset) == 2


def test_dataset_from_list():
    dataset = Dataset.from_list([
        {"input": "hello", "expected": "world"},
        {"input": "foo"},
    ])
    assert len(dataset) == 2
    assert dataset.cases[0].input == "hello"
    assert dataset.cases[1].expected is None
