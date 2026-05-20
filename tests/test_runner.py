from llm_eval import Dataset, Runner, ExactMatch

def test_runner_sync_pipeline():
    dataset = Dataset.from_list([
        {"input": "hello", "expected": "HELLO"},
    ])
    runner = Runner(llm_fn=lambda x: x.upper(), scorers=[ExactMatch()])
    results = runner.run(dataset)
    assert results.cases[0].output == "HELLO"
    assert results.cases[0].scores["exact_match"] == 1.0


async def test_runner_async_pipeline():
    dataset = Dataset.from_list([
        {"input": "hello", "expected": "HELLO"},
    ])
    runner = Runner(llm_fn=lambda x: x.upper(), scorers=[ExactMatch()])
    results = await runner.run_async(dataset)
    assert results.cases[0].output == "HELLO"
    assert results.cases[0].scores["exact_match"] == 1.0

def flaky_pipeline(text):
    if text == "bad":
        raise ValueError("intentional failure")
    return text.upper()

def test_runner_pipeline_failure():
    dataset = Dataset.from_list([
        {"input": "hello", "expected": "HELLO"},
        {"input": "bad", "expected": "BAD"},
    ])
    runner = Runner(llm_fn=flaky_pipeline, scorers=[ExactMatch()])
    results = runner.run(dataset)
    assert results.cases[0].scores["exact_match"] == 1.0
    assert results.cases[1].scores["exact_match"] is None
