# llm-eval

Local-first evals for LLM apps. No server, no signup, just a library.

---

## Why

When you change your prompt or swap models, you have no idea if things got better or worse. llm-eval gives you a repeatable way to measure that — run your pipeline against test cases, score the outputs, and compare runs over time.

---

## Install

```
pip install git+https://github.com/utsavv/llm-eval
```

Or clone and install locally:

```
git clone https://github.com/utsavv/llm-eval
cd llm-eval
pip install -e ".[dev]"
```

---

## Quickstart

**1. Write your test cases** (`tests.jsonl`):

```jsonl
{"input": "My card got charged twice", "expected": "billing"}
{"input": "App crashes on upload", "expected": "technical"}
{"input": "I can't log in", "expected": "account"}
```

**2. Write your config** (`eval_config.py`):

```python
from llm_eval import ExactMatch

def pipeline(text):
    # call your LLM here, return a string
    return my_llm(text)

scorers = [ExactMatch()]
dataset_path = "tests.jsonl"
```

**3. Run it:**

```
llm-eval run eval_config.py
```

**4. Save and compare runs:**

```
llm-eval run eval_config.py --output run1.json
# make changes to your pipeline
llm-eval run eval_config.py --output run2.json
llm-eval compare run1.json run2.json
```

---

## Scorers

| Scorer | What it checks | Needs |
|---|---|---|
| `ExactMatch` | Output matches expected exactly | Nothing |
| `EmbeddingSimilarity` | Output is semantically close to expected | OpenAI API key |
| `LLMJudge` | Another LLM grades output on a rubric | Anthropic API key |

---

## The config file

Your `eval_config.py` must define three things:

```python
pipeline      # function (str) -> str
scorers       # list of scorer instances
dataset_path  # path to your .jsonl file
```

---

## Reading the results

- `mean` — average score across all cases
- `p50` — median, typical performance
- `p95` — 95th percentile, how bad your worst cases are
- `n` — number of cases scored

In the compare view, green deltas are improvements, red are regressions.

---

## Out of scope (v1)

Web UI, tracing, dataset versioning, cost tracking, PyPI publishing. See issues for the roadmap.

---

## License

MIT
