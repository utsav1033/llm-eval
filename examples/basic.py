from llm_eval import Dataset, Runner, ExactMatch


def pipeline(text):
    text = text.lower()
    if any(w in text for w in ["charge", "bill", "refund", "payment"]):
        return "billing"
    if any(w in text for w in ["login", "password", "account"]):
        return "account"
    if any(w in text for w in ["crash", "error", "bug", "upload"]):
        return "technical"
    if any(w in text for w in ["add", "feature", "dark mode", "would"]):
        return "feature_request"
    return "complaint"


dataset = Dataset.from_jsonl("examples/tests.jsonl")
runner = Runner(llm_fn=pipeline, scorers=[ExactMatch()])
results = runner.run(dataset)
results.print()
results.save("examples/run1.json")
