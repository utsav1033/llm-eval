from llm_eval import ExactMatch

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

scorers = [ExactMatch()]
dataset_path = "examples/tests.jsonl"
