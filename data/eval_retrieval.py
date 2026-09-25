"""Small retrieval-quality eval set: question -> expected source doc. Run after any
chunking/threshold/document change to check nothing regressed."""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from agents.qualitative import QualitativeAgent

EVAL_SET = [
    ("What is our security policy on multi-factor authentication?", "security_policy.md"),
    ("For customers with orders over $1,000, what approval process applies?", "customer_support_policy.md"),
    ("What's our churn escalation threshold?", "customer_support_policy.md"),
    ("How many days a week can employees work remotely?", "remote_work_policy.md"),
    ("What's our customer activation target?", "onboarding_guide.md"),
    ("What is the capital of France?", None),
    ("What's our policy on employee parental leave?", None),
    ("Tell me a joke.", None),
]


def run():
    agent = QualitativeAgent()
    hits = 0
    for question, expected in EVAL_SET:
        result = agent.answer(question)
        top_source = result.citations[0].source if result.citations and result.found else None
        ok = (top_source == expected) if expected else (not result.found)
        hits += ok
        status = "OK  " if ok else "FAIL"
        print(f"{status}  expected={expected}  got={top_source}  found={result.found}")
    print(f"\n{hits}/{len(EVAL_SET)} passed")


if __name__ == "__main__":
    run()