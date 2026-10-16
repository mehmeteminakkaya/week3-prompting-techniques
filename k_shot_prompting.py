"""K-shot prompting — ticket classification.

Task: classify a customer message into one of: billing, shipping, returns,
technical, other. Use in-context EXAMPLES (k-shot) so the model outputs exactly the
label, lowercase, with no extra text.

Only edit YOUR_SYSTEM_PROMPT.
Run:  python k_shot_prompting.py
"""
from ollama import chat

MODEL = "llama3.1:8b"
NUM_RUNS = 5

# TODO: Fill this in! Include a few labeled examples (k-shot) in the system prompt.
YOUR_SYSTEM_PROMPT = """You are a support-ticket classifier for an online store.
Classify the customer's message into exactly one label:
billing, shipping, returns, technical, other

- billing: charges, payments, invoices, refunds of money already charged
- shipping: delivery status, tracking, late or lost packages
- returns: sending an item back, exchanges, return requests
- technical: website or app errors, crashes, login problems
- other: anything else

Reply with the label only: one lowercase word, no punctuation, no explanation.

Examples:
Message: I was billed for an order I cancelled.
Label: billing

Message: My tracking number hasn't updated in five days.
Label: shipping

Message: The jacket doesn't fit, how do I return it?
Label: returns

Message: I can't log in, the page keeps showing an error.
Label: technical

Message: Do you have a store in Istanbul?
Label: other"""

CASES = [
    ("My card was charged twice for order 4412.", "billing"),
    ("Where is my package? It's been two weeks.", "shipping"),
    ("I want to send the shoes back for a refund.", "returns"),
    ("The app crashes every time I open my cart.", "technical"),
]


def classify(system_prompt: str, message: str) -> str:
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": message},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for message, expected in CASES:
        got = classify(YOUR_SYSTEM_PROMPT, message)
        ok = got == expected
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {message!r} -> got {got!r}, expected {expected!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
