"""RAG — policy-grounded answers.

Task: answer a customer question using ONLY the retrieved help-center text. If the
answer isn't in the docs, the model must say it doesn't know (don't hallucinate).

Fill in YOUR_SYSTEM_PROMPT and YOUR_RETRIEVER.
Run:  python rag.py
"""
import os

from ollama import chat

MODEL = "llama3.1:8b"
DATA = os.path.join(os.path.dirname(__file__), "data", "help_center.txt")


def load_chunks() -> list[str]:
    with open(DATA, encoding="utf-8") as f:
        # one chunk per non-empty line
        return [line.strip() for line in f if line.strip()]


STOPWORDS = {
    "a", "an", "the", "is", "are", "do", "does", "you", "your", "i", "my", "me", "to",
    "of", "in", "on", "for", "and", "or", "what", "what's", "how", "long", "have",
    "can", "it", "item", "we", "our",
}


def _keywords(text: str) -> set[str]:
    words = (w.strip("?.,!:;'\"").lower() for w in text.split())
    # crude stemming: first 5 letters, so "returns"/"return" and
    # "internationally"/"international" share a keyword
    return {w[:5] for w in words if w and w not in STOPWORDS}


# Keyword-overlap retriever: rank chunks by shared keywords, keep the top 3 with any overlap.
def YOUR_RETRIEVER(question: str, chunks: list[str]) -> list[str]:
    q = _keywords(question)
    scored = [(len(q & _keywords(c)), c) for c in chunks]
    return [c for score, c in sorted(scored, key=lambda s: -s[0]) if score > 0][:3]


YOUR_SYSTEM_PROMPT = """You are a help-center assistant for an online store.
Answer the question using ONLY the information in the Context. Do not use outside
knowledge and do not guess.
If the Context does not contain the answer, reply exactly:
"I don't know based on our help center."
Keep answers to one or two sentences and reuse the Context's wording where possible."""

CASES = [
    ("How long do I have to return an item?", ["30 days"]),
    ("Do you ship internationally?", ["international"]),
    ("What's your CEO's home address?", ["don't know", "do not know", "not sure", "cannot"]),
]


def answer(question: str) -> str:
    context = "\n".join(YOUR_RETRIEVER(question, load_chunks()))
    resp = chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": YOUR_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
        ],
        options={"temperature": 0.0},
    )
    return resp.message.content.strip().lower()


def run() -> bool:
    passed = 0
    for question, needles in CASES:
        got = answer(question)
        ok = any(n.lower() in got for n in needles)
        passed += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {question!r} -> {got[:80]!r}")
    print(f"{passed}/{len(CASES)} correct")
    return passed == len(CASES)


if __name__ == "__main__":
    run()
