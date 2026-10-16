# Week 3 — Prompting Techniques: The Support-Desk Assistant

**Name:** Mehmet Emin Akkaya
**Model:** `llama3.1:8b` via Ollama 0.40.5 (local, RTX 4060 Laptop 8 GB), `temperature=0.0`
**Result:** all four checks pass — k-shot 4/4, chain-of-thought 3/3, tool calling PASS, RAG 3/3
([`final_runs.txt`](final_runs.txt) has the full output).

To see what each technique actually *changed*, I also ran each task without it (same model,
same temperature) and compare against that baseline below.

---

## 1. K-shot prompting — ticket classification (`k_shot_prompting.py`)

**Final prompt:**
```text
You are a support-ticket classifier for an online store.
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
Label: other
```
The examples are deliberately *not* the test messages (no leakage), one per label.

**Sample input / output:**
`"My card was charged twice for order 4412."` → `billing`

**Baselines:**
- Empty prompt → the model answers as a support agent: *"i'm so sorry to hear that your card
  was charged twice…"* (no label at all).
- Zero-shot ("Classify … as billing, shipping, returns, technical, or other.") → the right
  category every time, but wrapped in prose: *'i would classify this message as "billing".
  the customer is inquiring about a duplicate charge…'* → **0/4** under the exact-match check.

**What I learned:** the model already knew the categories; the examples fixed the
**output format** — k-shot showed it that an answer is one bare word, which instructions alone
did not achieve (0/4 → 4/4).

---

## 2. Chain-of-thought — refund eligibility (`chain_of_thought.py`)

**Final prompt** (the policy is appended by the script):
```text
You decide whether an order is eligible for a refund under the policy below.
Think step by step. Check each rule in order and write one line per rule.
Judge each rule ONLY on its own question; never let one rule's result change another's.
1. Days since purchase: is it within 30 days? (PASS or FAIL)
2. Condition: is the item unused / in original condition? (PASS or FAIL; if not mentioned, PASS)
3. Final sale: is it a final-sale item? (FAIL if it is final sale, otherwise PASS)
If any rule FAILs the answer is NO; if all rules PASS the answer is YES.

The very last line of your reply must be exactly one of:
DECISION: YES
DECISION: NO
Write nothing after that line.
```

**Sample input / output:**
```text
Input:  Bought 5 days ago but it was a final-sale item.
Output: 1. Days since purchase: is it within 30 days? (PASS)
        2. Condition: is the item unused / in original condition? (PASS)
        3. Final sale: is it a final-sale item? (FAIL)

        DECISION: NO
```

**Baseline:** a direct-answer prompt ("Answer with DECISION: YES or DECISION: NO") got
2/3 — on *"Bought 45 days ago, unused."* the last line wasn't a parseable decision (`?`).

**Iteration:** my first CoT prompt also passed 3/3, but its reasoning for the final-sale case
was wrong: *"1. Days since purchase: FAIL (5 days is within 30 days, but it's a final-sale
item)"* — right decision, wrong step. Adding "Judge each rule ONLY on its own question" fixed
the step (now PASS / PASS / FAIL as shown above).

**What I learned:** CoT works best when *I* supply the steps (one line per rule) — it turned a
fuzzy judgement into a checklist and made the output gradeable, and reading the steps exposed
a wrong reason that the pass/fail check alone would have hidden.

---

## 3. Tool calling — order lookup (`tool_calling.py`)

**Final prompt:**
```text
You are a support assistant for an online store.
You do not know any order's status yourself. When a customer asks about an order,
you must first call the get_order_status tool with the order number from their
message, then answer in one short sentence using the tool's result.
```
**Tool schema (the TODOs):**
```python
"description": "Look up the current status of a customer's order (e.g. processing, "
               "shipped, delivery date). Call this whenever the customer asks about "
               "an order and mentions its order number.",
"order_id": {"type": "string",
             "description": "The order number from the customer's message, digits only (e.g. '4412')."}
```

**Sample input / output (full round trip, tool result fed back to the model):**
```text
User:        Hi, can you check on order 4412 for me?
Model call:  get_order_status({"order_id": "4412"})
Tool result: shipped, arriving Friday
Model reply: Your order 4412 has shipped and is expected to arrive on Friday.
             Is there anything else I can help you with?
```

**Baseline:** honestly, with an empty system prompt *and* empty tool descriptions,
llama3.1:8b still called `get_order_status(order_id="4412")` — the function name plus a
message containing a number was enough for this easy case.

**What I learned:** the model never runs anything itself — it only emits a structured call
that *my code* executes and feeds back; the schema is the contract, and the descriptions
matter for *when* to call and *what format* the argument takes (here "digits only"), even
though this simple case worked without them.

---

## 4. RAG — grounded answers (`rag.py`)

**Retriever:** keyword overlap. Lower-case the words, drop stopwords, cut each word to its
first 5 letters (a crude stem, so `returns`/`return` and `internationally`/`international`
match), score each help-center line by shared keywords, return the top 3 with score > 0.
```text
"How long do I have to return an item?"  -> the two "Returns:" lines
"Do you ship internationally?"           -> the international shipping line
"What's your CEO's home address?"        -> [] (no overlap, empty context)
```

**Final prompt:**
```text
You are a help-center assistant for an online store.
Answer the question using ONLY the information in the Context. Do not use outside
knowledge and do not guess.
If the Context does not contain the answer, reply exactly:
"I don't know based on our help center."
Keep answers to one or two sentences and reuse the Context's wording where possible.
```

**Sample input / output:**
`"How long do I have to return an item?"` → *"you may return most items within 30 days of
delivery for a full refund."*
`"What's your CEO's home address?"` → *"i don't know based on our help center."*

**Baseline (no retrieval, no grounding prompt):**
- Returns → generic advice about "the store or website you purchased from" (no "30 days").
- Shipping → *"Yes, we ship internationally. We strive to make our products accessible…"* —
  plausible-sounding but **invented**: with no context the model has no idea what this store
  does.
- CEO → *"I can't provide that information"* — a refusal, but phrased so the check (which
  looks for "cannot"/"don't know") would fail.

**What I learned:** retrieval supplies facts the model doesn't have, and the grounding prompt
plus a fixed "I don't know" sentence turns "make up something plausible" into "answer from the
text or refuse in a predictable way".

---

## Closing: what I learned across all four
LLMs are good at structured tasks because they already have the underlying skill — this 8B
model picked the right category for every ticket zero-shot, and extracted the order id with
no help at all. What it lacks is **discipline about the interface**: it wanted to
chat instead of emit a label, it wrote decisions the parser couldn't read, it let one rule's
outcome leak into another, and with no documents it confidently invented a shipping policy.
Every technique here is scaffolding for exactly that gap: examples pin the output format
(k-shot), an explicit checklist pins the reasoning (CoT), a schema pins the action (tool
calling), and retrieved text plus a refusal rule pins the facts (RAG). The other lesson is
that a passing check isn't proof: the CoT case passed with a wrong reason, so I read the
model's intermediate output, not just the final line. This is the same pattern as the
Gemini CLI lab — a coding agent is a capable model wrapped in a lot of format, tool, and
context scaffolding.
