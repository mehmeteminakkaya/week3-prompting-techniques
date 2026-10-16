# Starter — Prompting Techniques

Fill-in-the-prompt stubs for four techniques. You only edit the `TODO` parts (the
prompt, and the retriever in `rag.py`).

## Setup
1. Install Ollama and start it: `ollama serve` (see https://ollama.com).
2. Pull the model: `ollama pull llama3.1:8b`.
3. Install the client: `pip install ollama`.

## Run each technique
Each file has a built-in PASS/FAIL check:
```bash
python k_shot_prompting.py
python chain_of_thought.py
python tool_calling.py
python rag.py
```
Iterate the prompt until the check passes.

## Files
- `k_shot_prompting.py` — ticket classification
- `chain_of_thought.py` — refund eligibility
- `tool_calling.py` — order lookup via a tool call
- `rag.py` — answer grounded only in `data/help_center.txt`

## Results
All four checks pass with `llama3.1:8b` (Ollama, temperature 0). See [`writeup.md`](writeup.md)
for the final prompts, sample input/output, and what each technique changed, and
[`final_runs.txt`](final_runs.txt) for the full check output.

## Week 3 submission
This repo is part 2 of my Week 3 submission. Part 1 (Gemini CLI dissection) is the pull
request [scottyUX/AI-Augmented-Software-Engineering#132](https://github.com/scottyUX/AI-Augmented-Software-Engineering/pull/132).
