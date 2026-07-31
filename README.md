# Resume Intelligence LLM — Domain-Specific Model Prototype

This repo documents how I'd replace GPT-4o Mini with a fine-tuned open-source model for a Resume Intelligence platform — parsing raw resume text into structured JSON. It covers the full assessment: model research, dataset creation, a fine-tuning pipeline design, an evaluation framework (with working code), and a minimal inference API.

**Chosen model throughout:** `Qwen2.5-3B-Instruct` (Apache 2.0), with `Qwen2.5-7B-Instruct` documented as an upgrade path. The reasoning is laid out in full in Part 1.

---

## How this maps to the assessment

| Rubric section | Marks | Folder | What's there |
|---|---|---|---|
| Part 1 — Research & Model Selection | 20 | [`part1_research/`](./part1_research) | Model comparison, pros/cons, why not other models |
| Part 2 — Dataset Creation | 25 | [`part2_dataset/`](./part2_dataset) | 50-example instruction-tuning dataset + scaling plan |
| Part 3 — Fine-Tuning / Adaptation | 25 | [`part3_finetuning/`](./part3_finetuning) | Full LoRA/QLoRA pipeline design (Option B) |
| Part 4 — Evaluation Framework | 20 | [`part4_evaluation/`](./part4_evaluation) | Metrics, hallucination detection, **working JSON-schema validator** |
| Part 5 — Engineering | 10 | [`part5_engineering/resume-parser-api/`](./part5_engineering/resume-parser-api) | Minimal FastAPI inference service |

---

## Quick start — running things yourself

**Validate the dataset against the schema** (Part 4's automated check):
```bash
cd part4_evaluation
python3 validate_outputs.py --predictions ../part2_dataset/resume_parsing_dataset.json
```
*(Note: `resume_parsing_dataset.json` is a list of `{instruction, input, output}` objects — point the validator at a list of just the `output` values to check them, e.g. by extracting them first. See `validate_outputs.py`'s docstring for the expected input format.)*

**Run the inference API** (Part 5):
```bash
cd part5_engineering/resume-parser-api
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
uvicorn app.main:app --reload --port 8000
```
Then visit `http://127.0.0.1:8000/docs` for interactive API docs, or see that folder's own README for `curl` examples.

---

## An honest note on scope

This was built under a tight deadline. Part 3 is a **pipeline design**, not an executed fine-tuning run (Option B, as the assessment explicitly allows when GPU access is a constraint). Part 5's model-calling layer is a **clearly labeled stub** — the API itself, its validation, and its error handling are real and tested; the actual model forward pass is not wired up, since a 3B model isn't loadable in most demo environments. Both of these scope decisions are explained in-line in their respective folders, not hidden.

Everything else — the dataset, the schema, the validator script, and the running API — is real and was tested before being included here.
