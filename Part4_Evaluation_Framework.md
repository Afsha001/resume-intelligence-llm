# Evaluation Framework Report
### Designing an Evaluation Methodology for the Resume Intelligence Model

---

## Objective

A resume-parsing model is only as trustworthy as the evaluation behind it. Because the output feeds directly into downstream hiring workflows, an error here isn't cosmetic — a misread phone number or a fabricated job title has real consequences for a candidate. This document lays out how we'd measure whether the fine-tuned model is actually good, not just fluent, and how we'd keep it that way as the model evolves.

The framework is organized around six questions: what to measure, how to measure parsing accuracy, how to catch hallucinations, how to enforce JSON correctness, how to compare model versions, and how to guard against regressions after retraining.

---

## 1. What Metrics Would We Use?

No single number can capture "is this a good resume parser," so we'd track a small stack of complementary metrics rather than chasing one aggregate score:

| Metric | What it captures | Why it matters here |
|---|---|---|
| **Field-level accuracy** | For each schema field (name, email, company, dates, etc.), the % of predictions that exactly match ground truth | Most granular signal — tells us *which* fields the model struggles with, not just an overall pass/fail |
| **Exact-match JSON accuracy** | % of full outputs that match ground truth exactly, key-for-key | The strictest metric; useful as a headline number for tracking progress over training runs |
| **Field-level F1 (for list fields)** | Precision/recall on array fields like `highlights`, `technologies`, `skills.technical` | Exact match is too harsh for lists — a model that extracts 4 of 5 highlights correctly shouldn't score 0 |
| **Normalized edit distance** | For string fields, how "close" a wrong answer is to correct (e.g., `"2021"` vs `"2021 "` vs `"Sep 2021"`) | Distinguishes near-misses (minor formatting) from genuine extraction failures |
| **Schema validity rate** | % of outputs that are syntactically valid JSON *and* conform to the target schema | A hard gate — if this isn't ~100%, nothing downstream can even be measured |
| **Hallucination rate** | % of predicted values that cannot be traced back to any span in the source resume text | Directly measures the model's tendency to invent information, which is the highest-risk failure mode for this platform |

Field-level accuracy and hallucination rate would be treated as the two headline metrics reported to stakeholders, since they map most directly to "does this system work" and "can we trust it," respectively.

---

## 2. How Would We Measure Parsing Accuracy?

We'd build a small **gold-standard test set** — resumes the model has never seen during fine-tuning, manually labeled by us to serve as ground truth (a held-out slice of the 50-example dataset, expanded as the dataset scales, following the strategy from Part 2).

For each test resume:
1. Run the model to get a predicted JSON output.
2. Align predicted fields against the gold JSON, field by field.
3. Score each field:
   - **Scalar fields** (name, email, phone, dates): exact match after light normalization (trim whitespace, lowercase for comparison purposes only).
   - **List fields** (highlights, skills, technologies): precision/recall/F1, since order and exact wording can legitimately vary while still being "correct."
   - **Nested list objects** (`work_experience`, `education`, `projects`): matched by best-alignment (e.g., matching predicted job entries to gold job entries by company+role similarity) before scoring their sub-fields — this avoids unfairly penalizing a model that gets the content right but lists two jobs in a different order.

This gives us both a single aggregate accuracy number and a field-by-field breakdown that tells us exactly where to focus further fine-tuning effort — for example, if `start_date`/`end_date` accuracy lags behind everything else, that's a signal to add more date-format diversity to the training set (as we intentionally did in Part 2).

---

## 3. How Would We Detect Hallucinations?

Hallucination is the highest-stakes failure mode for a resume parser: a model confidently inventing a company name, a degree, or a skill the candidate never listed. We'd approach detection in three layers:

1. **Span-grounding check (automatable).** For every string value the model outputs (company names, institutions, skills, etc.), check whether that string — or a close fuzzy match — actually appears somewhere in the source resume text. Values with no reasonable match are flagged as likely hallucinations. This is cheap to automate and catches the most blatant cases.
2. **Null-field discipline test.** We'd deliberately include test resumes with missing fields (as our dataset already does — 12 examples with no phone, 15 with no education section, etc.) and check that the model correctly outputs `null` or `[]` rather than fabricating a plausible-looking value to "fill the gap." A model that hallucinates a graduation year when none is present is a bigger risk than one that simply gets a date wrong.
3. **LLM-as-judge spot-check.** For a sampled subset of outputs, use a stronger model (e.g., GPT-4o or Claude) as a judge: given the source resume and the predicted JSON, ask it to flag any field whose value isn't supported by the text. This catches subtler hallucinations the span-grounding check might miss (e.g., a plausible-sounding but incorrect job title inferred from context rather than invented from nothing).

The hallucination rate would be reported separately from accuracy, since a low-accuracy-but-honest model (says "I don't know" via null) is meaningfully safer than a high-accuracy-looking model that occasionally fabricates confidently.

---

## 4. How Would We Validate JSON Correctness?

This is the one part of the framework we've fully automated rather than just designed. We define the target schema formally (JSON Schema, Draft-7) and validate every model output against it programmatically using Python's `jsonschema` library.

**`resume_schema.json`** — the formal schema definition matching the Part 2 target structure (all six top-level sections, required keys, and correct types for every field).

**`validate_outputs.py`** — a script that takes a batch of model outputs and reports:
- The % that are syntactically valid JSON at all (catches outright generation failures — truncated output, stray text before/after the JSON, etc.)
- The % that additionally conform to the schema (correct keys, correct types, no missing required fields)
- A breakdown of *which* fields most commonly fail validation, which directly informs what to prioritize in the next fine-tuning pass

We tested this validator against our 50-example dataset plus two deliberately corrupted examples (one missing a required key, one with a wrong field type) to confirm it actually catches structural errors rather than just rubber-stamping everything:

```
Total examples evaluated: 52
Syntactically valid JSON: 52/52 (100.0%)
Schema-conformant JSON:   50/52 (96.2%)

Top violation types:
  - schema_violation:personal_info: 1
  - schema_violation:personal_info.links: 1
```

This would run automatically as part of the evaluation pipeline for every model checkpoint, before any accuracy or hallucination scoring even happens — a model that can't produce valid JSON reliably isn't ready to be scored on anything else.

---

## 5. How Would We Compare Two Model Versions?

We'd treat this as a straightforward **paired comparison on the same held-out test set**, not two separate evaluation runs compared informally:

1. Run both model versions (e.g., baseline Qwen2.5-3B vs. LoRA-fine-tuned Qwen2.5-3B) against the identical gold-standard test set.
2. Compute the full metric stack from Section 1 for each version, side by side.
3. For the headline metrics (field-level accuracy, hallucination rate), run a significance check (e.g., McNemar's test on per-example pass/fail) rather than eyeballing a percentage-point difference — with a small test set, a 2-3 point swing can easily be noise rather than genuine improvement.
4. Break results down **by edge-case category** (the same categories used to design the Part 2 dataset — missing fields, layout noise, multilingual, academic/publications, etc.), not just in aggregate. A new model version might improve overall accuracy while quietly regressing on one category (e.g., getting better at clean resumes but worse at OCR-noisy ones) — aggregate numbers alone would hide that.

The output of this process would be a short comparison table, not just a single "which is better" verdict, since different model versions may be preferable for different deployment contexts (e.g., a version stronger on multilingual resumes might be worth deploying regionally even if its aggregate score is marginally lower).

---

## 6. How Would We Perform Regression Testing After Retraining?

Every time the model is retrained or fine-tuned again (new data added, hyperparameters changed, base model upgraded), we'd run it against a **fixed regression suite** before it's allowed to replace the production model:

1. **Frozen gold test set.** The same held-out test set is used every time — never rotated or expanded casually — so scores are directly comparable across retraining runs over time.
2. **Automated gate on schema validity.** Using `validate_outputs.py`, any retrained model scoring below a minimum schema-conformance threshold (e.g., 98%) is automatically rejected before human review even begins.
3. **Category-level regression check.** Compare the new model's per-category accuracy (Section 5) against the previous production model's per-category accuracy. Any category that drops by more than a defined threshold (e.g., 5 percentage points) is flagged for investigation, even if the aggregate score improved — this is how we'd catch the "improved overall but broke multilingual parsing" scenario before it reaches production.
4. **Hallucination rate ceiling.** The hallucination rate (Section 3) is treated as a hard ceiling, not just a tracked metric — a retrained model is not deployed if its hallucination rate increases versus the current production model, even if accuracy improves, since fabricated candidate data is a worse failure mode than a missed field.

This turns "did retraining help" from a subjective judgment call into a repeatable, mostly automated checklist — which matters because retraining will happen repeatedly over the platform's life, and manual re-evaluation every time isn't sustainable.

---

## Automation Summary (Bonus)

The JSON-validity layer of this framework (Section 4) is fully automated and included as working code:
- `resume_schema.json` — formal JSON Schema definition of the target output structure
- `validate_outputs.py` — CLI tool that scores any batch of model outputs against the schema and reports both an aggregate pass rate and a per-field violation breakdown

This is deliberately the layer we automated first, since it's the cheapest to build, catches the most catastrophic failure mode (broken JSON that downstream systems can't even parse), and gates every other evaluation step described above.
