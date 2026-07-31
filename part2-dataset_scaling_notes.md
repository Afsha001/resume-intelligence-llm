# Scaling the Dataset to Thousands of Examples

The 50 hand-crafted examples were built to cover ~15 distinct edge-case
categories (missing fields, layout noise, date-format variants, domain
diversity, etc.) so that scaling is a matter of **multiplying diversity
axes**, not just generating more of the same pattern.

## Approach

1. **Template + slot-filling with an LLM-in-the-loop.**
   Define ~15-20 resume "archetypes" (the categories used here: clean
   single-column, multi-column/OCR-noisy, freelancer, academic/publications,
   fresher/projects-only, career-changer, non-English name, table-dump,
   etc.). For each archetype, use a strong LLM (e.g., GPT-4o or Claude) to
   generate N synthetic resumes with randomized names, companies, dates,
   and skills, conditioned on the archetype description — then have the
   *same* model (or a second model, for cross-checking) produce the
   matching structured JSON output.

2. **Programmatic field randomization.**
   Maintain pools of realistic names (multi-region, to preserve name
   diversity), companies, job titles, universities, and skill sets. Mix
   template resumes with randomly sampled field values so no two examples
   are near-duplicates, while keeping generation cheap and fast.

3. **Real resume mining (with care).**
   Scrape or source anonymized/publicly shared resume examples (e.g. from
   resume-template sites, not real candidates' private data), strip PII,
   and use them as additional structural variety — especially for layout
   noise that's hard to synthesize convincingly (real PDF-to-text
   artifacts, broken tables, inconsistent bullet symbols).

4. **Human-in-the-loop validation on a sample, not the whole set.**
   Once at thousands of examples, full manual review isn't feasible.
   Instead: (a) auto-validate every example against the JSON schema
   (required keys, types, no extra fields), (b) sample ~5% for manual
   review per generation batch, (c) use a second LLM as a "judge" to
   flag likely hallucinations (fields in the output not traceable to the
   input text) before human review.

5. **Deliberate class-balance tracking.**
   Track counts of each edge-case category as generation scales (e.g. via
   a metadata tag per example, stripped before final export) so the
   dataset doesn't silently converge to mostly "clean" resumes, which
   would make the fine-tuned model brittle on exactly the cases that
   matter most in production.

## Why this matters more than raw volume

The instruction explicitly notes dataset **quality** over quantity — a
model fine-tuned on 50 well-designed edge cases will generalize far better
than one fine-tuned on 5,000 near-identical "clean" resumes. Scaling should
preserve the *ratio* of edge-case diversity established in this seed set,
not just multiply volume.
