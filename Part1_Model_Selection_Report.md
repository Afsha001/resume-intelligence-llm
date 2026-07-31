# Research & Model Selection Report
### Replacing GPT-4o Mini with an Open-Source LLM for a Resume Intelligence Platform

---

## Objective

GPT-4o Mini currently powers our Resume Intelligence Platform's parsing pipeline, converting unstructured resume text into structured JSON. This document evaluates open-source alternatives and recommends a replacement that reduces per-token cost, keeps candidate PII inside our own infrastructure, and can be fine-tuned specifically for resume-parsing accuracy — something a closed API cannot offer.

---

## 1. Which Open-Source Model Would I Choose?

**Primary recommendation: Qwen2.5-3B-Instruct**
**Upgrade path (if accuracy demands it): Qwen2.5-7B-Instruct**

Resume parsing is fundamentally a structured information-extraction task rather than a deep-reasoning task: the model needs to read messy, inconsistently formatted text and reliably map it onto a fixed JSON schema. This favors a small, fine-tunable model over a large general-purpose one — a smaller model that has been taught the exact schema and edge cases outperforms a larger, generic model on this narrow task, at a fraction of the cost.

Qwen2.5-3B-Instruct is proposed as the default. Qwen2.5-7B-Instruct is documented as a fallback for cases where a single model must also handle long, multi-page executive CVs or heavily multilingual candidate pools without degrading accuracy.

---

## 2. Why This Model?

1. **Structured-output strength.** The Qwen2.5 release notes highlight explicit improvements in "understanding structured data (e.g., tables) and generating structured outputs, especially JSON" — precisely the failure mode that matters most for this platform (malformed JSON breaks downstream systems).
2. **Fine-tuning feasibility.** At 3B parameters, LoRA/QLoRA fine-tuning is feasible on a single free-tier GPU (Colab/Kaggle T4, 16GB VRAM), which keeps iteration cost near zero during development.
3. **Open weights, permissive license.** Apache 2.0 licensing allows unrestricted commercial use, self-hosting, and fine-tuning — none of which GPT-4o Mini permits.
4. **PII stays in-house.** Resumes contain names, emails, and phone numbers. Self-hosting removes the need to send this data to a third-party API, simplifying data-privacy compliance.
5. **Community precedent.** The Qwen2.5 family has extensive published fine-tuning examples for extraction-style tasks, reducing implementation risk compared to less-documented alternatives.

---

## 3–6. Technical Comparison

| Criterion | Qwen2.5-3B-Instruct (primary) | Qwen2.5-7B-Instruct (upgrade path) | GPT-4o Mini |
|---|---|---|---|
| **Parameter size** | 3B (disclosed) | 7.61B (disclosed) | Undisclosed |
| **Hardware — fine-tuning** | Feasible on free-tier GPU (Colab/Kaggle T4, 16GB VRAM) via LoRA/QLoRA | Requires ~24GB VRAM (RTX 4090/L4 class) — beyond free-tier | Not applicable — no fine-tuning access |
| **Hardware — inference** | Runs on a single consumer GPU (8–12GB VRAM), quantizable to run on CPU for low-throughput use | Single GPU, 16–24GB VRAM recommended | None — fully managed API |
| **Context window** | 32K tokens | 128K tokens, up to 8K generated tokens | 128K tokens |
| **License** | Apache 2.0 — free commercial use, full weight access | Apache 2.0 | Proprietary — governed by provider ToS, no weight access |
| **Multilingual support** | Supported | Supported across 29+ languages — stronger fit for international resume pools | Strong, but data leaves internal infrastructure |
| **Deployment mode** | Self-hosted (vLLM / Ollama / TGI) | Self-hosted | Remote API only |
| **Data privacy** | Fully on-premise / private cloud | Fully on-premise / private cloud | Candidate PII transmitted to third-party endpoint |

---

## 7. Pros & Cons

**Pros**
- No per-token API cost — fixed infrastructure cost regardless of resume volume.
- Full control over model weights: fine-tunable, quantizable, versionable, and not subject to vendor deprecation.
- Candidate PII never leaves our environment, simplifying compliance posture.
- Fine-tuning on our own resume dataset directly targets the platform's actual failure modes (missing fields, layout noise, non-English names) rather than relying on general-purpose capability.

**Cons**
- Requires internal ownership of GPU hosting, monitoring, and scaling (no longer "someone else's problem").
- Baseline (pre-fine-tuning) reasoning ability on ambiguous or contradictory resume content is below GPT-4o Mini's — this gap needs to be closed through the fine-tuning dataset, not assumed away.
- 3B variant's 32K context window can clip unusually long CVs (rare in practice, but a known edge case); the 7B upgrade path exists specifically to cover this.

---

## 8. Why Not Other Popular Models?

| Model considered | Reason not chosen as primary |
|---|---|
| **Llama-3.2-3B** | Comparable size and license, but published benchmarks show weaker structured-JSON consistency than Qwen2.5 at the same parameter count. |
| **Phi-3-mini (3.8B)** | MIT-licensed and strong on general reasoning-per-parameter, but historically less consistent at strict JSON-schema adherence than Qwen2.5 — a critical requirement here. |
| **Gemma-2-2B** | Smaller and faster, but noticeably weaker instruction-following on complex, multi-field extraction tasks. |
| **DeepSeek-R1-Distill-Qwen-1.5B** | An interesting reasoning-distilled option, but its inherited chain-of-thought tendency risks polluting strict JSON-only output unless explicitly suppressed — added complexity without a clear accuracy payoff for a non-reasoning task. |
| **DeepSeek-V3 / DeepSeek-R1 (full)** | 671B total parameters (37B active, Mixture-of-Experts). Requires multi-GPU server-grade infrastructure for inference alone; fine-tuning is not feasible on typical hardware. Disqualified on cost and infrastructure grounds — using it would defeat the cost/privacy rationale for going open-source in the first place. |
| **Llama-3.3-70B / other 70B-class models** | Requires 2–4x A100-class GPUs (160GB+ VRAM). Resume extraction is an information-extraction task, not a complex-reasoning task — a 70B model is hardware overkill relative to the difficulty of the task. |

---

## Conclusion

Qwen2.5-3B-Instruct is recommended as the primary replacement for GPT-4o Mini: it is small enough to fine-tune cheaply, licensed for unrestricted commercial self-hosting, and specifically strong at structured JSON generation — the platform's core requirement. Qwen2.5-7B-Instruct is documented as a defined upgrade path for edge cases (long-form executive CVs, heavily multilingual candidates) rather than the default, keeping the primary deployment cost-efficient. Next step: benchmark both variants after LoRA fine-tuning on the resume-parsing dataset to confirm the 3B model's accuracy is sufficient before ruling out the 7B upgrade.
