# Part 3 — Fine-Tuning / Adaptation (Option B: Pipeline Design)

## 1. Model Choice

Sticking with **Qwen2.5-3B-Instruct** as the base model, consistent with Part 1. The core reasoning holds: resume-to-JSON parsing is an extraction task, not a reasoning-heavy one, so a 3B model is enough horsepower once it's adapted to the schema. It's Apache 2.0 licensed (no legal friction), fits comfortably on a single free-tier T4 (16GB VRAM) with LoRA/QLoRA, and the 32K context window covers even long, multi-page academic CVs with publication lists. The 7B variant stays on the shelf as an upgrade path if we later need better multilingual handling or the dataset grows large enough to justify the extra capacity.

## 2. Dataset Preparation

The 50-example set gets reformatted into Qwen2.5's chat template — system instruction + user turn (raw resume text) + assistant turn (the target JSON, serialized with consistent key ordering and no trailing whitespace). Consistent serialization matters here because the model is partly learning *formatting discipline*, not just content extraction.

**Split:** with only 50 examples, a strict 80/10/10 split leaves validation and test at 5 examples each — too thin to trust, especially with ~15 edge-case categories to cover. A more honest split is **40 train / 5 validation / 5 test**, but chosen deliberately rather than randomly: validation and test each get a *stratified* sample that includes at least one instance from the trickiest categories (OCR-noisy layout, missing fields, non-English names, freelancer/no-company, publications-heavy academic CV). Random splitting risks putting all the hard cases in train and none in test, which would make eval numbers meaningless. This is closer to careful manual curation than a train_test_split() call.

Tokenization uses Qwen2.5's own tokenizer, with the input+output pair packed into a single sequence and loss masked over the prompt portion (instruction + input), so the model is only penalized for getting the output JSON wrong, not for "predicting" the resume text.

**Scaling note:** Part 2 anticipates growing this to low thousands of examples over time (synthetic generation + real anonymized resumes + active-learning on failure cases from Part 4's evaluation). Once we're in the hundreds-to-thousands range, the split ratio flips back toward standard 80/10/10 with random stratified sampling, and the hyperparameters below would need revisiting — more data supports a higher rank and more epochs before overfitting becomes a concern.

## 3. Training Strategy

**QLoRA over LoRA.** Full LoRA training in fp16/bf16 fits on a T4, but QLoRA (4-bit NF4 quantized base weights + LoRA adapters) leaves more VRAM headroom, which matters for stability on free-tier GPUs and gives room to bump batch size slightly if needed. Given a 3B model, the quality trade-off from 4-bit quantization is negligible for an extraction task like this — we're not asking the model to do delicate reasoning where quantization noise would bite.

**Target modules:** standard attention projections for Qwen2.5 — `q_proj`, `k_proj`, `v_proj`, `o_proj` — plus optionally the MLP `gate_proj`/`up_proj`/`down_proj` if early runs show the model struggling with schema-specific vocabulary (e.g., section headers it hasn't seen much of). Starting with attention-only keeps the adapter small and the overfitting risk lower.

| Hyperparameter | Value | Why |
|---|---|---|
| LoRA rank (r) | 8 | Low capacity on purpose — 50 examples can't support a high-rank adapter without memorizing them |
| LoRA alpha | 16 | Standard 2× rank ratio, keeps adapter contribution scaled reasonably |
| Learning rate | 1e-4 | Conservative for LoRA; high enough to adapt, low enough not to overwrite base behavior fast |
| Batch size | 4 (effective, via gradient accumulation) | T4 VRAM constraint; accumulation keeps effective batch stable |
| Epochs | 3–4, with early stopping | Small dataset saturates fast; validation loss is checked every epoch and training stops the moment it stops improving |
| Optimizer | paged_adamw_8bit | Memory-efficient variant that plays well with QLoRA on limited VRAM |

The overfitting risk here is real and specific: 50 examples means the model could start memorizing surface patterns (e.g., "if the resume mentions X company, output Y") rather than learning the general extraction skill. Low rank, low epoch count, and validation-loss-triggered early stopping are the three levers pulling against that, in order of how much they matter — rank first, because a high-rank adapter has enough parameters to memorize the training set outright regardless of epoch count.

## 4. Hardware Requirements

**Training:** a single T4 (16GB VRAM, free-tier Colab/Kaggle) is sufficient for QLoRA fine-tuning of a 3B model at this dataset size — 4-bit base weights plus LoRA adapters comfortably fit with room for gradient accumulation. An L4 or A10 (24GB) would speed things up and remove any need for gradient accumulation tricks, but isn't required.

**Inference (post-training):** noticeably lighter — the merged model (base + adapter) can run in fp16 on a T4, or even more efficiently in 4-bit for serving. No training-time optimizer states or gradients to keep in memory, so inference could realistically run on cheaper/smaller hardware than training required, or on CPU with acceptable latency for a non-real-time batch-processing use case like resume parsing.

## 5. Evaluation Plan

No new framework needed here — this plugs directly into Part 4. The 5-example held-out test split defined above is scored using the same field-level accuracy, exact-match JSON accuracy, list-field F1, hallucination rate, and schema validity metrics already defined, gated through the existing `validate_outputs.py` validator. Before/after comparison (base Qwen2.5-3B-Instruct vs. the fine-tuned version) runs through Part 4's paired-comparison approach on the identical test examples, so any improvement is directly attributable to fine-tuning rather than dataset differences.

## 6. Estimated Costs

At 50 examples, this is genuinely cheap. On free-tier Colab/Kaggle (T4), training time for 3–4 epochs over 40 examples is on the order of minutes, not hours — cost is effectively $0 aside from time. On a paid cloud GPU (e.g., a single A10 or L4 on a provider like Lambda or RunPod), rates run roughly $0.50–$1.20/hour depending on provider and spot vs. on-demand pricing, and total training time at this scale would still likely be under 30 minutes, so total cost is well under $1 for the actual training run.

The honest caveat: this cost estimate is only meaningful at the current 50-example scale. Per the Part 2 scaling plan, once the dataset grows into the thousands, training time scales roughly linearly with data size (more so if rank/epochs also increase to make use of the extra data), so costs would move from "negligible" to "a few dollars per run" — still modest, but worth tracking once we're iterating frequently.

## 7. Potential Risks and Mitigation

| Risk | Mitigation |
|---|---|
| Overfitting on 50 examples | Low LoRA rank (8), max 3–4 epochs, early stopping on validation loss, stratified val/test split to catch it early |
| Schema drift / inconsistent JSON on edge cases outside the 50 examples | Keep the schema validator (`validate_outputs.py`) in the loop at inference time, not just eval time, so malformed output is caught and can trigger a retry or fallback before it reaches a downstream system |
| Catastrophic forgetting of base model's general language ability | LoRA/QLoRA inherently limits this (base weights frozen), but we'd also spot-check the fine-tuned model on a few generic instruction-following prompts unrelated to resumes, to confirm it hasn't degraded outside the target task |
| Hallucination risk doesn't disappear post-fine-tuning | Fine-tuning teaches format and extraction patterns, not truthfulness — Part 4's span-grounding and null-field discipline test still runs on the fine-tuned model's outputs exactly as it would on the base model's, since this risk is orthogonal to training |

## Conclusion

Option B was the right call given tonight's deadline and no GPU access — designing the pipeline properly is more useful to evaluate than a rushed, unverified training run would be. The real next step, if this were executed, would be to actually run the QLoRA fine-tune on Colab's free T4 using the 40/5/5 split above, then run the Part 4 evaluation framework on both base and fine-tuned models to see if the design assumptions here (particularly the overfitting mitigations) hold up in practice.
