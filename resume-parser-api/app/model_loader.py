"""
Model-calling layer.

This module isolates the one piece of this prototype that is NOT real:
the actual forward pass through Qwen2.5-3B-Instruct. Everything else in
the app (routing, validation, error handling) is genuine, running code.

In a real deployment, `generate_structured_output` would either:
  (a) load the fine-tuned adapter with `transformers` + `peft` and run
      `model.generate(...)` directly, or
  (b) call out to a served endpoint (e.g. a vLLM server hosting the
      merged fine-tuned weights) over HTTP.

Neither is wired up here since a 3B model isn't something this demo
environment can load or run.
"""

import logging

from app.config import settings

logger = logging.getLogger(__name__)

# Example of what a real prompt template would look like -- kept here so
# the stub below is at least structurally honest about how the real call
# would be built, even though it never reaches a model.
PROMPT_TEMPLATE = """You are a resume-parsing assistant. Extract the following \
resume text into the JSON schema you were fine-tuned on. Use null for \
missing scalar fields and [] for missing list fields. Do not invent \
information that is not present in the text.

Resume text:
{resume_text}

JSON output:"""


def generate_structured_output(resume_text: str) -> dict:
    """
    Given raw resume text, return a structured dict matching resume_schema.json.

    # TODO: replace with real model call.
    # Real implementation sketch:
    #
    #   from transformers import AutoModelForCausalLM, AutoTokenizer
    #   tokenizer = AutoTokenizer.from_pretrained(settings.model_name)
    #   model = AutoModelForCausalLM.from_pretrained(
    #       settings.model_name, device_map=settings.device
    #   )
    #   prompt = PROMPT_TEMPLATE.format(resume_text=resume_text)
    #   inputs = tokenizer(prompt, return_tensors="pt").to(settings.device)
    #   output_ids = model.generate(
    #       **inputs,
    #       max_new_tokens=settings.max_new_tokens,
    #       temperature=settings.temperature,
    #   )
    #   raw_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #   return json.loads(extract_json(raw_text))
    #
    # For this demo, we return a fixed, schema-valid stub so the rest of
    # the API (validation, error handling, response shape) can be
    # exercised end-to-end without a loaded model.
    """
    logger.info(
        "generate_structured_output called (STUBBED, no real model loaded) "
        "model_name=%s device=%s",
        settings.model_name,
        settings.device,
    )

    return {
        "personal_info": {
            "full_name": None,
            "email": None,
            "phone": None,
            "links": [],
        },
        "work_experience": [],
        "projects": [],
        "research_publications": [],
        "skills": {"technical": [], "soft": []},
        "education": [],
    }
