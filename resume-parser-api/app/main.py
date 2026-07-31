"""
Resume Intelligence API -- minimal prototype.

Single endpoint that takes raw resume text and returns structured JSON
matching resume_schema.json. See app/model_loader.py for what's real vs
stubbed in the model-calling layer.
"""

import logging

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from app.config import settings
from app.model_loader import generate_structured_output
from app.schema import SchemaValidationError, validate_resume_output

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Resume Intelligence API",
    description="Prototype: parses raw resume text into structured JSON.",
    version="0.1.0",
)


class ParseResumeRequest(BaseModel):
    resume_text: str


class ParseResumeResponse(BaseModel):
    result: dict


@app.get("/health")
def health_check():
    return {"status": "ok", "model_name": settings.model_name, "device": settings.device}


@app.post("/parse-resume", response_model=ParseResumeResponse)
def parse_resume(payload: ParseResumeRequest):
    # Pydantic handles a missing/wrong-typed "resume_text" field with its
    # own 422. We explicitly check for empty/whitespace-only text here so
    # that case gets a clean, deliberate 400 instead.
    resume_text = payload.resume_text
    if not resume_text or not resume_text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="resume_text must not be empty.",
        )

    try:
        raw_output = generate_structured_output(resume_text)
    except Exception:
        logger.exception("Unexpected error during model inference")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong while generating the resume parse. "
            "Please try again or contact support if this persists.",
        )

    try:
        validate_resume_output(raw_output)
    except SchemaValidationError as e:
        logger.error("Model output failed schema validation: %s", e.errors)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=(
                "The model produced output that didn't match the expected "
                "resume schema, so it was rejected rather than returned. "
                "This has been logged for review."
            ),
        )

    return ParseResumeResponse(result=raw_output)


# --- Catch-all: never leak a raw stack trace to the client ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An unexpected error occurred. Please try again."},
    )
