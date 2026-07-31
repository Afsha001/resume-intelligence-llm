# Resume Intelligence API (Prototype)

A minimal FastAPI service demonstrating the inference flow for parsing raw
resume text into structured JSON, as designed in Parts 1–4 of this
assessment. This is a prototype, not a production system — see the note
at the bottom on what's real vs. stubbed.

## Project structure

```
resume-parser-api/
├── app/
│   ├── main.py             # FastAPI app, routes, error handling
│   ├── config.py           # env-based settings
│   ├── schema.py            # loads resume_schema.json, validates output
│   ├── model_loader.py       # model-calling layer (stubbed, see below)
│   └── resume_schema.json    # single source of truth for output shape (from Part 4)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install dependencies** (Python 3.10+ recommended):

   ```bash
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables:**

   ```bash
   cp .env.example .env
   ```

   Then edit `.env` if you want to change the model name, device, or
   generation settings. Defaults work out of the box for local testing.

3. **Run the server:**

   ```bash
   uvicorn app.main:app --reload --port 8000
   ```

   The API will be available at `http://127.0.0.1:8000`. Interactive docs
   (Swagger UI) are auto-generated at `http://127.0.0.1:8000/docs`.

## Usage example

**Request:**

```bash
curl -X POST http://127.0.0.1:8000/parse-resume \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Jane Doe, Software Engineer at Acme Corp. jane@example.com"}'
```

**Response:**

```json
{
  "result": {
    "personal_info": {
      "full_name": null,
      "email": null,
      "phone": null,
      "links": []
    },
    "work_experience": [],
    "projects": [],
    "research_publications": [],
    "skills": { "technical": [], "soft": [] },
    "education": []
  }
}
```

(The response above is empty/null-filled because the model call is
currently stubbed — see below. The response *shape* is what a real
fine-tuned model would populate.)

**Error cases:**

```bash
# empty input -> 400
curl -X POST http://127.0.0.1:8000/parse-resume \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "   "}'

# missing field -> 422 (FastAPI's built-in request validation)
curl -X POST http://127.0.0.1:8000/parse-resume \
  -H "Content-Type: application/json" \
  -d '{}'
```

## What's real vs. stubbed

The FastAPI app, request/response validation, the `/parse-resume` and
`/health` endpoints, the JSON-schema validation step (reusing
`resume_schema.json` from Part 4), and all error handling (400 for empty
input, 500 for schema-invalid output, 500 for unexpected exceptions with
no stack trace leaked to the client) are real, running code — you can
start the server and hit it exactly as shown above. What is **not** real
is `generate_structured_output()` in `app/model_loader.py`: it currently
returns a fixed, schema-valid placeholder instead of actually running
Qwen2.5-3B-Instruct, since a 3B model isn't something this environment
can load. That function is clearly marked with a `# TODO: replace with
real model call` comment and a sketch of what the real `transformers`
call would look like.
