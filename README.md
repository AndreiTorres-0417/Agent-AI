# Academic Review MVP (FastAPI + OpenAI)

Chat-first academic-review AI agent MVP for final project demo.

## Features
- Browser frontend at `/`
- `POST /chat`: academic-paper assistant conversation
- `POST /analyze_text`: analyze pasted text
- `POST /analyze_text_stream`: analyze pasted text with live status events
- `POST /analyze_docx`: parse and analyze uploaded `.docx`
- `POST /analyze_docx_stream`: parse and analyze uploaded `.docx` with live status events
- `GET /health`: service + OpenAI and local Ollama status
- Structured JSON feedback:
  - structure/format issues
  - academic quality issues
  - citation/consistency issues
  - prioritized suggestions
  - optional rewrite suggestions
- Resilient deterministic fallback when OpenAI fails/unavailable
- Model switch between OpenAI `gpt-4.1-mini` and local Ollama `gemma3:1b`
- Review-mode switch for Format Checker and General Academic Review
- APA 7 and IEEE compliance findings for citations, references, and table/figure naming
- General academic-review findings for tone, grammar, wording, clarity, structure, and citation needs
- Step 1 assistant receives the analyzed paper context for follow-up discussion
- Deterministic IEEE-only formatter for pasted text and `.docx` text extraction
- Export formatted IEEE preview as a downloadable `.docx`
- Grouped and deduplicated IEEE findings to avoid repetitive rule-by-rule output
- Live backend status updates during analysis, including streamed OpenAI output progress
- Inline highlighted draft excerpts with hover explanations

## Setup
1. Create virtual environment and install dependencies:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
copy .env.example .env
```
The app loads `.env` automatically at startup. Edit it if needed:
- `OPENAI_API_KEY` (required for assistant chat and AI-powered document review)
- `OPENAI_CHAT_MODEL` (default `openai:gpt-4.1-mini`)
- `ANALYSIS_MODEL` (default `openai:gpt-4.1-mini`)
- `OLLAMA_URL` (default `http://localhost:11434`)

3. Create an OpenAI API key from `https://platform.openai.com/api-keys` and add it to `.env`:
```bash
OPENAI_API_KEY=your_key_here
```
Keep `.env` private. Revoke and replace any key that has been pasted into chat, source code, or a public location.

Optional local Gemma assistant and document analysis:
```bash
ollama pull gemma3:1b
ollama serve
```
Select `Local Gemma 3 1B` under AI Model. The selection applies to both assistant chat and document analysis.

4. Run API:
```bash
uvicorn main:app --reload --port 8000
```

5. Open docs:
- App UI: `http://localhost:8000/`
- `http://localhost:8000/docs`

## Demo flow
1. Open `http://localhost:8000/`.
2. Ask the academic-paper assistant for help with planning, structure, citations, or revisions.
3. Paste a draft in the text box or switch to DOCX upload.
4. Choose Format Checker for APA 7 or IEEE compliance, or General Academic Review for writing feedback.
5. Review the structured feedback cards and discuss the analyzed paper in Step 1.

## Endpoint examples

### 1) Academic-paper assistant chat
```bash
curl -X POST "http://localhost:8000/chat" ^
  -H "Content-Type: application/json" ^
  -d "{\"session_id\":\"demo-1\",\"message\":\"Hi, help me review my paper.\"}"
```

### 2) Analyze text
```bash
curl -X POST "http://localhost:8000/analyze_text" ^
  -H "Content-Type: application/json" ^
  -d "{\"text\":\"Your draft text here...\",\"metadata\":{\"citation_style\":\"APA\"}}"
```

### 3) Analyze docx
```bash
curl -X POST "http://localhost:8000/analyze_docx" ^
  -F "file=@sample.docx"
```

### 4) Health
```bash
curl "http://localhost:8000/health"
```

## Notes
- If the selected model is unavailable during format checking, the service returns a deterministic fallback response with `fallback_used: true`.
- This MVP stores chat session state in memory (`chat_sessions`) for demo simplicity.
