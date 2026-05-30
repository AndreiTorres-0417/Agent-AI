# Academic Review MVP (FastAPI + Ollama)

Chat-first academic-review AI agent MVP for final project demo.

## Features
- Browser frontend at `/`
- `POST /chat`: intake Q&A flow
- `POST /analyze_text`: analyze pasted text
- `POST /analyze_text_stream`: analyze pasted text with live status events
- `POST /analyze_docx`: parse and analyze uploaded `.docx`
- `POST /analyze_docx_stream`: parse and analyze uploaded `.docx` with live status events
- `GET /health`: service + Ollama status
- Structured JSON feedback:
  - structure/format issues
  - academic quality issues
  - citation/consistency issues
  - prioritized suggestions
  - optional rewrite suggestions
- Resilient fallback when Ollama fails/unavailable
- Frontend model switch between local Ollama models and hosted Groq `llama-3.1-8b-instant`
- Format mode switch for General Review, APA 7, and IEEE
- Deterministic IEEE-only formatter for pasted text and `.docx` text extraction
- Export formatted IEEE preview as a downloadable `.docx`
- Grouped and deduplicated IEEE findings to avoid repetitive rule-by-rule output
- Live backend status updates during analysis, including streamed Ollama output progress
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
Edit `.env` if needed:
- `OLLAMA_URL` (default `http://localhost:11434`)
- `OLLAMA_MODEL` (default `deepseek-r1:8b`)
- `GROQ_API_KEY` (optional, required only for the Groq hosted model)

3. Start Ollama and pull model (example):
```bash
ollama pull deepseek-r1:8b
ollama serve
```

Optional hosted fast model:
1. Create a free Groq API key from `https://console.groq.com/keys`.
2. Add it to `.env`:
```bash
GROQ_API_KEY=your_key_here
```
3. Select `Groq Llama 3.1 8B Instant` in the frontend model dropdown.

4. Run API:
```bash
uvicorn main:app --reload --port 8000
```

5. Open docs:
- App UI: `http://localhost:8000/`
- `http://localhost:8000/docs`

## Demo flow
1. Open `http://localhost:8000/`.
2. Paste a draft directly into the chat panel for immediate analysis.
3. Or paste a draft in the text box / switch to DOCX upload.
4. Review the structured feedback cards.

## Endpoint examples

### 1) Chat intake
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
- If Ollama is down or returns invalid output, service returns a safe fallback structured response with `fallback_used: true`.
- This MVP stores chat session state in memory (`chat_sessions`) for demo simplicity.
