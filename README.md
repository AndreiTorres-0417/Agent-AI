# Academic Review Agent

A web-based AI agent prototype that reviews academic drafts, checks IEEE-style citation/reference issues, highlights relevant text, and supports follow-up chat about the reviewed paper.

## System Architecture

```text
User
  ↓
Browser UI
  ↓
FastAPI Backend
  ↓
Input Processing
  ├── Pasted text
  └── DOCX text extraction
  ↓
Review Engine
  ├── LLM academic review
  ├── Rule-based IEEE citation checks
  └── Deterministic fallback if the model fails
  ↓
Structured Output
  ├── Summary
  ├── Academic quality issues
  ├── Citation/reference issues
  ├── Priority fixes
  └── Highlighted draft excerpts
  ↓
Chat Memory
  ↓
Section-aware follow-up discussion
```

The agent accepts a draft, processes it through an LLM and rule-based checks, returns structured feedback, stores the review context, and lets the user ask follow-up questions about specific sections such as the introduction or conclusion.

## Libraries And Tools Used

- **Python**: backend programming language
- **FastAPI**: API server and web routes
- **Uvicorn**: local development server
- **OpenAI API**: hosted LLM option
- **Ollama**: local LLM runtime option
- **Gemma 3 1B**: local model option through Ollama
- **python-docx**: DOCX text extraction and DOCX export support
- **Pydantic**: request and response validation
- **python-dotenv**: environment variable loading from `.env`
- **HTML / CSS / JavaScript**: browser frontend
- **localStorage**: browser-side persistence for chat, settings, and draft text

## Setup Instructions

### 1. Create A Virtual Environment

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\activate
```

### 2. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy the example environment file:

```powershell
copy .env.example .env
```

Open `.env` and add your OpenAI API key if using OpenAI:

```env
OPENAI_API_KEY=your_key_here
```

Optional environment variables:

```env
OPENAI_CHAT_MODEL=openai:gpt-4.1-mini
ANALYSIS_MODEL=openai:gpt-4.1-mini
OLLAMA_URL=http://localhost:11434
OLLAMA_TIMEOUT_SECONDS=600
```

### 4. Optional Local Model Setup

Install and run Ollama, then pull Gemma:

```powershell
ollama pull gemma3:1b
ollama serve
```

The app can use either OpenAI or Local Gemma from the AI Model dropdown.

### 5. Run The App

```powershell
uvicorn main:app --reload --port 8000
```

Open the web app:

```text
http://localhost:8000/
```

Open API docs:

```text
http://localhost:8000/docs
```

## Basic Usage

1. Start the FastAPI server.
2. Open `http://localhost:8000/`.
3. Paste an academic draft or upload a `.docx` file.
4. Click **Analyze Draft**.
5. Review the generated summary, issues, suggestions, and highlights.
6. Ask follow-up questions in the chat, such as:

```text
how's the conclusion?
```

The chat will use the reviewed paper context and underline the section being discussed.
