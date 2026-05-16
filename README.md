# CreatorFlow AI — Backend

AI-powered YouTube content generation engine. This service exposes a **FastAPI** REST API that orchestrates a multi-agent LangGraph pipeline to produce topics, scripts, SEO packages, community posts, thumbnail prompts, and marketing strategies — all in one streaming generation run.

---

## Table of Contents

- [Overview](#overview)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Setup](#setup)
- [Environment Variables](#environment-variables)
- [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Agents & Prompts](#agents--prompts)
- [Database](#database)
- [Channel Profiles](#channel-profiles)

---

## Overview

The backend is a standalone Python service. The frontend (separate repo) communicates with it over HTTP. The core intelligence lives in a **LangGraph** workflow that chains specialised agents:

```
Topic Agent → Script Agent → SEO Agent → Content Agent → Critic Agent
```

Results are streamed back to the client via **Server-Sent Events (SSE)**, so the UI updates in real time as each agent completes its step.

---

## Tech Stack

| Layer | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| AI Orchestration | LangGraph + LangChain |
| LLM Providers | HuggingFace Inference API · OpenRouter · OpenAI |
| Database | SQLite via SQLAlchemy + aiosqlite |
| Package Manager | [uv](https://github.com/astral-sh/uv) (recommended) or pip |
| Validation | Pydantic v2 |
| Observability | LangSmith (optional tracing) |

---

## Project Structure

```
backend/
├── app/
│   ├── main.py               # FastAPI app entry point, CORS config
│   ├── config.py             # Pydantic Settings — reads from .env
│   ├── agents/
│   │   ├── base_agent.py     # Shared LLM setup & JSON parsing helpers
│   │   ├── topic_agent.py    # Topic generation + novelty/virality scoring
│   │   ├── script_agent.py   # Video script generation
│   │   ├── seo_agent.py      # Title, description & tags generation
│   │   ├── content_agent.py  # Community posts, thumbnail & marketing prompts
│   │   └── critic_agent.py   # Quality critique of generated content
│   ├── api/
│   │   └── routes.py         # All API route handlers
│   ├── models/               # SQLAlchemy ORM models + DB init
│   ├── prompts/              # Plain-text prompt templates (one file per agent)
│   ├── utils/
│   │   ├── channel_profile.py  # Channel profile CRUD helpers
│   │   └── logger.py           # Workflow step logger (SSE log feed)
│   └── workflow/
│       ├── graph.py          # LangGraph workflow definition & node wiring
│       └── state.py          # Typed workflow state schema
├── data/
│   └── content_history.db    # SQLite database (auto-created on first run)
├── requirements.txt
├── pyproject.toml
├── .env.example              # Template — copy to .env and fill in your keys
└── .gitignore
```

---

## Prerequisites

- **Python 3.10+**
- **uv** package manager — install with:
  ```bash
  pip install uv
  ```
- At least **one** LLM provider API key (see [Environment Variables](#environment-variables))

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Nirikshan95/CreatorFlow-AI--Backend.git
cd CreatorFlow-AI--Backend
```

### 2. Create and activate a virtual environment

```bash
uv venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 4. Create the data directory

```bash
mkdir data
```

### 5. Configure environment variables

```bash
copy .env.example .env   # Windows
cp .env.example .env     # macOS / Linux
```

Then open `.env` and fill in your API keys (see the next section).

---

## Environment Variables

Copy `.env.example` to `.env` and set the values below.  
You only need **one** LLM provider — the system falls back automatically.

```env
# ── LLM Providers (at least one required) ──────────────────────────────────

# Option A: HuggingFace Inference API (primary)
HUGGINGFACEHUB_API_TOKEN=hf_...

# Option B: OpenRouter (free-tier models available)
OPENROUTER_API_KEY=sk-or-...

# Option C: OpenAI
OPENAI_API_KEY=sk-...

# ── Model Selection (defaults shown) ───────────────────────────────────────

# HuggingFace models
HF_HEAVY_MODEL=Qwen/Qwen2.5-72B-Instruct
HF_FLASH_MODEL=Qwen/Qwen2.5-32B-Instruct

# OpenRouter models
OR_HEAVY_MODEL=google/gemini-2.0-pro-exp-02-05:free
OR_FLASH_MODEL=google/gemini-2.0-flash-lite-preview-02-05:free
OR_FALLBACK_MODEL=meta-llama/llama-3.3-70b-instruct:free

# ── Application ────────────────────────────────────────────────────────────

DATABASE_PATH=data/content_history.db
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=True

# ── LangSmith Tracing (optional) ───────────────────────────────────────────

LANGCHAIN_TRACING_V2=false
LANGCHAIN_API_KEY=your_langchain_api_key_here
LANGCHAIN_PROJECT=CreatorFlow-AI

# ── Channel Profile ─────────────────────────────────────────────────────────

CHANNEL_PROFILE_PATH=data/channel_profile.json
```

> **Note:** If none of the LLM provider keys are set, the agents will raise a configuration error on startup.

---

## Running the Server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at **`http://localhost:8000`**.

Once running, visit **`http://localhost:8000/docs`** for the interactive Swagger UI, or **`http://localhost:8000/redoc`** for ReDoc.

Health check:
```
GET http://localhost:8000/health
```

---

## API Reference

All routes are prefixed with `/api/v1`.

### Content Generation

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/generate/stream` | **Primary** — streams SSE events as agents complete |
| `POST` | `/api/v1/generate` | Synchronous generation (waits for full result) |
| `POST` | `/api/v1/generate/workflow` | Alias for the synchronous workflow endpoint |

#### Streaming endpoint query parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `category` | string | `null` | Content category (e.g. `"confidence"`, `"communication"`) |
| `num_topics` | int | `5` | Number of topic candidates to generate |
| `script_type` | string | `"descriptive"` | Script style (`"descriptive"` or `"storytelling"`) |
| `channel_profile_id` | string | `null` | ID of a saved channel profile to inject into prompts |

**Example:**
```
GET /api/v1/generate/stream?category=communication&num_topics=3&script_type=descriptive
```

SSE event types returned:

| `step` value | Payload | Meaning |
|---|---|---|
| `fetch_past_topics` | — | Memory loaded |
| `generate_topics` | — | Topics generated |
| `generate_script` | — | Script written |
| `generate_seo` | — | SEO package ready |
| `generate_content` | — | Community post & thumbnail done |
| `memory_summary` | `summary` | Condensed past-topic list |
| `heartbeat` | — | Keep-alive ping (every 15 s) |
| `log` | `message` | Internal workflow log message |
| `final` | `data` | Complete generated content object |
| `error` | `message` | Generation failed |

---

### Content History

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/history` | Last 50 generated content records |
| `GET` | `/api/v1/content/{video_id}` | Full detail for one record |
| `GET` | `/api/v1/past-topics` | Plain list of all past topic strings |
| `GET` | `/api/v1/past-topics-summary` | LLM-condensed summary of past topics |

---

### Channel Profiles

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/channel-profiles` | List all saved profiles |
| `POST` | `/api/v1/channel-profiles` | Create a new profile |
| `PUT` | `/api/v1/channel-profiles/{id}` | Update a profile by ID |
| `DELETE` | `/api/v1/channel-profiles/{id}` | Delete a profile by ID |
| `GET` | `/api/v1/channel-profile` | Get the default (legacy single) profile |
| `PUT` | `/api/v1/channel-profile` | Update the default profile |

---

## Agents & Prompts

Each agent has a corresponding plain-text prompt template in `app/prompts/`. You can edit these files to change AI behaviour **without touching Python code**.

| Agent | Prompt file | Purpose |
|-------|-------------|---------|
| `TopicAgent` | `topic_generation.txt` | Generates video topic candidates with novelty & virality scores |
| `ScriptAgent` | `script_generation.txt` | Writes a full video script for the chosen topic |
| `SEOAgent` | `seo_title.txt`, `seo_description.txt`, `seo_package.txt` | Produces optimised title, description, and tags |
| `ContentAgent` | `community_post.txt`, `post_image_prompt.txt`, `thumbnail_prompt.txt`, `marketing_strategy.txt` | Community post, image prompts, and marketing strategy |
| `CriticAgent` | `content_critic.txt` | Evaluates output quality |

Prompts support `{placeholder}` variables that are substituted at runtime (e.g. `{channel_info}`, `{past_topics_summary}`).

---

## Database

SQLite database is stored at `data/content_history.db` and is created automatically on first run.

Each record stores:

- `video_id` — UUID identifier
- `topic` — chosen topic string
- `category` — content category
- `keywords` — extracted keywords
- `title` — SEO-optimised title
- `script_data` — full script JSON
- `seo_data` — title, description, tags JSON
- `community_post` — community tab post text
- `thumbnail_prompt` — image generation prompt
- `marketing_strategy` — promotion plan
- `novelty_score` / `virality_score` — 0–10 float scores
- `critique_data` — critic agent output
- `created_at` — timestamp

---

## Channel Profiles

Channel profiles let you inject brand-specific context into every generation run (channel name, intro lines, default hashtags, social links, etc.).

Profiles are stored as JSON files at the path configured by `CHANNEL_PROFILE_PATH`. Multiple profiles are supported — pick one per generation via the `channel_profile_id` query parameter.
