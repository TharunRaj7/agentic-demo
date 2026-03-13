# Agentic Demo

A step-by-step demo that walks you through how AI agent development works — from a basic LLM chat all the way to a fully decoupled MCP-powered agent. Built around a fictional **Lakewood University** campus assistant.

## Purpose

This project is designed to show how modern AI applications evolve in complexity:

| Step | File | What it demonstrates |
|------|------|----------------------|
| 1 & 2 | `app_step1_and_2.py` | Plain LLM chat with no tools — watch it hallucinate or admit it doesn't know campus-specific facts |
| 3 | `app_step3.py` | Agent with a `campus_info` tool — reads a local campus guide to give grounded answers |
| 4 | `app_step4.py` | Agent gains a second tool: `get_events` — hits a live mock API to fetch upcoming events |
| 5 | `app_step5.py` | Full local agent with three tools: `campus_info`, `get_events`, and `book_room` |
| 7 & 8 | `app_mcp.py` | MCP-powered agent — tool logic is fully decoupled via the [Model Context Protocol](https://modelcontextprotocol.io/) |

Each app displays a **Context Panel** so you can see exactly what is being sent to and received from the model at every step.

## Prerequisites

- **Python 3.10+**
- **pip**
- An [OpenRouter](https://openrouter.ai) API key (free tier available)

## Setup

**1. Clone the repo and enter the directory**

```bash
git clone <repo-url>
cd agentic-demo
```

**2. Create and activate a virtual environment**

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Configure your API key**

```bash
cp .env.example .env
```

Open `.env` and replace `your_openrouter_api_key_here` with your actual key:

```
OPENROUTER_API_KEY=sk-or-...
```

## Running the demos

### Steps 1 & 2 — Basic LLM chat

```bash
streamlit run app_step1_and_2.py
```

### Step 3 — Agent with campus info tool

```bash
streamlit run app_step3.py
```

### Step 4 — Agent with campus info + events tools

Start the mock API first, then the app in a second terminal:

```bash
uvicorn mock_api:app --port 8000
streamlit run app_step4.py
```

### Step 5 — Full local agent (all three tools)

```bash
uvicorn mock_api:app --port 8000
streamlit run app_step5.py
```

### Steps 7 & 8 — MCP-powered agent

```bash
uvicorn mock_api:app --port 8000
streamlit run app_mcp.py
```

## Project structure

```
agentic-demo/
├── app_step1_and_2.py  # Steps 1 & 2: plain LLM chat
├── app_step3.py        # Step 3: single tool (campus info)
├── app_step4.py        # Step 4: two tools (campus info + events)
├── app_step5.py        # Step 5: three tools (+ room booking)
├── app_mcp.py          # Steps 7 & 8: MCP-powered agent
├── mcp_server.py       # MCP server exposing campus tools
├── mock_api.py         # FastAPI mock backend (events + room booking)
├── llm_client.py       # Thin wrapper around the OpenRouter API
├── utils.py            # Shared UI helpers (context panel)
├── campus_guide.txt    # Static campus knowledge base
├── .env.example        # Environment variable template
└── requirements.txt    # Python dependencies
```
