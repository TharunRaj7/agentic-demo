"""
Step 4 — Agent + campus_info + get_events Tools
================================================

The assistant gains a second tool: `get_events`, which hits the live mock API
to retrieve upcoming campus events (optionally filtered by category).

**Requires the mock API to be running:**
    uvicorn mock_api:app --port 8000

Try asking:
  - "What tech events are coming up?"
  - "What time does the library close on Saturday?"
  - "Are there any wellness events this semester?"

Run with:
    streamlit run app_step4.py
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

from llm_client import MODEL, get_client
from utils import (
    add_context_event,
    init_context_log,
    render_context_panel,
    render_tool_call_badge,
    render_tool_result_badge,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful campus assistant for Lakewood University. "
    "Use the campus_info tool for questions about campus hours, buildings, "
    "dining, services, transportation, and policies. "
    "Use the get_events tool to look up upcoming campus events. "
    "Always use the appropriate tool rather than guessing."
)

CAMPUS_GUIDE_PATH = os.path.join(os.path.dirname(__file__), "campus_guide.txt")
MOCK_API_BASE = "http://localhost:8000"

# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "campus_info",
            "description": (
                "Search the Lakewood University campus guide for information "
                "about campus hours, buildings, dining, student services, "
                "transportation, policies, and other campus facilities."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "The topic or question to look up, "
                            "e.g. 'library hours', 'dining options', 'parking'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_events",
            "description": (
                "Retrieve upcoming campus events from the live events API. "
                "Optionally filter by category. "
                "Available categories: career, academic, arts, wellness, tech."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "description": (
                            "Optional event category to filter by: "
                            "career, academic, arts, wellness, or tech"
                        ),
                    }
                },
                "required": [],
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _campus_info(query: str) -> str:  # noqa: ARG001
    with open(CAMPUS_GUIDE_PATH, encoding="utf-8") as fh:
        return fh.read()


def _get_events(category: str | None = None) -> str:
    params = {}
    if category:
        params["category"] = category
    try:
        resp = requests.get(f"{MOCK_API_BASE}/events", params=params, timeout=5)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except requests.ConnectionError:
        return (
            "Error: Could not connect to the mock API. "
            "Make sure it is running with: uvicorn mock_api:app --port 8000"
        )
    except requests.HTTPError as exc:
        return f"Error from events API: {exc}"


def _execute_tool(name: str, args: dict) -> str:
    if name == "campus_info":
        return _campus_info(args.get("query", ""))
    if name == "get_events":
        return _get_events(args.get("category"))
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 8


def run_agent(user_input: str) -> tuple[str, list[dict]]:
    """
    Run the tool-calling agent loop.

    Returns
    -------
    (final_answer, tool_events)
    """
    api_messages: list[dict] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        *st.session_state["messages"],
        {"role": "user", "content": user_input},
    ]

    tool_events: list[dict] = []

    for _ in range(MAX_ITERATIONS):
        response = get_client().chat.completions.create(
            model=MODEL,
            messages=api_messages,
            tools=TOOLS,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            # Some models occasionally return an empty string instead of a reply.
            # Fall back to a helpful message instead of surfacing a blank response.
            final = msg.content or (
                "I wasn't able to generate an answer to that question. "
                "Try asking more directly about campus hours, buildings, dining, "
                "student services, transportation, policies, or upcoming events."
            )
            return final, tool_events

        # Use "" instead of None — some models reject null content on follow-up calls.
        api_messages.append(
            {
                "role": "assistant",
                "content": msg.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ],
            }
        )

        for tc in msg.tool_calls:
            args = json.loads(tc.function.arguments)
            result = _execute_tool(tc.function.name, args)
            tool_events.append({"id": tc.id, "name": tc.function.name, "args": args, "result": result})
            api_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

    return "I wasn't able to complete the request within the allowed steps.", tool_events


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Step 4 — Agent + get_events",
    page_icon="🎓",
    layout="wide",
)

init_context_log()

# ---------------------------------------------------------------------------
# API health banner
# ---------------------------------------------------------------------------

try:
    health = requests.get(f"{MOCK_API_BASE}/health", timeout=2)
    api_ok = health.status_code == 200
except Exception:
    api_ok = False

if not api_ok:
    st.warning(
        "⚠️ **Mock API is not running.** The `get_events` tool will fail until you start it:\n\n"
        "```\nuvicorn mock_api:app --port 8000\n```",
        icon="🔌",
    )

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🎓 Campus Assistant — Step 4: Agent + get_events Tool")

st.info(
    "**What's new:** A second tool, `get_events`, calls the live mock API at "
    "`localhost:8000/events`. The agent chooses the right tool based on the "
    "question — campus guide for facility info, live API for events.\n\n"
    "Try: *'What tech events are coming up?'* or "
    "*'Are there any career fairs this semester?'*",
    icon="💡",
)

col_chat, col_ctx = st.columns([3, 2], gap="large")

# ---------------------------------------------------------------------------
# Chat panel
# ---------------------------------------------------------------------------

with col_chat:
    st.subheader("Chat")

    for msg in st.session_state.get("messages", []):
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about campus or upcoming events…")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)

        add_context_event("user_message", user_input, label=f"User: {user_input[:60]}")

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    final_answer, tool_events = run_agent(user_input)
                except Exception as exc:
                    final_answer = f"⚠️ API error: {exc}"
                    tool_events = []

            for event in tool_events:
                render_tool_call_badge(event["name"], event["args"])
                render_tool_result_badge(event["name"], event["result"])
                add_context_event(
                    "tool_call",
                    {"tool": event["name"], "args": event["args"]},
                    label=f"Tool call: {event['name']}({json.dumps(event['args'])})",
                )
                add_context_event(
                    "tool_result",
                    event["result"],
                    label=f"Tool result: {event['name']} → {str(event['result'])[:60]}",
                )

            if not final_answer:
                final_answer = (
                    "_The model returned an empty response. "
                    "Try rephrasing your question._"
                )
            st.markdown(final_answer)

        st.session_state["messages"].append({"role": "user", "content": user_input})
        st.session_state["messages"].append({"role": "assistant", "content": final_answer})
        add_context_event("assistant_message", final_answer, label=f"Assistant: {final_answer[:60]}")
        add_context_event(
            "raw_messages",
            [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state["messages"],
            label="Full messages sent to API",
        )

        st.rerun()

    if st.session_state.get("messages"):
        if st.button("🗑️ Clear conversation", key="clear"):
            st.session_state["messages"] = []
            st.session_state["context_log"] = []
            st.rerun()

# ---------------------------------------------------------------------------
# Context panel
# ---------------------------------------------------------------------------

with col_ctx:
    render_context_panel(system_prompt=SYSTEM_PROMPT)
