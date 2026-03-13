"""
Step 5 — Full Local Agent (campus_info + get_events + book_room)
================================================================

The complete local agent with all three tools:
  1. campus_info   – campus guide lookup (no network)
  2. get_events    – live events from mock API
  3. book_room     – reserve a study / meeting room via mock API

**Requires the mock API to be running:**
    uvicorn mock_api:app --port 8000

Try asking:
  - "Book a Hartwell study room for Alice on April 5th at 2pm for 2 hours"
  - "What career events are coming up?"
  - "What time does the library close on Sunday?"
  - "Reserve an innovation hub room for 4 people on April 10 at 10am, name: Bob"

Run with:
    streamlit run app_step5.py
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
    "You have three tools available:\n"
    "- campus_info: look up campus hours, buildings, dining, services, and policies\n"
    "- get_events: retrieve upcoming campus events (optionally by category)\n"
    "- book_room: reserve a study or meeting room on campus\n\n"
    "Always use the appropriate tool. For room bookings, collect all required "
    "details (room_type, date, start_time, duration_hours, name) before calling "
    "book_room. Valid room types: hartwell, innovation_hub, student_union."
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
    {
        "type": "function",
        "function": {
            "name": "book_room",
            "description": (
                "Reserve a campus study or meeting room. "
                "Valid room types: hartwell (2–8 people), "
                "innovation_hub (4–12 people), student_union (up to 20 people). "
                "Bookings are 1–3 hours; up to 7 days in advance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "room_type": {
                        "type": "string",
                        "description": (
                            "The type of room to book: "
                            "'hartwell', 'innovation_hub', or 'student_union'"
                        ),
                    },
                    "date": {
                        "type": "string",
                        "description": "Booking date in ISO format, e.g. '2026-04-05'",
                    },
                    "start_time": {
                        "type": "string",
                        "description": "Start time in 24-hour format, e.g. '14:00'",
                    },
                    "duration_hours": {
                        "type": "integer",
                        "description": "Duration in whole hours: 1, 2, or 3",
                    },
                    "name": {
                        "type": "string",
                        "description": "Full name of the person making the booking",
                    },
                    "purpose": {
                        "type": "string",
                        "description": "Optional: brief description of the booking purpose",
                    },
                },
                "required": ["room_type", "date", "start_time", "duration_hours", "name"],
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


def _book_room(
    room_type: str,
    date: str,
    start_time: str,
    duration_hours: int,
    name: str,
    purpose: str | None = None,
) -> str:
    payload: dict = {
        "room_type": room_type,
        "date": date,
        "start_time": start_time,
        "duration_hours": duration_hours,
        "name": name,
    }
    if purpose:
        payload["purpose"] = purpose
    try:
        resp = requests.post(f"{MOCK_API_BASE}/book-room", json=payload, timeout=5)
        resp.raise_for_status()
        return json.dumps(resp.json(), indent=2)
    except requests.ConnectionError:
        return (
            "Error: Could not connect to the mock API. "
            "Make sure it is running with: uvicorn mock_api:app --port 8000"
        )
    except requests.HTTPError as exc:
        try:
            detail = exc.response.json().get("detail", str(exc))
        except Exception:
            detail = str(exc)
        return f"Booking failed: {detail}"


def _execute_tool(name: str, args: dict) -> str:
    if name == "campus_info":
        return _campus_info(args.get("query", ""))
    if name == "get_events":
        return _get_events(args.get("category"))
    if name == "book_room":
        return _book_room(
            room_type=args["room_type"],
            date=args["date"],
            start_time=args["start_time"],
            duration_hours=args["duration_hours"],
            name=args["name"],
            purpose=args.get("purpose"),
        )
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 10


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
            return msg.content or "", tool_events

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
    page_title="Step 5 — Full Local Agent",
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
        "⚠️ **Mock API is not running.** The `get_events` and `book_room` tools "
        "will fail until you start it:\n\n"
        "```\nuvicorn mock_api:app --port 8000\n```",
        icon="🔌",
    )

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🎓 Campus Assistant — Step 5: Full Local Agent")

st.info(
    "**What's new:** The agent now has all three tools: `campus_info`, "
    "`get_events`, and `book_room`. It can look up campus info, check events, "
    "*and* make real bookings via the mock API — all in one conversation.\n\n"
    "Try: *'Book a Hartwell study room for Alice on 2026-04-05 at 14:00 for 2 hours'* "
    "or *'What wellness events are coming up and can I book a room for my study group?'*",
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

    user_input = st.chat_input("Ask about campus, events, or book a room…")

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
