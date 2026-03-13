"""
Step 3 — Agent + campus_info Tool
==================================

The assistant can now call a `campus_info` tool that reads the campus guide.
Watch the Context Panel to see the agent decide to call the tool, receive the
result, and then synthesise a grounded answer.

Try asking:
  - "What time does the library close on Friday?"
  - "Where is the IT Help Desk?"
  - "What dining options are near the library?"

Run with:
    streamlit run app_step3.py
"""

from __future__ import annotations

import json
import os

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
    "When students ask about campus information — hours, locations, dining, "
    "services, transportation, or policies — always use the campus_info tool "
    "to look up accurate information. Never guess; use the tool."
)

CAMPUS_GUIDE_PATH = os.path.join(os.path.dirname(__file__), "campus_guide.txt")

# ---------------------------------------------------------------------------
# Tool definition + implementation
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
    }
]


def _campus_info(query: str) -> str:  # noqa: ARG001 – query reserved for future filtering
    """Return the full campus guide text."""
    with open(CAMPUS_GUIDE_PATH, encoding="utf-8") as fh:
        return fh.read()


def _execute_tool(name: str, args: dict) -> str:
    if name == "campus_info":
        return _campus_info(args.get("query", ""))
    return f"Unknown tool: {name}"


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 6


def run_agent(user_input: str) -> tuple[str, list[dict]]:
    """
    Run the tool-calling agent loop.

    Returns
    -------
    (final_answer, tool_events)
        final_answer : str  – the last assistant text reply
        tool_events  : list[dict] – each dict has keys: id, name, args, result
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

        # Serialise assistant message (with tool_calls) back into the history.
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
    page_title="Step 3 — Agent + campus_info",
    page_icon="🎓",
    layout="wide",
)

init_context_log()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🎓 Campus Assistant — Step 3: Agent + campus_info Tool")

st.info(
    "**What's new:** The assistant now has a `campus_info` tool that reads "
    "the campus guide. Watch the **Context Panel** on the right to see the "
    "agent call the tool, receive the result, and build a grounded answer — "
    "no more hallucinations!\n\n"
    "Try: *'What time does Hartwell Library close on Friday?'* or "
    "*'Where is the IT Help Desk?'*",
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

    user_input = st.chat_input("Ask anything about campus…")

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
