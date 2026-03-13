"""
Steps 7+8 — MCP-Powered Campus Agent
======================================

The agent no longer hard-codes tool implementations. Instead it:
  1. Spawns mcp_server.py as a subprocess (stdio transport)
  2. Discovers available tools dynamically via the MCP protocol
  3. Converts MCP tool schemas → OpenAI function-calling format
  4. Routes every tool call the LLM makes back through the MCP session

This is the full MCP pattern: the agent is decoupled from tool logic entirely.

**Requires the mock API to be running:**
    uvicorn mock_api:app --port 8000

Run with:
    streamlit run app_mcp.py
"""

from __future__ import annotations

import asyncio
import json
import os
import sys

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
    "You have tools available to look up campus information, retrieve upcoming "
    "events, and book study or meeting rooms.\n\n"
    "Always use the appropriate tool. For room bookings, collect all required "
    "details (room_type, date, start_time, duration_hours, name) before calling "
    "book_room. Valid room types: hartwell, innovation_hub, student_union."
)

MOCK_API_BASE = "http://localhost:8000"

MCP_SERVER_PATH = os.path.join(os.path.dirname(__file__), "mcp_server.py")


# ---------------------------------------------------------------------------
# MCP helpers — run async operations from sync Streamlit context
# ---------------------------------------------------------------------------


def _get_event_loop() -> asyncio.AbstractEventLoop:
    """Return a reusable event loop stored in session state."""
    if "event_loop" not in st.session_state:
        loop = asyncio.new_event_loop()
        st.session_state["event_loop"] = loop
    return st.session_state["event_loop"]


def run_async(coro):
    """Run an async coroutine from a synchronous Streamlit context."""
    loop = _get_event_loop()
    return loop.run_until_complete(coro)


# ---------------------------------------------------------------------------
# MCP tool discovery & execution
# ---------------------------------------------------------------------------


async def _list_mcp_tools() -> list[dict]:
    """
    Spawn mcp_server.py and return its tools in OpenAI function-calling format.
    """
    from mcp import StdioServerParameters, stdio_client
    from mcp.client.session import ClientSession

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools_result = await session.list_tools()
            return _mcp_tools_to_openai(tools_result.tools)


async def _call_mcp_tool(tool_name: str, tool_args: dict) -> str:
    """
    Spawn mcp_server.py, call one tool, and return the result as a string.
    """
    from mcp import StdioServerParameters, stdio_client
    from mcp.client.session import ClientSession

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER_PATH],
    )
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, tool_args)
            parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    parts.append(content.text)
                else:
                    parts.append(str(content))
            return "\n".join(parts) if parts else "(no output)"


def _mcp_tools_to_openai(mcp_tools) -> list[dict]:
    """Convert MCP Tool objects to the OpenAI function-calling schema format."""
    openai_tools = []
    for tool in mcp_tools:
        # MCP tool.inputSchema is already a JSON-schema dict
        parameters = tool.inputSchema if tool.inputSchema else {"type": "object", "properties": {}}
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": parameters,
                },
            }
        )
    return openai_tools


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------

MAX_ITERATIONS = 10


def run_agent(user_input: str, tools: list[dict]) -> tuple[str, list[dict]]:
    """
    Run the MCP-powered tool-calling agent loop.

    Parameters
    ----------
    user_input : str
    tools : list[dict]
        OpenAI-format tool schemas fetched from the MCP server.

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
            tools=tools,
            tool_choice="auto",
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content or "", tool_events

        api_messages.append(
            {
                "role": "assistant",
                "content": msg.content,
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
            result = run_async(_call_mcp_tool(tc.function.name, args))
            tool_events.append(
                {"id": tc.id, "name": tc.function.name, "args": args, "result": result}
            )
            api_messages.append(
                {"role": "tool", "tool_call_id": tc.id, "content": result}
            )

    return "I wasn't able to complete the request within the allowed steps.", tool_events


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Step 7+8 — MCP Agent",
    page_icon="🎓",
    layout="wide",
)

init_context_log()

# ---------------------------------------------------------------------------
# Load MCP tool schemas once per session
# ---------------------------------------------------------------------------

if "mcp_tools" not in st.session_state:
    with st.spinner("Connecting to MCP server and discovering tools…"):
        try:
            st.session_state["mcp_tools"] = run_async(_list_mcp_tools())
            st.session_state["mcp_tools_error"] = None
        except Exception as exc:
            st.session_state["mcp_tools"] = []
            st.session_state["mcp_tools_error"] = str(exc)

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

if st.session_state.get("mcp_tools_error"):
    st.error(
        f"Failed to connect to MCP server: {st.session_state['mcp_tools_error']}",
        icon="🚨",
    )

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🎓 Campus Assistant — Steps 7+8: MCP Agent")

mcp_tools: list[dict] = st.session_state.get("mcp_tools", [])
tool_names = [t["function"]["name"] for t in mcp_tools]

st.info(
    "**What's new:** The agent no longer hard-codes any tool logic. It spawns "
    "`mcp_server.py` as a subprocess, discovers tools via the **Model Context "
    "Protocol**, and routes every tool call through the MCP session.\n\n"
    f"**Tools discovered from MCP server:** `{'`, `'.join(tool_names) if tool_names else 'none — check server'}`\n\n"
    "Try: *'Book an innovation hub room for 6 people on 2026-04-10 at 10:00 for 2 hours, name: Bob'* "
    "or *'What career events are coming up?'*",
    icon="🔌",
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
        if not mcp_tools:
            st.error("No tools available from the MCP server. Cannot process requests.")
        else:
            with st.chat_message("user"):
                st.markdown(user_input)

            add_context_event("user_message", user_input, label=f"User: {user_input[:60]}")

            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        final_answer, tool_events = run_agent(user_input, mcp_tools)
                    except Exception as exc:
                        final_answer = f"⚠️ Error: {exc}"
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

                st.markdown(final_answer)

            st.session_state["messages"].append({"role": "user", "content": user_input})
            st.session_state["messages"].append({"role": "assistant", "content": final_answer})
            add_context_event(
                "assistant_message", final_answer, label=f"Assistant: {final_answer[:60]}"
            )
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

    if mcp_tools:
        with st.expander(f"🔌 MCP Tools ({len(mcp_tools)} discovered)", expanded=False):
            st.code(
                json.dumps([t["function"] for t in mcp_tools], indent=2),
                language="json",
            )
