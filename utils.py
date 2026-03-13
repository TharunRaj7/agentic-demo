"""
Shared utilities for the Campus Assistant demo.

Provides:
  - render_context_panel()  – draws the right-hand context panel in Streamlit
  - add_context_event()     – appends an event to st.session_state["context_log"]
  - init_context_log()      – ensures context_log exists in session state
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st


# ---------------------------------------------------------------------------
# Session-state helpers
# ---------------------------------------------------------------------------


def init_context_log() -> None:
    """Ensure session_state keys used by the context panel exist."""
    if "context_log" not in st.session_state:
        st.session_state["context_log"] = []
    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def add_context_event(event_type: str, payload: Any, label: str = "") -> None:
    """
    Append one event to the context log.

    Parameters
    ----------
    event_type : str
        One of: "system_prompt", "user_message", "assistant_message",
                "tool_call", "tool_result", "raw_messages"
    payload : Any
        The data to display (str, dict, list, …).
    label : str
        Optional short label shown as the expander title.
    """
    st.session_state["context_log"].append(
        {"type": event_type, "label": label, "payload": payload}
    )


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------

_TYPE_ICON: dict[str, str] = {
    "system_prompt":      "🔧",
    "user_message":       "👤",
    "assistant_message":  "🤖",
    "tool_call":          "⚙️",
    "tool_result":        "📦",
    "raw_messages":       "📋",
}

_TYPE_COLOR: dict[str, str] = {
    "system_prompt":      "#e8f4fd",
    "user_message":       "#f0f7f0",
    "assistant_message":  "#fafafa",
    "tool_call":          "#fff8e1",
    "tool_result":        "#fce4ec",
    "raw_messages":       "#f3e5f5",
}


def _pretty(payload: Any) -> str:
    """Return a pretty-printed JSON string for any payload."""
    if isinstance(payload, str):
        try:
            return json.dumps(json.loads(payload), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, TypeError):
            return payload
    return json.dumps(payload, indent=2, ensure_ascii=False, default=str)


def render_context_panel(system_prompt: str | None = None) -> None:
    """
    Render the full context panel into the current Streamlit column.

    Call this inside the right column after the chat column has been drawn,
    passing the current system prompt string if one is in use.

    Example
    -------
    col_chat, col_ctx = st.columns([3, 2])
    with col_chat:
        render_chat()
    with col_ctx:
        render_context_panel(system_prompt=SYSTEM_PROMPT)
    """
    st.markdown("### Context Panel")
    st.caption("Live view of what the model sees — messages, tool calls, and raw JSON.")

    # ── System Prompt ──────────────────────────────────────────────────────
    if system_prompt:
        with st.expander("🔧 System Prompt", expanded=False):
            st.code(system_prompt, language="markdown")

    # ── Full Message History (raw JSON) ────────────────────────────────────
    messages = st.session_state.get("messages", [])
    if messages:
        with st.expander(f"📋 Full Message History ({len(messages)} messages)", expanded=False):
            st.json(messages)

    st.divider()

    # ── Event log ──────────────────────────────────────────────────────────
    log: list[dict] = st.session_state.get("context_log", [])

    if not log:
        st.info("No events yet. Send a message to see the agent's reasoning here.")
        return

    st.markdown(f"**{len(log)} event(s) logged this session**")

    for i, event in enumerate(log):
        etype = event.get("type", "unknown")
        label = event.get("label") or etype.replace("_", " ").title()
        icon = _TYPE_ICON.get(etype, "📌")
        payload = event.get("payload")

        # Skip types already shown above to avoid duplication
        if etype in ("system_prompt", "raw_messages"):
            continue

        color = _TYPE_COLOR.get(etype, "#ffffff")
        title = f"{icon} [{i+1}] {label}"

        with st.expander(title, expanded=(etype in ("tool_call", "tool_result"))):
            if isinstance(payload, str):
                st.markdown(payload)
            else:
                st.code(_pretty(payload), language="json")


def render_tool_call_badge(tool_name: str, args: dict | None = None) -> None:
    """
    Display a compact inline badge for a tool call inside the chat panel.
    Call this in the chat column when streaming assistant messages.
    """
    args_str = json.dumps(args, ensure_ascii=False) if args else "{}"
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:#fff8e1;
            border:1px solid #f9a825;
            border-radius:6px;
            padding:4px 10px;
            font-size:0.82em;
            margin:4px 0;
            font-family:monospace;
        ">
        ⚙️ <strong>tool call</strong>: <code>{tool_name}</code><br/>
        <span style="color:#555">{args_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_tool_result_badge(tool_name: str, result: Any) -> None:
    """Display a compact inline badge for a tool result inside the chat panel."""
    result_str = _pretty(result)
    # Truncate long results in the badge
    if len(result_str) > 300:
        result_str = result_str[:297] + "…"
    st.markdown(
        f"""
        <div style="
            display:inline-block;
            background:#fce4ec;
            border:1px solid #e91e63;
            border-radius:6px;
            padding:4px 10px;
            font-size:0.82em;
            margin:4px 0;
            font-family:monospace;
        ">
        📦 <strong>tool result</strong>: <code>{tool_name}</code><br/>
        <span style="color:#555">{result_str}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
