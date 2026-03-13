"""
Step 1 + 2 — Basic LLM Chat (no tools)
=======================================

STEP 1: Plain LLM chat — the assistant knows nothing about Lakewood University.
         Ask it "What time does the library close?" and watch it hallucinate or admit it doesn't know.

STEP 2: Same app, but now look at the Context Panel on the right to understand exactly
         what is being sent to the model (system prompt + message history).

Run with:
    streamlit run app_step1.py
"""

from __future__ import annotations

import streamlit as st

from llm_client import chat
from utils import add_context_event, init_context_log, render_context_panel

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful university assistant. "
    "Answer student questions as best you can."
)

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Step 1 — Basic LLM Chat",
    page_icon="🎓",
    layout="wide",
)

init_context_log()

# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("🎓 Campus Assistant — Step 1: Basic LLM Chat")

st.info(
    "**What to notice:** This is a plain LLM with no campus knowledge. "
    "Try asking *'What time does Hartwell Library close?'* or "
    "*'What events are coming up this week?'* — the model will either "
    "hallucinate or admit it doesn't know. "
    "Check the **Context Panel** on the right to see exactly what is sent to the model.",
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
        st.session_state["messages"].append({"role": "user", "content": user_input})
        add_context_event("user_message", user_input, label=f"User: {user_input[:60]}")

        with st.chat_message("user"):
            st.markdown(user_input)

        api_messages = [{"role": "system", "content": SYSTEM_PROMPT}] + st.session_state["messages"]

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    reply = chat(api_messages)
                except Exception as exc:
                    reply = f"⚠️ API error: {exc}"
            st.markdown(reply)

        st.session_state["messages"].append({"role": "assistant", "content": reply})
        add_context_event("assistant_message", reply, label=f"Assistant: {reply[:60]}")
        add_context_event("raw_messages", api_messages, label="Full messages sent to API")

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
