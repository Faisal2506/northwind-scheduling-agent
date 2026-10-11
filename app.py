"""Chat plus the tool log. Run with: streamlit run app.py"""

from __future__ import annotations

import json
import os
import traceback

import streamlit as st
from dotenv import load_dotenv

from agent.logs import log, recent, setup_logging
from agent.runtime import build_graph, build_prompt, load_reinforcement_files, make_chat_model, run_turn
from clinic.fixtures import make_session

st.set_page_config(page_title="Northwind Clinic", layout="wide")
load_dotenv()
setup_logging()


def _boot() -> None:
    if "session" in st.session_state:
        return
    patches = load_reinforcement_files()
    st.session_state.patches = patches
    st.session_state.prompt = build_prompt(patches)
    st.session_state.session = make_session()
    st.session_state.graph = build_graph(make_chat_model())
    st.session_state.chat = []
    st.session_state.turns = []


def _reset() -> None:
    for key in ("session", "graph", "prompt", "patches", "chat", "turns"):
        st.session_state.pop(key, None)
    _boot()


st.title("Northwind Clinic")
try:
    _boot()
except Exception as exc:
    from agent.logs import log

    log.exception("Northwind failed to start")
    st.error(str(exc))
    st.code(traceback.format_exc())
    st.stop()

session = st.session_state.session
caption = "Fictional patients only. The log is the clinic, not the model's wording."
if st.session_state.patches:
    caption += " Reinforcement: " + ", ".join(patch["id"] for patch in st.session_state.patches)
model_name = os.environ.get("AGENT_MODEL", "gemini")
if model_name == "fake":
    st.error(
        "This page is the offline stand-in, not Gemini. Stop this server (Ctrl+C) and start it again "
        "without AGENT_MODEL=fake."
    )
st.caption(caption + f" Model: {model_name}.")

with st.sidebar:
    st.subheader("Patients in this clinic")
    st.markdown(
        """
- Maya Chen, 1990-04-12
- Maya Patel, 1985-01-03
- Owen Brooks, 1978-11-20
- Priya Shah, 1992-07-07
        """
    )
    st.caption("Today in the clinic is 2026-10-12. A name that is not on this list comes back as not found.")
    if st.button("New conversation"):
        _reset()
        st.rerun()

def _show_log(turns: list[dict]) -> None:
    state = session.public_state()
    st.markdown(
        f"**Phase** `{state['phase']}`  \n"
        f"**Patient** `{state['resolved_patient_id'] or '—'}`  \n"
        f"**Last error** `{state['last_error'] or '—'}`"
    )
    st.markdown("**Calendar**")
    st.json(session.clinic.snapshot(), expanded=False)
    if not turns:
        st.info("Send a message. Each tool call is listed here and printed in the terminal.")
        return
    for index, turn in enumerate(reversed(turns), start=1):
        label = f"Turn {len(turns) - index + 1}: {turn['user'][:48]}"
        with st.expander(label, expanded=(index == 1)):
            st.markdown("**Patient**")
            st.write(turn["user"])
            st.markdown("**Agent**")
            st.write(turn["reply"])
            if not turn["tools"]:
                st.warning("No tool was called on this turn.")
            for call in turn["tools"]:
                st.markdown(f"**{call['tool']}**")
                st.code(
                    json.dumps({"args": call["args"], "result": call["result"]}, indent=2),
                    language="json",
                )


def _print_turn(user_text: str, reply: str, tools: list[dict]) -> None:
    print(f"\n--- patient: {user_text}", flush=True)
    if not tools:
        print("tool: none", flush=True)
    for call in tools:
        print(f"tool: {call['tool']}", flush=True)
        print(json.dumps(call["result"], indent=2), flush=True)
    print(f"agent: {reply}\n", flush=True)


chat_tab, log_tab = st.tabs(["Conversation", "Logs"])

with chat_tab:
    for message in st.session_state.chat:
        with st.chat_message(message["role"]):
            st.write(message["text"])
    if st.session_state.turns:
        last = st.session_state.turns[-1]
        names = ", ".join(call["tool"] for call in last["tools"]) or "no tool called"
        st.caption(f"Last turn tools: {names}. Open the Logs tab for the full trace.")

with log_tab:
    st.subheader("Python log")
    st.caption("Gemini HTTP calls and tool calls. The same lines are in the terminal running this app.")
    st.code(recent(), language="text")
    st.subheader("Clinic trace")
    _show_log(st.session_state.turns)

draft = st.chat_input("Message the scheduler")

if draft and draft.strip():
    user_text = draft.strip()
    before = len(session.tool_trace)
    with chat_tab:
        with st.spinner("Checking the clinic..."):
            try:
                reply = run_turn(session, user_text, st.session_state.prompt, st.session_state.graph)
            except Exception as exc:
                reply = str(exc)
                log.error("turn failed: %s", reply.split("\n", 1)[0][:300])
    tools = session.tool_trace[before:]
    _print_turn(user_text, reply, tools)
    st.session_state.chat.append({"role": "user", "text": user_text})
    st.session_state.chat.append({"role": "assistant", "text": reply})
    st.session_state.turns.append({"user": user_text, "reply": reply, "tools": tools})
    st.rerun()
