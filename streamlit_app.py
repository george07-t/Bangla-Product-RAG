"""
Streamlit UI for Bangla Product RAG.
Uses FastAPI backend at http://127.0.0.1:8000 by default.
"""
import uuid
import os
from typing import Any, Dict, List

import requests
import streamlit as st

st.set_page_config(page_title="Bangla Product RAG", page_icon="🛒", layout="wide")

DEFAULT_API_BASE = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")


def _init_state() -> None:
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "messages" not in st.session_state:
        st.session_state.messages = []


def _post_chat(api_base: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    url = f"{api_base.rstrip('/')}/chat"
    resp = requests.post(url, json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _post_reset(api_base: str, session_id: str) -> None:
    url = f"{api_base.rstrip('/')}/reset"
    requests.post(url, json={"session_id": session_id}, timeout=30)


def _render_debug(data: Dict[str, Any]) -> None:
    with st.expander("Response Details", expanded=False):
        c1, c2, c3 = st.columns(3)
        c1.metric("Retrieval (ms)", f"{data.get('retrieval_ms', 0):.2f}")
        c2.metric("LLM (ms)", f"{data.get('llm_ms', 0):.2f}")
        c3.metric("Total (ms)", f"{data.get('total_ms', 0):.2f}")

        st.write(f"Session turn: {data.get('session_turn')}")
        st.write(f"Rewritten: {data.get('was_rewritten')}")
        st.write(f"Tracked entity: {data.get('tracked_entity')}")
        st.write(f"Original query: {data.get('original_query')}")
        st.write(f"Rewritten query: {data.get('rewritten_query')}")

        products: List[Dict[str, Any]] = data.get("retrieved_products", [])
        if products:
            st.write("Top retrieved products")
            st.table(
                [
                    {
                        "name": p.get("name"),
                        "category": p.get("category"),
                        "price": p.get("price"),
                        "unit": p.get("unit"),
                        "score": round(float(p.get("_score", 0.0)), 4),
                    }
                    for p in products
                ]
            )


def main() -> None:
    _init_state()

    st.title("Bangla Product RAG")
    st.caption("Streamlit UI powered by FastAPI backend")

    with st.sidebar:
        st.header("Backend")
        api_base = st.text_input("FastAPI base URL", value=DEFAULT_API_BASE)

        st.header("Query Settings")
        response_mode = st.selectbox("Response mode", options=["fast", "llm"], index=0)
        llm_model = st.text_input("LLM model", value="gpt-4o-mini")
        top_k = st.slider("Top K", min_value=1, max_value=10, value=5)

        st.header("Session")
        st.text_input("Session ID", key="session_id")

        if st.button("New Session", use_container_width=True):
            st.session_state.session_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.success("New session created")

        if st.button("Reset Server Session", use_container_width=True):
            try:
                _post_reset(api_base, st.session_state.session_id)
                st.session_state.messages = []
                st.success("Server session reset")
            except Exception as ex:
                st.error(f"Reset failed: {ex}")

    for msg in st.session_state.messages:
        role = msg.get("role", "assistant")
        with st.chat_message(role):
            st.markdown(msg.get("content", ""))
            details = msg.get("details")
            if details and role == "assistant":
                _render_debug(details)

    prompt = st.chat_input("বাংলায় প্রশ্ন লিখুন...")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    payload = {
        "message": prompt,
        "session_id": st.session_state.session_id,
        "top_k": top_k,
        "llm_model": llm_model,
        "response_mode": response_mode,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                data = _post_chat(api_base, payload)
                answer = data.get("response", "No response")
                st.markdown(answer)
                _render_debug(data)
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "details": data}
                )
            except requests.HTTPError as http_ex:
                msg = f"API error: {http_ex}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})
            except Exception as ex:
                msg = f"Request failed: {ex}"
                st.error(msg)
                st.session_state.messages.append({"role": "assistant", "content": msg})


if __name__ == "__main__":
    main()
