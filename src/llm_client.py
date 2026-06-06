"""
llm_client.py
-------------
Single interface for LLM calls. Supports:
  - Ollama Cloud (gemma4:31b) as primary
  - Groq (llama-3.3-70b-versatile) as fallback

Uses raw httpx rather than LangChain's ChatOllama because Ollama Cloud
requires Bearer auth + follow_redirects=True, which ChatOllama doesn't
expose cleanly. Wrapped as a LangChain RunnableLambda for LCEL chains.
"""

import os
import httpx
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain_core.runnables import RunnableLambda

load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
LLM_PROVIDER    = os.getenv("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "https://ollama.com")
OLLAMA_API_KEY  = os.getenv("OLLAMA_API_KEY", "")
OLLAMA_MODEL    = os.getenv("OLLAMA_MODEL", "gemma4:31b")
GROQ_API_KEY    = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL      = "llama-3.3-70b-versatile"


# ── Message formatting helpers ────────────────────────────────────────────────

def _lc_to_ollama_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain message objects to Ollama API format."""
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
    }
    result = []
    for m in messages:
        role = role_map.get(m.type, "user")
        result.append({"role": role, "content": m.content})
    return result


def _lc_to_groq_messages(messages: list[BaseMessage]) -> list[dict]:
    """Convert LangChain messages to OpenAI-compatible format (Groq uses this)."""
    return _lc_to_ollama_messages(messages)  # same format


# ── Ollama Cloud call ─────────────────────────────────────────────────────────

def _call_ollama(messages: list[BaseMessage], temperature: float = 0.1) -> str:
    """
    Call Ollama Cloud API.
    Critical: follow_redirects=True — Ollama Cloud redirects the POST.
    """
    url = f"{OLLAMA_BASE_URL}/api/chat"
    payload = {
        "model": OLLAMA_MODEL,
        "messages": _lc_to_ollama_messages(messages),
        "stream": False,
        "options": {"temperature": temperature},
    }
    headers = {
        "Authorization": f"Bearer {OLLAMA_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(follow_redirects=True, timeout=120.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    return resp.json()["message"]["content"]


# ── Groq call ─────────────────────────────────────────────────────────────────

def _call_groq(messages: list[BaseMessage], temperature: float = 0.1) -> str:
    """Call Groq API (OpenAI-compatible endpoint)."""
    url = "https://api.groq.com/openai/v1/chat/completions"
    payload = {
        "model": GROQ_MODEL,
        "messages": _lc_to_groq_messages(messages),
        "temperature": temperature,
        "max_tokens": 1024,
    }
    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }
    with httpx.Client(timeout=60.0) as client:
        resp = client.post(url, json=payload, headers=headers)
        resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ── Unified call ──────────────────────────────────────────────────────────────

def call_llm(messages: list[BaseMessage], temperature: float = 0.1) -> str:
    """
    Call the configured LLM. Falls back to Groq if Ollama fails.
    Always uses temperature=0.1 for financial precision unless overridden.
    """
    provider = LLM_PROVIDER.lower()

    if provider == "ollama":
        try:
            return _call_ollama(messages, temperature)
        except Exception as e:
            print(f"[llm_client] Ollama failed ({e}), falling back to Groq")
            return _call_groq(messages, temperature)
    else:
        return _call_groq(messages, temperature)


# ── LangChain-compatible wrapper ──────────────────────────────────────────────

def make_llm_runnable(temperature: float = 0.1) -> RunnableLambda:
    """
    Returns a LangChain RunnableLambda that accepts a list of BaseMessage
    and returns a string. Drop-in for use in LCEL chains.

    Usage:
        llm = make_llm_runnable()
        chain = prompt | llm
    """
    def _invoke(messages: list[BaseMessage]) -> str:
        return call_llm(messages, temperature=temperature)

    return RunnableLambda(_invoke)
