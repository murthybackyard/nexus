"""
nexus.llm — Snowflake Cortex LLM access.

Provides:
    MODELS          — supported chat-model names
    EMBED_MODELS    — supported embedding-model names
    call_cortex     — invoke a Cortex chat model and return raw text

The session must be created by the caller (typically nexus.vector_store).
This module imports the session lazily so it doesn't pull Snowflake
connection setup into utils-level callers.
"""

import json as _json
import re
from typing import Any, Dict, List, Optional


# Detect optional native Cortex Python API. Falls back
# to AI_COMPLETE via SQL if unavailable. Both paths use the active session.
try:
    from snowflake.cortex import complete as _cortex_complete
    HAS_CORTEX_PY = True
except ImportError:
    _cortex_complete = None
    HAS_CORTEX_PY = False

# Lazy session import to avoid circular imports.
def _get_session():
    """Lazy import of the Snowflake session created in nexus.vector_store.
    Importing at call time (not module load time) lets `from nexus.llm
    import call_cortex` work even before any Snowflake setup runs."""
    from nexus.vector_store import session
    return session

MODELS = {
    "Claude Opus 4.7": "claude-opus-4-7",
    "GPT-5":           "openai-gpt-5",
    "Gemini 3.1 Pro":  "gemini-3.1-pro",
}

def call_cortex(model_id: str, prompt: str, temperature: float = 0.2,
                max_tokens: int = 8000, top_p: float = None,
                guardrails: bool = None) -> str:
    """
    Call Cortex completion via the active Snowflake session.

    Uses snowflake.cortex.complete when available (snowflake-ml-python
    installed). Otherwise falls back to SELECT AI_COMPLETE(...) SQL,
    which only needs snowflake-snowpark-python.

    top_p and guardrails are optional - when None they are omitted from
    the options dict, so callers not needing them behave as before.
    """
    opts = {"temperature": temperature, "max_tokens": max_tokens}
    if top_p is not None:
        opts["top_p"] = top_p
    if guardrails is not None:
        opts["guardrails"] = guardrails

    # Preferred: native Python API
    if HAS_CORTEX_PY:
        try:
            return _cortex_complete(
                model=model_id,
                prompt=prompt,
                session=session,
                options=opts,
            )
        except Exception:
            pass  # fall through to SQL path

    # Fallback: AI_COMPLETE via SQL with OBJECT_CONSTRUCT.
    # We build the options object dynamically so the SQL matches `opts`.
    obj_pairs, obj_params = [], []
    for k, v in opts.items():
        obj_pairs.append(f"'{k}', ?")
        obj_params.append(v)
    obj_expr = f"OBJECT_CONSTRUCT({', '.join(obj_pairs)})"
    sql = f"""
        SELECT AI_COMPLETE(
            ?,
            ?,
            {obj_expr}
        ) AS response
    """
    row = _get_session().sql(
        sql, params=[model_id, prompt] + obj_params
    ).collect()
    return row[0]["RESPONSE"] if row else ""

EMBED_MODELS = {
    # label shown in UI : (cortex model name, dimension, table)
    "Arctic Embed M (768d, faster)": (
        "snowflake-arctic-embed-m", 768, "ARTIFACT_VECTORS_768"
    ),
    "Arctic Embed L v2.0 (1024d, higher quality)": (
        "snowflake-arctic-embed-l-v2.0", 1024, "ARTIFACT_VECTORS_1024"
    ),
}

