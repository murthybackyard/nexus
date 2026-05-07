"""
Main Streamlit tab visibility.

Edit defaults here, or override at runtime with environment variable
``NEXUS_APP_TAB_VISIBILITY`` (JSON object), e.g.::

    set NEXUS_APP_TAB_VISIBILITY={"quick_go":true,"reverse_engineering":false}

Keys: quick_go, reverse_engineering, forward_engineering, view_artifacts
"""
from __future__ import annotations

import json
import os
from typing import Dict, List, Tuple

TAB_KEYS_ORDER: List[str] = [
    "quick_go",
    "reverse_engineering",
    "forward_engineering",
    "view_artifacts",
]

TAB_LABELS: Dict[str, str] = {
    "quick_go": "⚡  Quick GO",
    "reverse_engineering": "⟲  Reverse Engineering",
    "forward_engineering": "⟳  Forward Engineering",
    "view_artifacts": "📚  View Artifacts",
}

# Default: all tabs visible
DEFAULT_TAB_VISIBILITY: Dict[str, bool] = {k: True for k in TAB_KEYS_ORDER}

#raw={"quick_go":true,"reverse_engineering":false,"forward_engineering":true,"view_artifacts":true}

def _parse_env_overrides() -> Dict[str, bool]:
    raw = os.environ.get("NEXUS_APP_TAB_VISIBILITY", "").strip()
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return {}
    if not isinstance(data, dict):
        return {}
    out: Dict[str, bool] = {}
    for k, v in data.items():
        if k in TAB_LABELS:
            out[k] = bool(v)
    return out


def get_tab_visibility() -> Dict[str, bool]:
    """Return visibility map; unknown keys ignored."""
    vis = dict(DEFAULT_TAB_VISIBILITY)
    vis.update(_parse_env_overrides())
    return vis


def build_tab_labels(vis: Dict[str, bool]) -> List[str]:
    """Labels for ``st.tabs`` in fixed order, skipping hidden tabs."""
    return [TAB_LABELS[k] for k in TAB_KEYS_ORDER if vis.get(k, True)]
