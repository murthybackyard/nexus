"""nexus.vector_store — STUB.

Phase 1 split: this module exists only to provide the `session` symbol
that nexus.llm depends on. The full vector-store functionality remains
in streamlit_app.py for now and will be moved here in Phase 2.

The `session` is imported lazily by streamlit_app.py at runtime, which
sets `nexus.vector_store.session = ...` after creating it.
"""
session = None  # populated by streamlit_app.py at startup
