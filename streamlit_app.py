"""
Data Engineering Co-Pilot — Snowflake Native Streamlit App
-----------------------------------------------------------
Claude Code-style UI for Reverse / Forward data engineering tasks.

Powered by Snowflake Cortex AI models:
  - claude-opus-4-7  (Anthropic)
  - openai-gpt-5     (OpenAI)
  - gemini-3.1-pro   (Google)

Authentication uses the active Snowflake session (no user/password needed).
Deploy as a Snowflake Native Streamlit App (Streamlit in Snowflake).
"""

import json
import re
import zipfile
import base64
import os
import time
from datetime import datetime
from io import BytesIO, StringIO
from typing import List, Optional
from urllib import error as urllib_error
from urllib import request as urllib_request

import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session

# Try to import the native Cortex Python API. It's only available when
# `snowflake-ml-python` is in the environment. If it isn't, we fall back
# to the AI_COMPLETE SQL function — both paths use the active session.

# ── nexus package imports — extracted modules ─────────────────────────────
# Phase 1 of split: pure-function modules (no Streamlit / Snowflake coupling).
# These have been moved out of this file but explicitly re-imported here so
# existing call sites continue to work without modification.
#
# We use explicit imports (not wildcards) because Python's `from X import *`
# skips names starting with underscore unless the module defines __all__.
# Most of our extracted helpers start with `_` (sanitizers, internal
# constants), so wildcards would silently miss them.
from nexus.utils import (
    _SQL_TYPE_CODES, _slug, _split_ddl_by_create_table,
    _split_narrative_by_section, _unwrap_json_string,
    clean_file_body, df_to_excel_bytes, extract_json,
    extract_mermaid_script, extract_sql_ddl,
    extract_text_from_upload, parse_markdown_table,
    parse_table_response, resolve_sql_type,
)
from nexus.llm import (EMBED_MODELS, MODELS, call_cortex,
                       HAS_CORTEX_PY, _cortex_complete)
from nexus.parsers import (
    _decode_dbt_planner_path_blob, _harvest_naked_blocks,
    _parse_dbt_json,
    _find_field, _parse_dsx_flat, _parse_dsx_xml,
    _scan_balanced_blocks, _split_top_level_commas,
    _sql_extract_objects, detect_tech_stack,
    parse_bods_xml, parse_control_m, parse_datastage_dsx,
    parse_dbt_plan, parse_dbt_project_from_response,
    parse_denodo_vql, parse_legacy_sql, parse_metadata_csv,
    parse_mssql_sql, parse_netezza_sql, parse_shell_script,
    parse_source_ddl_or_metadata,
    parse_ssis_dtsx, parse_sttm_csv,
)
from nexus.prompts import (
    _mermaid_safe_id, DASHBOARD_TYPES,
    MEDALLION_DV_FILE_CONSTRAINTS, MEDALLION_DV_FULL_SPEC,
    _DV_STANDARDS_BY_ARTIFACT, _DV_STD_ABBREVIATIONS,
    _DV_STD_HASH, _DV_STD_LINK_RULES, _DV_STD_METADATA_COLS,
    _DV_STD_NAMING, _DV_STD_SATELLITE_RULES, _dv_standards_block,
    build_business_vault_dbt_prompt,
    build_business_vault_mermaid_prompt,
    build_business_vault_narrative_prompt,
    build_business_vault_sql_prompt,
    build_data_catalog_prompt, build_data_domain_prompt,
    build_dbt_file_prompt, build_dbt_planner_prompt,
    build_dbt_tests_prompt, build_forward_catalog_prompt,
    build_forward_domains_prompt, build_forward_sttm_prompt,
    build_lineage_mermaid, build_raw_vault_dbt_prompt,
    build_raw_vault_mermaid_prompt, build_raw_vault_narrative_prompt,
    build_raw_vault_sql_prompt, build_raw_vault_validation_prompt,
    build_semantic_model_mermaid_prompt,
    build_semantic_model_prompt, build_semantic_model_sql_prompt,
    build_source_to_hub_prompt, build_sttm_prompt,
    build_transformation_rules_prompt,
)
from nexus.app_tabs_config import build_tab_labels, get_tab_visibility
from nexus.prompt_instructions import (
    CHAT_ASSISTANT_SYSTEM,
    forward_business_vault_dbt_codegen_context,
    forward_raw_vault_dbt_codegen_context,
    quick_go_business_vault_dbt_codegen_context,
    quick_go_raw_vault_dbt_codegen_context,
)
from nexus.dbt_sanitizers import (
    VECTOR_DB, VECTOR_SCHEMA, VECTOR_STAGE,
    NATIVE_DBT_PROFILE_ACCOUNT, NATIVE_DBT_PROFILE_USER,
    NATIVE_DBT_PROFILE_ROLE, NATIVE_DBT_PROFILE_WAREHOUSE,
    RAW_SOURCE_SCHEMA, RAW_VAULT_SCHEMA, BUSINESS_VAULT_SCHEMA,
    SNOWFLAKE_NATIVE_DBT_VERSION,
    _SCAFFOLD_PACKAGES_YML, _SCAFFOLD_DBT_PROJECT_RV,
    _SCAFFOLD_DBT_PROJECT_BV, _RECORD_SOURCE_MACRO,
    _SCAFFOLD_PROFILES_YML,
    _AUDIT_COL_SAFE_SQL, _DBT_NODE_PREFIX_RE,
    _best_stage_model_match, _build_stub_staging_model,
    _ensure_bronze_shims_for_missing_refs,
    _ensure_dbt_runtime_profiles, _ensure_forward_dbt_scaffold,
    _ensure_raw_source_catalog, _final_native_dbt_safety_sanitize,
    _infer_raw_columns_from_dv_metadata,
    _infer_raw_tables_from_model_paths,
    _infer_staging_metadata_from_callers,
    _native_snowflake_profiles_yml, _normalize_dbt_node_refs,
    _normalize_generated_file_body,
    _normalize_source_raw_table_casing,
    _normalize_sources_yml_table_casing,
    _packages_yml_requires_dbt_deps,
    _pin_dbt_project_object_version,
    _repair_missing_stage_references,
    _sanitize_dbt_project_yml_for_snowflake_native,
    _sanitize_jinja_in_yaml_metadata,
    _split_sql_script_into_statements, _sql_dbt_version_suffix,
    _strip_disallowed_show_parameter_sql,
    _strip_hooks_from_sql_model, _strip_query_tag_from_sql_model,
    _validated_native_dbt_version, _verify_dbt_project_version,
)

# Bind the Snowflake session into nexus.vector_store so nexus.llm.call_cortex
# can find it. The session is created later in this file via get_session().
import nexus.vector_store as _nx_vs  # noqa: E402
# `_nx_vs.session = session` is set right after `session = get_session()` below.
# ──────────────────────────────────────────────────────────────────────────




# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Data Engineering Co-Pilot",
    page_icon="✴",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────────────────────────────────────
# CLAUDE CODE-STYLE CSS
# ─────────────────────────────────────────────────────────────────────────────
CLAUDE_CSS = """
<style>
    /* Import Claude's typeface fallbacks */
    @import url('https://fonts.googleapis.com/css2?family=Tiempos+Headline&family=Inter:wght@400;500;600&display=swap');

    /* Base app background */
    .stApp {
        background: #FFFFFF;
        color: #1F2937;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: #FFFFFF;
        border-right: 1px solid #E5E7EB;
    }
    section[data-testid="stSidebar"] .stMarkdown h3 {
        font-size: 13px;
        font-weight: 500;
        color: #3D3929;
        margin-bottom: 4px;
    }

    /* Main title */
    .claude-greeting {
        font-family: 'Tiempos Headline', 'Georgia', serif;
        font-size: 44px;
        font-weight: 400;
        color: #141413;
        text-align: center;
        margin-top: 40px;
        margin-bottom: 8px;
        letter-spacing: -0.5px;
    }
    .claude-star {
        color: #C96442;
        font-size: 36px;
        margin-right: 12px;
    }
    .claude-subtitle {
        text-align: center;
        color: #6B6456;
        font-size: 14px;
        margin-bottom: 32px;
    }
    .plan-badge {
        text-align: center;
        margin-bottom: 20px;
    }
    .plan-badge span {
        background: #F3F4F6;
        color: #374151;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 12px;
    }

    /* ─── Chat input — Claude Code cream aesthetic ─────────────────────
       Snowflake's SiS iframe applies its own theme that overrides
       config.toml. We use max-specificity selectors (html body chain)
       and universal descendants to defeat emotion-cache class churn. */

    /* STEP 1: paint the bottom dock area cream */
    html body div[data-testid="stBottom"],
    html body div[data-testid="stBottom"] > *,
    html body div[data-testid="stBottomBlockContainer"],
    html body div[data-testid="stAppViewContainer"] > footer {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border-top: 1px solid #E5E7EB !important;
        box-shadow: none !important;
    }
    html body div[data-testid="stBottom"]::before,
    html body div[data-testid="stBottom"]::after {
        display: none !important;
        background: transparent !important;
    }

    /* STEP 2: force transparency on EVERY descendant of chat_input */
    html body div[data-testid="stChatInput"] *,
    html body div[data-testid="stChatInputTextArea"] *,
    html body .stChatInput * {
        background: transparent !important;
        background-color: transparent !important;
    }

    /* STEP 3: paint the pill wrapper white — target the actual
       form element AND every likely container candidate.
       stChatInput contains a <form> which is the real pill. */
    html body div[data-testid="stChatInput"] form,
    html body div[data-testid="stChatInput"] > div,
    html body .stChatInput form,
    html body .stChatInput > div {
        background: #FFFFFF !important;
        background-color: #FFFFFF !important;
        border: 1px solid #E8E6DC !important;
        border-radius: 20px !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.04) !important;
        padding: 4px 8px 4px 16px !important;
    }
    html body div[data-testid="stChatInput"] form:focus-within,
    html body .stChatInput form:focus-within {
        border-color: #C96442 !important;
        box-shadow: 0 0 0 3px rgba(201, 100, 66, 0.15) !important;
    }

    /* STEP 4: textarea text + caret */
    html body div[data-testid="stChatInput"] textarea,
    html body .stChatInput textarea {
        color: #141413 !important;
        font-size: 15px !important;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
        caret-color: #C96442 !important;
        border: none !important;
        outline: none !important;
        box-shadow: none !important;
    }
    html body div[data-testid="stChatInput"] textarea::placeholder {
        color: #A39C8B !important;
        opacity: 1 !important;
    }

    /* STEP 5: send button — orange pill */
    html body div[data-testid="stChatInput"] button,
    html body div[data-testid="stChatInputSubmitButton"],
    html body .stChatInput button {
        background: #C96442 !important;
        background-color: #C96442 !important;
        border: none !important;
        border-radius: 50% !important;
        width: 32px !important;
        height: 32px !important;
        min-width: 32px !important;
        color: #FFFFFF !important;
    }
    html body div[data-testid="stChatInput"] button:hover {
        background: #B85537 !important;
        background-color: #B85537 !important;
    }
    html body div[data-testid="stChatInput"] button:disabled {
        background: #E8E6DC !important;
        background-color: #E8E6DC !important;
    }
    html body div[data-testid="stChatInput"] button svg,
    html body div[data-testid="stChatInput"] button svg path,
    html body div[data-testid="stChatInput"] button * {
        fill: #FFFFFF !important;
        color: #FFFFFF !important;
        stroke: #FFFFFF !important;
    }
    html body div[data-testid="stChatInput"] button:disabled svg,
    html body div[data-testid="stChatInput"] button:disabled svg path {
        fill: #A39C8B !important;
        stroke: #A39C8B !important;
    }

    /* Tabs styled like Claude pills */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: transparent;
        border-bottom: none;
        justify-content: center;
        margin-top: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 10px;
        padding: 8px 20px;
        font-weight: 500;
        font-size: 14px;
        color: #3D3929;
    }
    .stTabs [aria-selected="true"] {
        background: #141413 !important;
        color: #FFFFFF !important;
        border-color: #141413 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 10px;
        color: #3D3929;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.15s ease;
    }
    .stButton > button:hover {
        background: #F9FAFB;
        border-color: #D1D5DB;
    }

    /* Primary button */
    .stButton > button[kind="primary"] {
        background: #C96442;
        color: #FFFFFF;
        border-color: #C96442;
    }
    .stButton > button[kind="primary"]:hover {
        background: #B85537;
        border-color: #B85537;
    }

    /* Chat message bubbles */
    .stChatMessage {
        background: transparent !important;
        border: none !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 10px;
        font-weight: 500;
    }

    /* Selectbox */
    .stSelectbox > div > div {
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 10px;
    }

    /* Hide Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E8E6DC;
        border-radius: 12px;
        padding: 16px;
    }

    /* Bottom model picker — scoped by Streamlit key so it doesn't affect
       other selectboxes elsewhere in the app */
    div[class*="st-key-model_picker_bottom"] div[data-baseweb="select"] > div {
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        min-height: 28px !important;
        color: #6B6456 !important;
        font-size: 13px !important;
    }
    div[class*="st-key-model_picker_bottom"] div[data-baseweb="select"] > div:hover {
        background: #F3F4F6 !important;
        border-radius: 10px !important;
    }
    div[class*="st-key-model_picker_bottom"] svg {
        color: #6B6456 !important;
    }

</style>
"""
st.markdown(CLAUDE_CSS, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# SNOWFLAKE SESSION (uses logged-in active session — no credentials needed)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def get_session():
    """Return the active Snowflake session inside Streamlit in Snowflake."""
    return get_active_session()


session = get_session()


_nx_vs.session = session  # nexus modules need this binding
# ─────────────────────────────────────────────────────────────────────────────
# MODEL REGISTRY
# ─────────────────────────────────────────────────────────────────────────────




















def build_artifacts_bundle(artifacts: dict,
                           source_filename: str = "",
                           metadata_summary: str = "") -> bytes:
    """
    Pack every generated artifact into a single ZIP for one-click
    download. Each artifact is materialized in its most useful formats:

      Raw Vault Model   → .md (narrative), .mermaid (ER), .sql (DDL),
                          combined .md bundle
      STTM              → .xlsx, .csv
      Data Catalog      → .xlsx, .csv
      Data Domain       → .xlsx, .csv
      Data Lineage      → .mermaid, source_to_hub.csv (when available),
                          graph.json, entity_column_inventory.csv/xlsx,
                          stage_to_stage_flows.csv/xlsx,
                          interactive.html (standalone SVG viewer)

    Also includes a top-level README.md, the raw metadata_summary used
    to generate everything, and a manifest.json enumerating the bundle.
    """
    buf = BytesIO()
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%SZ")
    manifest = {
        "generated_utc": stamp,
        "source_files":  source_filename,
        "artifacts":     [],
    }

    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # ── Top-level README ─────────────────────────────────────────
        readme = [
            "# Data Engineering Co-Pilot — Artifacts Bundle",
            "",
            f"**Generated (UTC):** {stamp}",
            f"**Source files:** {source_filename or '—'}",
            "",
            "## Contents",
            "",
        ]

        # ── Raw metadata summary used as LLM input ──────────────────
        if metadata_summary:
            zf.writestr("00_metadata_summary.md", metadata_summary)
            readme.append("- `00_metadata_summary.md` — parsed metadata "
                          "used as LLM context")

        for key, art in artifacts.items():
            kind    = art.get("kind", "")
            label   = art.get("label", key)
            content = art.get("content", {}) or {}
            folder  = _slug(key)
            entry = {"key": key, "label": label, "kind": kind,
                     "files": []}

            if kind == "raw_vault":
                # Narrative Markdown
                nm = content.get("narrative_md") or ""
                if nm.strip():
                    p = f"{folder}/narrative.md"
                    zf.writestr(p, nm)
                    entry["files"].append(p)
                # Mermaid
                mer = content.get("mermaid") or ""
                if mer.strip():
                    p = f"{folder}/er_diagram.mermaid"
                    zf.writestr(p, mer)
                    entry["files"].append(p)
                # SQL DDL
                sql = content.get("sql") or ""
                if sql.strip():
                    p = f"{folder}/raw_vault.sql"
                    zf.writestr(p, sql)
                    entry["files"].append(p)
                # Combined .md bundle (same format as the single-button
                # download already in the UI)
                bundle = (
                    f"# {label}\n\n{nm}\n\n"
                    f"## ER Diagram\n\n```mermaid\n{mer}\n```\n\n"
                    f"## Snowflake DDL\n\n```sql\n{sql}\n```\n"
                )
                p = f"{folder}/{_slug(key)}.md"
                zf.writestr(p, bundle)
                entry["files"].append(p)

            elif kind == "table":
                df = content.get("df")
                raw = content.get("raw", "") or ""
                if df is not None and not df.empty:
                    # XLSX (primary)
                    try:
                        xb = df_to_excel_bytes(df, sheet_name=label[:31])
                        p = f"{folder}/{_slug(key)}.xlsx"
                        zf.writestr(p, xb)
                        entry["files"].append(p)
                    except Exception as e:
                        entry["xlsx_error"] = str(e)
                    # CSV (always)
                    p = f"{folder}/{_slug(key)}.csv"
                    zf.writestr(p, df.to_csv(index=False))
                    entry["files"].append(p)
                # Raw LLM response — useful for debugging parse failures
                if raw:
                    p = f"{folder}/_raw_response.txt"
                    zf.writestr(p, raw)
                    entry["files"].append(p)

            elif kind == "lineage":
                mer = content.get("mermaid") or ""
                s2h_df = content.get("source_to_hub")
                rv_tables = content.get("rv_tables", {}) or {}
                graph_data = content.get("graph", {}) or {}
                mapping_raw = content.get("mapping_raw", "") or ""

                if mer.strip():
                    p = f"{folder}/lineage.mermaid"
                    zf.writestr(p, mer)
                    entry["files"].append(p)

                if s2h_df is not None and not s2h_df.empty:
                    p = f"{folder}/source_to_hub.csv"
                    zf.writestr(p, s2h_df.to_csv(index=False))
                    entry["files"].append(p)
                    try:
                        xb = df_to_excel_bytes(
                            s2h_df, sheet_name="Source to Hub"
                        )
                        p = f"{folder}/source_to_hub.xlsx"
                        zf.writestr(p, xb)
                        entry["files"].append(p)
                    except Exception:
                        pass

                # Structured graph as JSON (for external tooling)
                if graph_data.get("nodes"):
                    # Strip the internal layout fields we add on render;
                    # export just the semantic graph.
                    clean_nodes = [
                        {k: v for k, v in n.items()
                         if k not in ("_y_idx", "_layer_size", "layer",
                                      "x", "y", "w", "h")}
                        for n in graph_data.get("nodes", [])
                    ]
                    clean = {"nodes": clean_nodes,
                             "edges": graph_data.get("edges", [])}
                    p = f"{folder}/graph.json"
                    zf.writestr(p, json.dumps(clean, indent=2))
                    entry["files"].append(p)

                # Raw Vault tables discovered in DDL (for quick lookup)
                if rv_tables:
                    p = f"{folder}/raw_vault_tables.json"
                    zf.writestr(p, json.dumps(rv_tables, indent=2))
                    entry["files"].append(p)

                if mapping_raw:
                    p = f"{folder}/_mapping_raw_response.txt"
                    zf.writestr(p, mapping_raw)
                    entry["files"].append(p)

                # Tabular lineage views (Quick GO Phase 1) — derived
                # from the parsed Reverse Engineering Inputs.
                inv_df = content.get("inventory_df")
                if inv_df is not None and not inv_df.empty:
                    p = f"{folder}/entity_column_inventory.csv"
                    zf.writestr(p, inv_df.to_csv(index=False))
                    entry["files"].append(p)
                    try:
                        xb = df_to_excel_bytes(
                            inv_df, sheet_name="Entity & Column Inventory"
                        )
                        p = f"{folder}/entity_column_inventory.xlsx"
                        zf.writestr(p, xb)
                        entry["files"].append(p)
                    except Exception:
                        pass

                flows_df = content.get("flows_df")
                if flows_df is not None and not flows_df.empty:
                    p = f"{folder}/stage_to_stage_flows.csv"
                    zf.writestr(p, flows_df.to_csv(index=False))
                    entry["files"].append(p)
                    try:
                        xb = df_to_excel_bytes(
                            flows_df,
                            sheet_name="Stage-to-Stage Flows",
                        )
                        p = f"{folder}/stage_to_stage_flows.xlsx"
                        zf.writestr(p, xb)
                        entry["files"].append(p)
                    except Exception:
                        pass

            elif kind == "dbt_project":
                # Each dbt project has a files dict: {relative_path: body}.
                # Preserve the full tree inside this artifact's folder.
                files = content.get("files") or {}
                raw   = content.get("raw", "") or ""
                for rel_path, body in files.items():
                    # Strip any leading slashes and prevent escaping
                    safe = rel_path.lstrip("/").replace("..", "")
                    p = f"{folder}/project/{safe}"
                    zf.writestr(p, body)
                    entry["files"].append(p)
                if raw:
                    p = f"{folder}/_codegen_raw_response.md"
                    zf.writestr(p, raw)
                    entry["files"].append(p)
                # Also bundle it as a standalone zip for easy deploy
                try:
                    project_name = _slug(key).lower()
                    pz_bytes = bundle_dbt_project(files, project_name)
                    p = f"{folder}/{project_name}.zip"
                    zf.writestr(p, pz_bytes)
                    entry["files"].append(p)
                except Exception:
                    pass

            elif kind == "markdown":
                # Free-form Markdown artifacts (e.g. Transformation Rules
                # & Business Logic from Quick GO Phase 1).
                md = content.get("markdown", "") or ""
                raw_md = content.get("raw", "") or ""
                if md.strip():
                    p = f"{folder}/{_slug(key)}.md"
                    zf.writestr(p, md)
                    entry["files"].append(p)
                if raw_md and raw_md != md:
                    p = f"{folder}/_raw_response.txt"
                    zf.writestr(p, raw_md)
                    entry["files"].append(p)

            elif kind == "validation":
                # Raw Vault Validation report — JSON + raw response.
                vj = content.get("json")
                raw_v = content.get("raw", "") or ""
                if vj is not None:
                    p = f"{folder}/{_slug(key)}.json"
                    try:
                        zf.writestr(p, json.dumps(vj, indent=2))
                        entry["files"].append(p)
                    except Exception:
                        pass
                if raw_v:
                    p = f"{folder}/_raw_response.txt"
                    zf.writestr(p, raw_v)
                    entry["files"].append(p)

            readme.append(f"- **{label}** (`{folder}/`)")
            for p in entry["files"]:
                readme.append(f"    - `{p}`")
            manifest["artifacts"].append(entry)

        zf.writestr("README.md", "\n".join(readme) + "\n")
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    return buf.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# VECTOR STORE
#
# Pipeline: artifacts → internal stage (versioned) → chunker →
#           Cortex embed → VECTOR column in Snowflake → semantic search.
#
# Design notes:
#   • One table per embedding dimension because Snowflake's VECTOR(FLOAT, N)
#     is fixed-size. 768 for arctic-embed-m, 1024 for arctic-embed-l-v2.0.
#   • Rows keyed by (version, data_domain, artifact_type, chunk_id) so a
#     single table serves every artifact and re-loading a version is an
#     idempotent DELETE-then-INSERT on (version).
#   • Data Domain artifact is the source of truth for entity → domain.
#     Everything else gets domain via lookup; unresolvable → UNCLASSIFIED.
# ─────────────────────────────────────────────────────────────────────────────


# Fixed Snowflake-native dbt profile for EXECUTE DBT PROJECT (not session-derived).
# Schema where dbt RAW VAULT models (silver/raw_vault/* — hubs, links,
# sats, staging) materialize. Distinct from RAW (landing) and from
# the dbt profile target schema (which holds Bronze + Gold by default).
# Schema where dbt BUSINESS VAULT models materialize. Same separation
# pattern as RAW_VAULT.

# Diagnostic: how many `generate_schema_name` macro definitions were
# stripped from a given prepared file map. Keyed by id(out_dict). Used
# by deploy sites to surface a confirmation message to the user so we
# can verify the strip ran.
_NEXUS_STRIP_COUNTERS: dict = {}




# Snowflake-managed dbt Core version for CREATE / EXECUTE DBT PROJECT.
# 1.10.15 has a runtime bug: the EXECUTE DBT PROJECT stored procedure runs
# `show parameters like 'TELEMETRY_ENABLE_INTERACTIVE_MODE' in session`
# during startup (in _udf_code.py line 1251), and the same stored procedure
# rejects SHOW PARAMETER as "Unsupported statement type". Self-conflict bug.
# 1.9.4 doesn't include that telemetry probe and runs cleanly.
# Run `SHOW DBT VERSIONS;` in Snowsight to see what's available in your account.






def _sql_single_quoted_literal(value: str) -> str:
    """Escape a Python string for use inside a SQL single-quoted literal."""
    return (value or "").replace("'", "''")












# Mapping from common audit-column names to safe SQL expressions to use
# when replacing Jinja inside `{%- set yaml_metadata -%}` blocks.
# These values become raw SQL fragments emitted by automate_dv (or our
# dv_*) macros, so they must be valid Snowflake SQL — not Jinja.


















def _fqn(table: str) -> str:
    """Fully-qualified table name."""
    return f"{VECTOR_DB}.{VECTOR_SCHEMA}.{table}"


def ensure_vector_infrastructure(session):
    """
    Create the database, schema, stage, and vector tables if they don't
    exist. Idempotent. Called once per persist / vectorization run.

    The fix here addresses the most common cause of "artifacts are not
    getting stored in stage" — the schema or database not existing in
    the target account. We CREATE IF NOT EXISTS at every level (DB →
    schema → stage → tables) and surface a clear error if any step
    fails so the user can see which permission they're missing.
    """
    # 1. Database — CREATE IF NOT EXISTS. If the role lacks CREATE
    #    DATABASE, fall back to USE DATABASE to confirm it's reachable.
    try:
        session.sql(
            f"CREATE DATABASE IF NOT EXISTS {VECTOR_DB}"
        ).collect()
    except Exception as e_db:
        # The role may not have CREATE DATABASE privilege but the DB
        # might already exist — try to USE it before giving up.
        try:
            session.sql(f"USE DATABASE {VECTOR_DB}").collect()
        except Exception as e_use:
            raise RuntimeError(
                f"Cannot create or use database {VECTOR_DB}. "
                f"CREATE failed: {e_db}. USE failed: {e_use}. "
                f"Ask an admin to GRANT USAGE on this database to "
                f"your current role, or to create the database for you."
            )

    # 2. Schema — CREATE IF NOT EXISTS, fully qualified.
    try:
        session.sql(
            f"CREATE SCHEMA IF NOT EXISTS {VECTOR_DB}.{VECTOR_SCHEMA}"
        ).collect()
    except Exception as e_sch:
        try:
            session.sql(
                f"USE SCHEMA {VECTOR_DB}.{VECTOR_SCHEMA}"
            ).collect()
        except Exception as e_use_sch:
            raise RuntimeError(
                f"Cannot create or use schema "
                f"{VECTOR_DB}.{VECTOR_SCHEMA}. CREATE failed: {e_sch}. "
                f"USE failed: {e_use_sch}. Ask an admin to GRANT USAGE "
                f"on this schema to your current role."
            )

    # 3. Stage — must be CREATEd in the target schema. The fully-
    #    qualified name handles cases where the session's default
    #    schema differs.
    try:
        session.sql(
            f"CREATE STAGE IF NOT EXISTS {_fqn(VECTOR_STAGE)} "
            f"COMMENT = 'Versioned staging area for Data Engineering "
            f"Co-Pilot artifacts'"
        ).collect()
    except Exception as e_stg:
        raise RuntimeError(
            f"Cannot create stage {_fqn(VECTOR_STAGE)}. {e_stg}. "
            f"Ask an admin to GRANT CREATE STAGE on schema "
            f"{VECTOR_DB}.{VECTOR_SCHEMA} to your current role."
        )

    # 4. Vector tables — same idempotent pattern.
    for _, (_, dim, table) in EMBED_MODELS.items():
        ddl = (
            f"CREATE TABLE IF NOT EXISTS {_fqn(table)} ("
            f"  ROW_ID STRING NOT NULL,"
            f"  VERSION STRING NOT NULL,"
            f"  DATA_DOMAIN STRING NOT NULL,"
            f"  ARTIFACT_TYPE STRING NOT NULL,"
            f"  ARTIFACT_LABEL STRING,"
            f"  CHUNK_ID STRING NOT NULL,"
            f"  ENTITY STRING,"
            f"  CONTENT STRING NOT NULL,"
            f"  METADATA VARIANT,"
            f"  EMBEDDING_MODEL STRING NOT NULL,"
            f"  EMBEDDING VECTOR(FLOAT, {dim}) NOT NULL,"
            f"  CREATED_AT TIMESTAMP_NTZ DEFAULT CURRENT_TIMESTAMP(),"
            f"  CONSTRAINT PK_{table} PRIMARY KEY (ROW_ID)"
            f") "
            f"COMMENT = 'Vector store for Data Engineering Co-Pilot - "
            f"dimension {dim}'"
        )
        try:
            session.sql(ddl).collect()
        except Exception as e_tbl:
            # Vector tables are optional for the staging path — don't
            # block stage upload over a vector-table failure. Surface
            # via a runtime warning the caller can show.
            import warnings
            warnings.warn(
                f"Could not create vector table {_fqn(table)}: {e_tbl}"
            )


def extract_domain_map(artifacts: dict) -> dict:
    """
    Build {entity_name_upper → domain} from the Data Domain artifact.
    Returns {} if the Data Domain artifact wasn't generated.
    """
    dom = artifacts.get("Data Domain", {}) or {}
    df = (dom.get("content") or {}).get("df")
    if df is None or df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    ent_col = cols.get("entity") or cols.get("entity_name")
    dom_col = cols.get("domain")
    if not ent_col or not dom_col:
        return {}
    mapping = {}
    for _, row in df.iterrows():
        ent = str(row[ent_col]).strip().upper()
        dm  = str(row[dom_col]).strip()
        if ent and dm:
            mapping[ent] = dm
    return mapping


def _resolve_domain(candidates: list, domain_map: dict) -> str:
    """
    Try each candidate string against the domain map, case-insensitively.
    For Hub/Link/Sat names we strip the prefix and try again.

    We also match substring-wise when a Hub name's stem is a prefix/suffix
    of a mapped entity name — this bridges the common case where Data
    Domain maps source entities like "CustomerFeed" but the Raw Vault has
    a consolidated "HUB_CUSTOMER" without the "Feed" suffix.
    """
    def _direct_lookup(s):
        if s in domain_map:
            return domain_map[s]
        # Substring bridge: any mapped entity contain s, or s contain
        # any mapped entity? Require ≥4 chars to avoid spurious matches
        # on generic fragments like "ID" or "DT".
        if len(s) >= 4:
            for mapped_ent, dm in domain_map.items():
                if s in mapped_ent or mapped_ent in s:
                    return dm
        return None

    for c in candidates:
        if not c:
            continue
        s = str(c).strip().upper()
        # Try the name as-is
        hit = _direct_lookup(s)
        if hit:
            return hit
        # Try stripping Data Vault prefixes
        for prefix in ("HUB_", "LNK_", "LINK_", "SAT_"):
            if s.startswith(prefix):
                stripped = s[len(prefix):]
                hit = _direct_lookup(stripped)
                if hit:
                    return hit
                break
    return "UNCLASSIFIED"






def _split_mermaid_entities(mermaid: str) -> list:
    """Split a Mermaid diagram into one chunk per entity block."""
    if not mermaid or not mermaid.strip():
        return []
    chunks = []
    # erDiagram entities: ENTITY_NAME { ... }
    for m in re.finditer(
        r'^([A-Z_][\w_]*)\s*\{([^}]*)\}',
        mermaid, re.MULTILINE
    ):
        entity = m.group(1)
        chunks.append((entity, m.group(0)))
    if not chunks:
        # Flowchart — one chunk per node declaration
        for m in re.finditer(
            r'^\s*([A-Za-z_]\w*)\s*\[([^\]]+)\]',
            mermaid, re.MULTILINE
        ):
            chunks.append((m.group(1), m.group(0).strip()))
    if not chunks:
        # Fallback: whole diagram as one chunk
        chunks.append(("(diagram)", mermaid.strip()))
    return chunks


def chunk_artifacts(artifacts: dict, domain_map: dict) -> list:
    """
    Turn every artifact into a list of chunks suitable for embedding.
    Each chunk is:
        {
          "artifact_type": str,  # key in artifacts dict
          "artifact_label": str,
          "chunk_id": str,       # unique within (artifact_type, version)
          "entity": str,         # for domain resolution & filtering
          "data_domain": str,    # resolved via domain_map
          "content": str,        # the text to embed
          "metadata": dict,      # free-form, stored as VARIANT
        }
    Chunking strategy varies by artifact kind:
      table  → one chunk per row
      raw_vault → narrative by section, DDL by CREATE TABLE
      lineage → one chunk per source-to-hub mapping + per Mermaid entity
    """
    chunks = []

    for key, art in artifacts.items():
        kind    = art.get("kind", "")
        label   = art.get("label", key)
        content = art.get("content", {}) or {}

        if kind == "raw_vault":
            # Narrative → one chunk per section heading
            narrative = content.get("narrative_md") or ""
            for i, (heading, body) in enumerate(
                _split_narrative_by_section(narrative)
            ):
                chunks.append({
                    "artifact_type":  f"{key} — Narrative",
                    "artifact_label": label,
                    "chunk_id":       f"narrative_{i:03d}",
                    "entity":         heading,
                    "data_domain":    _resolve_domain([heading], domain_map),
                    "content":        body,
                    "metadata":       {"section": heading,
                                       "source": "narrative"},
                })
            # DDL → one chunk per CREATE TABLE
            ddl = content.get("sql") or ""
            for i, stmt in enumerate(_split_ddl_by_create_table(ddl)):
                # Extract table name from the statement
                m = re.search(
                    r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+'
                    r'(?:[A-Za-z_][\w]*\.)?([A-Za-z_][\w]*)',
                    stmt, re.IGNORECASE
                )
                tname = m.group(1).upper() if m else f"stmt_{i}"
                chunks.append({
                    "artifact_type":  f"{key} — DDL",
                    "artifact_label": label,
                    "chunk_id":       f"ddl_{tname}",
                    "entity":         tname,
                    "data_domain":    _resolve_domain([tname], domain_map),
                    "content":        stmt,
                    "metadata":       {"table_name": tname,
                                       "source": "ddl"},
                })
            # Mermaid → per-entity chunks (optional; indexing helps)
            mermaid = content.get("mermaid") or ""
            for i, (entity, body) in enumerate(
                _split_mermaid_entities(mermaid)
            ):
                chunks.append({
                    "artifact_type":  f"{key} — ER Diagram",
                    "artifact_label": label,
                    "chunk_id":       f"er_{entity}_{i:03d}",
                    "entity":         entity,
                    "data_domain":    _resolve_domain([entity], domain_map),
                    "content":        body,
                    "metadata":       {"entity": entity,
                                       "source": "mermaid"},
                })

        elif kind == "table":
            df = content.get("df")
            if df is None or df.empty:
                continue
            # Try to find an entity column for domain resolution
            cols = {c.lower(): c for c in df.columns}
            ent_col = (cols.get("entity") or cols.get("entity_name")
                       or cols.get("source table") or cols.get("source_table")
                       or cols.get("target table") or cols.get("target_table"))
            for i, row in df.iterrows():
                # Render the row as "col: val" lines — dense but readable
                lines = []
                for c in df.columns:
                    v = str(row[c]) if row[c] is not None else ""
                    if v and v.lower() != "nan":
                        lines.append(f"{c}: {v}")
                content_text = "\n".join(lines)
                ent = str(row[ent_col]).strip() if ent_col else ""
                chunks.append({
                    "artifact_type":  key,
                    "artifact_label": label,
                    "chunk_id":       f"row_{i:05d}",
                    "entity":         ent,
                    "data_domain":    _resolve_domain([ent], domain_map),
                    "content":        content_text,
                    "metadata":       {"row_index": int(i),
                                       "columns": list(df.columns)},
                })

        elif kind == "lineage":
            # Source-to-Hub mapping — one chunk per row
            s2h_df = content.get("source_to_hub")
            if s2h_df is not None and not s2h_df.empty:
                cols = {c.lower(): c for c in s2h_df.columns}
                src_col = cols.get("source entity") or cols.get("source_entity")
                hub_col = cols.get("hub name") or cols.get("hub_name")
                for i, row in s2h_df.iterrows():
                    lines = []
                    for c in s2h_df.columns:
                        v = str(row[c])
                        if v and v.lower() != "nan":
                            lines.append(f"{c}: {v}")
                    src = str(row[src_col]).strip() if src_col else ""
                    hub = str(row[hub_col]).strip() if hub_col else ""
                    chunks.append({
                        "artifact_type":  f"{key} — Source-to-Hub",
                        "artifact_label": label,
                        "chunk_id":       f"s2h_{i:05d}",
                        "entity":         src or hub,
                        "data_domain":    _resolve_domain(
                            [src, hub], domain_map
                        ),
                        "content":        "\n".join(lines),
                        "metadata":       {"source_entity": src,
                                           "hub": hub,
                                           "row_index": int(i)},
                    })
            # Graph nodes — one chunk per node
            graph = content.get("graph") or {}
            for n in graph.get("nodes", []):
                label_text = str(n.get("label", ""))
                chunks.append({
                    "artifact_type":  f"{key} — Graph Node",
                    "artifact_label": label,
                    "chunk_id":       f"node_{n['id']}",
                    "entity":         label_text,
                    "data_domain":    _resolve_domain(
                        [label_text], domain_map
                    ),
                    "content":        (
                        f"Node: {label_text}\n"
                        f"Kind: {n.get('kind', '')}\n"
                        f"Group: {n.get('group', '')}"
                    ),
                    "metadata":       {"node_id": n["id"],
                                       "kind": n.get("kind", ""),
                                       "group": n.get("group", "")},
                })

        elif kind == "dbt_project":
            # One chunk per file in the dbt project. Each file is self-
            # contained (a model, macro, test, yml) so row-per-file is
            # the right granularity for semantic search.
            files = content.get("files") or {}
            for i, (rel_path, body) in enumerate(files.items()):
                if not body or not str(body).strip():
                    continue
                # Pull entity from filename: stg_customer.sql → stg_customer
                base = rel_path.rsplit("/", 1)[-1]
                entity = re.sub(r"\.(sql|yml|yaml|md)$", "", base,
                                flags=re.IGNORECASE)
                # Classify by directory for metadata
                parts = rel_path.split("/")
                category = "root"
                for p in parts:
                    if p in ("models", "macros", "tests", "seeds",
                             "snapshots", "analyses"):
                        category = p
                        break
                # Truncate very long files — keep the first ~6KB which
                # captures config + first several CTEs for most models.
                body_str = str(body)
                if len(body_str) > 6000:
                    body_str = body_str[:6000] + "\n-- [truncated] --"
                chunks.append({
                    "artifact_type":  f"{key} — {category}",
                    "artifact_label": label,
                    "chunk_id":       f"file_{i:04d}_{_slug(rel_path)}",
                    "entity":         entity,
                    "data_domain":    _resolve_domain(
                        [entity], domain_map
                    ),
                    "content":        f"# FILE: {rel_path}\n\n{body_str}",
                    "metadata":       {"file_path": rel_path,
                                       "category": category,
                                       "file_size": len(str(body))},
                })

    return chunks


def upload_artifacts_to_stage(session, artifacts: dict, version: str,
                              source_filename: str = "",
                              metadata_summary: str = "") -> dict:
    """
    Write the artifact bundle to the internal stage under
    @{stage}/v<version>/. Returns {"path", "file_count", "bytes"}.

    The function is defensive: if anything goes wrong (no artifacts,
    empty bundle, missing schema, missing perms, put_stream failure)
    we raise a clean RuntimeError with actionable text rather than a
    bare Snowpark exception. Callers should catch RuntimeError and
    surface the message via st.error.
    """
    if not artifacts:
        raise RuntimeError(
            "No artifacts to stage. Run Phase 1 (and optionally Phase 2) "
            "first so there is something to persist."
        )

    import io as _io
    try:
        bundle_bytes = build_artifacts_bundle(
            artifacts, source_filename=source_filename,
            metadata_summary=metadata_summary,
        )
    except Exception as e_bundle:
        raise RuntimeError(
            f"Bundle construction failed before staging: {e_bundle}. "
            f"This usually indicates a corrupted artifact in session "
            f"state — try Reset Quick GO and re-run."
        )
    if not bundle_bytes:
        raise RuntimeError(
            "Bundle was built but came back empty (0 bytes). Nothing "
            "to upload."
        )

    # Safe version slug for path use
    ver_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", version.strip()) or "v0"
    stage_path = f"@{_fqn(VECTOR_STAGE)}/v{ver_slug}"
    fname = f"artifacts_v{ver_slug}.zip"
    full_target = f"{stage_path}/{fname}"

    # Use put_stream which takes a file-like object — no local disk needed.
    # The bundle zip is one file; the chunk-level processing reads it back
    # via session.file.get_stream() rather than from session_state so that
    # vectorization always reflects what actually landed in the stage
    # (source of truth).
    stream = _io.BytesIO(bundle_bytes)
    try:
        session.file.put_stream(
            stream, full_target,
            auto_compress=False, overwrite=True,
        )
    except Exception as e_put:
        raise RuntimeError(
            f"put_stream failed when writing to {full_target}: {e_put}. "
            f"Check that your role has WRITE on the stage "
            f"{_fqn(VECTOR_STAGE)}."
        )

    # Verify the file actually landed by listing it back. This catches
    # silent failures where put_stream reports OK but nothing was
    # actually written (rare, but useful for diagnostics).
    landed_size = 0
    try:
        rows = session.sql(
            f"LIST {full_target}"
        ).collect()
        for r in rows:
            row_dict = r.as_dict() if hasattr(r, "as_dict") else dict(r)
            for col in ("size", "SIZE", "Size"):
                if col in row_dict and row_dict[col] is not None:
                    try:
                        landed_size = int(row_dict[col])
                    except (ValueError, TypeError):
                        pass
                    break
            if landed_size:
                break
    except Exception:
        # Listing is best-effort — don't fail the operation just
        # because LIST returned nothing parseable.
        pass

    if not landed_size:
        # Listing says size is 0 OR no rows — the upload silently failed.
        # Best to surface this rather than report success.
        raise RuntimeError(
            f"Upload to {full_target} appears to have silently failed "
            f"(LIST shows size=0 or no rows). Check stage permissions "
            f"and re-run."
        )

    return {
        "path": full_target,
        "file_count": 1,
        "bytes": len(bundle_bytes),
        "landed_bytes": landed_size,
        "version_slug": ver_slug,
    }


def deploy_dbt_project_to_stage(session, files: dict,
                                project_slug: str,
                                version_slug: str = "latest") -> str:
    """
    Upload each dbt project file individually to the stage at
    @<VECTOR_STAGE>/dbt_runs/<project_slug>/<version_slug>/<path>.
    Individual files (not zipped) are required by CREATE DBT PROJECT
    FROM '@stage_path' — it expects to see dbt_project.yml at the root.

    Returns the fully-qualified stage path usable in CREATE DBT PROJECT.
    """
    import io as _io
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_",
                  (project_slug or "").strip()) or "dbt_project"
    ver  = re.sub(r"[^A-Za-z0-9._-]+", "_",
                  (version_slug or "").strip()) or "latest"
    base = f"@{_fqn(VECTOR_STAGE)}/dbt_runs/{slug}/{ver}"

    # Clean any prior files at this path so removed files don't linger.
    try:
        session.sql(f"REMOVE {base}").collect()
    except Exception:
        pass  # Stage may not exist or path may be empty — both fine.

    uploaded = 0
    for rel_path, body in files.items():
        safe = rel_path.lstrip("/").replace("..", "")
        stream = _io.BytesIO(
            body.encode("utf-8") if isinstance(body, str) else body
        )
        session.file.put_stream(
            stream, f"{base}/{safe}",
            auto_compress=False, overwrite=True,
        )
        uploaded += 1
    return base


def create_and_execute_dbt_project(session, stage_path: str,
                                   project_name: str,
                                   args: str = "run --target dev",
                                   dbt_version: str = None,
                                   run_deps_first: bool = True,
                                   packages_yml_body: Optional[str] = None,
                                   ) -> dict:
    """
    Create (or replace) a DBT PROJECT object pointing at stage_path,
    then execute it with the supplied args. Returns a dict with the
    query ID and the execution history row if available.

    Per Snowflake docs: CREATE DBT PROJECT + EXECUTE DBT PROJECT
    manage the full lifecycle; EXECUTE uses the Snowflake-native dbt
    runtime so you don't need dbt Core installed locally.

    ``DBT_VERSION`` is always pinned (default ``SNOWFLAKE_NATIVE_DBT_VERSION``)
    so Snowflake does not fall back to dbt 1.9.4, which can trigger
    ``SHOW PARAMETER`` failures in the native job runtime.

    When ``packages_yml_body`` is provided and lists no remote packages,
    the automatic ``dbt deps`` pre-step is skipped.
    """
    pname = re.sub(r"\W+", "_",
                   (project_name or "").strip()) or "dbt_project"
    fq_project = f"{VECTOR_DB}.{VECTOR_SCHEMA}.{pname}"
    ver_sql = _sql_dbt_version_suffix(dbt_version)

    # CREATE or REPLACE the project object. Drop first so any old
    # DBT_VERSION pin (e.g. 1.10.15) doesn't survive into the new object —
    # CREATE OR REPLACE alone is not always sufficient on Snowflake.
    try:
        session.sql(f"DROP DBT PROJECT IF EXISTS {fq_project}").collect()
    except Exception:
        pass
    create_sql = (
        f"CREATE OR REPLACE DBT PROJECT {fq_project} "
        f"FROM '{stage_path}'{ver_sql}"
    )
    session.sql(create_sql).collect()
    _pin_dbt_project_object_version(session, fq_project, dbt_version)
    # If the project object's DBT_VERSION ended up wrong, surface that
    # before EXECUTE — running on 1.10.15 will fire the SHOW PARAMETER
    # bug and the user will spend hours chasing it otherwise.
    actual_ver = _verify_dbt_project_version(session, fq_project)
    if actual_ver:
        expected = _validated_native_dbt_version(dbt_version)
        if actual_ver.strip() != expected.strip():
            try:
                import streamlit as _st
                _st.error(
                    f"⚠ DBT_VERSION mismatch on `{fq_project}`: "
                    f"expected `{expected}`, got `{actual_ver}`. "
                    f"`SHOW PARAMETER` runtime errors are likely. "
                    f"Drop manually and redeploy: "
                    f"`DROP DBT PROJECT {fq_project};`"
                )
            except Exception:
                pass

    exec_rows = []
    deps_sql = ""
    args_clean = (args or "").strip()
    starts_with_deps = args_clean.lower().startswith("deps")
    want_deps = run_deps_first and not starts_with_deps
    if want_deps and packages_yml_body is not None:
        want_deps = _packages_yml_requires_dbt_deps(packages_yml_body)
    target_match = re.search(r"--target\s+([A-Za-z0-9_\-]+)", args_clean)
    deps_args = "deps"
    if target_match:
        deps_args = f"deps --target {target_match.group(1)}"
    deps_args_lit = _sql_single_quoted_literal(deps_args)
    main_args_lit = _sql_single_quoted_literal(
        args_clean or "build --target dev"
    )
    if want_deps:
        deps_sql = (
            f"EXECUTE DBT PROJECT {fq_project} "
            f"ARGS='{deps_args_lit}'"
        )
        session.sql(deps_sql).collect()

    # Build EXECUTE statement
    exec_sql = (
        f"EXECUTE DBT PROJECT {fq_project} "
        f"ARGS='{main_args_lit}'"
    )

    # Execute. If package-install state is flaky, retry deps+execute once.
    try:
        exec_rows = session.sql(exec_sql).collect()
    except Exception as e:
        emsg = str(e)
        needs_deps_retry = (
            "dbt_packages" in emsg
            or "Run \"dbt deps\"" in emsg
            or "package(s) installed" in emsg
        )
        if not needs_deps_retry:
            raise
        if packages_yml_body is not None and not _packages_yml_requires_dbt_deps(
            packages_yml_body
        ):
            raise
        deps_sql = (
            f"EXECUTE DBT PROJECT {fq_project} "
            f"ARGS='{deps_args_lit}'"
        )
        session.sql(deps_sql).collect()
        exec_rows = session.sql(exec_sql).collect()
    # Snowflake returns the query_id of the EXECUTE itself via
    # LAST_QUERY_ID() immediately after
    try:
        qid_row = session.sql(
            "SELECT LAST_QUERY_ID() AS QID"
        ).collect()
        query_id = qid_row[0]["QID"] if qid_row else ""
    except Exception:
        query_id = ""

    # Pull the execution row for details
    history = []
    try:
        hist_rows = session.sql(f"""
            SELECT *
            FROM TABLE(INFORMATION_SCHEMA.DBT_PROJECT_EXECUTION_HISTORY(
                OBJECT_NAME => '{pname}',
                OBJECT_TYPE => 'DBT PROJECT',
                RESULT_LIMIT => 5
            ))
            ORDER BY START_TIME DESC
        """).collect()
        for r in hist_rows:
            d = r.as_dict()
            history.append({k: (str(v) if v is not None else "")
                            for k, v in d.items()})
    except Exception:
        pass

    return {
        "project_name":  fq_project,
        "query_id":      query_id,
        "deps_sql":      deps_sql,
        "execute_sql":   exec_sql,
        "exec_rows":     [r.as_dict() for r in exec_rows]
                         if exec_rows else [],
        "history":       history,
        "stage_path":    stage_path,
    }


def _github_api_json(url: str, token: str,
                     method: str = "GET",
                     payload: Optional[dict] = None) -> dict:
    """
    Minimal GitHub REST helper using urllib so no extra dependencies are needed.
    Raises RuntimeError with useful details on failure.
    """
    headers = {
        "Accept": "application/vnd.github+json",
        "Authorization": f"Bearer {token}",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "nexus-agent-streamlit",
    }
    data = None
    if payload is not None:
        data = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    req = urllib_request.Request(url=url, method=method,
                                 data=data, headers=headers)
    last_err: Optional[Exception] = None
    # Retries for transient network/runtime issues seen in hosted runtimes:
    # <urlopen error [Errno 16] Device or resource busy>
    for attempt in range(1, 5):
        try:
            with urllib_request.urlopen(req, timeout=45) as resp:
                body = resp.read().decode("utf-8")
                if not body:
                    return {}
                return json.loads(body)
        except urllib_error.HTTPError as e:
            # HTTP-level failures are usually deterministic; don't retry.
            try:
                msg = e.read().decode("utf-8")
            except Exception:
                msg = str(e)
            raise RuntimeError(f"GitHub API {e.code}: {msg}") from e
        except urllib_error.URLError as e:
            last_err = e
            reason = str(getattr(e, "reason", e))
            transient = ("Errno 16" in reason or "resource busy" in reason.lower()
                         or "timed out" in reason.lower()
                         or "temporarily unavailable" in reason.lower())
            if attempt < 4 and transient:
                time.sleep(0.4 * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"GitHub API error: {e}") from e
        except OSError as e:
            last_err = e
            msg = str(e)
            if attempt < 4 and ("Errno 16" in msg
                                or "resource busy" in msg.lower()):
                time.sleep(0.4 * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"GitHub API error: {e}") from e
        except Exception as e:
            last_err = e
            if attempt < 4:
                time.sleep(0.4 * (2 ** (attempt - 1)))
                continue
            raise RuntimeError(f"GitHub API error: {e}") from e
    raise RuntimeError(f"GitHub API error: {last_err}")


def resolve_github_token(session_token: str = "") -> str:
    """
    Resolve GitHub PAT from session value, Streamlit secrets, or env vars.
    Supports common key shapes:
      GITHUB_TOKEN / github_token / GITHUB_PAT / github.pat / github.token
    """
    if session_token and session_token.strip():
        return session_token.strip()

    # 1) Streamlit secrets
    try:
        sec = st.secrets
    except Exception:
        sec = {}

    direct_keys = [
        "GITHUB_TOKEN", "github_token", "GITHUB_PAT", "github_pat",
        "PAT", "pat",
    ]
    for k in direct_keys:
        try:
            v = sec.get(k, "")
        except Exception:
            v = ""
        if isinstance(v, str) and v.strip():
            return v.strip()

    for section in ("github", "git", "auth", "tokens"):
        try:
            block = sec.get(section, {})
        except Exception:
            block = {}
        if isinstance(block, dict):
            for k in ("token", "pat", "github_token", "github_pat"):
                v = block.get(k, "")
                if isinstance(v, str) and v.strip():
                    return v.strip()

    # 2) Environment variables
    for k in (
        "GITHUB_TOKEN", "github_token",
        "GITHUB_PAT", "github_pat",
        "GH_TOKEN", "gh_token",
    ):
        v = os.getenv(k, "")
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def publish_dbt_project_to_github(
    files: dict,
    token: str,
    owner: str,
    repo: str,
    subfolder: str,
    branch: str = "main",
    auto_init_repo: bool = True,
    commit_message_prefix: str = "nexus dbt publish",
) -> dict:
    """
    Publish dbt files to GitHub repo under <subfolder>/<relative_path>.
    Uses Contents API and creates one commit per file update.
    """
    if not files:
        return {"published": 0, "failed": 0, "errors": []}
    if not token.strip():
        raise ValueError("GitHub token is required.")

    sf = re.sub(r"[^A-Za-z0-9._/-]+", "_", subfolder.strip("/")) or "dbt"
    safe_files = {}
    for rel_path, body in files.items():
        rp = rel_path.strip().replace("\\", "/").lstrip("/")
        if not rp or ".." in rp:
            continue
        safe_files[rp] = body

    errors = []
    # Preflight repository + branch.
    repo_meta = _github_api_json(
        f"https://api.github.com/repos/{owner}/{repo}",
        token,
        method="GET",
    )
    default_branch = (repo_meta.get("default_branch") or "main").strip()
    target_branch = (branch or "").strip() or default_branch
    repo_size = int(repo_meta.get("size") or 0)

    def _branch_exists(br: str) -> bool:
        try:
            _github_api_json(
                f"https://api.github.com/repos/{owner}/{repo}/branches/{br}",
                token,
                method="GET",
            )
            return True
        except Exception:
            return False

    def _init_repo_if_empty(init_branch: str) -> None:
        blob = _github_api_json(
            f"https://api.github.com/repos/{owner}/{repo}/git/blobs",
            token,
            method="POST",
            payload={"content": "# nexus\n", "encoding": "utf-8"},
        )
        tree = _github_api_json(
            f"https://api.github.com/repos/{owner}/{repo}/git/trees",
            token,
            method="POST",
            payload={
                "tree": [{
                    "path": "README.md",
                    "mode": "100644",
                    "type": "blob",
                    "sha": blob["sha"],
                }]
            },
        )
        commit = _github_api_json(
            f"https://api.github.com/repos/{owner}/{repo}/git/commits",
            token,
            method="POST",
            payload={
                "message": "Initialize repository",
                "tree": tree["sha"],
            },
        )
        _github_api_json(
            f"https://api.github.com/repos/{owner}/{repo}/git/refs",
            token,
            method="POST",
            payload={
                "ref": f"refs/heads/{init_branch}",
                "sha": commit["sha"],
            },
        )

    if repo_size == 0 and auto_init_repo:
        _init_repo_if_empty(target_branch)
    elif not _branch_exists(target_branch):
        target_branch = default_branch if _branch_exists(default_branch) else ""
        if not target_branch:
            raise RuntimeError(
                "No usable branch found. Create a branch or initialize repo."
            )

    published = 0
    for rel_path, body in safe_files.items():
        target_path = f"{sf}/{rel_path}"
        url = (f"https://api.github.com/repos/{owner}/{repo}/contents/"
               f"{target_path}")
        sha = None
        try:
            existing = _github_api_json(
                f"{url}?ref={target_branch}", token, method="GET"
            )
            sha = existing.get("sha")
        except RuntimeError as e:
            # 404 means create-new; anything else should fail.
            if "404" not in str(e):
                errors.append((rel_path, str(e)))
                continue

        content_bytes = (
            body.encode("utf-8") if isinstance(body, str) else body
        )
        payload = {
            "message": (
                f"{commit_message_prefix}: {target_path}"
            ),
            "content": base64.b64encode(content_bytes).decode("ascii"),
            "branch": target_branch,
        }
        if sha:
            payload["sha"] = sha
        try:
            _github_api_json(url, token, method="PUT", payload=payload)
            published += 1
        except Exception as e:
            errors.append((rel_path, str(e)))

    return {
        "published": published,
        "failed": len(errors),
        "errors": errors,
        "subfolder": sf,
        "branch": target_branch,
        "repo_url": f"https://github.com/{owner}/{repo}",
    }


def embed_and_store(session, chunks: list, version: str,
                    embed_model: str, dim: int, table: str,
                    progress_cb=None) -> int:
    """
    Generate embeddings via Cortex and insert into the vector table.
    Deletes any rows with this (version, embed_model) before insert so
    re-running the pipeline is idempotent.

    Returns number of rows inserted.
    """
    if not chunks:
        return 0

    ver_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", version.strip()) or "v0"
    full_table = _fqn(table)

    # Idempotency: clear any previous rows for this (version, model)
    session.sql(
        f"DELETE FROM {full_table} "
        f"WHERE VERSION = ? AND EMBEDDING_MODEL = ?",
        params=[ver_slug, embed_model],
    ).collect()

    # Batch the inserts — one row at a time is slow but most reliable in
    # SiS where bulk binds across VECTOR columns can be finicky. Callers
    # see progress via progress_cb.
    inserted = 0
    for i, c in enumerate(chunks):
        row_id = f"{ver_slug}::{embed_model}::{c['artifact_type']}::{c['chunk_id']}"
        # Cortex EMBED function name depends on dimension
        embed_fn = f"SNOWFLAKE.CORTEX.EMBED_TEXT_{dim}"
        session.sql(
            f"""
            INSERT INTO {full_table}
              (ROW_ID, VERSION, DATA_DOMAIN, ARTIFACT_TYPE,
               ARTIFACT_LABEL, CHUNK_ID, ENTITY, CONTENT, METADATA,
               EMBEDDING_MODEL, EMBEDDING)
            SELECT ?, ?, ?, ?, ?, ?, ?, ?, PARSE_JSON(?), ?,
                   {embed_fn}(?, ?)
            """,
            params=[
                row_id, ver_slug, c["data_domain"], c["artifact_type"],
                c["artifact_label"], c["chunk_id"], c.get("entity", ""),
                c["content"], json.dumps(c.get("metadata", {})),
                embed_model, embed_model, c["content"],
            ],
        ).collect()
        inserted += 1
        if progress_cb and (i % 5 == 0 or i == len(chunks) - 1):
            progress_cb(inserted, len(chunks))

    return inserted


def semantic_search(session, query: str, table: str, dim: int,
                    embed_model: str, version: str = None,
                    data_domain: str = None, artifact_type: str = None,
                    top_k: int = 10) -> "pd.DataFrame":
    """
    Find the most semantically similar chunks using vector cosine
    similarity. Optional filters on version, data_domain, artifact_type.
    """
    filters = ["EMBEDDING_MODEL = ?"]
    params  = [embed_model]
    if version:
        ver_slug = re.sub(r"[^A-Za-z0-9._-]+", "_", version.strip()) or "v0"
        filters.append("VERSION = ?")
        params.append(ver_slug)
    if data_domain and data_domain != "(any)":
        filters.append("DATA_DOMAIN = ?")
        params.append(data_domain)
    if artifact_type and artifact_type != "(any)":
        filters.append("ARTIFACT_TYPE = ?")
        params.append(artifact_type)

    where = " AND ".join(filters)
    embed_fn = f"SNOWFLAKE.CORTEX.EMBED_TEXT_{dim}"
    sql = f"""
        SELECT
            VERSION,
            DATA_DOMAIN,
            ARTIFACT_TYPE,
            ARTIFACT_LABEL,
            CHUNK_ID,
            ENTITY,
            CONTENT,
            VECTOR_COSINE_SIMILARITY(
                EMBEDDING,
                {embed_fn}(?, ?)
            ) AS SIMILARITY
        FROM {_fqn(table)}
        WHERE {where}
        ORDER BY SIMILARITY DESC
        LIMIT {int(top_k)}
    """
    # The query embedding binds come first so they're before the filter binds
    all_params = [embed_model, query] + params
    df = session.sql(sql, params=all_params).to_pandas()
    return df


def list_stored_versions(session) -> "pd.DataFrame":
    """
    List all (version, embedding_model) combinations present in the
    vector tables, along with row counts and domain counts. Used by the
    View Artifacts tab to populate the version dropdown.
    """
    rows = []
    for label, (model, dim, table) in EMBED_MODELS.items():
        try:
            r = session.sql(f"""
                SELECT
                    VERSION,
                    EMBEDDING_MODEL,
                    COUNT(*)                         AS CHUNK_COUNT,
                    COUNT(DISTINCT DATA_DOMAIN)      AS DOMAIN_COUNT,
                    COUNT(DISTINCT ARTIFACT_TYPE)    AS TYPE_COUNT,
                    MIN(CREATED_AT)                  AS FIRST_LOADED,
                    MAX(CREATED_AT)                  AS LAST_LOADED
                FROM {_fqn(table)}
                GROUP BY VERSION, EMBEDDING_MODEL
                ORDER BY LAST_LOADED DESC
            """).collect()
            for row in r:
                rows.append({
                    "version":      row["VERSION"],
                    "model":        row["EMBEDDING_MODEL"],
                    "model_label":  label,
                    "dim":          dim,
                    "table":        table,
                    "chunk_count":  row["CHUNK_COUNT"],
                    "domain_count": row["DOMAIN_COUNT"],
                    "type_count":   row["TYPE_COUNT"],
                    "first_loaded": row["FIRST_LOADED"],
                    "last_loaded":  row["LAST_LOADED"],
                })
        except Exception:
            # Table doesn't exist yet — skip
            continue
    return pd.DataFrame(rows)


def list_stage_bundles(session) -> list:
    """
    List ZIP bundles stored in the artifacts stage. Returns a list of
    {path, size, last_modified, version_slug} dicts, sorted newest first.
    """
    try:
        rows = session.sql(
            f"LIST @{_fqn(VECTOR_STAGE)}"
        ).collect()
    except Exception:
        return []

    def _row_get(row, *candidates, default=""):
        """
        Safely pull a value out of a Snowpark Row. Tries each candidate
        key (attribute-then-subscript style) and falls back to default.
        LIST @stage column names can differ by Snowflake version —
        sometimes 'name', sometimes '"name"', sometimes uppercase —
        so we try several.
        """
        # Snowpark Row supports .as_dict()
        try:
            d = row.as_dict()
        except Exception:
            d = None
        if d:
            # Try each candidate against both the dict and uppercased keys
            upper_map = {k.upper(): v for k, v in d.items()}
            for c in candidates:
                if c in d:
                    return d[c] if d[c] is not None else default
                if c.upper() in upper_map:
                    v = upper_map[c.upper()]
                    return v if v is not None else default
        # Last-ditch: subscript access
        for c in candidates:
            try:
                v = row[c]
                if v is not None:
                    return v
            except Exception:
                pass
        return default

    out = []
    for r in rows:
        name = _row_get(r, "name", "NAME")
        if not name:
            continue
        if not str(name).endswith(".zip"):
            continue
        # name looks like: artifacts_stage/v1.0.0/artifacts_v1.0.0.zip
        parts = str(name).split("/")
        ver = ""
        for p in parts:
            if p.startswith("v") and len(p) > 1:
                ver = p[1:]
                break
        out.append({
            "path":          "@" + str(name),
            "size":          _row_get(r, "size", "SIZE", default=0),
            "last_modified": _row_get(r, "last_modified", "LAST_MODIFIED"),
            "version_slug":  ver,
            "filename":      parts[-1] if parts else str(name),
        })
    # Sort newest-first. last_modified from LIST @stage is an RFC 2822
    # string ("Wed, 16 Apr 2026 12:00:00 GMT") which doesn't sort
    # lexicographically in the right order, so we sort by version_slug
    # descending as the primary key and fall back to last_modified. For
    # semver-ish versions ("1.0.0", "2.0.0") this gives the intuitive
    # newest-first ordering.
    out.sort(
        key=lambda d: (d["version_slug"], str(d["last_modified"])),
        reverse=True,
    )
    return out


def load_artifacts_from_stage(session, stage_path: str) -> dict:
    """
    Download the ZIP bundle from the stage and reconstruct an artifacts
    dict with the same shape as st.session_state.artifacts — so the
    existing render_artifacts() function can display it unchanged.

    Returns:
        {
          "artifacts":        {key: {kind, label, content}},
          "metadata_summary": str (or ""),
          "source_filename":  str (from manifest),
          "version":          str,
        }
    """
    # Download via session.file.get_stream — returns a file-like object
    try:
        stream = session.file.get_stream(stage_path)
        zip_bytes = stream.read()
    except Exception as e:
        raise RuntimeError(f"Could not read {stage_path}: {e}")

    import io as _io
    zf = zipfile.ZipFile(_io.BytesIO(zip_bytes))

    # Parse manifest to get the list of artifacts + folder structure
    try:
        manifest = json.loads(zf.read("manifest.json"))
    except Exception:
        manifest = {"artifacts": []}

    # Read metadata summary if present
    metadata_summary = ""
    try:
        metadata_summary = zf.read("00_metadata_summary.md").decode()
    except Exception:
        pass

    source_filename = manifest.get("source_files", "") or ""
    artifacts = {}

    for entry in manifest.get("artifacts", []):
        key   = entry.get("key", "")
        label = entry.get("label", key)
        kind  = entry.get("kind", "")
        files = entry.get("files", [])
        content = {}

        if kind == "raw_vault":
            for f in files:
                low = f.lower()
                try:
                    data = zf.read(f).decode()
                except Exception:
                    continue
                if low.endswith("narrative.md"):
                    content["narrative_md"] = data
                elif low.endswith(".mermaid"):
                    content["mermaid"] = data
                elif low.endswith(".sql"):
                    content["sql"] = data
            content.setdefault("mermaid_raw", "")
            content.setdefault("sql_raw", "")

        elif kind == "table":
            # Prefer .csv (stable to parse). If absent, try xlsx.
            df = None
            for f in files:
                if f.lower().endswith(".csv"):
                    try:
                        df = pd.read_csv(
                            StringIO(zf.read(f).decode()),
                            dtype=str, keep_default_na=False,
                        )
                        break
                    except Exception:
                        continue
            if df is None:
                for f in files:
                    if f.lower().endswith(".xlsx"):
                        try:
                            df = pd.read_excel(
                                _io.BytesIO(zf.read(f)), dtype=str,
                            )
                            break
                        except Exception:
                            continue
            # Also preserve the raw LLM response if present (for debug)
            raw = ""
            for f in files:
                if f.endswith("_raw_response.txt"):
                    try:
                        raw = zf.read(f).decode()
                    except Exception:
                        pass
            content = {"df": df, "raw": raw}

        elif kind == "lineage":
            # Reconstruct: mermaid + source_to_hub df + graph dict +
            # rv_tables
            mermaid = ""
            s2h_df = None
            graph = {}
            rv_tables = {}
            for f in files:
                low = f.lower()
                try:
                    if low.endswith(".mermaid"):
                        mermaid = zf.read(f).decode()
                    elif low.endswith("source_to_hub.csv"):
                        s2h_df = pd.read_csv(
                            StringIO(zf.read(f).decode()),
                            dtype=str, keep_default_na=False,
                        )
                    elif low.endswith("graph.json"):
                        graph = json.loads(zf.read(f))
                    elif low.endswith("raw_vault_tables.json"):
                        rv_tables = json.loads(zf.read(f))
                except Exception:
                    continue
            content = {
                "mermaid":       mermaid,
                "source_to_hub": s2h_df,
                "graph":         graph,
                "rv_tables":     rv_tables,
                "mapping_raw":   "",
            }
        elif kind == "dbt_project":
            # Files live under {folder}/project/{rel_path}. Walk back
            # into the files dict. Skip the aux .zip and
            # _codegen_raw_response.md (they'll be re-derived on
            # demand from the files dict).
            project_files = {}
            raw = ""
            # Find the "project/" prefix for this artifact's folder
            for f in files:
                if "/project/" in f:
                    # e.g. "Raw_Vault_dbt/project/models/hub_customer.sql"
                    parts = f.split("/project/", 1)
                    if len(parts) == 2 and parts[1]:
                        rel = parts[1]
                        try:
                            project_files[rel] = zf.read(f).decode()
                        except Exception:
                            pass
                elif f.endswith("_codegen_raw_response.md"):
                    try:
                        raw = zf.read(f).decode()
                    except Exception:
                        pass
            content = {"files": project_files, "raw": raw}
        else:
            continue  # unknown kind — skip

        artifacts[key] = {"kind": kind, "label": label, "content": content}

    # Derive version from manifest.generated_utc or caller can overlay
    version = manifest.get("generated_utc", "")

    return {
        "artifacts":        artifacts,
        "metadata_summary": metadata_summary,
        "source_filename":  source_filename,
        "version":          version,
    }


def render_artifacts(artifacts: dict, key_prefix: str = 'rev'):
    """
    Render a generated-artifacts dict with the same layout used in the
    Reverse Engineering tab. key_prefix disambiguates widget keys when
    the same artifacts structure is rendered in more than one place
    (e.g. Reverse Engineering tab and View Artifacts tab).
    """
    if not artifacts:
        st.info("No artifacts to display.")
        return
    for key, artifact in artifacts.items():
        kind    = artifact["kind"]
        label   = artifact["label"]
        content = artifact["content"]

        with st.expander(f"✦  {label}", expanded=True):

            # ── Raw Vault → sub-tabs: Narrative | ER Diagram | SQL DDL
            if kind == "raw_vault":
                rv_tab1, rv_tab2, rv_tab3 = st.tabs(
                    ["📖 Model Overview", "🗺 ER Diagram (Mermaid)",
                     "🧱 Snowflake DDL"]
                )
                with rv_tab1:
                    st.markdown(content.get("narrative_md") or
                                "_No narrative produced._")

                with rv_tab2:
                    mer = (content.get("mermaid") or "").strip()
                    mer_raw = content.get("mermaid_raw", "")
                    # Heuristic: Mermaid script should contain one of
                    # these keywords to be a valid diagram
                    looks_valid = bool(re.search(
                        r'\b(erDiagram|flowchart|graph|sequenceDiagram)\b',
                        mer
                    ))
                    if mer and looks_valid:
                        try:
                            render_mermaid(mer, height=600)
                        except Exception as e:
                            st.error(f"Mermaid render error: {e}")
                        st.markdown("**Mermaid source:**")
                        st.code(mer, language="text")
                    else:
                        st.warning(
                            "Could not extract a valid Mermaid "
                            "diagram from the model's response."
                        )
                        if mer_raw:
                            st.markdown("**Model's raw response:**")
                            st.code(mer_raw[:4000], language="text")

                with rv_tab3:
                    sql = (content.get("sql") or "").strip()
                    sql_raw = content.get("sql_raw", "")
                    looks_valid = bool(re.search(
                        r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE',
                        sql, re.IGNORECASE
                    ))
                    if sql and looks_valid:
                        st.code(sql, language="sql")
                    else:
                        st.warning(
                            "Could not extract valid SQL DDL from "
                            "the model's response."
                        )
                        if sql_raw:
                            st.markdown("**Model's raw response:**")
                            st.code(sql_raw[:4000], language="text")

                # Download bundle as Markdown
                bundle_md = (
                    f"# {label}\n\n"
                    f"{content.get('narrative_md', '')}\n\n"
                    f"## ER Diagram\n\n```mermaid\n"
                    f"{content.get('mermaid', '')}\n```\n\n"
                    f"## Snowflake DDL\n\n```sql\n"
                    f"{content.get('sql', '')}\n```\n"
                )
                c1, c2 = st.columns(2)
                c1.download_button(
                    f"⬇ Download {label} (.md)",
                    data=bundle_md,
                    file_name=f"{key.replace(' ', '_')}.md",
                    mime="text/markdown",
                    key=f"{key_prefix}_dl_md_{key}",
                    use_container_width=True,
                )
                c2.download_button(
                    f"⬇ Download DDL (.sql)",
                    data=content.get("sql") or "",
                    file_name=f"{key.replace(' ', '_')}.sql",
                    mime="text/plain",
                    key=f"{key_prefix}_dl_sql_{key}",
                    use_container_width=True,
                )

            # ── STTM / Data Catalog / Data Domain → DataFrame
            elif kind == "table":
                df = content.get("df")
                raw = content.get("raw", "")
                if df is not None and not df.empty:
                    st.dataframe(df, use_container_width=True,
                                 hide_index=True)

                    dl1, dl2 = st.columns(2)
                    # Excel download (primary)
                    try:
                        xlsx_bytes = df_to_excel_bytes(
                            df, sheet_name=label[:31]
                        )
                        dl1.download_button(
                            f"⬇ Download {label} (.xlsx)",
                            data=xlsx_bytes,
                            file_name=f"{key.replace(' ', '_')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            key=f"{key_prefix}_dl_xlsx_{key}",
                            use_container_width=True,
                        )
                    except Exception as e:
                        dl1.warning(f"Excel export unavailable: {e}")

                    # CSV download (fallback, always works)
                    dl2.download_button(
                        f"⬇ Download {label} (.csv)",
                        data=df.to_csv(index=False),
                        file_name=f"{key.replace(' ', '_')}.csv",
                        mime="text/csv",
                        key=f"{key_prefix}_dl_csv_{key}",
                        use_container_width=True,
                    )
                else:
                    st.warning(
                        "Could not parse a table from the model's "
                        "response. Showing raw output below so you "
                        "can see what was produced."
                    )
                    if raw:
                        with st.expander(
                            "Raw model response",
                            expanded=True,
                        ):
                            st.code(raw[:6000], language="text")
                            st.download_button(
                                f"⬇ Download raw response",
                                data=raw,
                                file_name=(
                                    f"{key.replace(' ', '_')}_"
                                    f"raw_response.txt"
                                ),
                                mime="text/plain",
                                key=f"{key_prefix}_dl_rawtxt_{key}",
                            )
                    # Diagnostics to help debug
                    pipe_line_count = sum(
                        1 for ln in raw.splitlines()
                        if ln.count("|") >= 2
                    )
                    st.caption(
                        f"Diagnostics: response length = "
                        f"{len(raw)} chars, "
                        f"lines with ≥2 pipes = {pipe_line_count}."
                    )
                    st.markdown("**Model's raw response:**")
                    st.code(raw, language="markdown")
                    st.download_button(
                        f"⬇ Download raw response (.txt)",
                        data=raw,
                        file_name=f"{key.replace(' ', '_')}_raw.txt",
                        mime="text/plain",
                        key=f"{key_prefix}_dl_raw_{key}",
                    )

            # ── Data Lineage → Interactive + Mermaid + mapping
            elif kind == "lineage":
                mer = (content.get("mermaid") or "").strip()
                graph = content.get("graph") or {}
                s2h_df = content.get("source_to_hub")
                rv_tables = content.get("rv_tables", {})
                mapping_raw = content.get("mapping_raw", "")

                lg_tab1, lg_tab2, lg_tab3, lg_tab4 = st.tabs([
                    "🕸 Interactive Graph",
                    "🖼 Mermaid Diagram",
                    "📋 Source → Hub Mapping",
                    "🧩 Mermaid Source",
                ])

                with lg_tab1:
                    st.caption(
                        "Click a node to highlight its upstream "
                        "and downstream lineage. Hover for column "
                        "details. Drag to rearrange, scroll to "
                        "zoom, drag the canvas to pan."
                    )
                    if graph and graph.get("nodes"):
                        try:
                            render_interactive_lineage(
                                graph, height=720
                            )
                        except Exception as e:
                            st.error(
                                f"Interactive renderer error: {e}"
                            )
                        st.caption(
                            f"Graph: **{len(graph.get('nodes', []))} "
                            f"nodes**, "
                            f"**{len(graph.get('edges', []))} edges**"
                        )
                    else:
                        st.warning(
                            "No graph data available — regenerate "
                            "the lineage artifact."
                        )

                with lg_tab2:
                    if mer and "flowchart" in mer:
                        try:
                            render_mermaid(mer, height=750)
                        except Exception as e:
                            st.error(f"Mermaid render error: {e}")
                        st.caption(
                            "**Legend:**  "
                            "📄 File  →  🟢 Stage/Entity  "
                            "══▶ mapped to Hub  "
                            "─▶ explicit DataStage link  "
                            "┈▶ inferred via shared columns  "
                            "🧡 Hub · 💜 Link · 💙 Satellite"
                        )
                    else:
                        st.warning(
                            "Could not render Mermaid diagram."
                        )

                with lg_tab3:
                    if s2h_df is not None and not s2h_df.empty:
                        st.dataframe(
                            s2h_df, use_container_width=True,
                            hide_index=True
                        )
                        st.download_button(
                            "⬇ Download mapping (.csv)",
                            data=s2h_df.to_csv(index=False),
                            file_name=(
                                f"{key.replace(' ', '_')}_"
                                f"source_to_hub.csv"
                            ),
                            mime="text/csv",
                            key=f"{key_prefix}_dl_s2h_{key}",
                        )
                    else:
                        st.info(
                            "No source-to-Hub mapping was produced. "
                            "This usually means the Raw Vault DDL "
                            "had no parseable Hub tables, or the "
                            "source entities had no recognizable "
                            "business keys."
                        )
                        if mapping_raw:
                            with st.expander(
                                "Show raw LLM response"
                            ):
                                st.code(mapping_raw[:4000],
                                        language="text")

                    if rv_tables:
                        st.markdown("**Raw Vault tables extracted "
                                    "from DDL:**")
                        rv_summary = pd.DataFrame([
                            {"Kind": "Hubs",
                             "Count": len(rv_tables.get("hubs", [])),
                             "Tables": ", ".join(
                                 rv_tables.get("hubs", [])
                             )},
                            {"Kind": "Links",
                             "Count": len(rv_tables.get("links", [])),
                             "Tables": ", ".join(
                                 rv_tables.get("links", [])
                             )},
                            {"Kind": "Satellites",
                             "Count": len(rv_tables.get("sats", [])),
                             "Tables": ", ".join(
                                 rv_tables.get("sats", [])
                             )},
                        ])
                        st.dataframe(rv_summary,
                                     use_container_width=True,
                                     hide_index=True)

                with lg_tab4:
                    if mer:
                        st.code(mer, language="text")
                        st.download_button(
                            "⬇ Download Mermaid (.mmd)",
                            data=mer,
                            file_name=(
                                f"{key.replace(' ', '_')}.mmd"
                            ),
                            mime="text/plain",
                            key=f"{key_prefix}_dl_mmd_{key}",
                        )
                    else:
                        st.warning("No Mermaid source available.")

            # ── dbt project → file tree + code viewer + zip download ──
            elif kind == "dbt_project":
                files = content.get("files") or {}
                raw   = content.get("raw", "") or ""

                if not files:
                    st.warning(
                        "No parsed files in this dbt project. "
                        "Showing raw codegen response."
                    )
                    if raw:
                        st.code(raw[:8000], language="markdown")
                else:
                    st.caption(
                        f"{len(files)} file"
                        f"{'s' if len(files) != 1 else ''} in this "
                        f"dbt project"
                    )
                    paths = sorted(files.keys())
                    picked = st.selectbox(
                        "Browse file",
                        paths,
                        index=0,
                        key=f"{key_prefix}_dbtfile_{key}",
                    )
                    lang = "sql"
                    if picked.endswith(".yml") or picked.endswith(".yaml"):
                        lang = "yaml"
                    elif picked.endswith(".md"):
                        lang = "markdown"
                    st.code(files[picked], language=lang)

                    # Download single file
                    st.download_button(
                        f"⬇ Download {picked}",
                        data=files[picked],
                        file_name=picked.split("/")[-1],
                        mime="text/plain",
                        key=(f"{key_prefix}_dl_dbtfile_{key}_"
                             f"{_slug(picked)}"),
                    )

                    # Download entire project as zip
                    try:
                        project_name = re.sub(r"\W+", "_",
                                              key.lower())
                        zb = bundle_dbt_project(files, project_name)
                        st.download_button(
                            f"⬇ Download {label} (.zip)",
                            data=zb,
                            file_name=f"{project_name}.zip",
                            mime="application/zip",
                            use_container_width=True,
                            key=f"{key_prefix}_dl_dbtzip_{key}",
                        )
                    except Exception as e:
                        st.warning(f"Could not build zip: {e}")

            # (Legacy markdown branch kept for safety)
            else:
                md_text = content if isinstance(content, str) else str(content)
                st.markdown(md_text)
                st.download_button(
                    f"⬇ Download {label} (.md)",
                    data=md_text,
                    file_name=f"{key.replace(' ', '_')}.md",
                    mime="text/markdown",
                    key=f"{key_prefix}_dl_md_{key}",
                )


# ─────────────────────────────────────────────────────────────────────────────
# FORWARD ENGINEERING HELPERS
#
# Pipeline: (1) pick reverse artifacts from stage -> (2) upload dashboard
# spec + (3) rules doc -> LLM generates -> (4) semantic model -> (5)
# business vault + STTM + catalog + domain -> (6) raw vault dbt -> (7)
# business vault dbt -> (8) dbt tests.
# ─────────────────────────────────────────────────────────────────────────────


# Cortex call options for code generation - tuned for better-quality
# code. Higher temperature than the default gives the LLM room to
# produce well-structured dbt projects without being overly conservative.
CODEGEN_OPTS = {
    "temperature": 0.4,
    "top_p":       0.9,
    "max_tokens":  8192,
    "guardrails":  False,
}




def summarize_reverse_artifacts(artifacts: dict,
                                max_chars: int = 18000) -> str:
    """
    Dense text summary of selected Reverse Engineering artifacts,
    suitable for pasting into Forward Engineering prompts.
    """
    parts = []
    for key, art in (artifacts or {}).items():
        kind    = art.get("kind", "")
        label   = art.get("label", key)
        content = art.get("content", {}) or {}
        parts.append(f"\n=== {label} ===")

        if kind == "raw_vault":
            nm = (content.get("narrative_md") or "").strip()
            if nm:
                parts.append(f"[Narrative]\n{nm}")
            sql = (content.get("sql") or "").strip()
            if sql:
                parts.append(f"[Snowflake DDL]\n{sql}")
            mer = (content.get("mermaid") or "").strip()
            if mer:
                parts.append(f"[Mermaid ER]\n{mer}")

        elif kind == "table":
            df = content.get("df")
            if df is not None and not df.empty:
                parts.append(
                    f"[Columns]: {', '.join(df.columns)}\n"
                    f"[Sample rows]:\n{df.head(30).to_csv(index=False)}"
                )

        elif kind == "lineage":
            mer = (content.get("mermaid") or "").strip()
            if mer:
                parts.append(f"[Lineage Mermaid]\n{mer}")
            s2h_df = content.get("source_to_hub")
            if s2h_df is not None and not s2h_df.empty:
                parts.append(
                    f"[Source -> Hub mapping]:\n"
                    f"{s2h_df.to_csv(index=False)}"
                )
            rv_tables = content.get("rv_tables") or {}
            if rv_tables:
                parts.append(f"[Raw Vault tables]: "
                             f"{json.dumps(rv_tables, indent=2)}")

    full = "\n\n".join(parts)
    if len(full) > max_chars:
        full = full[:max_chars] + (
            f"\n\n[... truncated at {max_chars} chars; "
            f"original was {len(full)} chars]"
        )
    return full

























# ─────────────────────────────────────────────────────────────────────────────
# Per-file dbt generation — reliable alternative to bulk codegen
#
# The bulk approach (one big prompt → one response containing 10+ files)
# is fragile: Cortex often hits max_tokens, produces inconsistent format,
# or adds prose that confuses parsers. Instead we do:
#
#   1. PLANNER call: ask the LLM for a list of file paths to generate
#      (tiny response — cannot fail)
#   2. WORKER calls: one call per file, with a focused prompt that
#      asks for ONLY the file body (no format markers, no JSON wrapping)
#
# Each worker call fits comfortably in 8192 output tokens, the response
# is plain file content (zero parsing ambiguity), and a single failed
# file doesn't kill the whole project — the rest still generate.
# ─────────────────────────────────────────────────────────────────────────────

# ─────────────────────────────────────────────────────────────────────────────
# MEDALLION + DATA VAULT 2.0 SYSTEM SPEC
# ─────────────────────────────────────────────────────────────────────────────
# This is the authoritative spec for the LLM-driven dbt code generation.
# It comes from a user-supplied prompt (see /docs or chat for source) and
# is the SINGLE source of truth for what "good" generated dbt looks like
# in this app. Both the planner and the per-file builder lead with this
# (the per-file builder uses a compressed slice to save tokens).
#
# Anything in this spec that conflicts with our Snowflake-native runtime
# constraints (e.g. external packages on snowflake-managed dbt projects,
# query_tag, persist_docs) is overridden by the post-generation
# sanitizers in `_prepare_files_for_snowflake_native_dbt`. The LLM is
# explicitly told NOT to emit those constructs so sanitizer load is low.
# ─────────────────────────────────────────────────────────────────────────────



# Compressed slice for per-file calls — keeps token cost reasonable when
# generating 12-24 files per project. Includes only what changes the
# generated content of a single file: audit columns, materialization
# rules, and the runtime hard constraints.


# ════════════════════════════════════════════════════════════════════════════
# DATA VAULT 2.0 STANDARDS (user-supplied, authoritative)
# ════════════════════════════════════════════════════════════════════════════
# These six constants are the bank's own DV 2.0 standards. They are
# injected into the Reverse Engineering artifact prompts (Lineage,
# STTM, Data Catalog, Data Domain, Raw Vault Model) so generated
# artifacts conform to the bank's naming, hashing, and modeling rules.
#
# `_DV_STANDARDS_BY_ARTIFACT` selects which standards apply to which
# artifact (e.g. abbreviations matter for Catalog naming; satellite
# rules only matter for Raw Vault).













# Selector: which standards apply to which Reverse Engineering artifact.
# Lineage is excluded from heavy DV standards because it's a pure
# source→target dataflow visualization — but abbreviations help label
# nodes consistently.






def _filter_dbt_paths_for_layer(paths: list, layer: str) -> list:
    """
    Keep only contract-compliant file paths for forward dbt generators.
    This prevents planner drift (extra infra files, wrong folders).
    """
    if layer == "raw_vault":
        allowed = [
            r"^models/silver/staging/stg_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/raw_vault/hubs/hub_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/raw_vault/links/lnk_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/raw_vault/sats/sat_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/raw_vault/schema\.yml$",
        ]
    elif layer == "business_vault":
        allowed = [
            r"^models/silver/business_vault/staging/stg_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/business_vault/hubs/hub_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/business_vault/links/lnk_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/business_vault/sats/sat_[A-Za-z0-9_]+\.sql$",
            r"^models/silver/business_vault/schema\.yml$",
        ]
    else:
        return paths

    out = []
    seen = set()
    for p in paths:
        pp = p.strip().strip("/").replace("\\", "/")
        if any(re.match(rx, pp) for rx in allowed):
            if pp not in seen:
                seen.add(pp)
                out.append(pp)
    return out




_LOCAL_AUTOMATE_DV_MACROS = """{# Local fallback macros for Snowflake DBT PROJECT execution.
   These mirror the automate_dv contract just enough to compile a Data Vault
   project end-to-end without the external package being installed.
   Hash keys are SHA1_BINARY of UPPER(TRIM(NVL(<col>,'^^'))) — same convention
   as automate_dv defaults — so downstream sat/link models can join on them. #}

{% macro _dv_source_model_name(source_model) %}
  {% if source_model is sequence and source_model is not string %}
    {{ return(source_model[0]) }}
  {% else %}
    {{ return(source_model) }}
  {% endif %}
{% endmacro %}

{# Build a hash expression for one or more business-key columns. Mirrors
   automate_dv's approach: uppercased, trimmed, NULLs replaced with '^^',
   multiple cols concatenated with '||~~||'. Accepts:
     - a string (one col):              'CUSTOMER_ID'
     - a list (multi-col HK / hashdiff):['CUSTOMER_ID', 'TENANT_ID']
     - a dict (hashdiff form):          {'is_hashdiff': true, 'columns': [...]}
                                         (we hash the 'columns' value). #}
{% macro _dv_hash_expr(cols) %}
  {%- if cols is mapping -%}
    {%- set cols = cols.get('columns', cols.get('source_column', [])) -%}
  {%- endif -%}
  {%- if cols is string -%}
    SHA1_BINARY(UPPER(TRIM(NVL(CAST({{ cols }} AS VARCHAR), '^^'))))
  {%- else -%}
    SHA1_BINARY(
      {%- for c in cols -%}
        UPPER(TRIM(NVL(CAST({{ c }} AS VARCHAR), '^^')))
        {%- if not loop.last %} || '||~~||' || {% endif -%}
      {%- endfor -%}
    )
  {%- endif -%}
{% endmacro %}

{# dv_stage: SELECT * FROM source PLUS any hashed_columns computed inline.
   hashed_columns is a dict like {'CUSTOMER_HK': 'CUSTOMER_ID', 'LNK_X_HK': ['A','B']}.
   derived_columns is a dict of {alias: sql_expression} pairs, passed through.

   Snowflake's `SELECT * EXCLUDE (...)` requires every excluded column to
   actually exist in the source. Earlier versions of this macro built the
   EXCLUDE list from derived/hashed keys unconditionally — but audit
   columns like CREATED_BY / EFFECTIVE_DATETIME / BATCH_ID don't exist in
   raw landing tables, so EXCLUDE errored with "column ... does not
   exist". We now inspect the upstream relation at compile time via
   adapter.get_columns_in_relation and EXCLUDE only the names that
   actually overlap with the source's columns. #}
{% macro dv_stage(include_source_columns=true, source_model=None,
                  derived_columns=None, hashed_columns=None) %}
  {%- set _override_names = [] -%}
  {%- if derived_columns -%}
    {%- for alias in derived_columns.keys() -%}
      {%- do _override_names.append(alias | upper) -%}
    {%- endfor -%}
  {%- endif -%}
  {%- if hashed_columns -%}
    {%- for hk_alias in hashed_columns.keys() -%}
      {%- do _override_names.append(hk_alias | upper) -%}
    {%- endfor -%}
  {%- endif -%}
  {% set sm = _dv_source_model_name(source_model) %}
  {%- set _exclude = [] -%}
  {%- if include_source_columns and _override_names | length > 0 and execute -%}
    {#- Inspect upstream columns; intersect with overrides. Wrap in
        try-graceful-fallback because the relation may not exist yet
        (e.g. on a fresh `dbt parse` before any upstream is built). -#}
    {%- set _src_rel = ref(sm) -%}
    {%- set _src_cols_obj = [] -%}
    {%- set _src_cols_names = [] -%}
    {%- set _ok = true -%}
    {%- if _src_rel is not none -%}
      {%- set _src_cols_obj = adapter.get_columns_in_relation(_src_rel) or [] -%}
      {%- for c in _src_cols_obj -%}
        {%- do _src_cols_names.append(c.name | upper) -%}
      {%- endfor -%}
    {%- endif -%}
    {%- for nm in _override_names -%}
      {%- if nm in _src_cols_names and nm not in _exclude -%}
        {%- do _exclude.append(nm) -%}
      {%- endif -%}
    {%- endfor -%}
  {%- endif -%}
  select
    {% if include_source_columns -%}
      {%- if _exclude | length > 0 -%}
        src.* exclude (
        {%- for c in _exclude -%}
          {{ c }}{% if not loop.last %}, {% endif %}
        {%- endfor -%}
        )
      {%- else -%}
        src.*
      {%- endif -%}
    {%- else -%}
      /* source cols suppressed */ NULL as _placeholder
    {%- endif -%}
    {%- if derived_columns -%}
      {%- for alias, expr in derived_columns.items() -%}
        ,
        {{ expr }} as {{ alias }}
      {%- endfor -%}
    {%- endif -%}
    {%- if hashed_columns -%}
      {%- for hk_alias, nk_cols in hashed_columns.items() -%}
        ,
        {{ _dv_hash_expr(nk_cols) }} as {{ hk_alias }}
      {%- endfor -%}
    {%- endif %}
  from {{ ref(sm) }} src
{% endmacro %}

{# dv_hub: select PK (computed inline), NK, LDTS, SOURCE — deduped on PK.
   We ALWAYS compute the hash key from src_nk inline rather than reading
   src_pk from the upstream model. This way the macro compiles whether the
   stage was a passthrough (no HK column) or a fully-hashed stage (HK
   present but ignored — we recompute it identically). #}
{% macro dv_hub(src_pk, src_nk, src_ldts, src_source, source_model) %}
  {% set sm = _dv_source_model_name(source_model) %}
  {%- set nk_list = src_nk if (src_nk is sequence and src_nk is not string) else [src_nk] -%}
  with src as (
    select
      {{ _dv_hash_expr(src_nk) }} as {{ src_pk }},
      {% for nk in nk_list -%}
      {{ nk }} as {{ nk }}{% if not loop.last %},{% endif %}
      {% endfor %},
      {{ src_ldts }} as {{ src_ldts }},
      {{ src_source }} as {{ src_source }}
    from {{ ref(sm) }}
  )
  select
    {{ src_pk }},
    {%- for nk in nk_list %}
    {{ nk }},
    {%- endfor %}
    min({{ src_ldts }}) as {{ src_ldts }},
    min({{ src_source }}) as {{ src_source }}
  from src
  where {{ src_pk }} is not null
  group by
    {{ src_pk }}
    {%- for nk in nk_list -%}
      , {{ nk }}
    {%- endfor %}
{% endmacro %}

{# dv_link: select PK, all FK cols, LDTS, SOURCE — deduped on PK. #}
{% macro dv_link(src_pk, src_fk, src_ldts, src_source, source_model) %}
  {% set sm = _dv_source_model_name(source_model) %}
  {%- set fk_list = src_fk if (src_fk is sequence and src_fk is not string) else [src_fk] -%}
  select
    {{ src_pk }},
    {% for fk in fk_list -%}
      {{ fk }},
    {% endfor -%}
    min({{ src_ldts }}) as {{ src_ldts }},
    min({{ src_source }}) as {{ src_source }}
  from {{ ref(sm) }}
  where {{ src_pk }} is not null
  group by
    {{ src_pk }}
    {%- for fk in fk_list -%}
      , {{ fk }}
    {%- endfor %}
{% endmacro %}

{# dv_sat: select PK, hashdiff, payload, eff, LDTS, SOURCE.
   Payload may be a list of columns; if so, list them explicitly. #}
{% macro dv_sat(src_pk, src_hashdiff, src_payload, src_eff,
                src_ldts, src_source, source_model) %}
  {% set sm = _dv_source_model_name(source_model) %}
  {%- set payload_list = src_payload if (src_payload is sequence and src_payload is not string) else [src_payload] -%}
  select
    {{ src_pk }},
    {{ src_hashdiff }},
    {%- for p in payload_list %}
    {{ p }},
    {%- endfor %}
    {%- if src_eff %}
    {{ src_eff }},
    {%- endif %}
    {{ src_ldts }},
    {{ src_source }}
  from {{ ref(sm) }}
  where {{ src_pk }} is not null
{% endmacro %}

{% macro dv_pit(source_model, src_pk, as_of_dates_table, satellites,
                stage_tables_ldts, src_ldts) %}
  select * from {{ ref(source_model) }}
{% endmacro %}

{% macro dv_bridge(source_model, src_pk, src_ldts, bridge_walk,
                   as_of_dates_table, stage_tables_ldts) %}
  select * from {{ ref(source_model) }}
{% endmacro %}
"""

_NATIVE_QUERY_TAG_OVERRIDE_MACROS = """{# ============================================================================
  nexus: Disable query_tag for Snowflake native dbt jobs.
  ============================================================================
  In dbt-snowflake's default snowflake__set_query_tag(), the FIRST thing it
  does is call get_current_query_tag() to capture the prior tag for restore,
  and get_current_query_tag() runs:
      show parameters like 'query_tag' in session
  Snowflake's native dbt runtime (EXECUTE DBT PROJECT) rejects that with
      Unsupported statement type 'SHOW PARAMETER'
  even when no query_tag is configured anywhere — because the macro
  unconditionally captures-and-restores.

  Fix: override the *root-project* set_query_tag/unset_query_tag pair (the
  ones the materialization wrapper actually calls). Per dbt docs, root-project
  macros with these names take precedence over the adapter defaults without
  needing dispatch. We also override every namespaced variant as a defense in
  depth in case dispatch order is ever reconfigured.

  Critically, our overrides return early WITHOUT calling get_current_query_tag,
  so the SHOW PARAMETERS path is never entered. #}

{# --- Root macros invoked by the materialization wrapper (highest priority) #}
{% macro set_query_tag() -%}
  {{ return('') }}
{%- endmacro %}

{% macro unset_query_tag(original_query_tag) -%}
  {{ return('') }}
{%- endmacro %}

{% macro get_current_query_tag() -%}
  {{ return('') }}
{%- endmacro %}

{# --- Default-namespace fallbacks (used if dispatch resolves to 'default') -#}
{% macro default__set_query_tag() -%}
  {{ return('') }}
{%- endmacro %}

{% macro default__unset_query_tag(original_query_tag) -%}
  {{ return('') }}
{%- endmacro %}

{% macro default__get_current_query_tag() -%}
  {{ return('') }}
{%- endmacro %}

{# --- Snowflake-namespace overrides (if adapter dispatches to 'snowflake') --#}
{% macro snowflake__set_query_tag() -%}
  {{ return('') }}
{%- endmacro %}

{% macro snowflake__unset_query_tag(original_query_tag) -%}
  {{ return('') }}
{%- endmacro %}

{% macro snowflake__get_current_query_tag() -%}
  {{ return('') }}
{%- endmacro %}
"""








def _snowflake_quote_ident(ident: str) -> str:
    """Double-quote a Snowflake identifier; escape embedded quotes."""
    s = (ident or "").strip()
    if not s:
        return '""'
    return '"' + s.replace('"', '""') + '"'


def _snowflake_type_guess_for_column(col: str) -> str:
    u = (col or "").upper()
    if any(
        x in u
        for x in ("_TS", "_AT", "TIMESTAMP", "DATETIME", "_DTM", "LOAD_DTS",
                  "EFFECTIVE", "EXPIRY")
    ):
        return "TIMESTAMP_NTZ"
    if u.endswith("_DT") or u.endswith("_DATE") or "DATE" in u:
        return "DATE"
    if "HASH" in u or u.endswith("_HK") or u.endswith("_SK"):
        return "VARCHAR(128)"
    if u.endswith("_ID") or u == "ID" or u.endswith("KEY"):
        return "VARCHAR(500)"
    return "VARCHAR(4000)"


def _extract_output_column_from_select_item(piece: str) -> str:
    """
    Given one comma-separated SELECT item, return the output column name
    (alias if present, else last simple identifier token).
    """
    p = (piece or "").strip()
    if not p:
        return ""
    m_as = re.search(r"\bas\s+([A-Za-z_][A-Za-z0-9_]*)\s*$", p, re.I)
    if m_as:
        return m_as.group(1)
    toks = re.findall(r"[A-Za-z_][A-Za-z0-9_]*", p)
    if not toks:
        return ""
    # Drop common SQL keywords at end
    kw = {"null", "true", "false", "as", "and", "or", "not"}
    while toks and toks[-1].lower() in kw:
        toks.pop()
    return toks[-1] if toks else ""




def _parse_select_columns_for_raw_source(body: str, raw_table: str) -> Optional[List[str]]:
    """
    If a staging model selects an explicit column list from
    ``source('raw', '<raw_table>')``, return those output column names.
    Returns None when only ``SELECT *`` (or parse fails).
    """
    if not body or not raw_table:
        return None
    b = re.sub(r"\{#.*?#\}", "", body, flags=re.S)
    b = re.sub(r"--[^\n]*", "", b)
    # Case-insensitive match on raw entity (DDL uses UPPER names; SQL may not).
    esc = re.escape((raw_table or "").strip().upper())
    pat = (
        r"select\s+(?P<sel>[\s\S]+?)\s+from\s+"
        r"\{\{\s*source\s*\(\s*['\"]raw['\"]\s*,\s*['\"]"
        + f"(?i:{esc})"
        + r"['\"]\s*\)\s*\}\}"
    )
    m = re.search(pat, b, re.I)
    if not m:
        return None
    sel = (m.group("sel") or "").strip()
    if not sel or re.fullmatch(r"\*", sel):
        return None
    depth = 0
    cur: List[str] = []
    cols: List[str] = []
    for ch in sel + ",":
        if ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            cur.append(ch)
        elif ch == "," and depth == 0:
            piece = "".join(cur).strip()
            if piece:
                cname = _extract_output_column_from_select_item(piece)
                if cname:
                    cols.append(cname)
            cur = []
        else:
            cur.append(ch)
    return cols if cols else None


def _parse_raw_table_names_from_sources_yml(yml: str) -> set:
    """Return table names listed under ``sources: - name: raw`` → ``tables:``."""
    names = set()
    if not yml or not isinstance(yml, str):
        return names
    in_raw = False
    in_tables = False
    raw_indent: Optional[int] = None
    for line in yml.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if re.match(r"-\s*name:\s*raw\s*$", stripped, re.I):
            in_raw = True
            in_tables = False
            raw_indent = indent
            continue
        if in_raw and raw_indent is not None:
            if (
                indent == raw_indent
                and stripped.startswith("- name:")
                and not re.match(r"-\s*name:\s*raw\s*$", stripped, re.I)
            ):
                in_raw = False
                in_tables = False
                continue
            if re.match(r"tables:\s*$", stripped, re.I) and indent > raw_indent:
                in_tables = True
                continue
            if in_tables and indent > raw_indent:
                m = re.match(r"-\s*name:\s*(.+)$", stripped, re.I)
                if m:
                    nm = m.group(1).strip().strip("'\"")
                    if nm and nm.lower() != "raw":
                        names.add(nm.upper())
                continue
            if in_tables and indent <= raw_indent:
                in_tables = False
    return names




def _collect_raw_landing_table_names(files: dict) -> List[str]:
    """
    Union of RAW entities from every signal we can find in the project:

      - ``source('raw', '<X>')`` calls anywhere in any file
      - ``ref('stg_<X>')`` and ``ref('br_<X>')`` calls
      - ``source_model='stg_<X>'`` and ``source_model=['stg_<X>']`` —
        common in automate_dv.hub / .link / .sat call sites that don't
        ref() the staging model directly
      - ``_sources.yml`` files under ``sources: - name: raw → tables:``
      - ``stg_<X>.sql`` / ``br_<X>.sql`` filenames

    Without the source_model and ref('br_*') signals, entities that only
    appeared in hub/sat call args or via bronze-only references were
    silently dropped, so their RAW landing tables never got created.
    """
    found: set = set()
    for _, body in (files or {}).items():
        if not isinstance(body, str):
            continue
        # source('raw', 'X')
        for m in re.finditer(
            r"source\s*\(\s*['\"]raw['\"]\s*,\s*['\"]([A-Za-z0-9_\-]+)['\"]\s*\)",
            body,
            flags=re.I,
        ):
            found.add(m.group(1).upper())
        # ref('stg_X') and ref('br_X')
        for m in re.finditer(
            r"ref\(\s*['\"]((?:stg|br)_[A-Za-z0-9_]+)['\"]\s*\)",
            body,
            flags=re.I,
        ):
            nm = m.group(1).lower()
            if nm.startswith("stg_") and len(nm) > 4:
                found.add(nm[4:].upper())
            elif nm.startswith("br_") and len(nm) > 3:
                found.add(nm[3:].upper())
        # source_model='stg_X'  or  source_model="stg_X"
        for m in re.finditer(
            r"source_model\s*=\s*['\"](stg|br)_([A-Za-z0-9_]+)['\"]",
            body,
            flags=re.I,
        ):
            found.add(m.group(2).upper())
        # source_model=['stg_X', ...]  (list literal) — capture every entry
        for lst in re.finditer(
            r"source_model\s*=\s*\[([^\]]+)\]",
            body,
            flags=re.I | re.S,
        ):
            for nm_match in re.finditer(
                r"['\"](stg|br)_([A-Za-z0-9_]+)['\"]",
                lst.group(1),
                flags=re.I,
            ):
                found.add(nm_match.group(2).upper())
    for p, body in (files or {}).items():
        lp = p.replace("\\", "/").lower()
        if lp.endswith("_sources.yml") or lp.endswith("/_sources.yml"):
            found |= _parse_raw_table_names_from_sources_yml(
                body if isinstance(body, str) else ""
            )
    found |= _infer_raw_tables_from_model_paths(files)
    return sorted(found)


def _build_raw_landing_ddl_script(files: dict) -> str:
    """
    Build a runnable Snowflake DDL script for RAW landing tables referenced
    by the Raw Vault dbt project (``source('raw', ...)``).

    Table names are collected from: ``source('raw', ...)`` calls,
    ``models/**/_sources.yml`` under ``name: raw``, and ``stg_*`` / ``br_*``
    model filenames. Column lists are inferred when staging SQL lists
    explicit columns before ``{{ source('raw', '<entity>') }}``; otherwise a
    small VARIANT stub table is emitted so the script is never empty and
    you can widen columns after inspecting landing data.

    RAW **table** identifiers are emitted in **UPPER_CASE** (Snowflake
    default); column names stay as inferred from staging SQL.
    """
    raw_tables = _collect_raw_landing_table_names(files or {})
    fq_schema = f"{VECTOR_DB}.{RAW_SOURCE_SCHEMA}"
    dbq = _snowflake_quote_ident(VECTOR_DB)
    schq = _snowflake_quote_ident(RAW_SOURCE_SCHEMA)
    lines: List[str] = [
        "-- ============================================================================\n",
        "-- RAW landing DDL generated from Raw Vault dbt models.\n",
        f"-- Target schema: {fq_schema}\n",
        "-- Tables are inferred from source('raw', ...), _sources.yml, and\n",
        "-- stg_* / br_* model names when explicit source() calls are absent.\n",
        "-- Run in a worksheet with a role that can create schema/tables.\n",
        "-- ============================================================================\n",
        f"CREATE SCHEMA IF NOT EXISTS {dbq}.{schq};\n",
    ]
    if not raw_tables:
        lines.append(
            "\n-- No RAW entities could be inferred from this dbt bundle.\n"
            "-- Add `source('raw', '<entity>')` in staging/bronze SQL or list\n"
            "-- tables under `sources: - name: raw` in a *_sources.yml file,\n"
            "-- then regenerate Raw Vault dbt or redeploy.\n"
        )
        return "".join(lines)

    for tbl in raw_tables:
        staging_bodies: List[str] = []
        for p, body in (files or {}).items():
            if not isinstance(body, str):
                continue
            lp = p.replace("\\", "/").lower()
            if "stg_" not in lp or not lp.endswith(".sql"):
                continue
            if re.search(
                rf"source\s*\(\s*['\"]raw['\"]\s*,\s*['\"](?i:{re.escape(tbl)})['\"]\s*\)",
                body,
                flags=re.I,
            ):
                staging_bodies.append(body)

        # Pass 1: explicit SELECT list referencing source('raw',...)
        cols: Optional[List[str]] = None
        col_source = "explicit SELECT list"
        for sb in staging_bodies:
            cols = _parse_select_columns_for_raw_source(sb, tbl)
            if cols:
                break

        # Pass 2: Data Vault metadata fallback (hashed_columns,
        # src_nk, src_payload). This is the common case when staging
        # uses automate_dv.stage(...) macros that don't have an
        # explicit SELECT list to parse.
        if not cols:
            cols = _infer_raw_columns_from_dv_metadata(files or {}, tbl)
            if cols:
                col_source = (
                    "Data Vault metadata (hashed_columns / src_nk / src_payload)"
                )

        # Snowflake landing tables: UPPER-case identifiers (unquoted OK).
        tu = tbl.upper()
        tbq = tu if re.fullmatch(r"[A-Z0-9_]+", tu) else _snowflake_quote_ident(tu)
        lines.append(f"\n-- ---- RAW entity: {tbl.upper()} ----\n")
        if cols:
            lines.append(f"-- Columns inferred from {col_source}.\n")
            col_defs = []
            for c in cols:
                ct = _snowflake_type_guess_for_column(c)
                col_defs.append(
                    f"  {_snowflake_quote_ident(c)} {ct}"
                )
            lines.append(
                f"CREATE OR REPLACE TABLE {dbq}.{schq}.{tbq} (\n"
                + ",\n".join(col_defs)
                + "\n) COMMENT = 'Inferred columns; review types and add missing fields as needed.';\n"
            )
        else:
            lines.append(
                f"-- No column metadata found for `{tbl}` "
                f"(no SELECT list, no hashed_columns, no src_nk/src_payload).\n"
                f"-- Stub landing table: widen/replace columns after you inspect payloads.\n"
            )
            lines.append(
                f"CREATE OR REPLACE TABLE {dbq}.{schq}.{tbq} (\n"
                f"  {_snowflake_quote_ident('RAW_RECORD')} VARIANT "
                f"COMMENT 'Raw row (JSON/CSV); cast in staging as needed.'\n"
                f") COMMENT = 'Auto stub for dbt source(raw, {tbl}); replace with real columns.';\n"
            )
    text = "".join(lines).strip() + "\n"
    return text if text.strip() else (
        "-- RAW DDL generator produced no output (unexpected).\n"
        f"CREATE SCHEMA IF NOT EXISTS {dbq}.{schq};\n"
    )


def _build_min_schema_yml(model_names: List[str]) -> str:
    """
    Build a minimal valid dbt schema.yml dictionary.
    Ensures Snowflake parser gets a proper YAML mapping.
    """
    uniq = []
    seen = set()
    for m in model_names:
        mm = (m or "").strip()
        if not mm or mm in seen:
            continue
        seen.add(mm)
        uniq.append(mm)
    if not uniq:
        return "version: 2\nmodels: []\n"
    lines = ["version: 2", "models:"]
    for m in sorted(uniq):
        lines.append(f"  - name: {m}")
        lines.append("    description: Auto-generated schema entry")
    return "\n".join(lines) + "\n"












def _prepare_files_for_snowflake_native_dbt(files: dict) -> dict:
    """
    Snowflake DBT PROJECT snapshots don't persist package installs from EXECUTE deps.
    Convert external-package macro calls to local fallback macros so builds run.
    """
    out = _ensure_dbt_runtime_profiles(files)
    for p, body in list(out.items()):
        if isinstance(body, str):
            out[p] = _normalize_generated_file_body(body)
    # Sanitize every dbt_project*.yml AND profiles.yml at any depth. The
    # sanitizer strips query_tag, persist_docs, on-run hooks, query-comment,
    # and appends a `dispatch:` block so our project's set_query_tag override
    # wins over dbt-snowflake's adapter default.
    for p in list(out.keys()):
        norm = p.replace("\\", "/").lower()
        leaf = norm.rsplit("/", 1)[-1]
        is_project = (
            leaf == "dbt_project.yml"
            or (leaf.startswith("dbt_project") and leaf.endswith(".yml"))
        )
        is_profiles = (leaf == "profiles.yml")
        if is_project or is_profiles:
            b = out.get(p)
            if isinstance(b, str):
                out[p] = _sanitize_dbt_project_yml_for_snowflake_native(b)
    # Remove external package requirements for native execution path.
    if "packages.yml" in out:
        out["packages.yml"] = "packages: []\n"
    out["macros/local_automate_dv.sql"] = _LOCAL_AUTOMATE_DV_MACROS
    # Force-write the query_tag override macros file. Always overwrite — if
    # the LLM happened to emit a same-named file with a different body, our
    # override must win (this is the file that prevents SHOW PARAMETERS).
    out["macros/native_query_tag_overrides.sql"] = (
        _NATIVE_QUERY_TAG_OVERRIDE_MACROS
    )
    # Strip any pre-existing `generate_schema_name` macro definitions from
    # other macro files BEFORE we write our canonical override. The LLM
    # sometimes emits files like `macros/native_schema_overrides.sql` that
    # also define `generate_schema_name`, and dbt rejects builds with two
    # macros of the same name. Our override is the one that must win because
    # the deploy pipeline pre-creates schemas (RAW_VAULT, BUSINESS_VAULT,
    # GOLD) with the names our macro produces.
    #
    # Strategy: scan for any `macro\s+generate_schema_name` phrase
    # (case-insensitive, any whitespace), then walk back to the opening
    # `{%` and forward to the closing `%}` of the matching `endmacro`.
    # This catches all Jinja whitespace-control variants (`{%-`, `-%}`,
    # minified `{%macro%}`, weird whitespace inside the tag, etc.)
    # without trying to write a perfect Jinja-aware regex.
    _PHRASE_RE = re.compile(
        r"macro\s+generate_schema_name", flags=re.I,
    )
    _OUR_SCHEMA_MACRO_PATH = "macros/get_schema_name.sql"
    stripped_count = 0
    for p in list(out.keys()):
        if p == _OUR_SCHEMA_MACRO_PATH:
            continue
        lp = p.replace("\\", "/").lower()
        if not lp.startswith("macros/") or not lp.endswith(".sql"):
            continue
        body = out.get(p)
        if not isinstance(body, str) or "generate_schema_name" not in body:
            continue

        # Iteratively excise every macro definition. Re-scan after each
        # removal in case the file declares the macro multiple times.
        keep_iterating = True
        while keep_iterating:
            keep_iterating = False
            m = _PHRASE_RE.search(body)
            if not m:
                break
            phrase_start = m.start()
            # Walk back from the phrase to find the opening `{%` (allow up
            # to 24 chars of whitespace + dash between `{%` and `macro`,
            # which is enough for any sane LLM output).
            open_search = body.rfind(
                "{%", max(0, phrase_start - 24), phrase_start,
            )
            if open_search < 0:
                # Phrase appears outside a Jinja tag (e.g. a SQL comment).
                # Mask out this occurrence so the loop terminates without
                # falling into an infinite cycle.
                body = body[:phrase_start] + "MACRO_xxx_GENERATE_SCHEMA_NAME" + body[m.end():]
                continue
            # Walk forward to find the matching `endmacro` and its closing %}
            after = m.end()
            end_m = re.search(r"endmacro", body[after:], flags=re.I)
            if not end_m:
                break
            end_idx = after + end_m.start()
            close_search = body.find("%}", end_idx)
            if close_search < 0:
                break
            close_end = close_search + 2
            body = (
                body[:open_search]
                + "{# nexus: removed conflicting generate_schema_name macro;\n"
                  "   the canonical override lives in "
                  "macros/get_schema_name.sql #}"
                + body[close_end:]
            )
            stripped_count += 1
            keep_iterating = True

        # Restore any masked phrases (false positives — phrase appeared
        # outside a Jinja tag, e.g. in a SQL comment).
        body = body.replace(
            "MACRO_xxx_GENERATE_SCHEMA_NAME",
            "macro generate_schema_name",
        )

        # If the file no longer contains any macro definition, drop it
        # entirely so dbt doesn't try to parse a near-empty file.
        if not re.search(r"\{%-?\s*macro\s+\w+", body, flags=re.I):
            del out[p]
        else:
            out[p] = body
    # Stash the strip count on a module-level dict so deploy sites can
    # surface it to the user. Keyed by id() of the returned dict so
    # nested calls don't trample each other.
    _NEXUS_STRIP_COUNTERS[id(out)] = stripped_count

    # Inject generate_schema_name override so that ``+schema: raw_vault``
    # in dbt_project.yml resolves to the *literal* RAW_VAULT schema (after
    # upper-casing), not dbt's default ``{target.schema}_raw_vault``
    # concatenation. Without this, models tagged +schema: raw_vault land
    # in a hybrid schema like INNOVATION_DEV_raw_vault, defeating the
    # per-layer separation. The custom macro:
    #   - returns target.schema (default) when no custom schema is set
    #   - returns the custom schema verbatim (uppercased) otherwise
    out[_OUR_SCHEMA_MACRO_PATH] = (
        "{# nexus: override default generate_schema_name so +schema in\n"
        "   dbt_project.yml resolves to a literal schema name (upper-cased)\n"
        "   rather than {target.schema}_{custom_schema_name}. This is what\n"
        "   makes RAW_VAULT and BUSINESS_VAULT separation work. #}\n"
        "{% macro generate_schema_name(custom_schema_name, node) -%}\n"
        "    {%- if custom_schema_name is none -%}\n"
        "        {{ target.schema }}\n"
        "    {%- else -%}\n"
        "        {{ custom_schema_name | trim | upper }}\n"
        "    {%- endif -%}\n"
        "{%- endmacro %}\n"
    )

    for p, body in list(out.items()):
        lp = p.lower()
        if not lp.endswith(".sql") or not isinstance(body, str):
            continue
        b = body
        b = _sanitize_jinja_in_yaml_metadata(b)
        b = _normalize_dbt_node_refs(b)
        b = _normalize_source_raw_table_casing(b)
        b = b.replace("automate_dv.stage(", "dv_stage(")
        b = b.replace("automate_dv.hub(", "dv_hub(")
        b = b.replace("automate_dv.link(", "dv_link(")
        b = b.replace("automate_dv.sat(", "dv_sat(")
        b = b.replace("automate_dv.pit(", "dv_pit(")
        b = b.replace("automate_dv.bridge(", "dv_bridge(")
        b = _strip_query_tag_from_sql_model(b)
        b = _strip_hooks_from_sql_model(b)
        out[p] = b

    # Force valid YAML dictionary for schema files used in Snowflake runs.
    rv_models = []
    bv_models = []
    for p in out:
        pp = p.replace("\\", "/").lower()
        if not pp.endswith(".sql"):
            continue
        name = p.replace("\\", "/").rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if pp.startswith("models/silver/raw_vault/"):
            rv_models.append(name)
        if pp.startswith("models/silver/business_vault/"):
            bv_models.append(name)

    if rv_models:
        out["models/silver/raw_vault/schema.yml"] = _build_min_schema_yml(
            rv_models
        )
    if bv_models:
        out["models/silver/business_vault/schema.yml"] = _build_min_schema_yml(
            bv_models
        )
    out = _ensure_raw_source_catalog(out)
    out = _repair_missing_stage_references(out)
    out = _ensure_bronze_shims_for_missing_refs(out)
    # Normalize table-name casing in EVERY *_sources.yml in the project.
    # This is the catch-all for LLM-emitted YAMLs (e.g. models/_sources.yml)
    # whose lowercase entries don't match the upper-cased source('raw',...)
    # calls in models. dbt's source resolution is exact-string match, so
    # a lowercase YAML entry + uppercase SQL ref = "source not found".
    out = _normalize_sources_yml_table_casing(out)
    out = _strip_disallowed_show_parameter_sql(out)
    out = _final_native_dbt_safety_sanitize(out)
    out["scripts/create_raw_landing_tables.sql"] = _build_raw_landing_ddl_script(
        out
    )
    return out


def _bind_profiles_to_current_session(session, files: dict) -> dict:
    """
    Overwrite ``profiles.yml`` with the fixed Snowflake-native profile used for
    ``EXECUTE DBT PROJECT`` (empty account/user per Snowflake docs; role,
    warehouse, database, and schema from ``NATIVE_DBT_PROFILE_*`` and
    ``VECTOR_*``). The ``session`` argument is kept for a stable call
    signature at deploy sites; values are not read from the session.
    """
    out = dict(files or {})
    out["profiles.yml"] = _native_snowflake_profiles_yml()
    return out




# DDL/DML keywords we'll actually send to Snowflake. Anything that doesn't
# begin with one of these (after stripping leading comment lines) is
# treated as a non-statement and skipped, even if our splitter passed it
# through. Belt-and-suspenders.
_EXECUTABLE_SQL_KEYWORDS = (
    "CREATE", "ALTER", "DROP", "USE", "GRANT", "REVOKE",
    "INSERT", "UPDATE", "DELETE", "MERGE", "TRUNCATE",
    "COMMENT", "SET", "BEGIN", "COMMIT", "ROLLBACK",
    "CALL", "EXECUTE", "COPY", "PUT", "GET", "REMOVE",
    "DESCRIBE", "DESC", "SHOW",
)


def _statement_starts_with_executable_keyword(stmt: str) -> bool:
    """
    Strip leading line/block comments and whitespace, then check whether
    the first SQL token is something we'd actually want to execute.
    Filters out comment-only fragments and stray text the splitter may
    have produced.
    """
    s = stmt.lstrip()
    # Strip leading comment lines and block comments
    while s:
        if s.startswith("--"):
            nl = s.find("\n")
            s = s[nl + 1:].lstrip() if nl >= 0 else ""
        elif s.startswith("/*"):
            end = s.find("*/")
            s = s[end + 2:].lstrip() if end >= 0 else ""
        else:
            break
    if not s:
        return False
    first_word = re.split(r"\s+", s, maxsplit=1)[0].upper()
    return first_word in _EXECUTABLE_SQL_KEYWORDS


def _execute_raw_landing_ddl(session, files: dict) -> dict:
    """
    Execute the auto-generated raw landing DDL so the bronze ``source('raw',
    ...)`` calls have something to read, AND ensure the per-layer dbt
    target schemas exist so models that use ``+schema: raw_vault`` or
    ``+schema: business_vault`` have a target schema to materialize into.

    The DDL script is built by ``_build_raw_landing_ddl_script`` and bundled
    into ``files`` under ``scripts/create_raw_landing_tables.sql``. We use
    a string-aware splitter (``_split_sql_script_into_statements``) that
    correctly skips ``;`` inside string literals — earlier versions used
    ``script.split(";")``, which broke on COMMENT clauses like
    ``'Raw row (JSON/CSV); cast in staging as needed.'`` and split single
    CREATE TABLE statements into multiple malformed fragments.

    Empty stub tables are fine for compilation: ``dbt build`` will succeed
    and downstream models will simply produce zero rows until real data is
    loaded into the RAW schema.

    Returns a result summary:
      {
        "executed":   int,   # number of statements run successfully
        "skipped":    int,   # number of comment-only / non-executable fragments
        "errors":     list,  # [(stmt_preview, error_str), ...]
        "schemas":    list,  # fully-qualified schema names ensured
      }

    If the bundle contains no DDL (e.g. business_vault layer with no raw
    sources), this is a no-op for tables but per-layer schemas are still
    created.
    """
    schemas_ensured = [
        f"{VECTOR_DB}.{RAW_SOURCE_SCHEMA}",
        f"{VECTOR_DB}.{RAW_VAULT_SCHEMA}",
        f"{VECTOR_DB}.{BUSINESS_VAULT_SCHEMA}",
        f"{VECTOR_DB}.GOLD",
    ]
    summary = {
        "executed": 0, "skipped": 0, "errors": [],
        "schemas": list(schemas_ensured),
    }

    # Always ensure per-layer schemas exist before any dbt model runs.
    # Empty stub tables in RAW won't help if dbt can't find the target
    # schema for raw_vault / business_vault models.
    for fq_schema in schemas_ensured:
        db_part, sch_part = fq_schema.split(".", 1)
        ddl = (
            f"CREATE SCHEMA IF NOT EXISTS "
            f"{_snowflake_quote_ident(db_part)}."
            f"{_snowflake_quote_ident(sch_part)}"
        )
        try:
            session.sql(ddl).collect()
            summary["executed"] += 1
        except Exception as e:
            preview = ddl[:160].replace("\n", " ")
            summary["errors"].append((preview, str(e)))

    ddl_script = (files or {}).get("scripts/create_raw_landing_tables.sql", "")
    if not isinstance(ddl_script, str) or not ddl_script.strip():
        return summary

    for stmt in _split_sql_script_into_statements(ddl_script):
        if not _statement_starts_with_executable_keyword(stmt):
            summary["skipped"] += 1
            continue
        try:
            session.sql(stmt).collect()
            summary["executed"] += 1
        except Exception as e:
            preview = stmt[:160].replace("\n", " ")
            summary["errors"].append((preview, str(e)))

    return summary




def generate_dbt_project_per_file(call_fn, context: str, layer: str,
                                  progress_cb=None) -> tuple:
    """
    Generate a full dbt project one file at a time.

    Args:
      call_fn:     function(prompt_str, opts_dict) -> response_str
      context:     the data model context to ground all files on
      layer:       "raw_vault" | "business_vault" | "dbt_tests"
      progress_cb: optional callable(done, total, current_file)

    Returns:
      (files_dict, plan_raw, errors_list)
      files_dict    : {path: content}
      plan_raw      : the raw planner response (for debugging)
      errors_list   : [(path, exception_str), ...] for any failures
    """
    # 1) Plan
    plan_raw = call_fn(
        build_dbt_planner_prompt(context, layer),
        {"temperature": 0.2, "max_tokens": 1500, "top_p": 0.9},
    )
    paths = parse_dbt_plan(plan_raw)
    paths = _filter_dbt_paths_for_layer(paths, layer)
    if not paths and layer == "raw_vault":
        paths = [
            "models/silver/staging/stg_entity.sql",
            "models/silver/raw_vault/hubs/hub_entity.sql",
            "models/silver/raw_vault/links/lnk_entity_relation.sql",
            "models/silver/raw_vault/sats/sat_entity_detail.sql",
            "models/silver/raw_vault/schema.yml",
        ]
    elif not paths and layer == "business_vault":
        paths = [
            "models/silver/business_vault/staging/stg_entity.sql",
            "models/silver/business_vault/hubs/hub_entity.sql",
            "models/silver/business_vault/links/lnk_entity_relation.sql",
            "models/silver/business_vault/sats/sat_entity_detail.sql",
            "models/silver/business_vault/schema.yml",
        ]

    if not paths:
        return {}, plan_raw, [("<planner>",
                              "No file paths returned by planner")]

    # Cap file count to keep runtime reasonable (planner may list many)
    paths = paths[:28]

    # 2) Generate each file
    files = {}
    errors = []
    for i, p in enumerate(paths, start=1):
        if progress_cb:
            try:
                progress_cb(i - 1, len(paths), p)
            except Exception:
                pass
        try:
            body = call_fn(
                build_dbt_file_prompt(p, context, layer),
                {"temperature": 0.4, "top_p": 0.9,
                 "max_tokens": 2500, "guardrails": False},
            )
            cleaned = clean_file_body(body)
            if cleaned.strip():
                files[p] = cleaned
            else:
                errors.append((p, "empty body"))
        except Exception as e:
            errors.append((p, str(e)))

    if progress_cb:
        try:
            progress_cb(len(paths), len(paths), "done")
        except Exception:
            pass

    _ensure_forward_dbt_scaffold(files, layer)

    if layer == "raw_vault" and files:
        files["scripts/create_raw_landing_tables.sql"] = (
            _build_raw_landing_ddl_script(files)
        )

    return files, plan_raw, errors




def bundle_dbt_project(files: dict, project_name: str = "dbt_project") -> bytes:
    """Pack {relative_path: content} into a ZIP."""
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for path, content in files.items():
            safe = path.lstrip("/\\").replace("..", "_")
            zf.writestr(safe, content)
        zf.writestr(
            "MANIFEST.txt",
            f"dbt project: {project_name}\n"
            f"files: {len(files)}\n"
            f"generated_utc: "
            f"{datetime.utcnow().strftime('%Y-%m-%dT%H-%M-%SZ')}\n"
        )
    return buf.getvalue()


def render_mermaid(script: str, height: int = 500):
    """Render a Mermaid diagram inline using the Mermaid JS CDN."""
    import streamlit.components.v1 as components
    # Strip any leftover fences
    script = script.strip()
    if script.startswith("```"):
        script = re.sub(r'^```(?:mermaid)?\s*', '', script)
        script = re.sub(r'```\s*$', '', script)
    # Escape for safe HTML embedding
    safe = script.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""
    <div style="background:#FFFFFF; border:1px solid #E8E6DC;
                border-radius:12px; padding:16px;">
      <pre class="mermaid" style="background:transparent; margin:0;">{safe}</pre>
    </div>
    <script type="module">
      import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
      mermaid.initialize({{
        startOnLoad: true,
        theme: 'default',
        themeVariables: {{
          primaryColor: '#F5F4EE',
          primaryTextColor: '#141413',
          primaryBorderColor: '#C96442',
          lineColor: '#6B6456',
          fontFamily: 'Inter, -apple-system, sans-serif'
        }}
      }});
    </script>
    """
    components.html(html, height=height, scrolling=True)


# ─────────────────────────────────────────────────────────────────────────────
# METADATA PARSERS
# ─────────────────────────────────────────────────────────────────────────────




































# DataStage column metadata uses ODBC/JDBC numeric SQL type codes.
# This table maps code → canonical SQL type name, which is what the LLM
# needs to emit correct "Source Data Type" and "Target Data Type" cells
# in the STTM. Without this, the summary only lists column NAMES and the
# model has to guess the type.




# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDERS
# Strategy: each prompt returns ONE simple artifact type the LLM handles well:
#   • Raw text / code (for Mermaid scripts and SQL DDL)
#   • Markdown pipe tables (for STTM & Data Catalog — then parsed to DataFrames)
#   • Markdown (for narratives)
# We avoid multi-key JSON because LLMs frequently mangle nested code inside
# JSON string values.
# ─────────────────────────────────────────────────────────────────────────────













# ─────────────────────────────────────────────────────────────────────────────
# DATA LINEAGE GRAPH
#
# Strategy: build a DETERMINISTIC Mermaid flowchart from what we already
# parsed (files → stages → cross-job flows → shared-column relationships)
# and then ask the LLM to do one narrow job: read the Raw Vault DDL and
# emit `STAGE -> HUB` edges so the graph terminates at the target model.
#
# This is more reliable than asking the LLM to invent the whole graph —
# the structural edges come from actual metadata, and the LLM only fills
# in the source-to-Hub resolution piece, which it is good at.
# ─────────────────────────────────────────────────────────────────────────────

def extract_raw_vault_tables(sql_ddl: str) -> dict:
    """
    Pull table names out of the Raw Vault DDL, classified by kind.
    Returns {"hubs": [...], "links": [...], "sats": [...]}.
    Used by the lineage graph so we can draw source→Hub edges even when
    the Raw Vault SQL is the only thing we know about the target side.
    """
    hubs, links, sats = [], [], []
    if not sql_ddl:
        return {"hubs": hubs, "links": links, "sats": sats}

    # CREATE [OR REPLACE] TABLE [schema.]NAME
    table_re = re.compile(
        r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+'
        r'(?:[A-Za-z_][\w]*\.)?([A-Za-z_][\w]*)',
        re.IGNORECASE
    )
    seen = set()
    for m in table_re.finditer(sql_ddl):
        name = m.group(1).upper()
        if name in seen:
            continue
        seen.add(name)
        if name.startswith("HUB_"):
            hubs.append(name)
        elif name.startswith("LNK_") or name.startswith("LINK_"):
            links.append(name)
        elif name.startswith("SAT_"):
            sats.append(name)
    return {"hubs": hubs, "links": links, "sats": sats}






def build_lineage_graph(parsed_metadata: dict, raw_vault_tables: dict,
                        source_to_hub_df) -> dict:
    """
    Same lineage information as build_lineage_mermaid() but returned as
    a structured graph dict suitable for an interactive renderer:

        {
          "nodes": [{"id", "label", "kind", "group", "title", ...}, ...],
          "edges": [{"from", "to", "kind", "label", "dashed"}, ...]
        }

    Node "kind" is one of: file, stage, hub, link, sat.
    Edge "kind" is one of: file_stage, stage_stage, stage_hub,
                           hub_link, hub_sat, cross_file.

    "title" is HTML content shown on hover (tooltip).
    "group" is the file name (for stages) or "__rv__" (for RV nodes).
    """
    nodes, edges = [], []
    if not parsed_metadata:
        return {"nodes": nodes, "edges": edges}

    entities = parsed_metadata.get("entities", {})
    cross_flows = parsed_metadata.get("cross_job_flows", [])
    files = parsed_metadata.get("files", [])

    def _esc(s):
        return (str(s).replace("&", "&amp;")
                       .replace("<", "&lt;")
                       .replace(">", "&gt;"))

    # File nodes
    for f in files:
        fname = f["filename"]
        nodes.append({
            "id":    "F_" + _mermaid_safe_id(fname),
            "label": fname,
            "kind":  "file",
            "group": fname,
            "title": f"<b>File:</b> {_esc(fname)}<br/>"
                     f"<b>Kind:</b> {_esc(f.get('kind', ''))}",
        })

    # Stage nodes
    stage_ids = {}
    for ent_key, meta in entities.items():
        sid = "S_" + _mermaid_safe_id(ent_key)
        stage_ids[ent_key] = sid
        cols = sorted(meta["columns"])
        typed = meta.get("columns_typed", {})
        col_lines = []
        for c in cols[:30]:
            t = typed.get(c)
            col_lines.append(f"&nbsp;&nbsp;• {_esc(c)}"
                             + (f" <i>{_esc(t)}</i>" if t else ""))
        if len(cols) > 30:
            col_lines.append(f"&nbsp;&nbsp;… and {len(cols) - 30} more")
        tooltip = (
            f"<b>Stage:</b> {_esc(meta['display_name'])}<br/>"
            f"<b>Job:</b> {_esc(meta.get('job_name') or '—')}<br/>"
            f"<b>File:</b> {_esc(meta['source_file'])}<br/>"
            f"<b>Type:</b> {_esc(meta.get('type', '—'))}<br/>"
            f"<b>Columns ({len(cols)}):</b><br/>"
            + "<br/>".join(col_lines)
        )
        label = meta["display_name"]
        if meta.get("job_name"):
            label = f"{label}\n({meta['job_name']})"
        nodes.append({
            "id":    sid,
            "label": label,
            "kind":  "stage",
            "group": meta["source_file"],
            "title": tooltip,
        })

        # File → Stage edge
        nodes_file_id = "F_" + _mermaid_safe_id(meta["source_file"])
        edges.append({
            "from":   nodes_file_id,
            "to":     sid,
            "kind":   "file_stage",
            "label":  "",
            "dashed": False,
        })

    # Raw Vault nodes
    hub_ids, link_ids, sat_ids = {}, {}, {}
    for name in raw_vault_tables.get("hubs", []):
        hid = "H_" + _mermaid_safe_id(name)
        hub_ids[name] = hid
        nodes.append({
            "id":    hid,
            "label": name,
            "kind":  "hub",
            "group": "__rv__",
            "title": f"<b>Hub:</b> {_esc(name)}<br/>"
                     f"<i>Deduplicated business concept</i>",
        })
    for name in raw_vault_tables.get("links", []):
        lid = "K_" + _mermaid_safe_id(name)
        link_ids[name] = lid
        nodes.append({
            "id":    lid,
            "label": name,
            "kind":  "link",
            "group": "__rv__",
            "title": f"<b>Link:</b> {_esc(name)}<br/>"
                     f"<i>Associative Hub-to-Hub relation</i>",
        })
    for name in raw_vault_tables.get("sats", []):
        tid = "T_" + _mermaid_safe_id(name)
        sat_ids[name] = tid
        nodes.append({
            "id":    tid,
            "label": name,
            "kind":  "sat",
            "group": "__rv__",
            "title": f"<b>Satellite:</b> {_esc(name)}<br/>"
                     f"<i>Descriptive, time-variant attributes</i>",
        })

    # Hub → Link edges (by name-stem inclusion)
    for lname, lid in link_ids.items():
        stem = lname[4:] if lname.startswith("LNK_") else lname[5:]
        for hname, hid in hub_ids.items():
            hstem = hname[4:]
            if hstem and hstem in stem:
                edges.append({"from": hid, "to": lid,
                              "kind": "hub_link", "label": "",
                              "dashed": True})

    # Hub/Link → Sat edges
    for sname, sid in sat_ids.items():
        core = sname[4:] if sname.startswith("SAT_") else sname
        matched_parent = None
        for hname, hid in hub_ids.items():
            hstem = hname[4:]
            if hstem and core.startswith(hstem):
                matched_parent = hid
                break
        if matched_parent is None:
            for lname, lid in link_ids.items():
                lstem = lname[4:] if lname.startswith("LNK_") else lname[5:]
                if lstem and core.startswith(lstem):
                    matched_parent = lid
                    break
        if matched_parent:
            edges.append({"from": matched_parent, "to": sid,
                          "kind": "hub_sat", "label": "",
                          "dashed": False})

    # Stage → Hub edges from LLM mapping
    added = set()
    if source_to_hub_df is not None and not source_to_hub_df.empty:
        cols = {c.lower(): c for c in source_to_hub_df.columns}
        src_col  = cols.get("source entity") or cols.get("source_entity")
        file_col = cols.get("source file") or cols.get("source_file")
        hub_col  = cols.get("hub name") or cols.get("hub_name")
        bk_col   = cols.get("business key column") or \
                   cols.get("business_key_column")
        conf_col = cols.get("confidence")
        reason_col = cols.get("reason")

        if src_col and hub_col:
            for _, row in source_to_hub_df.iterrows():
                src = str(row[src_col]).strip()
                hub = str(row[hub_col]).strip().upper()
                file_hint = (str(row[file_col]).strip()
                             if file_col else "")
                if not src or hub in ("", "NONE", "NAN"):
                    continue
                if hub not in hub_ids:
                    continue  # hallucinated
                matched = None
                for k, m in entities.items():
                    if m["display_name"] == src:
                        if (not file_hint
                                or file_hint in m["source_file"]
                                or m["source_file"] in file_hint):
                            matched = k
                            break
                if matched is None:
                    for k, m in entities.items():
                        if m["display_name"] == src:
                            matched = k
                            break
                if matched is None:
                    continue
                key = (matched, hub)
                if key in added:
                    continue
                added.add(key)

                bk = str(row[bk_col]).strip() if bk_col else ""
                conf = str(row[conf_col]).strip() if conf_col else ""
                reason = str(row[reason_col]).strip() if reason_col else ""
                label_parts = []
                if bk:
                    label_parts.append(bk)
                edge_label = " / ".join(label_parts)
                tooltip = (
                    f"<b>{_esc(src)}</b> → <b>{_esc(hub)}</b>"
                    + (f"<br/><b>Key:</b> {_esc(bk)}" if bk else "")
                    + (f"<br/><b>Confidence:</b> {_esc(conf)}" if conf else "")
                    + (f"<br/><b>Reason:</b> {_esc(reason)}" if reason else "")
                )
                edges.append({
                    "from":   stage_ids[matched],
                    "to":     hub_ids[hub],
                    "kind":   "stage_hub",
                    "label":  edge_label,
                    "dashed": False,
                    "title":  tooltip,
                })

    # Within-file explicit stage-to-stage links
    file_stage_to_key = {
        (m["source_file"], m["display_name"]): k
        for k, m in entities.items()
    }
    for f in files:
        if f["kind"] != "datastage":
            continue
        for l in f["parsed"].get("links", []):
            fk = file_stage_to_key.get((f["filename"], l["from_stage"]))
            tk = file_stage_to_key.get((f["filename"], l["to_stage"]))
            if fk and tk and fk != tk:
                edges.append({
                    "from":   stage_ids[fk],
                    "to":     stage_ids[tk],
                    "kind":   "stage_stage",
                    "label":  l.get("link_name", "") or "",
                    "dashed": False,
                })

    # Cross-file inferred edges (shared columns)
    for cf in cross_flows:
        if cf.get("kind", "").startswith("cross-file"):
            from_file = cf.get("file", "").split("↔")[0].strip()
            to_file = cf.get("file", "").split("↔")[-1].strip()
            fk = file_stage_to_key.get((from_file, cf["from_stage"]))
            tk = file_stage_to_key.get((to_file, cf["to_stage"]))
            if fk and tk and fk != tk:
                shared = cf.get("shared_columns", [])
                edges.append({
                    "from":   stage_ids[fk],
                    "to":     stage_ids[tk],
                    "kind":   "cross_file",
                    "label":  f"shared: {', '.join(shared[:3])}"
                              + (" …" if len(shared) > 3 else ""),
                    "dashed": True,
                    "title":  "<b>Cross-file inferred flow</b><br/>"
                              "via shared columns: "
                              + _esc(", ".join(shared)),
                })

    return {"nodes": nodes, "edges": edges}


def _compute_lineage_layout(graph: dict) -> dict:
    """
    Compute node positions for a layered LR layout using longest-path
    layering (a.k.a. Sugiyama-style) without external libs.

    Returns the graph with each node augmented with {x, y, layer}.
    Node positions are in an abstract coordinate grid; the SVG renderer
    maps them to pixels.
    """
    nodes = graph.get("nodes", [])
    edges = graph.get("edges", [])
    if not nodes:
        return graph

    node_by_id = {n["id"]: n for n in nodes}

    # Build adjacency
    out_adj = {n["id"]: [] for n in nodes}
    in_adj  = {n["id"]: [] for n in nodes}
    for e in edges:
        if e["from"] in out_adj and e["to"] in in_adj:
            out_adj[e["from"]].append(e["to"])
            in_adj[e["to"]].append(e["from"])

    # Layer assignment: each node's layer = 1 + max(layer of any parent),
    # with roots (no parents) at layer 0. Handle cycles by iterating a
    # bounded number of times.
    layer = {n["id"]: 0 for n in nodes}
    for _ in range(len(nodes) + 2):
        changed = False
        for nid in layer:
            parents = in_adj[nid]
            if parents:
                want = max(layer[p] for p in parents) + 1
                if want > layer[nid]:
                    layer[nid] = want
                    changed = True
        if not changed:
            break

    # Override: force file nodes to layer 0 and Raw Vault nodes to the
    # right-most layers, so the visual reads File → Stage → Hub → Sat.
    max_layer = max(layer.values()) if layer else 0
    for n in nodes:
        if n["kind"] == "file":
            layer[n["id"]] = 0
        elif n["kind"] == "hub":
            layer[n["id"]] = max(layer[n["id"]], max_layer - 1)
        elif n["kind"] == "link":
            layer[n["id"]] = max(layer[n["id"]], max_layer)
        elif n["kind"] == "sat":
            layer[n["id"]] = max(layer[n["id"]], max_layer + 1)

    # Group by layer
    layers = {}
    for nid, lv in layer.items():
        layers.setdefault(lv, []).append(nid)

    # Sort within layer: group stages by file (stable order), RV by kind
    for lv, ids in layers.items():
        def sort_key(nid):
            n = node_by_id[nid]
            k_order = {"file": 0, "stage": 1, "hub": 2, "link": 3,
                       "sat": 4}.get(n["kind"], 9)
            return (k_order, n.get("group", ""), n["label"])
        ids.sort(key=sort_key)

    # Assign y positions within each layer
    for lv in layers:
        ids = layers[lv]
        for i, nid in enumerate(ids):
            node_by_id[nid]["layer"] = lv
            node_by_id[nid]["_y_idx"] = i
            node_by_id[nid]["_layer_size"] = len(ids)

    return {"nodes": list(node_by_id.values()), "edges": edges,
            "n_layers": max(layers) + 1 if layers else 1}


def render_interactive_lineage(graph: dict, height: int = 720):
    """
    Render an interactive lineage graph as inline SVG — no external
    JS libraries, no CDN dependencies. Works in locked-down Snowflake
    network policies where external scripts are blocked.

    Supports:
      - CLICK a node → highlights upstream + downstream lineage;
        everything else dims. Click empty canvas to clear.
      - Hover tooltip showing column/key/confidence details.
      - Scroll to zoom, drag the canvas to pan.
      - Fit-view and Clear-selection buttons.
      - Hierarchical LR layout matching the Mermaid diagram.
    """
    import streamlit.components.v1 as components
    import json as _json

    if not graph or not graph.get("nodes"):
        components.html(
            "<div style='padding:20px;color:#6B6456;font-family:Inter,"
            "sans-serif;'>No lineage graph data available.</div>",
            height=80,
        )
        return

    # Compute layout
    laid_out = _compute_lineage_layout(graph)

    # Style per kind
    STYLE = {
        "file":  {"bg": "#F0EEE6", "stroke": "#6B6456",
                  "sw": 1.5, "w": 160, "h": 36, "rx": 4},
        "stage": {"bg": "#E8F0E8", "stroke": "#3D6B3D",
                  "sw": 1.5, "w": 170, "h": 46, "rx": 6},
        "hub":   {"bg": "#F9E4D4", "stroke": "#C96442",
                  "sw": 2.5, "w": 160, "h": 42, "rx": 4},
        "link":  {"bg": "#E4E0F9", "stroke": "#6B4CC9",
                  "sw": 2,   "w": 150, "h": 42, "rx": 4},
        "sat":   {"bg": "#E4EEF9", "stroke": "#4C80C9",
                  "sw": 1.5, "w": 180, "h": 38, "rx": 18},
    }

    # Map each node to pixel coordinates in an abstract SVG space.
    # We use a fixed layout grid: columns (layers) are LAYER_W apart;
    # within a column rows are LAYER_ROW_H apart, centered vertically.
    LAYER_W     = 260
    LAYER_ROW_H = 82
    TOP_MARGIN  = 60
    LEFT_MARGIN = 50

    # Determine per-layer max row count to compute tall enough canvas
    n_layers = laid_out.get("n_layers", 1)
    max_rows = 1
    for n in laid_out["nodes"]:
        if n.get("_layer_size", 1) > max_rows:
            max_rows = n["_layer_size"]

    svg_w = LEFT_MARGIN * 2 + max(1, n_layers) * LAYER_W
    svg_h = TOP_MARGIN  * 2 + max(1, max_rows)  * LAYER_ROW_H

    # Position nodes
    for n in laid_out["nodes"]:
        s = STYLE.get(n["kind"], STYLE["stage"])
        lv = n.get("layer", 0)
        idx = n.get("_y_idx", 0)
        size = n.get("_layer_size", 1)
        # Center the layer vertically within the canvas
        layer_total_h = size * LAYER_ROW_H
        layer_top = (svg_h - layer_total_h) / 2
        n["x"] = LEFT_MARGIN + lv * LAYER_W + (LAYER_W - s["w"]) / 2
        n["y"] = layer_top + idx * LAYER_ROW_H + (LAYER_ROW_H - s["h"]) / 2
        n["w"] = s["w"]
        n["h"] = s["h"]

    graph_json = _json.dumps({
        "nodes": laid_out["nodes"],
        "edges": laid_out["edges"],
        "style": STYLE,
        "svg_w": svg_w,
        "svg_h": svg_h,
    })

    # Build the HTML. Everything is inline — no CDN.
    # We use \""" multi-line strings carefully: JS is wrapped in a
    # script tag with Python-side value interpolation through one
    # placeholder only.
    html = """
<div id="lineage-root" style="background:#FFFFFF; border:1px solid #E8E6DC;
     border-radius:12px; padding:0; position:relative;
     height:__HEIGHT__px; width:100%;
     font-family: Inter, -apple-system, sans-serif; overflow:hidden;">

  <div id="lineage-toolbar" style="position:absolute; top:10px; left:10px;
       z-index:10; background:rgba(255,255,255,0.92);
       border:1px solid #E8E6DC; border-radius:8px; padding:6px 10px;
       font-size:12px; color:#3D3929; display:flex; gap:8px;
       align-items:center; user-select:none;">
    <span style="font-weight:600;">Interactive Lineage</span>
    <span style="color:#8A8370;">·</span>
    <button id="btn-fit" style="border:1px solid #E8E6DC; background:#F5F4EE;
            color:#3D3929; padding:3px 8px; border-radius:6px;
            cursor:pointer; font-size:12px;">Fit view</button>
    <button id="btn-zoomin" style="border:1px solid #E8E6DC;
            background:#F5F4EE; color:#3D3929; padding:3px 10px;
            border-radius:6px; cursor:pointer; font-size:14px;
            font-weight:600;">+</button>
    <button id="btn-zoomout" style="border:1px solid #E8E6DC;
            background:#F5F4EE; color:#3D3929; padding:3px 10px;
            border-radius:6px; cursor:pointer; font-size:14px;
            font-weight:600;">−</button>
    <button id="btn-clear" style="border:1px solid #E8E6DC;
            background:#F5F4EE; color:#3D3929; padding:3px 8px;
            border-radius:6px; cursor:pointer; font-size:12px;">
            Clear selection</button>
    <span id="zoom-label" style="color:#8A8370; font-size:11px;
          margin-left:4px;">100%</span>
  </div>

  <div id="lineage-legend" style="position:absolute; top:10px; right:10px;
       z-index:10; background:rgba(255,255,255,0.92);
       border:1px solid #E8E6DC; border-radius:8px; padding:8px 10px;
       font-size:11px; color:#3D3929; line-height:1.6; user-select:none;">
    <div><span style="display:inline-block; width:10px; height:10px;
         background:#F0EEE6; border:1.5px solid #6B6456;
         border-radius:2px; margin-right:4px;"></span>File</div>
    <div><span style="display:inline-block; width:10px; height:10px;
         background:#E8F0E8; border:1.5px solid #3D6B3D;
         border-radius:2px; margin-right:4px;"></span>Stage/Entity</div>
    <div><span style="display:inline-block; width:10px; height:10px;
         background:#F9E4D4; border:2px solid #C96442;
         border-radius:2px; margin-right:4px;"></span>Hub</div>
    <div><span style="display:inline-block; width:10px; height:10px;
         background:#E4E0F9; border:2px solid #6B4CC9;
         border-radius:2px; margin-right:4px;"></span>Link</div>
    <div><span style="display:inline-block; width:10px; height:10px;
         background:#E4EEF9; border:1.5px solid #4C80C9;
         border-radius:9px; margin-right:4px;"></span>Satellite</div>
  </div>

  <div id="lineage-panel" style="position:absolute; bottom:10px; left:10px;
       z-index:10; background:rgba(255,255,255,0.96);
       border:1px solid #E8E6DC; border-radius:8px; padding:10px 12px;
       font-size:12px; color:#3D3929; max-width:440px;
       max-height:240px; overflow-y:auto; display:none;
       box-shadow: 0 2px 8px rgba(0,0,0,0.08);">
    <div id="panel-content"></div>
  </div>

  <div id="lineage-tooltip" style="position:absolute; z-index:20;
       background:#FFFFFF; border:1px solid #E8E6DC; border-radius:6px;
       padding:8px 10px; font-size:11px; color:#3D3929;
       box-shadow: 0 2px 8px rgba(0,0,0,0.12); pointer-events:none;
       max-width:340px; line-height:1.5; display:none;"></div>

  <svg id="lineage-svg" width="100%" height="100%"
       style="display:block; cursor:grab;" preserveAspectRatio="xMidYMid meet">
    <defs>
      <marker id="arrow-default" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#8A8370"></path>
      </marker>
      <marker id="arrow-hub" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="8" markerHeight="8" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#C96442"></path>
      </marker>
      <marker id="arrow-crossfile" viewBox="0 0 10 10" refX="9" refY="5"
              markerWidth="7" markerHeight="7" orient="auto-start-reverse">
        <path d="M 0 0 L 10 5 L 0 10 z" fill="#6B4CC9"></path>
      </marker>
    </defs>
    <g id="lineage-viewport"></g>
  </svg>
</div>

<style>
  #lineage-svg .node rect, #lineage-svg .node ellipse {
    transition: opacity 180ms ease;
    cursor: pointer;
  }
  #lineage-svg .node text { pointer-events: none; user-select: none; }
  #lineage-svg .edge path {
    transition: opacity 180ms ease, stroke-width 180ms ease;
    fill: none;
  }
  #lineage-svg .edge.highlight path { stroke-width: 3; }
  #lineage-svg .dim { opacity: 0.15 !important; }
  #lineage-svg .node:hover rect, #lineage-svg .node:hover ellipse {
    filter: brightness(0.97);
  }
</style>

<script>
(function() {
  var GRAPH = __GRAPH__;

  var svg   = document.getElementById('lineage-svg');
  var vp    = document.getElementById('lineage-viewport');
  var tip   = document.getElementById('lineage-tooltip');
  var panel = document.getElementById('lineage-panel');
  var pc    = document.getElementById('panel-content');
  var zl    = document.getElementById('zoom-label');

  // Initial viewBox = whole content
  var W = GRAPH.svg_w, H = GRAPH.svg_h;
  var view = { x: 0, y: 0, w: W, h: H };

  function applyViewBox() {
    svg.setAttribute('viewBox',
      view.x + ' ' + view.y + ' ' + view.w + ' ' + view.h);
    var pct = Math.round((W / view.w) * 100);
    zl.textContent = pct + '%';
  }
  applyViewBox();

  // Build adjacency for upstream/downstream walks
  var outAdj = {}, inAdj = {};
  var edgeById = {};
  var nodeById = {};
  GRAPH.nodes.forEach(function(n) { nodeById[n.id] = n; });
  GRAPH.edges.forEach(function(e, i) {
    e._id = 'e' + i;
    edgeById[e._id] = e;
    (outAdj[e.from] = outAdj[e.from] || []).push(e);
    (inAdj[e.to]    = inAdj[e.to]    || []).push(e);
  });

  // ── Render edges first (behind nodes) ────────────────────────────
  GRAPH.edges.forEach(function(e, i) {
    var a = nodeById[e.from], b = nodeById[e.to];
    if (!a || !b) return;
    var x1 = a.x + a.w, y1 = a.y + a.h / 2;
    var x2 = b.x,        y2 = b.y + b.h / 2;
    var mx = (x1 + x2) / 2;
    var d = 'M ' + x1 + ' ' + y1 +
            ' C ' + mx + ' ' + y1 + ', ' + mx + ' ' + y2 +
            ', ' + x2 + ' ' + y2;

    var stroke = '#8A8370', dash = '', marker = 'arrow-default',
        sw = 1.3;
    if (e.kind === 'stage_hub') {
      stroke = '#C96442'; sw = 2.2; marker = 'arrow-hub';
    } else if (e.kind === 'cross_file') {
      stroke = '#6B4CC9'; dash = '6 4'; marker = 'arrow-crossfile';
    } else if (e.kind === 'hub_link') {
      dash = '3 3';
    } else if (e.dashed) {
      dash = '4 3';
    }

    var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'edge');
    g.setAttribute('data-edge-id', e._id);

    var p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    p.setAttribute('d', d);
    p.setAttribute('stroke', stroke);
    p.setAttribute('stroke-width', sw);
    if (dash) p.setAttribute('stroke-dasharray', dash);
    p.setAttribute('marker-end', 'url(#' + marker + ')');
    g.appendChild(p);

    // Edge label (business key / shared cols)
    if (e.label) {
      var lblX = mx, lblY = (y1 + y2) / 2 - 4;
      var tt = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      tt.setAttribute('x', lblX);
      tt.setAttribute('y', lblY);
      tt.setAttribute('text-anchor', 'middle');
      tt.setAttribute('font-size', '9');
      tt.setAttribute('font-family', 'Inter, sans-serif');
      tt.setAttribute('fill', '#6B6456');
      tt.setAttribute('paint-order', 'stroke');
      tt.setAttribute('stroke', '#FFFFFF');
      tt.setAttribute('stroke-width', '3');
      tt.setAttribute('stroke-linejoin', 'round');
      tt.textContent = e.label;
      g.appendChild(tt);
    }

    // Hover tooltip on edge
    if (e.title) {
      p.style.cursor = 'help';
      p.addEventListener('mouseenter', function(ev) { showTip(e.title, ev); });
      p.addEventListener('mousemove',  function(ev) { moveTip(ev); });
      p.addEventListener('mouseleave', function() { hideTip(); });
    }

    vp.appendChild(g);
  });

  // ── Render nodes ─────────────────────────────────────────────────
  GRAPH.nodes.forEach(function(n) {
    var s = GRAPH.style[n.kind] || GRAPH.style.stage;
    var g = document.createElementNS('http://www.w3.org/2000/svg', 'g');
    g.setAttribute('class', 'node');
    g.setAttribute('data-node-id', n.id);
    g.setAttribute('data-kind', n.kind);

    var shape;
    if (n.kind === 'sat') {
      shape = document.createElementNS('http://www.w3.org/2000/svg',
                                       'ellipse');
      shape.setAttribute('cx', n.x + n.w / 2);
      shape.setAttribute('cy', n.y + n.h / 2);
      shape.setAttribute('rx', n.w / 2);
      shape.setAttribute('ry', n.h / 2);
    } else {
      shape = document.createElementNS('http://www.w3.org/2000/svg',
                                       'rect');
      shape.setAttribute('x', n.x);
      shape.setAttribute('y', n.y);
      shape.setAttribute('width', n.w);
      shape.setAttribute('height', n.h);
      shape.setAttribute('rx', s.rx);
    }
    shape.setAttribute('fill', s.bg);
    shape.setAttribute('stroke', s.stroke);
    shape.setAttribute('stroke-width', s.sw);
    g.appendChild(shape);

    // Label — split on \\n, center vertically
    var lines = String(n.label || '').split('\\n');
    var fs = n.kind === 'hub' ? 13 : (n.kind === 'sat' ? 11 : 12);
    var startY = n.y + n.h / 2 - ((lines.length - 1) * fs * 0.6);
    lines.forEach(function(ln, i) {
      var t = document.createElementNS('http://www.w3.org/2000/svg', 'text');
      t.setAttribute('x', n.x + n.w / 2);
      t.setAttribute('y', startY + i * fs * 1.2 + fs * 0.35);
      t.setAttribute('text-anchor', 'middle');
      t.setAttribute('dominant-baseline', 'middle');
      t.setAttribute('font-size', fs);
      t.setAttribute('font-family', 'Inter, -apple-system, sans-serif');
      t.setAttribute('font-weight', n.kind === 'hub' ? '600' : '500');
      t.setAttribute('fill', '#141413');
      t.textContent = ln;
      g.appendChild(t);
    });

    // Events
    if (n.title) {
      shape.addEventListener('mouseenter', function(ev) {
        showTip(n.title, ev);
      });
      shape.addEventListener('mousemove', function(ev) { moveTip(ev); });
      shape.addEventListener('mouseleave', function() { hideTip(); });
    }
    shape.addEventListener('click', function(ev) {
      ev.stopPropagation();
      highlightLineage(n.id);
    });

    vp.appendChild(g);
  });

  // ── Upstream / downstream walk ───────────────────────────────────
  function walkUp(nid, seen) {
    (inAdj[nid] || []).forEach(function(e) {
      if (seen.nodes.has(e.from)) return;
      seen.nodes.add(e.from);
      seen.edges.add(e._id);
      walkUp(e.from, seen);
    });
  }
  function walkDown(nid, seen) {
    (outAdj[nid] || []).forEach(function(e) {
      if (seen.nodes.has(e.to)) return;
      seen.nodes.add(e.to);
      seen.edges.add(e._id);
      walkDown(e.to, seen);
    });
  }

  function highlightLineage(nid) {
    var seen = {nodes: new Set([nid]), edges: new Set()};
    walkUp(nid, seen);
    walkDown(nid, seen);

    Array.prototype.forEach.call(
      vp.querySelectorAll('.node'), function(g) {
        if (seen.nodes.has(g.getAttribute('data-node-id'))) {
          g.classList.remove('dim');
        } else {
          g.classList.add('dim');
        }
      });
    Array.prototype.forEach.call(
      vp.querySelectorAll('.edge'), function(g) {
        var kept = seen.edges.has(g.getAttribute('data-edge-id'));
        if (kept) { g.classList.remove('dim'); g.classList.add('highlight'); }
        else      { g.classList.add('dim');    g.classList.remove('highlight'); }
      });

    showPanel(nid, seen);
  }

  function clearHighlight() {
    Array.prototype.forEach.call(
      vp.querySelectorAll('.dim'),       function(e) {
        e.classList.remove('dim');
      });
    Array.prototype.forEach.call(
      vp.querySelectorAll('.highlight'), function(e) {
        e.classList.remove('highlight');
      });
    panel.style.display = 'none';
  }

  function showPanel(nid, seen) {
    var n = nodeById[nid];
    if (!n) { panel.style.display = 'none'; return; }
    var html = '<div style="margin-bottom:6px;"><b>' +
      String(n.label).replace(/\\n/g, ' ') + '</b>' +
      ' <span style="color:#8A8370; font-size:11px;">[' +
      n.kind + ']</span></div>';
    if (n.title) {
      html += '<div style="color:#3D3929; font-size:11px; ' +
              'line-height:1.5;">' + n.title + '</div>';
    }
    var upCount = seen.nodes.size - 1;
    html += '<div style="margin-top:8px; padding-top:8px; ' +
            'border-top:1px solid #E8E6DC; color:#6B6456; font-size:11px;">' +
            'Connected lineage: <b>' + upCount + '</b> node' +
            (upCount === 1 ? '' : 's') + ', <b>' +
            seen.edges.size + '</b> edge' +
            (seen.edges.size === 1 ? '' : 's') + '</div>';
    pc.innerHTML = html;
    panel.style.display = 'block';
  }

  // Blank-canvas click clears
  svg.addEventListener('click', function(ev) {
    if (ev.target === svg || ev.target === vp ||
        ev.target.tagName === 'path') { /* edges don't clear */ }
    if (ev.target === svg || ev.target.tagName === 'svg') {
      clearHighlight();
    }
  });

  // ── Tooltip ──────────────────────────────────────────────────────
  function showTip(html, ev) {
    tip.innerHTML = html;
    tip.style.display = 'block';
    moveTip(ev);
  }
  function moveTip(ev) {
    var root = document.getElementById('lineage-root');
    var rect = root.getBoundingClientRect();
    var x = ev.clientX - rect.left + 12;
    var y = ev.clientY - rect.top  + 12;
    // Clamp inside panel
    var tw = tip.offsetWidth, th = tip.offsetHeight;
    if (x + tw > rect.width)  x = ev.clientX - rect.left - tw - 12;
    if (y + th > rect.height) y = ev.clientY - rect.top  - th - 12;
    tip.style.left = x + 'px';
    tip.style.top  = y + 'px';
  }
  function hideTip() { tip.style.display = 'none'; }

  // ── Pan (drag) ───────────────────────────────────────────────────
  var dragging = false, lastX = 0, lastY = 0;
  svg.addEventListener('mousedown', function(ev) {
    if (ev.target !== svg && ev.target !== vp) return;
    dragging = true; lastX = ev.clientX; lastY = ev.clientY;
    svg.style.cursor = 'grabbing';
  });
  window.addEventListener('mousemove', function(ev) {
    if (!dragging) return;
    var dx = ev.clientX - lastX, dy = ev.clientY - lastY;
    lastX = ev.clientX; lastY = ev.clientY;
    var rect = svg.getBoundingClientRect();
    var scaleX = view.w / rect.width;
    var scaleY = view.h / rect.height;
    view.x -= dx * scaleX;
    view.y -= dy * scaleY;
    applyViewBox();
  });
  window.addEventListener('mouseup', function() {
    dragging = false; svg.style.cursor = 'grab';
  });

  // ── Zoom (wheel) ─────────────────────────────────────────────────
  svg.addEventListener('wheel', function(ev) {
    ev.preventDefault();
    var rect = svg.getBoundingClientRect();
    // Cursor position in content coords
    var cx = view.x + (ev.clientX - rect.left) / rect.width  * view.w;
    var cy = view.y + (ev.clientY - rect.top)  / rect.height * view.h;
    var factor = ev.deltaY < 0 ? 0.85 : 1.18;
    var newW = Math.max(W * 0.15, Math.min(W * 5, view.w * factor));
    var newH = newW * (view.h / view.w);
    view.x = cx - (cx - view.x) * (newW / view.w);
    view.y = cy - (cy - view.y) * (newH / view.h);
    view.w = newW; view.h = newH;
    applyViewBox();
  }, { passive: false });

  // ── Buttons ──────────────────────────────────────────────────────
  function fit() {
    view = { x: 0, y: 0, w: W, h: H };
    applyViewBox();
  }
  function zoomBy(factor) {
    var cx = view.x + view.w / 2;
    var cy = view.y + view.h / 2;
    var newW = Math.max(W * 0.15, Math.min(W * 5, view.w * factor));
    var newH = newW * (view.h / view.w);
    view.x = cx - newW / 2; view.y = cy - newH / 2;
    view.w = newW; view.h = newH;
    applyViewBox();
  }
  document.getElementById('btn-fit').addEventListener('click', fit);
  document.getElementById('btn-zoomin').addEventListener('click',
    function() { zoomBy(0.8); });
  document.getElementById('btn-zoomout').addEventListener('click',
    function() { zoomBy(1.25); });
  document.getElementById('btn-clear').addEventListener('click',
    clearHighlight);
})();
</script>
"""
    html = (html.replace("__HEIGHT__", str(height))
                .replace("__GRAPH__", graph_json))
    components.html(html, height=height + 20, scrolling=False)



def init_state():
    defaults = {
        "selected_model":   "Claude Opus 4.7",
        "chat_history":     [],
        "parsed_metadata":  None,
        "metadata_summary": None,
        "source_filename":  None,
        "artifacts":        {},  # {artifact_name: markdown_content}
        # Forward Engineering state
        "fwd_source_artifacts": None,  # loaded reverse artifacts
        "fwd_source_version":   None,  # which version was loaded
        "fwd_dashboard_text":   None,  # extracted dashboard text
        "fwd_dashboard_type":   None,
        "fwd_rules_text":       None,  # extracted rules text
        "fwd_artifacts":        {},    # generated forward artifacts
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


init_state()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — Claude Code style
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ✴ &nbsp; Co-Pilot")
    st.markdown("---")

    if st.button("＋ &nbsp; New chat", use_container_width=True):
        st.session_state.chat_history = []
        st.session_state.artifacts = {}
        st.rerun()

    st.markdown("&nbsp;&nbsp;📁 &nbsp; Projects")
    st.markdown("&nbsp;&nbsp;⚙️ &nbsp; Customize")
    st.markdown("&nbsp;&nbsp;✦ &nbsp; Artifacts")

    st.markdown("---")

    # Show active session info
    try:
        ctx = session.sql(
            "SELECT CURRENT_USER() U, CURRENT_ROLE() R, CURRENT_WAREHOUSE() W"
        ).collect()[0]
        st.markdown("### Session")
        st.caption(f"**User:** {ctx['U']}")
        st.caption(f"**Role:** {ctx['R']}")
        st.caption(f"**Warehouse:** {ctx['W']}")
    except Exception as e:
        st.caption(f"Session info unavailable: {e}")

    if st.session_state.artifacts:
        st.markdown("---")
        st.markdown("### Generated Artifacts")
        for name in st.session_state.artifacts:
            st.caption(f"• {name}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN HEADER
# ─────────────────────────────────────────────────────────────────────────────
hour = datetime.now().hour
greeting = ("Morning" if hour < 12 else "Afternoon" if hour < 18 else "Evening")

try:
    user_name = session.sql("SELECT first_name FROM SNOWFLAKE.ACCOUNT_USAGE.USERS WHERE name = CURRENT_USER();").collect()[0][0]
except Exception:
    user_name = "there"

st.markdown(
    f'<div class="plan-badge"><span>Snowflake Cortex · '
    f'{st.session_state.selected_model}</span></div>',
    unsafe_allow_html=True,
)
st.markdown(
    f'<div class="claude-greeting">'
    f'<span class="claude-star">✴</span>{greeting}, {user_name}'
    f'</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="claude-subtitle">Data Engineering Co-Pilot — '
    'powered by Snowflake Cortex</div>',
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# DBT DEPLOY SECTION HELPER — shared between Forward Engineering and
# Quick GO tabs. See render_dbt_deploy_section() docstring.
# ─────────────────────────────────────────────────────────────────────────────
def render_dbt_deploy_section(arts: dict, key_prefix: str) -> None:
    """Render the Build & Deploy DBT Project section.

    Extracted so both the Forward Engineering tab and the Quick GO
    tab can render the same flow without duplicating ~460 lines of
    Snowflake-native dbt deploy / execute logic.

    Args:
      arts:        the artifacts dict from session state (e.g.
                   st.session_state['fwd_artifacts'] for Forward
                   tab, or st.session_state['qg_artifacts'] for
                   Quick GO).
      key_prefix:  prefix for all Streamlit widget keys so the
                   same section can render in two tabs without
                   duplicate-key collisions. Pass 'fwd' for the
                   Forward tab and 'qg' for Quick GO.
    """
    # ── STEP 12: Run in dbt workspace ─────────────────────────
    st.markdown("---")
    st.markdown("##### Run dbt code (dbt Projects on Snowflake)")
    st.caption(
        f"Deploys the chosen dbt artifact to "
        f"`@{VECTOR_DB}.{VECTOR_SCHEMA}.{VECTOR_STAGE}/"
        f"dbt_runs/<project>/` and runs it with "
        f"`EXECUTE DBT PROJECT`. Snowflake's native dbt runtime is "
        f"used — no external dbt install needed. Requires your "
        f"role to have CREATE DBT PROJECT on the target schema. "
        f"Creates/executions pin `DBT_VERSION = {SNOWFLAKE_NATIVE_DBT_VERSION}`. "
        f"Deploy prep strips `query_tag` (dbt-snowflake otherwise runs "
        f"`SHOW PARAMETERS`, which native dbt rejects as `SHOW PARAMETER`)."
    )

    # Which dbt projects exist among arts?
    dbt_options = [
        k for k, v in arts.items()
        if v.get("kind") == "dbt_project"
           and (v.get("content", {}) or {}).get("files")
    ]
    if not dbt_options:
        st.info(
            "Generate at least one dbt project (⑥, ⑦, or ⑧) to "
            "enable dbt execution."
        )
    else:
        # Quick create options for the two canonical projects.
        raw_art = arts.get("Raw Vault dbt", {}) or {}
        raw_proj_files = (
            (raw_art.get("content", {}) or {}).get("files", {}) or {}
        )
        bv_art = arts.get("Business Vault dbt", {}) or {}
        bv_proj_files = (
            (bv_art.get("content", {}) or {}).get("files", {}) or {}
        )
        q1, q2 = st.columns(2)
        create_raw_proj = q1.button(
            "🧱 Create `raw_vault_proj`",
            use_container_width=True,
            key=f"{key_prefix}_create_raw_proj",
            disabled=not raw_proj_files,
        )
        create_bv_proj = q2.button(
            "🧱 Create `business_vault_proj`",
            use_container_width=True,
            key=f"{key_prefix}_create_bv_proj",
            disabled=not bv_proj_files,
        )
        if create_raw_proj or create_bv_proj:
            quick_name = (
                "raw_vault_proj" if create_raw_proj
                else "business_vault_proj"
            )
            quick_files = raw_proj_files if create_raw_proj else bv_proj_files
            quick_slug = (
                "raw_vault_dbt" if create_raw_proj else "business_vault_dbt"
            )
            ver_tag = re.sub(
                r"[^A-Za-z0-9._-]+", "_",
                (st.session_state.get("fwd_persist_version", "")
                 or "latest").strip()
            ) or "latest"
            try:
                quick_files = _prepare_files_for_snowflake_native_dbt(
                    quick_files
                )
                # Surface strip diagnostic so we can see the strip ran.
                _strip_n = _NEXUS_STRIP_COUNTERS.pop(id(quick_files), 0)
                if _strip_n:
                    st.info(
                        f"✓ Stripped {_strip_n} conflicting "
                        f"`generate_schema_name` macro definition(s) "
                        f"from generated project before deploy."
                    )
                quick_files = _bind_profiles_to_current_session(
                    session, quick_files
                )
                with st.spinner(
                    f"Deploying + creating DBT PROJECT `{quick_name}`…"
                ):
                    ensure_vector_infrastructure(session)
                    stage_dir = deploy_dbt_project_to_stage(
                        session, quick_files, quick_slug, ver_tag
                    )
                    # Create RAW landing tables BEFORE the DBT PROJECT
                    # is built. Bronze views read from source('raw', ...)
                    # via models/bronze/_sources.yml; without these
                    # tables existing in Snowflake, the bronze layer
                    # fails to compile and `dbt build` errors.
                    ddl_summary = _execute_raw_landing_ddl(
                        session, quick_files
                    )
                    if ddl_summary["executed"]:
                        schemas_list = ", ".join(
                            f"`{s}`" for s in ddl_summary["schemas"]
                        )
                        st.info(
                            f"✓ Ensured per-layer schemas ({schemas_list}) "
                            f"and created/replaced "
                            f"{ddl_summary['executed']} object(s). "
                            "RAW tables are empty stubs — load real data "
                            "before running `dbt build`."
                        )
                    if ddl_summary["errors"]:
                        err_lines = "\n".join(
                            f"  • {p}: {e}"
                            for p, e in ddl_summary["errors"][:5]
                        )
                        st.warning(
                            f"⚠ {len(ddl_summary['errors'])} DDL "
                            f"statement(s) failed (others succeeded "
                            f"and dbt may still build):\n{err_lines}"
                        )
                    ver_sql = _sql_dbt_version_suffix(None)
                    _fq = f"{VECTOR_DB}.{VECTOR_SCHEMA}.{quick_name}"
                    # Force-drop any previous project object so an old
                    # DBT_VERSION pin (e.g. 1.10.15 from a previous app
                    # version) doesn't survive CREATE OR REPLACE.
                    try:
                        session.sql(
                            f"DROP DBT PROJECT IF EXISTS {_fq}"
                        ).collect()
                    except Exception as _de:
                        st.caption(
                            f"(pre-drop note: {str(_de)[:120]})"
                        )
                    session.sql(
                        f"CREATE OR REPLACE DBT PROJECT {_fq} "
                        f"FROM '{stage_dir}'{ver_sql}"
                    ).collect()
                    _pin_dbt_project_object_version(
                        session, _fq, None,
                    )
                    _actual_ver = _verify_dbt_project_version(
                        session, _fq,
                    )
                    if _actual_ver:
                        _expected = _validated_native_dbt_version(None)
                        if _actual_ver.strip() != _expected.strip():
                            st.error(
                                f"⚠ DBT_VERSION mismatch on "
                                f"`{_fq}`: expected `{_expected}`, "
                                f"got `{_actual_ver}`. The "
                                f"`SHOW PARAMETER` runtime bug in "
                                f"1.10.15 will fire. Drop the project "
                                f"manually and redeploy: "
                                f"`DROP DBT PROJECT {_fq};`"
                            )
                        else:
                            st.caption(
                                f"✓ DBT_VERSION pinned to "
                                f"`{_actual_ver}` on `{_fq}`"
                            )
                    # Install packages only when packages.yml lists remotes;
                    # empty packages avoids an extra native job (and adapter churn).
                    if _packages_yml_requires_dbt_deps(
                        quick_files.get("packages.yml", "")
                    ):
                        dq = _sql_single_quoted_literal("deps")
                        session.sql(
                            f"EXECUTE DBT PROJECT "
                            f"{VECTOR_DB}.{VECTOR_SCHEMA}.{quick_name} "
                            f"ARGS='{dq}'"
                        ).collect()
                ran_deps = _packages_yml_requires_dbt_deps(
                    quick_files.get("packages.yml", "")
                )
                st.success(
                    f"✓ Created `{VECTOR_DB}.{VECTOR_SCHEMA}.{quick_name}` "
                    f"(dbt {SNOWFLAKE_NATIVE_DBT_VERSION}) from `{stage_dir}`"
                    + (
                        " after `dbt deps`."
                        if ran_deps
                        else " — skipped `deps` (no remote packages)."
                    )
                )
            except Exception as e:
                st.error(f"Create DBT PROJECT failed: {e}")

        rc1, rc2 = st.columns([2, 3])
        chosen_dbt = rc1.selectbox(
            "dbt project to run",
            dbt_options,
            key=f"{key_prefix}_dbt_run_project",
        )
        if not chosen_dbt and dbt_options:
            chosen_dbt = dbt_options[0]
        dbt_args = rc2.text_input(
            "dbt command & args",
            value="build --target dev",
            help=(
                "Examples: `deps` (install packages), "
                "`compile`, `run --target dev`, "
                "`test`, `build --select tag:silver`"
                ". The app auto-runs `deps` before build/run."
            ),
            key=f"{key_prefix}_dbt_run_args",
        )

        default_proj_name = (
            "raw_vault_proj"
            if "raw vault" in chosen_dbt.lower()
            else ("business_vault_proj"
                  if "business vault" in chosen_dbt.lower()
                  else re.sub(r"\W+", "_", chosen_dbt.lower()))
        )
        project_obj_name = st.text_input(
            "Snowflake DBT PROJECT object name",
            value=default_proj_name,
            key=f"{key_prefix}_dbt_project_name",
            help=(
                "Use `raw_vault_proj` for Raw Vault dbt and "
                "`business_vault_proj` for Business Vault dbt."
            ),
        )

        run_col1, run_col2, run_col3 = st.columns(3)
        deploy_only = run_col1.button(
            "☁ Deploy to stage only",
            use_container_width=True,
            key=f"{key_prefix}_dbt_deploy",
            help=(
                "Uploads individual files to stage so you can "
                "CREATE DBT PROJECT or open in a Snowflake "
                "workspace manually."
            ),
        )
        create_proj = run_col2.button(
            "🧱 Create DBT PROJECT",
            use_container_width=True,
            key=f"{key_prefix}_dbt_create_project",
        )
        run_now = run_col3.button(
            "▶ Build + Deploy DBT PROJECT",
            use_container_width=True,
            type="primary",
            key=f"{key_prefix}_dbt_execute",
            disabled=not dbt_args.strip(),
        )

        if deploy_only or create_proj or run_now:
            art     = arts[chosen_dbt]
            files   = (art.get("content", {}) or {}).get("files") or {}
            files = _prepare_files_for_snowflake_native_dbt(files)
            # Surface strip diagnostic so we can see the strip ran.
            _strip_n = _NEXUS_STRIP_COUNTERS.pop(id(files), 0)
            if _strip_n:
                st.info(
                    f"✓ Stripped {_strip_n} conflicting "
                    f"`generate_schema_name` macro definition(s) "
                    f"from generated project before deploy."
                )
            files = _bind_profiles_to_current_session(session, files)
            slug    = re.sub(r"\W+", "_", chosen_dbt.lower())
            ver_tag = re.sub(
                r"[^A-Za-z0-9._-]+", "_",
                        (st.session_state.get("fwd_persist_version", "")
                 or "latest").strip()
            ) or "latest"

            try:
                with st.spinner(
                    f"Ensuring stage + uploading {len(files)} "
                    f"files to @…/dbt_runs/{slug}/{ver_tag}/…"
                ):
                    ensure_vector_infrastructure(session)
                    stage_dir = deploy_dbt_project_to_stage(
                        session, files, slug, ver_tag
                    )
                st.success(f"✓ Deployed to `{stage_dir}`")
                st.caption(
                    f"To use in a Snowflake workspace, reference: "
                    f"`{stage_dir}/dbt_project.yml`"
                )

                # Create RAW landing tables AND per-layer dbt schemas
                # (RAW_VAULT, BUSINESS_VAULT) so bronze source('raw',...)
                # calls resolve and silver/gold models have target
                # schemas to materialize into. These are needed whenever
                # dbt build runs — whether triggered now via create_proj
                # or later from a workspace.
                ddl_summary = _execute_raw_landing_ddl(session, files)
                if ddl_summary["executed"]:
                    schemas_list = ", ".join(
                        f"`{s}`" for s in ddl_summary["schemas"]
                    )
                    st.info(
                        f"✓ Ensured per-layer schemas ({schemas_list}) "
                        f"and created/replaced "
                        f"{ddl_summary['executed']} object(s). "
                        "RAW tables are empty stubs — load real data "
                        "before running `dbt build`."
                    )
                if ddl_summary["errors"]:
                    err_lines = "\n".join(
                        f"  • {p}: {e}"
                        for p, e in ddl_summary["errors"][:5]
                    )
                    st.warning(
                        f"⚠ {len(ddl_summary['errors'])} DDL "
                        f"statement(s) failed (others succeeded "
                        f"and dbt may still build):\n{err_lines}"
                    )

                if create_proj:
                    proj_clean = re.sub(
                        r"\W+", "_", project_obj_name.strip()
                    ) or f"{slug}_{ver_tag}"
                    ver_sql = _sql_dbt_version_suffix(None)
                    pkg_needs_deps = _packages_yml_requires_dbt_deps(
                        files.get("packages.yml", "")
                    )
                    _fq2 = f"{VECTOR_DB}.{VECTOR_SCHEMA}.{proj_clean}"
                    with st.spinner(
                        f"Creating DBT PROJECT {proj_clean}…"
                    ):
                        # Force-drop any previous version so an old
                        # DBT_VERSION pin (e.g. 1.10.15) doesn't
                        # survive CREATE OR REPLACE.
                        try:
                            session.sql(
                                f"DROP DBT PROJECT IF EXISTS {_fq2}"
                            ).collect()
                        except Exception as _de2:
                            st.caption(
                                f"(pre-drop note: {str(_de2)[:120]})"
                            )
                        create_sql = (
                            f"CREATE OR REPLACE DBT PROJECT {_fq2} "
                            f"FROM '{stage_dir}'{ver_sql}"
                        )
                        session.sql(create_sql).collect()
                        _pin_dbt_project_object_version(
                            session, _fq2, None,
                        )
                        _actual_ver2 = _verify_dbt_project_version(
                            session, _fq2,
                        )
                        if _actual_ver2:
                            _expected2 = _validated_native_dbt_version(None)
                            if _actual_ver2.strip() != _expected2.strip():
                                st.error(
                                    f"⚠ DBT_VERSION mismatch on "
                                    f"`{_fq2}`: expected "
                                    f"`{_expected2}`, got "
                                    f"`{_actual_ver2}`. Drop manually "
                                    f"and redeploy: "
                                    f"`DROP DBT PROJECT {_fq2};`"
                                )
                            else:
                                st.caption(
                                    f"✓ DBT_VERSION pinned to "
                                    f"`{_actual_ver2}`"
                                )
                        if pkg_needs_deps:
                            deps_lit = _sql_single_quoted_literal("deps")
                            session.sql(
                                f"EXECUTE DBT PROJECT {_fq2} "
                                f"ARGS='{deps_lit}'"
                            ).collect()
                    st.success(
                        f"✓ Created DBT PROJECT "
                        f"`{VECTOR_DB}.{VECTOR_SCHEMA}.{proj_clean}`"
                        + (
                            " and ran `dbt deps` (remote packages declared)."
                            if pkg_needs_deps
                            else f" (dbt {SNOWFLAKE_NATIVE_DBT_VERSION}; "
                            f"no remote packages — skipped `deps`)."
                        )
                    )

                if run_now:
                    proj_clean = re.sub(
                        r"\W+", "_", project_obj_name.strip()
                    ) or f"{slug}_{ver_tag}"
                    build_args = dbt_args.strip() or "build --target dev"
                    with st.spinner(
                        f"CREATE DBT PROJECT + EXECUTE DBT "
                        f"PROJECT {proj_clean} ARGS='"
                        f"{build_args}'…"
                    ):
                        result = create_and_execute_dbt_project(
                            session,
                            stage_path=stage_dir,
                            project_name=proj_clean,
                            args=build_args,
                            packages_yml_body=files.get("packages.yml"),
                        )
                    st.success(
                        f"✓ Executed `{result['project_name']}`"
                    )

                    with st.expander("dbt execution details",
                                     expanded=True):
                        st.markdown(
                            f"**Query ID:** "
                            f"`{result['query_id'] or '—'}`  \n"
                            f"**Stage path:** "
                            f"`{result['stage_path']}`"
                        )
                        if result.get("deps_sql"):
                            st.markdown("**Dependency install step:**")
                            st.code(result["deps_sql"],
                                    language="sql")
                        st.code(result["execute_sql"],
                                language="sql")
                        if result.get("exec_rows"):
                            st.markdown("**EXECUTE output:**")
                            st.dataframe(
                                pd.DataFrame(result["exec_rows"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                        if result.get("history"):
                            st.markdown(
                                "**DBT_PROJECT_EXECUTION_HISTORY "
                                "(latest runs):**"
                            )
                            st.dataframe(
                                pd.DataFrame(result["history"]),
                                use_container_width=True,
                                hide_index=True,
                            )
                        st.caption(
                            "To download dbt artifacts (manifest, "
                            "compiled SQL, logs) from this run, "
                            "use: `SELECT SYSTEM$LOCATE_DBT_"
                            f"ARTIFACTS('{result['query_id']}');`"
                        )
            except Exception as e:
                st.error(f"dbt deploy/execute failed: {e}")
                try:
                    emsg = str(e)
                    m_qid = re.search(
                        r"\b([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b",
                        emsg,
                        flags=re.I,
                    )
                    if m_qid:
                        qid = m_qid.group(1)
                        log_rows = session.sql(
                            "SELECT SYSTEM$GET_DBT_LOG("
                            f"'{_sql_single_quoted_literal(qid)}'"
                            ") AS DBT_LOG"
                        ).collect()
                        if log_rows:
                            raw_log = str(log_rows[0]["DBT_LOG"] or "")
                            with st.expander(
                                "Native dbt error log excerpt",
                                expanded=True,
                            ):
                                st.code(raw_log[:12000], language="text")
                except Exception:
                    pass
                st.caption(
                    "Common causes: (1) role missing CREATE DBT "
                    "PROJECT privilege, (2) dbt Projects on "
                    "Snowflake not enabled for the account, "
                    "(3) the fixed `profiles.yml` role/warehouse/database/"
                    "schema cannot be assumed by the native runtime "
                    f"(see NATIVE_DBT_PROFILE_* and VECTOR_* in app code), "
                    "(4) network/package dependency restrictions in "
                    "Snowflake runtime, (5) residual `query_tag` in models "
                    "(prep strips it — redeploy after upgrading the app)."
                )





# ─────────────────────────────────────────────────────────────────────────────
# TABS — ``_render_tab_*`` bodies defined here; ``st.tabs`` is invoked once
# near the bottom (before model picker) so chrome stays outside tabs.
# Visibility: ``nexus/app_tabs_config.py`` or env ``NEXUS_APP_TAB_VISIBILITY``.
# ─────────────────────────────────────────────────────────────────────────────


def _render_tab_quickgo():
    # ═════════════════════════════════════════════════════════════════════════════
    # QUICK GO TAB — orchestrate full reverse + forward pipeline in one place
    # ═════════════════════════════════════════════════════════════════════════════
    # Self-contained: writes to qg_* session-state slots, never touches the
    # state used by the Reverse / Forward tabs (per user spec: "Always
    # regenerate fresh in Quick GO — independent state").
    #
    # Phases:
    #   Phase 1 (START button):    Lineage → STTM → Catalog → Transformation
    #                              Rules → Raw Vault Model → Raw Vault
    #                              Validation
    #   Pause for review.
    #   Phase 2 (Continue button): Business Vault → Forward STTM → Forward
    #                              Catalog → Semantic Data Model → Raw
    #                              Vault dbt → Business Vault dbt
    #   Phase 3 (Next button):     hands off to Build/Deploy DBT Project
    #
    # Domino's-style horizontal tracker on top, expandable result panels below.

    st.markdown("#### ⚡ Quick GO — end-to-end pipeline in one place")
    st.caption(
        "Upload reverse-engineering inputs and forward-engineering "
        "inputs once. **START** runs Lineage → STTM → Catalog → "
        "Transformation Rules → Raw Vault Model → Raw Vault Validation. "
        "**Continue** runs Business Vault → STTM → Catalog → Semantic "
        "Data Model → Raw Vault dbt → Business Vault dbt. **Next** "
        "hands off to Build/Deploy DBT Project."
    )

    # ─────────────────────────────────────────────────────────────────────
    # Independent session-state init for Quick GO. Prefix `qg_` ensures no
    # collision with the existing tabs' state.
    # ─────────────────────────────────────────────────────────────────────
    if "qg_phase" not in st.session_state:
        st.session_state["qg_phase"] = 0   # 0=inputs, 1=phase1 done, 2=phase2 done
    if "qg_artifacts" not in st.session_state:
        st.session_state["qg_artifacts"] = {}  # {step_name: {"content": ...}}
    if "qg_step_status" not in st.session_state:
        # status is one of: "pending", "running", "done", "error"
        st.session_state["qg_step_status"] = {}
    if "qg_metadata_summary" not in st.session_state:
        st.session_state["qg_metadata_summary"] = ""

    # ─────────────────────────────────────────────────────────────────────
    # 9-step tracker definition. Each step has a label, an icon, and a
    # phase index (1 = reverse, 2 = forward).
    # ─────────────────────────────────────────────────────────────────────
    QG_STEPS = [
        {"key": "Data Lineage",       "label": "Lineage",       "icon": "🔀", "phase": 1},
        {"key": "STTM",               "label": "STTM",          "icon": "🗺",  "phase": 1},
        {"key": "Data Catalog",       "label": "Catalog",       "icon": "📚", "phase": 1},
        {"key": "Transformation Rules", "label": "Rules",       "icon": "📜", "phase": 1},
        {"key": "Raw Vault Model",    "label": "Raw Vault",     "icon": "🏛", "phase": 1},
        {"key": "Raw Vault Validation", "label": "RV Validate", "icon": "🛡", "phase": 1},
        {"key": "Business Vault",     "label": "Biz Vault",     "icon": "🏢", "phase": 2},
        {"key": "Forward STTM",       "label": "Fwd STTM",      "icon": "🗺",  "phase": 2},
        {"key": "Forward Catalog",    "label": "Fwd Catalog",   "icon": "📚", "phase": 2},
        {"key": "Semantic Data Model", "label": "Semantic",     "icon": "✨", "phase": 2},
        {"key": "Raw Vault dbt",      "label": "RV dbt",        "icon": "⚙",  "phase": 2},
        {"key": "Business Vault dbt", "label": "BV dbt",        "icon": "⚙",  "phase": 2},
    ]

    def _qg_render_tracker():
        """Render the Domino's-style horizontal pizza tracker — 9 stages
        with status icons (pending ○, running animated spinner, done ✓,
        error ✗) plus arrow separators between consecutive tiles."""
        status_map = st.session_state["qg_step_status"]

        # 17 columns: 9 tiles interleaved with 8 narrow arrow columns.
        # Tile width 4, arrow width 1 → arrows visually small but clear.
        col_widths = []
        for i in range(len(QG_STEPS)):
            col_widths.append(4)            # tile
            if i < len(QG_STEPS) - 1:
                col_widths.append(1)        # arrow
        cols = st.columns(col_widths)

        # Inject a one-shot CSS keyframe for the animated spinner glyph.
        # `nx-spin` rotates a unicode partial-circle 360° on a 1s loop,
        # so the running step has *real* motion (not just a static ⏳).
        # Streamlit dedupes identical markdown so this is safe to render
        # on every rerun.
        st.markdown("""
<style>
@keyframes nx-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.nx-spinner {
  display: inline-block;
  animation: nx-spin 1s linear infinite;
  font-size: 18px;
}
</style>""", unsafe_allow_html=True)

        for i, step in enumerate(QG_STEPS):
            status = status_map.get(step["key"], "pending")
            if status == "done":
                badge_html = "<div style='font-size:18px;'>✅</div>"
                bg = "rgba(34, 197, 94, 0.15)"
                border = "#22c55e"
            elif status == "running":
                # Animated CSS spinner glyph — replaces static ⏳
                badge_html = (
                    "<div class='nx-spinner'>◐</div>"
                )
                bg = "rgba(234, 179, 8, 0.18)"
                border = "#eab308"
            elif status == "error":
                badge_html = "<div style='font-size:18px;'>❌</div>"
                bg = "rgba(239, 68, 68, 0.15)"
                border = "#ef4444"
            else:
                badge_html = "<div style='font-size:18px;'>○</div>"
                bg = "#F5F4EE"
                border = "rgba(120, 120, 120, 0.35)"
            phase_marker = "①" if step["phase"] == 1 else "②"

            tile_col = cols[i * 2]
            with tile_col:
                st.markdown(
                    f"<div style='text-align:center; padding:10px 6px; "
                    f"border-radius:10px; border:1.5px solid {border}; "
                    f"background:{bg}; min-height:96px;'>"
                    f"<div style='font-size:11px; opacity:0.65; "
                    f"margin-bottom:2px;'>{phase_marker} {step['icon']}</div>"
                    f"<div style='font-weight:600; font-size:13px; "
                    f"line-height:1.15; margin-bottom:4px;'>{step['label']}</div>"
                    f"{badge_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Arrow separator (between tiles, not after the last one)
            if i < len(QG_STEPS) - 1:
                arrow_col = cols[i * 2 + 1]
                # Arrow color shifts to a green tint when the LEFT tile is
                # done — visually shows progress flowing forward.
                arrow_color = ("#22c55e"
                               if status == "done"
                               else "rgba(120,120,120,0.45)")
                with arrow_col:
                    st.markdown(
                        f"<div style='text-align:center; "
                        f"padding-top:36px; font-size:22px; "
                        f"color:{arrow_color}; font-weight:bold;'>→</div>",
                        unsafe_allow_html=True,
                    )
    def _qg_render_tracker_header():
        """Render the Domino's-style horizontal pizza tracker — 9 stages
        with status icons (pending ○, running animated spinner, done ✓,
        error ✗) plus arrow separators between consecutive tiles."""
        status_map = st.session_state["qg_step_status"]

        # 17 columns: 9 tiles interleaved with 8 narrow arrow columns.
        # Tile width 4, arrow width 1 → arrows visually small but clear.
        col_widths = []
        for i in range(len(QG_STEPS)):
            col_widths.append(2)            # tile
            if i < len(QG_STEPS) - 1:
                col_widths.append(1)        # arrow
        cols = st.columns(col_widths)

        # Inject a one-shot CSS keyframe for the animated spinner glyph.
        # `nx-spin` rotates a unicode partial-circle 360° on a 1s loop,
        # so the running step has *real* motion (not just a static ⏳).
        # Streamlit dedupes identical markdown so this is safe to render
        # on every rerun.
        st.markdown("""
<style>
@keyframes nx-spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
.nx-spinner {
  display: inline-block;
  animation: nx-spin 1s linear infinite;
  font-size: 10px;
}
</style>""", unsafe_allow_html=True)

        for i, step in enumerate(QG_STEPS):
            status = status_map.get(step["key"], "pending")
            if status == "done":
                badge_html = "<div style='font-size:10px;'>✅</div>"
                bg = "rgba(34, 197, 94, 0.15)"
                border = "#22c55e"
            elif status == "running":
                # Animated CSS spinner glyph — replaces static ⏳
                badge_html = (
                    "<div class='nx-spinner'>◐</div>"
                )
                bg = "rgba(234, 179, 8, 0.18)"
                border = "#eab308"
            elif status == "error":
                badge_html = "<div style='font-size:10px;'>❌</div>"
                bg = "rgba(239, 68, 68, 0.15)"
                border = "#ef4444"
            else:
                badge_html = "<div style='font-size:10px;'>○</div>"
                bg = "rgba(120, 120, 120, 0.08)"
                border = "rgba(120, 120, 120, 0.35)"
            phase_marker = "①" if step["phase"] == 1 else "②"

            tile_col = cols[i * 2]
            with tile_col:
                st.markdown(
                    f"<div style='text-align:center; padding:5px 3px; "
                    f"border-radius:8px; border:0px solid {border}; "
                    f"background:{bg}; min-height:20px;'>"
                    f"<div style='font-size:9px; opacity:0.30; "
                    f"margin-bottom:1px;'>{phase_marker} {step['icon']}</div>"
                    f"<div style='font-weight:200; font-size:8px; "
                    f"line-height:1; margin-bottom:2px;'>{step['label']}</div>"
                    f"{badge_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            # Arrow separator (between tiles, not after the last one)
            if i < len(QG_STEPS) - 1:
                arrow_col = cols[i * 2 + 1]
                # Arrow color shifts to a green tint when the LEFT tile is
                # done — visually shows progress flowing forward.
                arrow_color = ("#22c55e"
                               if status == "done"
                               else "rgba(120,120,120,0.45)")
                with arrow_col:
                    st.markdown(
                        f"<div style='text-align:center; "
                        f"padding-top:5px; font-size:8px; "
                        f"color:{arrow_color}; font-weight:bold;'>→</div>",
                        unsafe_allow_html=True,
                    )
    def _qg_set_status(key: str, status: str):
        st.session_state["qg_step_status"][key] = status

    # ─────────────────────────────────────────────────────────────────────
    # ── INPUT SECTION 1: Reverse Engineering inputs (parallel to tab 2)
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("🟦 :blue[1. Reverse Engineering Inputs]",
                     expanded=(st.session_state["qg_phase"] == 0)):
        with st.container():
            st.caption(
                "Files are "
                "tagged per tech stack. Uploaded STTM / "
                "Mapping CSVs seed the analysis as ground truth."
            )

            QG_TECH_ZONES = [
                ("datastage", "IBM DataStage (.dsx, .xml, sequence)",
                 ["dsx", "xml", "txt"]),
                ("bods",      "SAP BODS (.xml)",                       ["xml"]),
                ("legacy_sql","Legacy SQL (Oracle/DB2/Teradata .sql)",
                 ["sql", "txt"]),
                ("netezza",   "Netezza SQL (.sql)",                    ["sql", "txt"]),
                ("controlm",  "Control-M jobs (.xml, .ctm, .txt)",
                 ["xml", "ctm", "txt"]),
                ("shell",     "Shell scripts (.sh, .ksh, .bash)",
                 ["sh", "ksh", "bash", "txt"]),
                ("ssis",      "SSIS packages (.dtsx)",                 ["dtsx", "xml"]),
                ("denodo",    "Denodo VQL (.vql)",                     ["vql", "txt"]),
                ("mssql",     "MS SQL Server (.sql)",                  ["sql", "txt"]),
                # Generic source DDL + tabular metadata. Accepts both
                # raw SQL DDL (CREATE TABLE / VIEW / etc.) and metadata
                # CSVs (table/column inventories). Auto-dispatches to
                # the right parser by file extension. This category is
                # intended to feed ALL Phase-1 artifacts (Lineage, STTM,
                # Catalog, Raw Vault Model) — the parsed entities flow
                # into qg_all_entities/qg_all_parsed alongside any other
                # source-tagged uploads.
                ("source_ddl", "Source DDL / Metadata (.sql, .csv)",
                 ["sql", "csv", "txt"]),
            ]

            rev_categories = [
                {"key": tech, "label": label, "exts": exts, "kind": "source"}
                for tech, label, exts in QG_TECH_ZONES
            ] + [
                {"key": "seed_sttm", "label": "Reference: Existing STTM CSV",
                 "exts": ["csv"], "kind": "seed_sttm"},
                {"key": "seed_mapping",
                 "label": "Reference: Attribute mapping CSV",
                 "exts": ["csv"], "kind": "seed_mapping"},
                {"key": "template_sttm",
                 "label": "Template: STTM target shape (CSV/XLSX)",
                 "exts": ["csv", "xlsx", "xls"],
                 "kind": "template_sttm"},
                {"key": "standards_dm",
                 "label": "Standards: Data Modeling Standards (PDF/MD/TXT)",
                 "exts": ["pdf", "md", "txt"],
                 "kind": "standards_dm"},
            ]
            rev_label_map = {c["key"]: c["label"] for c in rev_categories}
            rev_ext_map = {c["key"]: c["exts"] for c in rev_categories}
            rev_kind_map = {c["key"]: c["kind"] for c in rev_categories}
            if "qg_rev_upload_rows" not in st.session_state:
                st.session_state["qg_rev_upload_rows"] = 1
            if "qg_show_all_artifacts" not in st.session_state:
                st.session_state["qg_show_all_artifacts"] = False

            qg_tagged_files = []
            qg_sttm_seed_file = None
            qg_mapping_seed_file = None
            qg_sttm_template_file = None
            qg_dm_standards_files = []  # multiple allowed
            st.markdown("###### Source documents")
            for i in range(st.session_state["qg_rev_upload_rows"]):
                c1, c2 = st.columns([1, 2])
                with c1:
                    selected_category = st.selectbox(
                        f"Category #{i + 1}",
                        options=[c["key"] for c in rev_categories],
                        format_func=lambda x: rev_label_map.get(x, x),
                        key=f"qg_rev_cat_{i}",
                    )
                with c2:
                    files_for_category = st.file_uploader(
                        f"Upload document #{i + 1}",
                        type=rev_ext_map.get(selected_category, []),
                        accept_multiple_files=True,
                        key=f"qg_rev_file_{i}",
                    )
                if files_for_category:
                    cat_kind = rev_kind_map.get(selected_category, "source")
                    if cat_kind == "source":
                        for uf in files_for_category:
                            qg_tagged_files.append({
                                "tech": selected_category,
                                "file": uf,
                                "divisions": [],
                            })
                    elif cat_kind == "seed_sttm":
                        qg_sttm_seed_file = files_for_category[0]
                    elif cat_kind == "seed_mapping":
                        qg_mapping_seed_file = files_for_category[0]
                    elif cat_kind == "template_sttm":
                        # Only one template is honored — last one wins.
                        qg_sttm_template_file = files_for_category[0]
                    elif cat_kind == "standards_dm":
                        # Multiple standards files concatenated.
                        qg_dm_standards_files.extend(files_for_category)

            if st.button("➕ Add reverse document", key="qg_add_rev_doc"):
                st.session_state["qg_rev_upload_rows"] += 1
                st.rerun()

            st.markdown("###### Reverse prompt controls")
            qg_reverse_prompt_file = st.file_uploader(
                "Upload reverse prompt text (txt/md) — optional",
                type=["txt", "md"],
                accept_multiple_files=False,
                key="qg_reverse_prompt_file",
            )
    

    # ─────────────────────────────────────────────────────────────────────
    # ── INPUT SECTION 2: Forward Engineering inputs (parallel to tab 3)
    # ─────────────────────────────────────────────────────────────────────
    with st.expander("🟩 :green[2. Forward Engineering Inputs]",
                     expanded=(st.session_state["qg_phase"] == 0)):
            st.caption(
                "Dashboard category + spec (PDF/Excel/Image), plus banking "
                "rules document."
            )

            d_col1, d_col2 = st.columns([1, 1])
            qg_forward_categories = (
                list(DASHBOARD_TYPES.keys())
                + ["Banking rules / internal knowledge"]
            )
            qg_forward_category = d_col1.selectbox(
                "Category",
                options=qg_forward_categories,
                key="qg_forward_category",
            )
            if qg_forward_category in DASHBOARD_TYPES:
                qg_dashboard_type = qg_forward_category
                d_col2.caption(
                    f"Scope: {DASHBOARD_TYPES.get(qg_dashboard_type, '—')}"
                )
            else:
                qg_dashboard_type = st.session_state.get(
                    "qg_dashboard_type",
                    list(DASHBOARD_TYPES.keys())[0],
                )
                d_col2.caption(
                    f"Dashboard scope in use: "
                    f"{DASHBOARD_TYPES.get(qg_dashboard_type, '—')}"
                )
            st.session_state["qg_dashboard_type"] = qg_dashboard_type

            if "qg_fwd_upload_rows" not in st.session_state:
                st.session_state["qg_fwd_upload_rows"] = 1
            qg_dashboard_files = []
            qg_rules_files = []
            qg_fwd_upload_types = (
                ["pdf", "xlsx", "xls", "txt", "md", "csv"]
                if qg_forward_category == "Banking rules / internal knowledge"
                else ["pdf", "xlsx", "xls", "png", "jpg",
                      "jpeg", "csv", "txt", "md"]
            )
            st.markdown("###### Forward documents")
            for i in range(st.session_state["qg_fwd_upload_rows"]):
                uf = st.file_uploader(
                    f"Upload document #{i + 1}",
                    type=qg_fwd_upload_types,
                    accept_multiple_files=True,
                    key=f"qg_fwd_file_{i}",
                )
                if uf:
                    if qg_forward_category == "Banking rules / internal knowledge":
                        qg_rules_files.extend(uf)
                    else:
                        qg_dashboard_files.extend(uf)
            if st.button("➕ Add forward document", key="qg_add_fwd_doc"):
                st.session_state["qg_fwd_upload_rows"] += 1
                st.rerun()

            qg_dashboard_description = ""
            st.markdown("###### Forward prompt controls")
            qg_forward_prompt_file = st.file_uploader(
                "Upload forward prompt text (txt/md) — optional",
                type=["txt", "md"],
                accept_multiple_files=False,
                key="qg_forward_prompt_file",
            )

    # ─────────────────────────────────────────────────────────────────────
    # ── PROGRESS TRACKER (boxed steps with arrows + animated spinner)
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#####  Pipeline Progress")
    _qg_render_tracker_header()
    # st.caption(
    #     "**Phase ①** = Reverse Engineering (4 steps: STTM → Lineage → "
    #     "Catalog → Raw Vault). **Phase ②** = Forward Engineering "
    #     "(5 steps: Business Vault → STTM → Catalog → RV dbt → BV dbt)."
    # )

    # ─────────────────────────────────────────────────────────────────────
    # ── ACTION BUTTONS — change behavior based on current phase
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    btn_l, btn_m, btn_r = st.columns([1, 1, 1])

    qg_phase = st.session_state["qg_phase"]
    start_clicked    = btn_l.button(
        "🚀 START", type="primary", use_container_width=True,
        disabled=(qg_phase != 0 or not qg_tagged_files),
        key="qg_start_btn",
    )
    continue_clicked = btn_m.button(
        "➡ Continue",
        type=("primary" if qg_phase == 1 else "secondary"),
        use_container_width=True,
        disabled=(qg_phase != 1),
        key="qg_continue_btn",
    )
    next_clicked     = btn_r.button(
        "✓ Next → Build/Deploy DBT Project",
        type="secondary", use_container_width=True,
        disabled=(qg_phase != 2),
        key="qg_next_btn",
    )

    if qg_phase == 0 and not qg_tagged_files:
        st.info("Upload at least one source file to enable START.")

    # Reset + view controls
    qg_ctl_l, qg_ctl_r = st.columns([1, 1])
    with qg_ctl_l:
        reset_clicked = st.button("🔄 Reset Quick GO", key="qg_reset_btn")
    with qg_ctl_r:
        qg_all_steps_done_top = all(
            s["key"] in (st.session_state.get("qg_artifacts", {}) or {})
            for s in QG_STEPS
        )
        qg_view_all_clicked = st.button(
            "📚 View all artifacts on this page",
            key="qg_view_all_artifacts_btn",
            disabled=not (qg_all_steps_done_top and
                          st.session_state.get("qg_phase", 0) >= 2),
        )

    if reset_clicked:
        for k in list(st.session_state.keys()):
            if k.startswith("qg_") and k not in ("qg_dashboard_type",
                                                  "qg_dashboard_desc"):
                # Wipe everything but text inputs (those re-render via key)
                if k in ("qg_phase", "qg_artifacts", "qg_step_status",
                         "qg_metadata_summary"):
                    st.session_state.pop(k, None)
        st.session_state["qg_phase"] = 0
        st.session_state["qg_artifacts"] = {}
        st.session_state["qg_step_status"] = {}
        st.rerun()

    if qg_view_all_clicked:
        st.session_state["qg_show_all_artifacts"] = True
        st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # ── PHASE 1: START → 4 reverse-engineering steps
    # ─────────────────────────────────────────────────────────────────────
    if start_clicked and qg_tagged_files:
        # --- Build metadata_summary from uploads (parallel logic to tab2) ---
        with st.spinner(
            f"Parsing {len(qg_tagged_files)} file(s) across "
            f"{len({t['tech'] for t in qg_tagged_files})} tech stack(s)…"
        ):
            QG_TECH_PARSERS = {
                "datastage":  parse_datastage_dsx,
                "bods":       parse_bods_xml,
                "legacy_sql": parse_legacy_sql,
                "netezza":    parse_netezza_sql,
                "mssql":      parse_mssql_sql,
                "controlm":   parse_control_m,
                "shell":      parse_shell_script,
                "ssis":       parse_ssis_dtsx,
                "denodo":     parse_denodo_vql,
                "source_ddl": parse_source_ddl_or_metadata,
            }
            qg_all_entities = {}
            # Per-file parsed dicts — needed by build_lineage_mermaid to
            # render file subgraphs and within-file stage→stage links.
            # Shape per entry: {filename, kind, tech, divisions, parsed}
            qg_all_parsed = []
            for tagged in qg_tagged_files:
                uf = tagged["file"]
                tech = tagged["tech"]
                divs = tagged["divisions"]
                try:
                    content = uf.read().decode("utf-8", errors="ignore")
                    uf.seek(0)
                except Exception:
                    continue
                parser = QG_TECH_PARSERS.get(tech)
                if not parser:
                    continue
                parsed = parser(content)
                qg_all_parsed.append({
                    "filename":  uf.name,
                    "kind":      tech,
                    "tech":      tech,
                    "divisions": divs,
                    "parsed":    parsed,
                })
                stage_key_map = {}
                for s in parsed.get("stages", []):
                    key = f"{uf.name}::{s['job_name']}::{s['stage_name']}"
                    stage_key_map[(s["job_name"], s["stage_name"])] = key
                    qg_all_entities.setdefault(key, {
                        "display_name":  s["stage_name"],
                        "columns":       set(),
                        "columns_typed": {},
                        "source_file":   uf.name,
                        "type":          s["stage_type"],
                        "job_name":      s["job_name"],
                        "tech":          tech,
                        "divisions":     divs,
                    })
                for c in parsed.get("columns", []):
                    key = stage_key_map.get(
                        (c["job_name"], c["stage_name"])
                    )
                    if key and key in qg_all_entities:
                        col_name = c["column_name"].upper()
                        qg_all_entities[key]["columns"].add(col_name)
                        resolved = resolve_sql_type(
                            c.get("sql_type", ""),
                            c.get("precision", ""), c.get("scale", ""),
                        )
                        if resolved and not qg_all_entities[key][
                            "columns_typed"
                        ].get(col_name):
                            qg_all_entities[key]["columns_typed"][
                                col_name
                            ] = resolved

            # Build the metadata_summary string. Mirrors the tab_reverse
            # logic but stripped to essentials: entity inventory + columns.
            summary_lines = [
                "# METADATA ANALYSIS",
                f"Files uploaded: {len(qg_tagged_files)} "
                f"({', '.join(t['file'].name for t in qg_tagged_files)})",
                f"Tech stacks: "
                f"{', '.join(sorted({t['tech'] for t in qg_tagged_files}))}",
                "",
                "## COMPLETE ENTITY INVENTORY",
            ]
            for key, meta in qg_all_entities.items():
                cols_typed = meta["columns_typed"]
                col_lines = []
                for c in sorted(meta["columns"]):
                    t = cols_typed.get(c, "")
                    col_lines.append(f"  - {c}{(' ' + t) if t else ''}")
                summary_lines.append("")
                summary_lines.append(
                    f"### {meta['display_name']} "
                    f"[{meta['tech']} | {meta['type']} | "
                    f"file: {meta['source_file']}]"
                )
                summary_lines.append("\n".join(col_lines) or "  (no columns)")

            # Cross-entity shared keys (for hub-detection in Raw Vault)
            from collections import defaultdict as _qg_dd
            col_to_entities = _qg_dd(set)
            for key, meta in qg_all_entities.items():
                for c in meta["columns"]:
                    col_to_entities[c].add(meta["display_name"])
            shared = {c: ents for c, ents in col_to_entities.items()
                      if len(ents) >= 2}
            if shared:
                summary_lines.append("")
                summary_lines.append("## SHARED BUSINESS KEY CANDIDATES")
                for c, ents in sorted(
                    shared.items(), key=lambda x: (-len(x[1]), x[0])
                )[:25]:
                    summary_lines.append(
                        f"- **{c}** appears in: "
                        f"{', '.join(sorted(ents))}"
                    )

            # Inject seed CSVs as ground truth
            if qg_sttm_seed_file is not None:
                try:
                    seed_text = qg_sttm_seed_file.read().decode(
                        "utf-8", errors="ignore"
                    )
                    qg_sttm_seed_file.seek(0)
                    seed_parsed = parse_sttm_csv(seed_text)
                    summary_lines.append("")
                    summary_lines.append(
                        "# GROUND-TRUTH STTM (seed — extend, do not override)"
                    )
                    summary_lines.append(
                        seed_parsed["sttm_df"].head(200).to_csv(index=False)
                    )
                except Exception as e:
                    st.warning(f"STTM seed unreadable: {e}")
            if qg_mapping_seed_file is not None:
                try:
                    seed_text = qg_mapping_seed_file.read().decode(
                        "utf-8", errors="ignore"
                    )
                    qg_mapping_seed_file.seek(0)
                    seed_parsed = parse_metadata_csv(seed_text)
                    summary_lines.append("")
                    summary_lines.append(
                        "# GROUND-TRUTH attribute mapping (seed)"
                    )
                    summary_lines.append(
                        seed_parsed["columns_df"].head(200).to_csv(
                            index=False
                        )
                    )
                except Exception as e:
                    st.warning(f"Mapping seed unreadable: {e}")

            # ── Read the optional STTM Template (CSV or XLSX) ──
            # This file's first non-empty line is the authoritative
            # output header for STTM and Forward STTM. Subsequent rows
            # serve as example values the model should mirror.
            qg_sttm_template_text = ""
            if qg_sttm_template_file is not None:
                try:
                    qg_sttm_template_text = extract_text_from_upload(
                        qg_sttm_template_file
                    )
                    # XLSX extraction prepends "--- Sheet: <name> ---\n"
                    # which would be picked up as the "header" line.
                    # Strip those marker lines so the real header is
                    # the first non-empty line we see.
                    qg_sttm_template_text = "\n".join(
                        ln for ln in qg_sttm_template_text.splitlines()
                        if not ln.strip().startswith("--- Sheet:")
                    )
                    if qg_sttm_template_text.strip():
                        st.caption(
                            f"📋 STTM template loaded "
                            f"({len(qg_sttm_template_text):,} chars)"
                        )
                except Exception as e:
                    st.warning(f"STTM template unreadable: {e}")
                    qg_sttm_template_text = ""
            st.session_state["qg_sttm_template_text"] = (
                qg_sttm_template_text
            )

            # ── Read the optional Data Modeling Standards files ──
            # Concatenate text from each (PDF text via pypdf; markdown
            # / plaintext directly). Used by Raw Vault and Business
            # Vault generators as additional design constraints.
            qg_dm_standards_text = ""
            if qg_dm_standards_files:
                parts = []
                for uf in qg_dm_standards_files:
                    try:
                        txt = extract_text_from_upload(uf)
                        if txt and txt.strip():
                            parts.append(
                                f"--- File: {uf.name} ---\n{txt.strip()}"
                            )
                    except Exception as e:
                        st.warning(
                            f"Standards file {uf.name} unreadable: {e}"
                        )
                qg_dm_standards_text = "\n\n".join(parts)
                if qg_dm_standards_text:
                    st.caption(
                        f"📐 Data modeling standards loaded "
                        f"({len(qg_dm_standards_files)} file(s), "
                        f"{len(qg_dm_standards_text):,} chars)"
                    )
            st.session_state["qg_dm_standards_text"] = qg_dm_standards_text

            qg_meta = "\n".join(summary_lines)
            qg_reverse_prompt_blob = ""
            if qg_reverse_prompt_file is not None:
                try:
                    qg_reverse_prompt_blob += (
                        "\n\n# Uploaded reverse prompt\n" +
                        extract_text_from_upload(qg_reverse_prompt_file)[:12000]
                    )
                except Exception as e:
                    st.warning(f"Reverse prompt file unreadable: {e}")
            if qg_reverse_prompt_blob.strip():
                qg_meta = (
                    qg_meta
                    + "\n\n# USER ADDITIONAL INSTRUCTIONS\n"
                    + qg_reverse_prompt_blob.strip()
                )
            st.session_state["qg_metadata_summary"] = qg_meta

        # --- Run the 4 reverse steps ---
        qg_model_id = MODELS[st.session_state.selected_model]
        qg_arts = st.session_state["qg_artifacts"]

        def _qg_call(prompt, max_tokens=16000):
            return call_cortex(
                qg_model_id, prompt,
                temperature=0.1, max_tokens=max_tokens,
            )

        # Live container that re-renders the boxed tracker as each step
        # transitions running → done, so the user sees the tile colors
        # and animated spinner glyph update in place.
        qg_tracker_live = st.empty()
        # User-specified order: STTM → Lineage → Catalog → Transformation
        # Rules → Raw Vault → Raw Vault Validation. The Transformation
        # Rules step runs BEFORE Raw Vault so its Markdown can be fed
        # into the Raw Vault generator as additional design context.
        # Lineage at step 2 is final (no second-pass enrichment) — it's
        # built strictly from the parsed Reverse Engineering Inputs.
        for step_key in ("STTM", "Data Lineage", "Data Catalog",
                         "Transformation Rules",
                         "Raw Vault Model", "Raw Vault Validation"):
            _qg_set_status(step_key, "running")
            with qg_tracker_live.container():
                _qg_render_tracker()
            try:
                if step_key == "STTM":
                    # STTM is generated BEFORE the Raw Vault Model. The
                    # `pre_raw_vault=True` mode produces a clean
                    # source-to-staging mapping with NO Hub / Link /
                    # Satellite / Raw Vault terminology. The vault
                    # design is intentionally a separate downstream
                    # concern — review STTM standalone first.
                    # If the user uploaded an STTM template, its header
                    # becomes the authoritative output schema.
                    raw = _qg_call(build_sttm_prompt(
                        qg_meta, pre_raw_vault=True,
                        sttm_template_text=st.session_state.get(
                            "qg_sttm_template_text", ""),
                    ))
                    df = parse_table_response(raw)
                    qg_arts["STTM"] = {
                        "kind": "table",
                        "label": "Source-to-Target Mapping",
                        "content": {"df": df, "raw": raw},
                    }

                elif step_key == "Data Lineage":
                    # Build a parsed_meta shape that build_lineage_mermaid
                    # actually consumes: entities + files (full dicts
                    # with kind + parsed) + cross_job_flows (empty here).
                    parsed_meta_for_lineage = {
                        "entities":        qg_all_entities,
                        "files":           qg_all_parsed,
                        "cross_job_flows": [],
                    }
                    rv_tables = {"hubs": [], "links": [], "sats": []}
                    # CRITICAL: pass None, not [], for source_to_hub_df.
                    # The function does `df.empty` which AttributeErrors
                    # on a list.
                    mermaid = build_lineage_mermaid(
                        parsed_meta_for_lineage, rv_tables,
                        source_to_hub_df=None,
                    )
                    # Also build the interactive graph (nodes + edges)
                    # so the result panel can render the rich interactive
                    # lineage view, matching the Reverse / View tab UX.
                    graph = build_lineage_graph(
                        parsed_meta_for_lineage, rv_tables,
                        source_to_hub_df=None,
                    )

                    # ── Tabular Data Lineage ─────────────────────────
                    # In addition to the graphical (Mermaid + interactive
                    # node-graph) views, derive two tabular views from
                    # the parsed Reverse Engineering Inputs:
                    #
                    #   1. Entity & Column Inventory — one row per
                    #      (file, job, stage, column) showing what was
                    #      parsed from each source artifact. This is
                    #      column-level lineage anchored on the source.
                    #   2. Stage-to-Stage Flows — one row per
                    #      within-file or cross-file dataflow edge,
                    #      showing how data moves between parsed stages.
                    #
                    # Both tables come strictly from the source uploads
                    # (qg_all_entities + qg_all_parsed) — no Raw Vault
                    # information is used.
                    inv_rows = []
                    for ent_key, meta in qg_all_entities.items():
                        cols = sorted(meta.get("columns", []) or [])
                        typed = meta.get("columns_typed", {}) or {}
                        if not cols:
                            # Still record the entity so it shows up in
                            # the inventory even if no columns parsed.
                            inv_rows.append({
                                "Source File": meta.get("source_file", ""),
                                "Tech Stack":  meta.get("tech", ""),
                                "Job":         meta.get("job_name", "") or "",
                                "Entity / Stage":
                                    meta.get("display_name", ""),
                                "Stage Type":  meta.get("type", "") or "",
                                "Column":      "",
                                "Data Type":   "",
                            })
                            continue
                        for c in cols:
                            inv_rows.append({
                                "Source File": meta.get("source_file", ""),
                                "Tech Stack":  meta.get("tech", ""),
                                "Job":         meta.get("job_name", "") or "",
                                "Entity / Stage":
                                    meta.get("display_name", ""),
                                "Stage Type":  meta.get("type", "") or "",
                                "Column":      c,
                                "Data Type":   typed.get(c, "") or "",
                            })

                    flow_rows = []
                    # Derive flow rows from the actual lineage graph
                    # edges that the renderer is drawing. This makes the
                    # table view automatically consistent with the
                    # graphical view AND covers every tech stack — not
                    # just DataStage. We include:
                    #   - file → stage edges (every stage's source file)
                    #   - stage → stage edges within the same file
                    #     (from explicit links in DataStage/BODS/SSIS/SQL
                    #     etc.)
                    #   - cross-file edges (inferred from shared cols)
                    # Build a node lookup once so we can resolve the
                    # human-readable labels for "from" and "to".
                    node_lookup = {
                        n.get("id"): n for n in graph.get("nodes", [])
                        if n.get("id")
                    }
                    # Map each "stage" node id back to (file, job) so
                    # the flow rows can show that context.
                    stage_meta = {}
                    for ent_key, meta in qg_all_entities.items():
                        # Mirror the id construction from
                        # build_lineage_graph: "S_" + _mermaid_safe_id.
                        try:
                            sid = "S_" + _mermaid_safe_id(ent_key)
                        except Exception:
                            sid = None
                        if sid:
                            stage_meta[sid] = {
                                "file": meta.get("source_file", ""),
                                "job":  meta.get("job_name", "") or "",
                                "tech": meta.get("tech", ""),
                                "name": meta.get("display_name", ""),
                            }

                    def _node_file(node_id):
                        n = node_lookup.get(node_id) or {}
                        # "stage" nodes carry the file in `group`; "file"
                        # nodes use group == filename too.
                        return n.get("group", "") or ""

                    def _node_label(node_id):
                        n = node_lookup.get(node_id) or {}
                        # Strip newline-suffixed job name we add in the
                        # rendered label so the table reads cleanly.
                        return (n.get("label", "") or "").split("\n")[0]

                    def _node_tech(node_id):
                        sm = stage_meta.get(node_id)
                        return sm["tech"] if sm else ""

                    def _node_job(node_id):
                        sm = stage_meta.get(node_id)
                        return sm["job"] if sm else ""

                    flow_kind_label = {
                        "file_stage":  "file → stage",
                        "stage_stage": "within-file",
                        "cross_file":  "cross-file",
                    }
                    for e in graph.get("edges", []):
                        ek = e.get("kind", "")
                        if ek not in flow_kind_label:
                            # Skip RV-related edges (none here, but be
                            # defensive in case rv_tables is non-empty
                            # in a future code path).
                            continue
                        frm = e.get("from")
                        to = e.get("to")
                        flow_rows.append({
                            "From File":  _node_file(frm),
                            "From Job":   _node_job(frm),
                            "From Stage": _node_label(frm),
                            "To File":    _node_file(to),
                            "To Job":     _node_job(to),
                            "To Stage":   _node_label(to),
                            "Tech Stack": _node_tech(to) or _node_tech(frm),
                            "Flow Type":  flow_kind_label[ek],
                        })

                    # Fallback: if the graph yielded no edges (rare —
                    # e.g. when entity-discovery succeeded but graph
                    # build skipped them), pull within-file links from
                    # the raw parsed metadata. This ensures the table
                    # always reflects whatever flow information we have.
                    if not flow_rows:
                        for f in qg_all_parsed:
                            fname = f.get("filename", "")
                            tech  = f.get("tech", "")
                            for l in (f.get("parsed", {}) or {}).get(
                                "links", []
                            ) or []:
                                frm_stage = l.get("from_stage", "") or ""
                                to_stage = l.get("to_stage", "") or ""
                                if not (frm_stage or to_stage):
                                    continue
                                flow_rows.append({
                                    "From File":  fname,
                                    "From Job":   l.get("job_name", "") or "",
                                    "From Stage": frm_stage,
                                    "To File":    fname,
                                    "To Job":     l.get("job_name", "") or "",
                                    "To Stage":   to_stage,
                                    "Tech Stack": tech,
                                    "Flow Type":  "within-file",
                                })
                        # Also add file→stage rows for every parsed
                        # entity so the table is never empty when we
                        # have entities to show.
                        for ent_key, meta in qg_all_entities.items():
                            flow_rows.append({
                                "From File":  meta.get("source_file", ""),
                                "From Job":   "",
                                "From Stage": meta.get("source_file", ""),
                                "To File":    meta.get("source_file", ""),
                                "To Job":     meta.get("job_name", "") or "",
                                "To Stage":   meta.get("display_name", ""),
                                "Tech Stack": meta.get("tech", ""),
                                "Flow Type":  "file → stage",
                            })

                    try:
                        inventory_df = pd.DataFrame(inv_rows)
                        flows_df = pd.DataFrame(flow_rows)
                    except Exception:
                        inventory_df = None
                        flows_df = None

                    qg_arts["Data Lineage"] = {
                        "kind": "lineage",
                        "label": "Data Lineage",
                        "content": {
                            "mermaid":       mermaid,
                            "graph":         graph,
                            "source_to_hub": None,
                            "rv_tables":     rv_tables,
                            "mapping_raw":   "",
                            # Tabular views — derived from source
                            # uploads only (no Raw Vault dependency).
                            "inventory_df":  inventory_df,
                            "flows_df":      flows_df,
                            # Stash the parsed_meta for completeness
                            # (no longer used for any second pass).
                            "_parsed_meta":  parsed_meta_for_lineage,
                        },
                    }

                elif step_key == "Data Catalog":
                    raw = _qg_call(build_data_catalog_prompt(qg_meta))
                    df = parse_table_response(raw)
                    qg_arts["Data Catalog"] = {
                        "kind": "table",
                        "label": "Data Catalog",
                        "content": {"df": df, "raw": raw},
                    }

                elif step_key == "Transformation Rules":
                    # ─────────────────────────────────────────────────
                    # Reverse-engineer every transformation, derivation,
                    # filter, lookup, surrogate-key strategy, type cast,
                    # and business rule from the parsed source. Output
                    # is structured Markdown — readable by a human and
                    # re-usable as ADDITIONAL CONTEXT for the Raw Vault
                    # Model generator (see "Raw Vault Model" step below).
                    # Strictly source-driven: input is `qg_meta` only.
                    # ─────────────────────────────────────────────────
                    rules_raw = _qg_call(
                        build_transformation_rules_prompt(qg_meta),
                        max_tokens=12000,
                    )
                    # Cortex sometimes returns content as a JSON-encoded
                    # string (literal "\n" sequences and outer quotes).
                    # Unwrap so newlines render as real newlines and
                    # Markdown looks readable. Then strip wrapping code
                    # fences (```markdown / ```md / ```) if present.
                    rules_md = _unwrap_json_string(rules_raw or "").strip()
                    if rules_md.startswith("```"):
                        first_nl = rules_md.find("\n")
                        if first_nl != -1:
                            rules_md = rules_md[first_nl + 1:]
                        if rules_md.rstrip().endswith("```"):
                            rules_md = rules_md.rstrip()[:-3].rstrip()
                    # Final safety: collapse stray escaped sequences that
                    # may have survived (some models double-escape).
                    if "\\n" in rules_md and "\n" not in rules_md:
                        rules_md = (rules_md.replace("\\n", "\n")
                                             .replace("\\t", "\t")
                                             .replace('\\"', '"'))
                    qg_arts["Transformation Rules"] = {
                        "kind": "markdown",
                        "label": "Transformation Rules & Business Logic",
                        "content": {
                            "markdown": rules_md,
                            "raw":      rules_raw,
                        },
                    }

                elif step_key == "Raw Vault Model":
                    # Pull in the transformation-rules Markdown produced
                    # by the previous step so the Raw Vault generator
                    # can preserve legacy business semantics. Falls back
                    # to empty string if the step was skipped or failed.
                    rules_md_for_rv = (
                        qg_arts.get("Transformation Rules", {})
                               .get("content", {})
                               .get("markdown", "") or ""
                    )
                    # Optional uploaded Data Modeling Standards.
                    dm_std_text = st.session_state.get(
                        "qg_dm_standards_text", "") or ""
                    nar = _unwrap_json_string(_qg_call(
                        build_raw_vault_narrative_prompt(
                            qg_meta, rules_md_for_rv,
                            dm_standards_text=dm_std_text,
                        )
                    ))
                    mer_raw = _qg_call(
                        build_raw_vault_mermaid_prompt(
                            qg_meta, rules_md_for_rv,
                            dm_standards_text=dm_std_text,
                        )
                    )
                    mer = extract_mermaid_script(mer_raw)
                    sql_raw = _qg_call(
                        build_raw_vault_sql_prompt(
                            qg_meta, rules_md_for_rv,
                            dm_standards_text=dm_std_text,
                        )
                    )
                    sql = extract_sql_ddl(sql_raw)
                    qg_arts["Raw Vault Model"] = {
                        "kind": "raw_vault",
                        "label": "Raw Vault Data Model",
                        "content": {
                            "narrative_md": nar,
                            "mermaid":      mer,
                            "mermaid_raw":  mer_raw,
                            "sql":          sql,
                            "sql_raw":      sql_raw,
                        },
                    }
                    # NOTE: STTM and Data Lineage are deliberately built
                    # from the Reverse Engineering Inputs ONLY (qg_meta /
                    # parsed source files). They are NOT enriched
                    # post-hoc with the Raw Vault DDL — keeping them
                    # source-driven preserves the "what's in the legacy
                    # system" view, independent of the proposed Raw
                    # Vault. Any source→Hub mapping needed downstream
                    # can be derived later from the Raw Vault artifact
                    # itself.

                elif step_key == "Raw Vault Validation":
                    # ─────────────────────────────────────────────────
                    # GenAI as INDEPENDENT validator. Re-uses the same
                    # model id, but the prompt repositions the model as
                    # an adversarial reviewer (not the generator). The
                    # output is strict JSON so the result panel can
                    # render gauges, a violations register, business-key
                    # confidence matrix, lineage matrix, and a final
                    # production-readiness scorecard.
                    # ─────────────────────────────────────────────────
                    rv_art = qg_arts.get("Raw Vault Model", {}) \
                                    .get("content", {})
                    sttm_art = qg_arts.get("STTM", {}) \
                                      .get("content", {})
                    cat_art = qg_arts.get("Data Catalog", {}) \
                                     .get("content", {})
                    lin_art = qg_arts.get("Data Lineage", {}) \
                                     .get("content", {})

                    # Render lineage source→hub df back to text so the
                    # validator can see which sources feed which hubs.
                    sth_text = ""
                    sth_for_val = lin_art.get("source_to_hub")
                    if sth_for_val is not None:
                        try:
                            if hasattr(sth_for_val, "to_csv"):
                                sth_text = sth_for_val.to_csv(index=False)
                        except Exception:
                            sth_text = ""

                    val_prompt = build_raw_vault_validation_prompt(
                        metadata_summary=qg_meta,
                        raw_vault_sql=rv_art.get("sql", "") or "",
                        raw_vault_narrative=rv_art.get(
                            "narrative_md", "") or "",
                        raw_vault_mermaid=rv_art.get("mermaid", "") or "",
                        sttm_text=sttm_art.get("raw", "") or "",
                        data_catalog_text=cat_art.get("raw", "") or "",
                        source_to_hub_text=sth_text,
                    )
                    val_raw = _qg_call(val_prompt, max_tokens=12000)

                    # Best-effort JSON parsing — the model may wrap the
                    # JSON in stray prose or fences despite instructions.
                    # Importantly: json.loads() can return any JSON value
                    # (str, list, number, bool, null) — but the rendering
                    # code expects a dict. Enforce dict-only here so the
                    # downstream `.get()` calls don't AttributeError.
                    val_json = None
                    val_parse_error = ""

                    def _accept_if_dict(obj):
                        return obj if isinstance(obj, dict) else None

                    def _repair_json(blob):
                        """Last-resort repair for common LLM JSON glitches:
                        trailing commas, single quotes around keys, smart
                        quotes, and JS-style comments. Returns repaired
                        blob — caller still needs to json.loads() it."""
                        if not blob:
                            return blob
                        s = blob
                        # Smart-quote replacements
                        s = (s.replace("\u201c", '"').replace("\u201d", '"')
                              .replace("\u2018", "'").replace("\u2019", "'"))
                        # Strip JS-style line comments
                        s = re.sub(r"//[^\n\r]*", "", s)
                        # Strip JS-style block comments
                        s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
                        # Strip trailing commas before } or ]
                        s = re.sub(r",\s*([}\]])", r"\1", s)
                        return s

                    # Pull the response through _unwrap_json_string first —
                    # Cortex sometimes returns the JSON as a JSON-encoded
                    # string (literal \" and \n sequences).
                    val_unwrapped = _unwrap_json_string(val_raw or "")
                    # Strip outer ```json fences if present.
                    _stripped = val_unwrapped.strip()
                    if _stripped.startswith("```"):
                        first_nl = _stripped.find("\n")
                        if first_nl != -1:
                            _stripped = _stripped[first_nl + 1:]
                        if _stripped.rstrip().endswith("```"):
                            _stripped = _stripped.rstrip()[:-3].rstrip()

                    # Pass 1: direct parse of the unwrapped/de-fenced text
                    try:
                        val_json = _accept_if_dict(json.loads(_stripped))
                    except Exception as e1:
                        val_parse_error = str(e1)

                    # Pass 2: brace-balanced extraction
                    if val_json is None:
                        start = _stripped.find("{")
                        if start >= 0:
                            depth = 0
                            in_str = False
                            esc = False
                            for i, ch in enumerate(_stripped[start:],
                                                   start=start):
                                if esc:
                                    esc = False
                                    continue
                                if ch == "\\":
                                    esc = True
                                    continue
                                if ch == '"':
                                    in_str = not in_str
                                    continue
                                if in_str:
                                    continue
                                if ch == "{":
                                    depth += 1
                                elif ch == "}":
                                    depth -= 1
                                    if depth == 0:
                                        candidate = _stripped[start:i + 1]
                                        try:
                                            val_json = _accept_if_dict(
                                                json.loads(candidate)
                                            )
                                        except Exception as e2:
                                            val_parse_error = str(e2)
                                        break

                    # Pass 3: shared utils' extract_json (handles fences,
                    # prose wrappers, and partial truncation)
                    if val_json is None:
                        try:
                            extracted = extract_json(val_unwrapped)
                            val_json = _accept_if_dict(extracted)
                        except Exception as e3:
                            val_parse_error = str(e3)

                    # Pass 4: repair common glitches and retry
                    if val_json is None:
                        try:
                            repaired = _repair_json(_stripped)
                            val_json = _accept_if_dict(json.loads(repaired))
                        except Exception as e4:
                            val_parse_error = str(e4)

                    # Pass 5: brace-balance on REPAIRED text
                    if val_json is None:
                        repaired = _repair_json(_stripped)
                        start = repaired.find("{")
                        if start >= 0:
                            depth = 0
                            in_str = False
                            esc = False
                            for i, ch in enumerate(repaired[start:],
                                                   start=start):
                                if esc:
                                    esc = False
                                    continue
                                if ch == "\\":
                                    esc = True
                                    continue
                                if ch == '"':
                                    in_str = not in_str
                                    continue
                                if in_str:
                                    continue
                                if ch == "{":
                                    depth += 1
                                elif ch == "}":
                                    depth -= 1
                                    if depth == 0:
                                        candidate = repaired[start:i + 1]
                                        try:
                                            val_json = _accept_if_dict(
                                                json.loads(candidate)
                                            )
                                        except Exception as e5:
                                            val_parse_error = str(e5)
                                        break

                    qg_arts["Raw Vault Validation"] = {
                        "kind": "validation",
                        "label": "Raw Vault Validation Report",
                        "content": {
                            "json":         val_json,
                            "raw":          val_raw,
                            "parse_error":  val_parse_error,
                        },
                    }

                _qg_set_status(step_key, "done")
            except Exception as e:
                _qg_set_status(step_key, "error")
                st.error(f"Step **{step_key}** failed: {e}")
                with qg_tracker_live.container():
                    _qg_render_tracker()
                break
        else:
            st.session_state["qg_phase"] = 1
            st.success(
                "✓ Phase ① complete (STTM, Lineage, Catalog, "
                "Transformation Rules, Raw Vault, Validation). Review "
                "the results — including the transformation rules and "
                "validation scorecard — then click **Continue** to "
                "proceed to forward-engineering."
            )
        
        #Final tracker refresh
        with qg_tracker_live.container():
            _qg_render_tracker()
        st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # ── PHASE 2: Continue → 5 forward-engineering steps
    # ─────────────────────────────────────────────────────────────────────
    
    if continue_clicked:
        # Build dashboard_text + rules_text
        qg_dashboard_parts = []
        if qg_dashboard_description and qg_dashboard_description.strip():
            qg_dashboard_parts.append(
                f"USER DESCRIPTION:\n{qg_dashboard_description}"
            )
        for uf in (qg_dashboard_files or []):
            try:
                txt = extract_text_from_upload(uf)
                if txt:
                    qg_dashboard_parts.append(
                        f"--- FILE: {uf.name} ---\n{txt[:10000]}"
                    )
            except Exception as e:
                st.warning(f"Could not read {uf.name}: {e}")
        qg_dashboard_text = "\n\n".join(qg_dashboard_parts)

        qg_rules_parts = []
        for uf in (qg_rules_files or []):
            try:
                txt = extract_text_from_upload(uf)
                if txt:
                    qg_rules_parts.append(
                        f"--- FILE: {uf.name} ---\n{txt[:12000]}"
                    )
            except Exception as e:
                st.warning(f"Could not read {uf.name}: {e}")
        qg_rules_text = "\n\n".join(qg_rules_parts)
        qg_forward_prompt_blob = ""
        if qg_forward_prompt_file is not None:
            try:
                qg_forward_prompt_blob += (
                    "\n\n# Uploaded forward prompt\n" +
                    extract_text_from_upload(qg_forward_prompt_file)[:12000]
                )
            except Exception as e:
                st.warning(f"Forward prompt file unreadable: {e}")
        if qg_forward_prompt_blob.strip():
            qg_dashboard_text = (
                qg_dashboard_text
                + "\n\n# USER ADDITIONAL INSTRUCTIONS\n"
                + qg_forward_prompt_blob.strip()
            )
            qg_rules_text = (
                qg_rules_text
                + "\n\n# USER ADDITIONAL INSTRUCTIONS\n"
                + qg_forward_prompt_blob.strip()
            )

        # Reverse summary = the metadata + the artifacts we just generated
        qg_meta = st.session_state["qg_metadata_summary"]
        qg_arts = st.session_state["qg_artifacts"]
        qg_model_id = MODELS[st.session_state.selected_model]

        # Build a compact reverse_summary for forward prompts: include
        # the Raw Vault DDL + STTM + Catalog as ground truth.
        rev_sections = ["# REVERSE-ENGINEERING ARTIFACTS (ground truth)"]
        if qg_arts.get("Raw Vault Model", {}).get("content", {}).get("sql"):
            rev_sections.append("\n## RAW VAULT DDL\n")
            rev_sections.append("```sql")
            rev_sections.append(
                qg_arts["Raw Vault Model"]["content"]["sql"][:30000]
            )
            rev_sections.append("```")
        if qg_arts.get("Raw Vault Model", {}).get("content", {}) \
                   .get("narrative_md"):
            rev_sections.append("\n## RAW VAULT NARRATIVE\n")
            rev_sections.append(
                qg_arts["Raw Vault Model"]["content"]["narrative_md"][:8000]
            )
        if qg_arts.get("STTM", {}).get("content", {}).get("raw"):
            rev_sections.append("\n## STTM (CSV, head)\n")
            rev_sections.append("```csv")
            rev_sections.append(
                qg_arts["STTM"]["content"]["raw"][:8000]
            )
            rev_sections.append("```")
        qg_reverse_summary = "\n".join(rev_sections)

        def _qg_fwd_call(prompt, max_tokens=8000):
            return call_cortex(
                qg_model_id, prompt,
                temperature=0.2, max_tokens=max_tokens,
            )

        qg_tracker_live2 = st.empty()
        for step_key in ("Business Vault", "Forward STTM",
                         "Forward Catalog", "Semantic Data Model",
                         "Raw Vault dbt", "Business Vault dbt"):
            _qg_set_status(step_key, "running")
            with qg_tracker_live2.container():
                _qg_render_tracker()
            try:
                if step_key == "Business Vault":
                    # Business Vault narrative depends on a "semantic
                    # context" — in Quick GO mode we use the dashboard
                    # text + reverse summary directly (no separate
                    # Semantic Model step).
                    sem_proxy = (
                        qg_dashboard_text
                        or "(no dashboard spec; design BV directly from "
                           "Raw Vault)"
                    )
                    # Optional uploaded Data Modeling Standards (same
                    # standards used for Raw Vault — these constrain
                    # naming, hash-key types, audit columns, etc.).
                    dm_std_text = st.session_state.get(
                        "qg_dm_standards_text", "") or ""
                    try:
                        nar = _qg_fwd_call(
                            build_business_vault_narrative_prompt(
                                qg_dashboard_type, qg_dashboard_text,
                                qg_rules_text, qg_reverse_summary, sem_proxy,
                                dm_standards_text=dm_std_text,
                            )
                        )
                        mer_raw = _qg_fwd_call(
                            build_business_vault_mermaid_prompt(
                                nar, qg_reverse_summary,
                                dm_standards_text=dm_std_text,
                            )
                        )
                        sql_raw = _qg_fwd_call(
                            build_business_vault_sql_prompt(
                                nar, qg_reverse_summary,
                                dm_standards_text=dm_std_text,
                            )
                        )
                    except Exception as _bv_e:
                        # Retry with reduced context to avoid request-size /
                        # token-pressure failures on larger uploads.
                        st.warning(
                            "Business Vault generation hit an error; retrying "
                            "with reduced context."
                        )
                        nar = _qg_fwd_call(
                            build_business_vault_narrative_prompt(
                                qg_dashboard_type,
                                (qg_dashboard_text or "")[:4000],
                                (qg_rules_text or "")[:2000],
                                (qg_reverse_summary or "")[:6000],
                                (sem_proxy or "")[:4000],
                                dm_standards_text=(dm_std_text or "")[:6000],
                            )
                        )
                        mer_raw = _qg_fwd_call(
                            build_business_vault_mermaid_prompt(
                                nar[:8000], (qg_reverse_summary or "")[:6000],
                                dm_standards_text=(dm_std_text or "")[:4000],
                            )
                        )
                        sql_raw = _qg_fwd_call(
                            build_business_vault_sql_prompt(
                                nar[:10000], (qg_reverse_summary or "")[:6000],
                                dm_standards_text=(dm_std_text or "")[:6000],
                            )
                        )
                    mer = extract_mermaid_script(mer_raw)
                    sql = extract_sql_ddl(sql_raw)
                    if not nar or not nar.strip():
                        nar = (
                            "## Overview\n"
                            "Business Vault fallback narrative generated. "
                            "Review and refine in the Forward Engineering tab.\n"
                        )
                    if not sql or not sql.strip():
                        sql = (
                            "-- === BUSINESS VAULT (FALLBACK) ===\n"
                            "-- SQL extraction returned empty output.\n"
                        )
                    qg_arts["Business Vault"] = {
                        "kind": "raw_vault",
                        "label": "Business Vault Data Model",
                        "content": {"narrative_md": nar,
                                    "mermaid": mer, "sql": sql,
                                    "mermaid_raw": mer_raw,
                                    "sql_raw": sql_raw},
                    }
                elif step_key == "Forward STTM":
                    bv_nar = qg_arts.get("Business Vault", {}) \
                                  .get("content", {}).get("narrative_md", "")
                    # Same uploaded STTM template that drove Phase 1's
                    # source-to-staging STTM also drives the forward
                    # RV → BV / Semantic STTM, so both artifacts have a
                    # consistent shape.
                    raw = _qg_fwd_call(build_forward_sttm_prompt(
                        bv_nar, qg_reverse_summary, qg_dashboard_type,
                        sttm_template_text=st.session_state.get(
                            "qg_sttm_template_text", ""),
                    ))
                    df = parse_table_response(raw)
                    qg_arts["Forward STTM"] = {
                        "kind": "table",
                        "label": "STTM (RV → BV / Semantic)",
                        "content": {"df": df, "raw": raw},
                    }

                elif step_key == "Forward Catalog":
                    bv_nar = qg_arts.get("Business Vault", {}) \
                                  .get("content", {}).get("narrative_md", "")
                    raw = _qg_fwd_call(build_forward_catalog_prompt(
                        bv_nar,  # use BV as semantic proxy
                        bv_nar,
                        qg_dashboard_type,
                    ))
                    df = parse_table_response(raw)
                    qg_arts["Forward Catalog"] = {
                        "kind": "table",
                        "label": "Data Catalog (Forward)",
                        "content": {"df": df, "raw": raw},
                    }

                elif step_key == "Semantic Data Model":
                    # ─────────────────────────────────────────────────
                    # Dimensional (Kimball star-schema) Semantic Model.
                    # Inputs (per user spec):
                    #   • Forward Engineering Inputs — dashboard spec,
                    #     rules, dashboard type, reverse summary
                    #   • Business Vault Data Model — narrative MD from
                    #     the just-completed BV step
                    #   • Forward Catalog — CSV from the just-completed
                    #     Fwd Catalog step
                    # Output: narrative + Mermaid + DDL artifact, stored
                    # under kind="raw_vault" so the existing renderer
                    # handles it (narrative panel, ER diagram, DDL with
                    # download).
                    # ─────────────────────────────────────────────────
                    bv_nar = qg_arts.get("Business Vault", {}) \
                                  .get("content", {}).get(
                                      "narrative_md", "") or ""
                    fwd_cat_csv = qg_arts.get("Forward Catalog", {}) \
                                       .get("content", {}).get(
                                           "raw", "") or ""
                    try:
                        sem_nar = _qg_fwd_call(build_semantic_model_prompt(
                            qg_dashboard_type, qg_dashboard_text,
                            qg_rules_text, qg_reverse_summary,
                            business_vault_md=bv_nar,
                            forward_catalog_csv=fwd_cat_csv,
                        ))
                        sem_mer_raw = _qg_fwd_call(
                            build_semantic_model_mermaid_prompt(
                                sem_nar, qg_dashboard_type,
                            )
                        )
                        sem_sql_raw = _qg_fwd_call(
                            build_semantic_model_sql_prompt(
                                sem_nar, qg_dashboard_type,
                            )
                        )
                    except Exception as _sm_e:
                        # Retry with reduced context on token pressure.
                        st.warning(
                            "Semantic Data Model generation hit an "
                            "error; retrying with reduced context."
                        )
                        sem_nar = _qg_fwd_call(build_semantic_model_prompt(
                            qg_dashboard_type,
                            (qg_dashboard_text or "")[:4000],
                            (qg_rules_text or "")[:2000],
                            (qg_reverse_summary or "")[:6000],
                            business_vault_md=(bv_nar or "")[:6000],
                            forward_catalog_csv=(
                                fwd_cat_csv or "")[:3000],
                        ))
                        sem_mer_raw = _qg_fwd_call(
                            build_semantic_model_mermaid_prompt(
                                (sem_nar or "")[:8000], qg_dashboard_type,
                            )
                        )
                        sem_sql_raw = _qg_fwd_call(
                            build_semantic_model_sql_prompt(
                                (sem_nar or "")[:10000], qg_dashboard_type,
                            )
                        )
                    sem_mer = extract_mermaid_script(sem_mer_raw)
                    sem_sql = extract_sql_ddl(sem_sql_raw)
                    if not sem_nar or not sem_nar.strip():
                        sem_nar = (
                            "## Overview\n"
                            "Semantic (Dimensional) Model fallback "
                            "narrative generated. Review and refine in "
                            "the Forward Engineering tab.\n"
                        )
                    if not sem_sql or not sem_sql.strip():
                        sem_sql = (
                            "-- === DIMENSIONS (FALLBACK) ===\n"
                            "-- DDL extraction returned empty output.\n"
                        )
                    qg_arts["Semantic Data Model"] = {
                        "kind": "raw_vault",  # reuse existing renderer
                        "label": "Semantic (Dimensional) Data Model",
                        "content": {
                            "narrative_md": sem_nar,
                            "mermaid":      sem_mer,
                            "mermaid_raw":  sem_mer_raw,
                            "sql":          sem_sql,
                            "sql_raw":      sem_sql_raw,
                        },
                    }

                elif step_key == "Raw Vault dbt":
                    # Use the SAME per-file planner the Forward Engineering
                    # tab uses (generate_dbt_project_per_file). The
                    # previous single-call approach via
                    # build_raw_vault_dbt_prompt + parse_dbt_project_from_response
                    # frequently produced empty results because:
                    #   1. The single response would truncate at max_tokens,
                    #      yielding a partial JSON array that parses to {}
                    #   2. Even when it parsed, the storage key was
                    #      content.project but the result-panel renderer
                    #      reads content.files — a structural mismatch
                    #      that made every successful generation appear
                    #      empty.
                    rv_sql = qg_arts.get("Raw Vault Model", {}) \
                                  .get("content", {}).get("sql", "")
                    sttm_csv = qg_arts.get("STTM", {}) \
                                   .get("content", {}).get("raw", "")
                    rv_codegen_ctx = quick_go_raw_vault_dbt_codegen_context(
                        rv_sql, sttm_csv,
                    )

                    def _qg_call_for_dbt(prompt, opts):
                        return call_cortex(
                            qg_model_id, prompt,
                            temperature=opts.get("temperature", 0.1),
                            max_tokens=opts.get("max_tokens", 2500),
                            top_p=opts.get("top_p"),
                        )

                    files, plan_raw, errors = (
                        generate_dbt_project_per_file(
                            _qg_call_for_dbt, rv_codegen_ctx,
                            "raw_vault",
                        )
                    )
                    qg_arts["Raw Vault dbt"] = {
                        "kind": "dbt_project",
                        "label": "Raw Vault dbt Project",
                        "content": {
                            "files":   files,        # {path: content}
                            "raw":     "=== PLAN ===\n" + plan_raw +
                                       ("\n\n=== ERRORS ===\n"
                                        + "\n".join(
                                            f"{p}: {e}"
                                            for p, e in errors)
                                        if errors else ""),
                            "errors":  errors,
                        },
                    }

                elif step_key == "Business Vault dbt":
                    # Same per-file planner as Forward Engineering's
                    # Business Vault dbt step. See note above the Raw
                    # Vault dbt branch for why the single-call approach
                    # was replaced.
                    bv_sql = qg_arts.get("Business Vault", {}) \
                                  .get("content", {}).get("sql", "")
                    bv_nar = qg_arts.get("Business Vault", {}) \
                                  .get("content", {}).get("narrative_md", "")
                    fwd_sttm_csv = qg_arts.get("Forward STTM", {}) \
                                        .get("content", {}).get("raw", "")
                    rv_sql = qg_arts.get("Raw Vault Model", {}) \
                                  .get("content", {}).get("sql", "")
                    bv_codegen_ctx = (
                        quick_go_business_vault_dbt_codegen_context(
                            bv_nar, bv_sql, rv_sql, fwd_sttm_csv,
                        )
                    )

                    def _qg_call_for_bv_dbt(prompt, opts):
                        return call_cortex(
                            qg_model_id, prompt,
                            temperature=opts.get("temperature", 0.1),
                            max_tokens=opts.get("max_tokens", 2500),
                            top_p=opts.get("top_p"),
                        )

                    files, plan_raw, errors = (
                        generate_dbt_project_per_file(
                            _qg_call_for_bv_dbt, bv_codegen_ctx,
                            "business_vault",
                        )
                    )
                    qg_arts["Business Vault dbt"] = {
                        "kind": "dbt_project",
                        "label": "Business Vault dbt Project",
                        "content": {
                            "files":   files,
                            "raw":     "=== PLAN ===\n" + plan_raw +
                                       ("\n\n=== ERRORS ===\n"
                                        + "\n".join(
                                            f"{p}: {e}"
                                            for p, e in errors)
                                        if errors else ""),
                            "errors":  errors,
                        },
                    }

                _qg_set_status(step_key, "done")
            except Exception as e:
                _qg_set_status(step_key, "error")
                st.error(f"Step **{step_key}** failed: {e}")
                with qg_tracker_live2.container():
                    _qg_render_tracker()
                break
        else:
            st.session_state["qg_phase"] = 2
            st.success(
                "✓ Phase ② complete (Business Vault, STTM, Catalog, "
                "Semantic Data Model, RV dbt, BV dbt). Click "
                "**Next → Build/Deploy DBT Project** to ship the "
                "generated dbt projects to Snowflake."
            )
        # Final tracker refresh
        with qg_tracker_live2.container():
            _qg_render_tracker()
        st.rerun()

    # ─────────────────────────────────────────────────────────────────────
    # ── PHASE 3: Next → optionally hand off to Forward Engineering tab
    # ─────────────────────────────────────────────────────────────────────
    if next_clicked:
        # Stage the generated artifacts for the Forward tab's deploy flow.
        # This is now optional — the deploy section below renders inline
        # in Quick GO too, so users don't have to switch tabs. The
        # handoff is kept so users who prefer the Forward tab UX (extra
        # vector-store + version-history controls) can still use it.
        st.session_state.setdefault("fwd_artifacts", {})
        for k in ("Business Vault", "Forward STTM", "Forward Catalog",
                  "Semantic Data Model",
                  "Raw Vault dbt", "Business Vault dbt"):
            if k in st.session_state["qg_artifacts"]:
                st.session_state["fwd_artifacts"][k] = \
                    st.session_state["qg_artifacts"][k]
        # Also stage Quick GO's reverse artifacts so the Forward tab's
        # "selected reverse version" picker has something to point at.
        st.session_state.setdefault("artifacts", {})
        for k in ("Data Lineage", "STTM", "Data Catalog",
                  "Raw Vault Model"):
            if k in st.session_state["qg_artifacts"]:
                st.session_state["artifacts"][k] = \
                    st.session_state["qg_artifacts"][k]

        st.success(
            "✓ Artifacts also copied to Forward Engineering tab. You "
            "can deploy from either tab — they share the same "
            "underlying logic."
        )
        st.session_state["qg_show_all_artifacts"] = True

    # ─────────────────────────────────────────────────────────────────────
    # ── BUILD / DEPLOY DBT PROJECT — same section rendered in Forward tab.
    #    Appears once the user has at least one generated dbt project.
    # ─────────────────────────────────────────────────────────────────────
    _qg_arts_for_deploy = st.session_state.get("qg_artifacts", {}) or {}
    _qg_has_dbt = any(
        v.get("kind") == "dbt_project"
        and (v.get("content", {}) or {}).get("files")
        for v in _qg_arts_for_deploy.values()
    )
    if _qg_has_dbt:
        st.markdown("---")
        # NOTE: render_dbt_deploy_section uses key_prefix to namespace
        # all widget keys; passing 'qg' here keeps it isolated from the
        # Forward tab's section which uses 'fwd'. The same artifacts
        # dict (qg_artifacts) is the source of truth in this tab.
        render_dbt_deploy_section(
            _qg_arts_for_deploy, key_prefix="qg",
        )

    # ─────────────────────────────────────────────────────────────────────
    # ── QUICK GO PERSIST: download bundle, copy to stage, store vectors
    # ─────────────────────────────────────────────────────────────────────
    qg_arts_for_persist = st.session_state.get("qg_artifacts", {}) or {}
    if qg_arts_for_persist:
        st.markdown("---")
        st.markdown("##### Persist Quick GO artifacts")

        qg_stage_version = st.text_input(
            "Version number for Quick GO artifacts",
            value=st.session_state.get("qg_persist_version", "qg-1.0.0"),
            key="qg_persist_version",
        )
        qg_vector_version = st.text_input(
            "Version number for vector storage",
            value=qg_stage_version,
            key="qg_vector_version_in",
        )

        qgc1, qgc2, qgc3 = st.columns(3)

        # Download all as zip
        try:
            qg_bundle = build_artifacts_bundle(
                qg_arts_for_persist,
                source_filename="quickgo",
                metadata_summary=(
                    st.session_state.get("qg_metadata_summary", "") or ""
                ),
            )
            qg_stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            qgc1.download_button(
                f"📦 Download all ({len(qg_arts_for_persist)} artifacts)",
                data=qg_bundle,
                file_name=f"quickgo_artifacts_{qg_stamp}.zip",
                mime="application/zip",
                use_container_width=True,
                key="qg_dl_all_bundle",
            )
        except Exception as e:
            qgc1.warning(f"Bundle error: {e}")

        # Copy to stage
        if qgc2.button(
            "☁ Copy to Snowflake stage",
            use_container_width=True,
            key="qg_upload_stage",
            disabled=not qg_stage_version.strip(),
        ):
            try:
                with st.spinner(
                    f"Ensuring infrastructure "
                    f"({VECTOR_DB}.{VECTOR_SCHEMA}.{VECTOR_STAGE})…"
                ):
                    ensure_vector_infrastructure(session)
                with st.spinner(f"Uploading to v{qg_stage_version}…"):
                    up = upload_artifacts_to_stage(
                        session,
                        qg_arts_for_persist,
                        version=qg_stage_version,
                        source_filename="quickgo",
                        metadata_summary=(
                            st.session_state.get("qg_metadata_summary", "") or ""
                        ),
                    )
                st.success(
                    f"✓ Staged at `{up['path']}` "
                    f"({up.get('landed_bytes', up['bytes']):,} bytes)"
                )
                # Persist last-success info so reruns can show it.
                st.session_state["qg_last_stage_path"] = up["path"]
            except RuntimeError as e:
                # Friendly errors raised by our hardened helpers.
                st.error(f"Stage upload failed: {e}")
            except Exception as e:
                # Anything else — show the underlying exception so the
                # user can copy/paste it to support.
                st.error(
                    f"Stage upload failed (unexpected error): "
                    f"{type(e).__name__}: {e}"
                )

        # Store as vectors
        qg_embed_label = qgc3.selectbox(
            "Embedding model",
            list(EMBED_MODELS.keys()),
            index=0,
            key="qg_vec_model",
            label_visibility="collapsed",
        )
        if not qg_embed_label:
            qg_embed_label = list(EMBED_MODELS.keys())[0]
        qg_embed_model, qg_embed_dim, qg_embed_table = \
            EMBED_MODELS[qg_embed_label]

        if qgc3.button(
            "🧠 Store as vectors",
            use_container_width=True,
            key="qg_vectorize",
            disabled=not qg_vector_version.strip(),
        ):
            try:
                with st.spinner("Preparing chunks and embedding…"):
                    ensure_vector_infrastructure(session)
                    dm = extract_domain_map(qg_arts_for_persist)
                    chunks = chunk_artifacts(qg_arts_for_persist, dm)
                    if not chunks:
                        st.warning("No chunkable artifacts found.")
                    else:
                        inserted = embed_and_store(
                            session,
                            chunks,
                            version=qg_vector_version,
                            embed_model=qg_embed_model,
                            dim=qg_embed_dim,
                            table=qg_embed_table,
                        )
                        st.success(
                            f"✓ Stored {inserted} vector rows in "
                            f"`{VECTOR_DB}.{VECTOR_SCHEMA}.{qg_embed_table}`"
                        )
            except Exception as e:
                st.error(f"Vectorization failed: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # ── RESULT PANELS — collapsible, per step
    # ─────────────────────────────────────────────────────────────────────
    if st.session_state["qg_artifacts"] and (
        st.session_state.get("qg_phase", 0) < 2
        or st.session_state.get("qg_show_all_artifacts", False)
    ):
        st.markdown("---")
        st.markdown("##### 📦 Generated artifacts")
        st.caption(
            "Each completed step expands inline. Click the headers to "
            "collapse / expand."
        )
        qg_show_full_artifact_content = True
        if st.session_state.get("qg_show_all_artifacts", False):
            qg_show_full_artifact_content = st.toggle(
                "Render full artifact details (can be heavy)",
                value=False,
                key="qg_show_all_full_details_toggle",
                help=(
                    "Turn this on to render full tables/diagrams/code for "
                    "every artifact. Keep off for a stable summary view."
                ),
            )
        qg_arts = st.session_state["qg_artifacts"]
        for step in QG_STEPS:
            key = step["key"]
            if key not in qg_arts:
                continue
            art = qg_arts[key]
            label = art.get("label", key)
            kind = art.get("kind", "")
            with st.expander(
                f"{step['icon']} {label}",
                expanded=(
                    st.session_state.get("qg_show_all_artifacts", False)
                    or (
                        st.session_state["qg_step_status"].get(key) == "done"
                        and step["phase"] == st.session_state["qg_phase"]
                    )
                ),
            ):
                content = art.get("content", {})
                if not qg_show_full_artifact_content:
                    if kind == "table":
                        df = content.get("df")
                        if df is not None and not df.empty:
                            st.caption(
                                f"Rows: {len(df):,} | "
                                f"Columns: {len(df.columns):,}"
                            )
                        else:
                            raw_len = len(content.get("raw", "") or "")
                            st.caption(f"No parsed table. Raw size: {raw_len:,} chars")
                    elif kind == "raw_vault":
                        nar = content.get("narrative_md") or ""
                        mer = content.get("mermaid") or ""
                        sql = content.get("sql") or ""
                        if nar:
                            st.markdown(nar[:800] + ("…" if len(nar) > 800 else ""))
                        st.caption(
                            f"Mermaid size: {len(mer):,} chars | "
                            f"SQL size: {len(sql):,} chars"
                        )
                    elif kind == "lineage":
                        graph = content.get("graph") or {}
                        inv_df = content.get("inventory_df")
                        flows_df = content.get("flows_df")
                        inv_n = (len(inv_df)
                                 if inv_df is not None
                                 and hasattr(inv_df, "__len__") else 0)
                        flow_n = (len(flows_df)
                                  if flows_df is not None
                                  and hasattr(flows_df, "__len__") else 0)
                        st.caption(
                            f"Nodes: {len(graph.get('nodes', [])):,} | "
                            f"Edges: {len(graph.get('edges', [])):,} | "
                            f"Inventory rows: {inv_n:,} | "
                            f"Flow rows: {flow_n:,}"
                        )
                    elif kind == "dbt_project":
                        files = content.get("files") or {}
                        errors = content.get("errors") or []
                        st.caption(
                            f"Files: {len(files):,} | Errors: {len(errors):,}"
                        )
                        if files:
                            st.caption(
                                "Sample files: "
                                + ", ".join(sorted(list(files.keys()))[:5])
                            )
                    elif kind == "validation":
                        vj = content.get("json")
                        if isinstance(vj, dict) and vj:
                            st.caption(
                                f"Overall: **{vj.get('overall_score', 0)}/100**  •  "
                                f"{vj.get('readiness_level', 'Unknown')}  •  "
                                f"Violations: {len(vj.get('violations', []) or []):,}"
                            )
                        else:
                            raw_len = len(content.get("raw", "") or "")
                            st.caption(
                                f"Validation report (parse pending). "
                                f"Raw size: {raw_len:,} chars"
                            )
                    elif kind == "markdown":
                        md = content.get("markdown", "") or ""
                        # Count top-level "## " sections as a quick
                        # complexity hint.
                        section_count = sum(
                            1 for ln in md.splitlines()
                            if ln.startswith("## ")
                        )
                        st.caption(
                            f"Markdown size: {len(md):,} chars  •  "
                            f"Sections: {section_count}"
                        )
                    else:
                        st.caption("Artifact generated.")
                    continue
                if kind == "table":
                    df = content.get("df")
                    if df is not None and not df.empty:
                        st.dataframe(df, use_container_width=True,
                                     hide_index=True)
                        st.download_button(
                            f"⬇ Download {label} CSV",
                            df.to_csv(index=False).encode(),
                            file_name=f"qg_{_slug(label)}.csv",
                            mime="text/csv",
                            key=f"qg_dl_{key}",
                        )
                    else:
                        st.warning(
                            "No tabular data parsed. Showing raw output."
                        )
                        st.code(content.get("raw", "")[:5000])

                elif kind == "raw_vault":
                    if content.get("narrative_md"):
                        st.markdown("**Narrative**")
                        st.markdown(content["narrative_md"])
                    if content.get("mermaid"):
                        st.markdown("**ER Diagram**")
                        render_mermaid(content["mermaid"], height=500)
                    if content.get("sql"):
                        st.markdown("**Snowflake DDL**")
                        st.code(content["sql"], language="sql")
                        st.download_button(
                            f"⬇ Download {label} DDL",
                            content["sql"].encode(),
                            file_name=f"qg_{_slug(label)}.sql",
                            mime="text/plain",
                            key=f"qg_dl_sql_{key}",
                        )

                elif kind == "markdown":
                    # Free-form Markdown artifact (e.g. Transformation
                    # Rules & Business Logic). Render the Markdown
                    # natively so headings, code fences, and bullets
                    # display properly. Provide a raw view + downloads.
                    md = content.get("markdown", "") or ""
                    raw_md = content.get("raw", "") or ""
                    if md.strip():
                        st.markdown(md)
                    else:
                        st.warning(
                            "No Markdown produced. Showing raw output."
                        )
                        st.code(raw_md[:5000] or "(empty)",
                                language="text")

                    dl_l, dl_r = st.columns([1, 1])
                    with dl_l:
                        st.download_button(
                            f"⬇ Download {label} (Markdown)",
                            (md or raw_md).encode(),
                            file_name=f"qg_{_slug(label)}.md",
                            mime="text/markdown",
                            key=f"qg_dl_md_{key}",
                        )
                    with dl_r:
                        if st.checkbox(
                            "Show raw model output",
                            key=f"qg_show_md_raw_{key}",
                        ):
                            st.code(raw_md[:8000] or "(empty)",
                                    language="markdown")

                elif kind == "lineage":
                    mer = (content.get("mermaid") or "").strip()
                    graph = content.get("graph") or {}
                    inv_df = content.get("inventory_df")
                    flows_df = content.get("flows_df")

                    # Sub-tabs: graphical views first, then the two
                    # tabular views derived from the parsed Reverse
                    # Engineering Inputs (no Raw Vault dependency).
                    qg_lg_tab1, qg_lg_tab2, qg_lg_tab3, qg_lg_tab4 = (
                        st.tabs([
                            "🕸 Interactive Graph",
                            "🖼 Mermaid Diagram",
                            "📋 Entity & Column Inventory",
                            "🔁 Stage-to-Stage Flows",
                        ])
                    )

                    with qg_lg_tab1:
                        if graph and graph.get("nodes"):
                            try:
                                render_interactive_lineage(
                                    graph, height=720,
                                )
                                st.caption(
                                    f"Graph: "
                                    f"**{len(graph.get('nodes', []))}"
                                    f" nodes**, "
                                    f"**{len(graph.get('edges', []))}"
                                    f" edges**"
                                )
                            except Exception as e:
                                st.error(
                                    f"Interactive renderer error: {e}"
                                )
                        else:
                            st.info(
                                "No nodes parsed from the uploaded "
                                "source files."
                            )

                    with qg_lg_tab2:
                        if mer and "flowchart" in mer:
                            render_mermaid(mer, height=600)
                        else:
                            st.info(
                                "No Mermaid diagram could be built from "
                                "the uploaded source files."
                            )

                    with qg_lg_tab3:
                        # Entity & Column Inventory — column-level
                        # lineage anchored on the source artifacts.
                        if inv_df is not None and not inv_df.empty:
                            st.caption(
                                f"Rows: {len(inv_df):,}  •  "
                                f"Files: "
                                f"{inv_df['Source File'].nunique():,}  •  "
                                f"Entities: "
                                f"{inv_df['Entity / Stage'].nunique():,}  •  "
                                f"Columns: "
                                f"{inv_df['Column'].replace('', pd.NA).dropna().nunique():,}"
                            )
                            st.dataframe(
                                inv_df, use_container_width=True,
                                hide_index=True,
                            )
                            st.download_button(
                                "⬇ Download Inventory CSV",
                                inv_df.to_csv(index=False).encode(),
                                file_name=(
                                    f"qg_{_slug(label)}_inventory.csv"
                                ),
                                mime="text/csv",
                                key=f"qg_dl_lg_inv_{key}",
                            )
                        else:
                            st.info(
                                "No entities/columns were parsed from "
                                "the uploaded source files."
                            )

                    with qg_lg_tab4:
                        # Stage-to-Stage Flows — file→stage, within-file,
                        # and cross-file dataflow edges.
                        if flows_df is not None and not flows_df.empty:
                            ft = flows_df["Flow Type"]
                            st.caption(
                                f"Rows: {len(flows_df):,}  •  "
                                f"file → stage: "
                                f"{(ft == 'file → stage').sum():,}  •  "
                                f"Within-file: "
                                f"{(ft == 'within-file').sum():,}  •  "
                                f"Cross-file: "
                                f"{(ft == 'cross-file').sum():,}"
                            )
                            st.dataframe(
                                flows_df, use_container_width=True,
                                hide_index=True,
                            )
                            st.download_button(
                                "⬇ Download Flows CSV",
                                flows_df.to_csv(index=False).encode(),
                                file_name=(
                                    f"qg_{_slug(label)}_flows.csv"
                                ),
                                mime="text/csv",
                                key=f"qg_dl_lg_flows_{key}",
                            )
                        else:
                            st.caption(
                                "No flow rows could be derived from the "
                                "uploaded source files. This usually "
                                "means the parsers did not extract any "
                                "entities. Check the Entity & Column "
                                "Inventory tab — if it is also empty, "
                                "the file content was not recognized "
                                "by the available parsers."
                            )

                elif kind == "validation":
                    # ─────────────────────────────────────────────────
                    # GenAI Raw Vault Validation report. Rendered as a
                    # multi-section dashboard:
                    #   1. Headline metrics (overall score, readiness)
                    #   2. Scorecard table (7 weighted dimensions)
                    #   3. Layer 1 — Source Interpretation Assessment
                    #   4. Layer 2 — Structural Compliance summary
                    #   5. Rule Violations Register
                    #   6. Business Key Confidence Matrix
                    #   7. Satellite Design Quality
                    #   8. Relationship Integrity
                    #   9. Lineage Completeness Matrix
                    #  10. Hash Standardization
                    #  11. Remediation Recommendations
                    #  12. Final Deliverables Checklist
                    #  13. Raw JSON / fallback raw text + downloads
                    # ─────────────────────────────────────────────────
                    _vj_raw = content.get("json")
                    vj = _vj_raw if isinstance(_vj_raw, dict) else {}
                    raw_blob = content.get("raw", "") or ""
                    parse_err = content.get("parse_error", "") or ""

                    if not vj:
                        # Parse failed — show raw response so user can
                        # inspect / debug. Most common cause is the
                        # model wrapping JSON in extra prose.
                        st.error(
                            "Could not parse the validation report as "
                            "JSON. Showing raw output below."
                            + (f" (Error: {parse_err})" if parse_err else "")
                        )
                        st.code(raw_blob[:8000] or "(empty)", language="json")
                        st.download_button(
                            f"⬇ Download {label} (raw)",
                            (raw_blob or "").encode(),
                            file_name=f"qg_{_slug(label)}_raw.txt",
                            mime="text/plain",
                            key=f"qg_dl_val_raw_{key}",
                        )
                    else:
                        # ── Headline metrics ─────────────────────────
                        overall = int(vj.get("overall_score", 0) or 0)
                        level = vj.get("readiness_level", "Unknown")
                        # Color the readiness banner by band.
                        if overall >= 90:
                            banner_color = "#22c55e"  # green
                            banner_emoji = "✅"
                        elif overall >= 75:
                            banner_color = "#84cc16"  # lime
                            banner_emoji = "🟢"
                        elif overall >= 60:
                            banner_color = "#eab308"  # amber
                            banner_emoji = "⚠"
                        else:
                            banner_color = "#ef4444"  # red
                            banner_emoji = "❌"

                        st.markdown(
                            f"<div style='padding:14px 18px; "
                            f"border-radius:10px; "
                            f"border:1.5px solid {banner_color}; "
                            f"background:rgba(120,120,120,0.06); "
                            f"margin-bottom:8px;'>"
                            f"<span style='font-size:22px;'>"
                            f"{banner_emoji}</span>"
                            f"&nbsp;&nbsp;<span style='font-size:24px; "
                            f"font-weight:700; color:{banner_color};'>"
                            f"{overall}/100</span>"
                            f"&nbsp;&nbsp;<span style='font-size:16px; "
                            f"font-weight:600;'>{level}</span>"
                            f"</div>",
                            unsafe_allow_html=True,
                        )

                        if vj.get("summary"):
                            st.markdown(
                                f"**Executive summary.** "
                                f"{vj.get('summary')}"
                            )

                        # ── 1. Scorecard ──────────────────────────────
                        sc = vj.get("scorecard") or []
                        if sc:
                            st.markdown("**📊 Production Readiness Scorecard**")
                            try:
                                sc_df = pd.DataFrame([{
                                    "Area":     row.get("area", ""),
                                    "Weight":   row.get("weight", 0),
                                    "Score":    row.get("score", 0),
                                    "Weighted": row.get("weighted", 0),
                                    "Comment":  row.get("comment", ""),
                                } for row in sc])
                                st.dataframe(
                                    sc_df, use_container_width=True,
                                    hide_index=True,
                                )
                            except Exception:
                                st.json(sc)

                        # ── 2. Source Interpretation Assessment ───────
                        si = vj.get("source_interpretation") or {}
                        if si:
                            st.markdown(
                                "**🔎 Layer 1 — Source Interpretation "
                                "Assessment**"
                            )
                            si_checks = [
                                ("Tables captured",
                                 si.get("tables_captured")),
                                ("Primary keys identified",
                                 si.get("primary_keys_identified")),
                                ("Business keys inferred",
                                 si.get("business_keys_inferred")),
                                ("FK relationships recognized",
                                 si.get("fk_relationships_recognized")),
                                ("Grain understood",
                                 si.get("grain_understood")),
                                ("CDC behavior understood",
                                 si.get("cdc_behavior_understood")),
                            ]
                            check_cols = st.columns(3)
                            for idx, (lbl, ok) in enumerate(si_checks):
                                with check_cols[idx % 3]:
                                    icon = "✅" if ok else "❌"
                                    st.caption(f"{icon} {lbl}")
                            lc_items = si.get("low_confidence_items") or []
                            me_items = si.get("missing_entities") or []
                            if lc_items:
                                st.markdown(
                                    "_Low-confidence inferences:_"
                                )
                                for it in lc_items:
                                    st.caption(f"• {it}")
                            if me_items:
                                st.markdown("_Missing entities:_")
                                for it in me_items:
                                    st.caption(f"• {it}")

                        # ── 3. Structural Compliance summary ──────────
                        struct = vj.get("structural_compliance") or {}
                        if struct:
                            st.markdown(
                                "**🏛 Layer 2 — Structural Compliance**"
                            )
                            sc_cols = st.columns(3)
                            sc_cols[0].caption(
                                ("✅" if struct.get("hubs_pass")
                                 else "❌") + " Hubs"
                            )
                            sc_cols[1].caption(
                                ("✅" if struct.get("links_pass")
                                 else "❌") + " Links"
                            )
                            sc_cols[2].caption(
                                ("✅" if struct.get("satellites_pass")
                                 else "❌") + " Satellites"
                            )
                            mm = struct.get("missing_metadata_columns") or []
                            if mm:
                                st.caption("_Missing metadata columns:_")
                                try:
                                    st.dataframe(
                                        pd.DataFrame([{
                                            "Entity": r.get("entity", ""),
                                            "Missing": ", ".join(
                                                r.get("missing", []) or []
                                            ),
                                        } for r in mm]),
                                        use_container_width=True,
                                        hide_index=True,
                                    )
                                except Exception:
                                    st.json(mm)
                            ap = struct.get("anti_patterns") or []
                            if ap:
                                st.caption("_Anti-patterns observed:_")
                                for it in ap:
                                    st.caption(f"• {it}")

                        # ── 4. Rule Violations Register ───────────────
                        viols = vj.get("violations") or []
                        if viols:
                            st.markdown(
                                f"**🚨 Rule Violations Register "
                                f"({len(viols)})**"
                            )
                            try:
                                v_df = pd.DataFrame([{
                                    "Severity":    v.get("severity", ""),
                                    "Rule":        v.get("rule", ""),
                                    "Entity":      v.get("entity", ""),
                                    "Description": v.get("description", ""),
                                    "Remediation": v.get("remediation", ""),
                                } for v in viols])
                                # Sort by severity for at-a-glance triage
                                sev_order = {"High": 0, "Medium": 1,
                                             "Low": 2}
                                v_df["_sev"] = v_df["Severity"].map(
                                    lambda s: sev_order.get(str(s), 9)
                                )
                                v_df = v_df.sort_values("_sev").drop(
                                    columns=["_sev"]
                                )
                                st.dataframe(
                                    v_df, use_container_width=True,
                                    hide_index=True,
                                )
                                st.download_button(
                                    f"⬇ Download Violations CSV",
                                    v_df.to_csv(index=False).encode(),
                                    file_name=(
                                        f"qg_{_slug(label)}_violations.csv"
                                    ),
                                    mime="text/csv",
                                    key=f"qg_dl_val_viols_{key}",
                                )
                            except Exception:
                                st.json(viols)
                        else:
                            st.success(
                                "No rule violations reported by the "
                                "validator."
                            )

                        # ── 5. Business Key Confidence Matrix ─────────
                        bkeys = vj.get("business_keys") or []
                        if bkeys:
                            st.markdown(
                                "**🔑 Business Key Confidence Matrix**"
                            )
                            try:
                                bk_df = pd.DataFrame([{
                                    "Hub":          b.get("hub", ""),
                                    "Business Key": b.get("business_key", ""),
                                    "Confidence":   b.get("confidence", ""),
                                    "Stability":    b.get("stability", ""),
                                    "Uniqueness":   b.get("uniqueness", ""),
                                    "Cross-System": b.get("cross_system", ""),
                                    "Non-Null":     b.get("non_nullable", ""),
                                    "Risk":         b.get("risk", ""),
                                    "Alternatives": ", ".join(
                                        b.get("alternatives", []) or []
                                    ),
                                    "Notes":        b.get("notes", ""),
                                } for b in bkeys])
                                st.dataframe(
                                    bk_df, use_container_width=True,
                                    hide_index=True,
                                )
                            except Exception:
                                st.json(bkeys)

                        # ── 6. Satellite Design Quality ───────────────
                        sats = vj.get("satellite_quality") or []
                        if sats:
                            st.markdown(
                                "**🛰 Satellite Design Quality**"
                            )
                            try:
                                sat_df = pd.DataFrame([{
                                    "Satellite":       s.get(
                                        "satellite", ""),
                                    "By Source":       s.get(
                                        "grouped_by_source", ""),
                                    "By Rate":         s.get(
                                        "grouped_by_rate", ""),
                                    "Single Source":   s.get(
                                        "single_record_source", ""),
                                    "Issues":          "; ".join(
                                        s.get("issues", []) or []),
                                    "Recommendation":  s.get(
                                        "recommendation", ""),
                                } for s in sats])
                                st.dataframe(
                                    sat_df, use_container_width=True,
                                    hide_index=True,
                                )
                            except Exception:
                                st.json(sats)

                        # ── 7. Relationship Integrity ─────────────────
                        rel = vj.get("relationship_integrity") or {}
                        if rel:
                            st.markdown(
                                "**🔗 Relationship Integrity**"
                            )
                            rel_checks = [
                                ("Links → valid Hubs",
                                 rel.get("all_links_reference_valid_hubs")),
                                ("Cardinality modeled",
                                 rel.get("cardinality_modeled")),
                                ("Many-to-many via Links",
                                 rel.get("many_to_many_via_links")),
                                ("Recursive Links modeled",
                                 rel.get("recursive_links_modeled")),
                                ("Txn vs static distinguished",
                                 rel.get(
                                     "transactional_vs_static_distinguished")),
                            ]
                            rel_cols = st.columns(3)
                            for idx, (lbl, ok) in enumerate(rel_checks):
                                with rel_cols[idx % 3]:
                                    icon = "✅" if ok else "❌"
                                    st.caption(f"{icon} {lbl}")
                            rel_issues = rel.get("issues") or []
                            if rel_issues:
                                st.caption("_Issues:_")
                                for it in rel_issues:
                                    st.caption(f"• {it}")

                        # ── 8. Lineage Completeness Matrix ────────────
                        lin = vj.get("lineage_completeness") or {}
                        if lin:
                            st.markdown(
                                "**🧬 Lineage Completeness Matrix**"
                            )
                            try:
                                lin_rows = [
                                    ("Hubs traceable to source",
                                     lin.get("hubs_traceable_to_source", "")),
                                    ("Links traceable to relationships",
                                     lin.get(
                                         "links_traceable_to_relationships",
                                         "")),
                                    ("Satellite attributes traceable",
                                     lin.get("satellite_attrs_traceable",
                                             "")),
                                    ("Transformations documented",
                                     lin.get("transformations_documented",
                                             "")),
                                    ("Column-level lineage",
                                     lin.get("column_level_lineage", "")),
                                ]
                                lin_df = pd.DataFrame(
                                    lin_rows,
                                    columns=["Dimension", "Status"],
                                )
                                st.dataframe(
                                    lin_df, use_container_width=True,
                                    hide_index=True,
                                )
                                pct = lin.get("completeness_pct")
                                if pct is not None:
                                    try:
                                        pct_v = float(pct) / 100.0
                                        if 0.0 <= pct_v <= 1.0:
                                            st.progress(
                                                pct_v,
                                                text=(
                                                    f"Lineage completeness: "
                                                    f"{int(float(pct))}%"
                                                ),
                                            )
                                    except Exception:
                                        pass
                                orphans = lin.get(
                                    "orphan_lineage_nodes") or []
                                if orphans:
                                    st.caption("_Orphan lineage nodes:_")
                                    for it in orphans:
                                        st.caption(f"• {it}")
                            except Exception:
                                st.json(lin)

                        # ── 9. Hash Standardization ───────────────────
                        hsh = vj.get("hash_standardization") or {}
                        if hsh:
                            st.markdown(
                                "**# Hash Standardization**"
                            )
                            hsh_cols = st.columns(2)
                            with hsh_cols[0]:
                                st.caption(
                                    f"**Algorithm:** "
                                    f"{hsh.get('algorithm', 'unknown')}"
                                )
                                st.caption(
                                    f"**Delimiter:** "
                                    f"{hsh.get('delimiter', '')}"
                                )
                                st.caption(
                                    f"**Null replacement:** "
                                    f"{hsh.get('null_replacement', '')}"
                                )
                            with hsh_cols[1]:
                                st.caption(
                                    ("✅" if hsh.get(
                                        "deterministic_ordering")
                                     else "❌") + " Deterministic ordering"
                                )
                                st.caption(
                                    ("✅" if hsh.get("type_normalization")
                                     else "❌") + " Type normalization"
                                )
                            hsh_issues = hsh.get("issues") or []
                            if hsh_issues:
                                st.caption("_Issues:_")
                                for it in hsh_issues:
                                    st.caption(f"• {it}")

                        # ── 10. Remediation Recommendations ───────────
                        recs = vj.get("remediation_recommendations") or []
                        if recs:
                            st.markdown(
                                f"**🛠 Remediation Recommendations "
                                f"({len(recs)})**"
                            )
                            try:
                                rec_df = pd.DataFrame([{
                                    "Priority": r.get("priority", ""),
                                    "Area":     r.get("area", ""),
                                    "Action":   r.get("action", ""),
                                } for r in recs])
                                pri_order = {"High": 0, "Medium": 1,
                                             "Low": 2}
                                rec_df["_p"] = rec_df["Priority"].map(
                                    lambda s: pri_order.get(str(s), 9)
                                )
                                rec_df = rec_df.sort_values("_p").drop(
                                    columns=["_p"]
                                )
                                st.dataframe(
                                    rec_df, use_container_width=True,
                                    hide_index=True,
                                )
                            except Exception:
                                st.json(recs)

                        # ── 11. Final Deliverables Checklist ──────────
                        ck = vj.get("deliverables_checklist") or {}
                        if ck:
                            st.markdown(
                                "**📦 Final Deliverables**"
                            )
                            ck_items = [
                                ("Validation Report",
                                 ck.get("validation_report")),
                                ("Rule Violations Register",
                                 ck.get("rule_violations_register")),
                                ("Business Key Confidence Matrix",
                                 ck.get("business_key_confidence_matrix")),
                                ("Lineage Completeness Matrix",
                                 ck.get("lineage_completeness_matrix")),
                                ("Remediation Recommendations",
                                 ck.get("remediation_recommendations")),
                                ("Production Readiness Scorecard",
                                 ck.get("production_readiness_scorecard")),
                            ]
                            ck_cols = st.columns(3)
                            for idx, (lbl, ok) in enumerate(ck_items):
                                with ck_cols[idx % 3]:
                                    icon = "✅" if ok else "○"
                                    st.caption(f"{icon} {lbl}")

                        # ── Downloads ────────────────────────────────
                        try:
                            json_bytes = json.dumps(
                                vj, indent=2
                            ).encode("utf-8")
                        except Exception:
                            json_bytes = (raw_blob or "").encode("utf-8")
                        st.download_button(
                            f"⬇ Download {label} (JSON)",
                            json_bytes,
                            file_name=f"qg_{_slug(label)}.json",
                            mime="application/json",
                            key=f"qg_dl_val_json_{key}",
                        )
                        if st.checkbox(
                            "Show raw validator output",
                            key=f"qg_show_val_raw_{key}",
                        ):
                            st.code(raw_blob[:10000], language="json")

                elif kind == "dbt_project":
                    files = content.get("files") or {}
                    errors = content.get("errors") or []
                    file_count = len(files) if isinstance(files, dict) else 0

                    if file_count == 0:
                        st.error(
                            "❌ dbt project generation produced no files. "
                            "Inspect the planner output below for clues."
                        )
                        plan_blob = content.get("raw", "")
                        # NOTE: Streamlit forbids nested expanders, so we
                        # use a checkbox toggle instead — this whole
                        # branch is already inside the per-step expander.
                        if plan_blob and st.checkbox(
                            "Show planner output (raw)",
                            key=f"qg_show_plan_{key}",
                        ):
                            st.code(plan_blob[:6000], language="text")
                    else:
                        st.success(
                            f"✅ dbt project generated with "
                            f"**{file_count} files**."
                        )

                        # Files-by-folder breakdown — flat (no nested
                        # expanders allowed). We render a compact
                        # dataframe so the user can scroll and search.
                        from collections import defaultdict as _qg_dd2
                        by_folder = _qg_dd2(list)
                        for path in sorted(files.keys()):
                            folder = (
                                path.split("/", 1)[0]
                                if "/" in path else "(root)"
                            )
                            by_folder[folder].append(path)

                        # Folder summary line
                        folder_summary = "  •  ".join(
                            f"`{folder}/` ({len(by_folder[folder])})"
                            for folder in sorted(by_folder.keys())
                        )
                        st.caption(f"**By folder:**  {folder_summary}")

                        # All files as a sortable dataframe
                        try:
                            import pandas as _qg_pd
                            file_df = _qg_pd.DataFrame([
                                {
                                    "path": p,
                                    "size (lines)": (
                                        len(files[p].splitlines())
                                        if isinstance(files[p], str)
                                        else 0
                                    ),
                                    "size (chars)": (
                                        len(files[p])
                                        if isinstance(files[p], str)
                                        else 0
                                    ),
                                }
                                for p in sorted(files.keys())
                            ])
                            st.dataframe(
                                file_df,
                                use_container_width=True,
                                hide_index=True,
                                height=min(
                                    400, 40 + 35 * min(len(files), 10),
                                ),
                            )
                        except Exception:
                            # Fallback: plain list
                            for p in sorted(files.keys()):
                                st.caption(f"`{p}`")

                        # Per-file preview via selectbox (still flat)
                        preview_path = st.selectbox(
                            "Preview a file",
                            options=["(none)"] + sorted(files.keys()),
                            key=f"qg_preview_{key}",
                        )
                        if preview_path and preview_path != "(none)":
                            body = files.get(preview_path) or ""
                            lang = (
                                "sql" if preview_path.endswith(".sql")
                                else "yaml" if preview_path.endswith(
                                    (".yml", ".yaml")
                                )
                                else "text"
                            )
                            st.code(body[:6000], language=lang)
                            if len(body) > 6000:
                                st.caption(
                                    f"(truncated to 6000 of "
                                    f"{len(body)} chars)"
                                )

                        # Build a zip of all files for one-click download
                        try:
                            import io as _qg_io
                            import zipfile as _qg_zf
                            buf = _qg_io.BytesIO()
                            with _qg_zf.ZipFile(
                                buf, "w", _qg_zf.ZIP_DEFLATED,
                            ) as zf:
                                for path, body in files.items():
                                    zf.writestr(path, body or "")
                            buf.seek(0)
                            st.download_button(
                                f"⬇ Download {label} (.zip)",
                                buf.getvalue(),
                                file_name=f"qg_{_slug(label)}.zip",
                                mime="application/zip",
                                key=f"qg_dl_zip_{key}",
                            )
                        except Exception as e:
                            st.caption(
                                f"(Download zip failed: {e})"
                            )

                    if errors:
                        # Flat (no nested expander). Use a checkbox to
                        # toggle the error list visibility.
                        if st.checkbox(
                            f"⚠ Show {len(errors)} file error(s)",
                            key=f"qg_show_errs_{key}",
                        ):
                            for p, err in errors[:20]:
                                st.caption(f"`{p}` — {str(err)[:200]}")



# ═════════════════════════════════════════════════════════════════════════════
# REVERSE ENGINEERING TAB
# ═════════════════════════════════════════════════════════════════════════════
def _render_tab_reverse():
    st.markdown("#### Upload metadata")
    st.caption(
        "Upload legacy code from any combination of 8 tech stacks. Files "
        "are grouped per tech stack so the parser can apply the right "
        "rules. Optionally tag each file with one or more bank divisions, "
        "and seed the analysis with an existing STTM or attribute mapping "
        "CSV. Cross-file and cross-tech relationships are detected via "
        "shared column names."
    )

    # ── Bank divisions: free-text tags users add per session, then assign
    # to individual files via multiselect.
    divisions_str = st.text_input(
        "Bank divisions (comma-separated tags) — optional",
        placeholder="Retail, Commercial, Wealth, Cards, …",
        key="rev_divisions_input",
    )
    division_options = [
        d.strip() for d in (divisions_str or "").split(",") if d.strip()
    ]

    # ── 8 tech-stack uploaders. Each accepts multiple files. The
    # `tech` tag is recorded per file for downstream parser dispatch.
    tech_zones = [
        ("datastage",
         "IBM DataStage (.dsx, .xml, sequence)",
         ["dsx", "xml", "txt"]),
        ("bods",
         "SAP BODS (.xml)",
         ["xml"]),
        ("legacy_sql",
         "Legacy SQL (Oracle/DB2/Teradata .sql)",
         ["sql", "txt"]),
        ("netezza",
         "Netezza SQL (.sql)",
         ["sql", "txt"]),
        ("controlm",
         "Control-M jobs (.xml, .ctm, .txt)",
         ["xml", "ctm", "txt"]),
        ("shell",
         "Shell scripts (.sh, .ksh, .bash)",
         ["sh", "ksh", "bash", "txt"]),
        ("ssis",
         "SSIS packages (.dtsx)",
         ["dtsx", "xml"]),
        ("denodo",
         "Denodo VQL (.vql)",
         ["vql", "txt"]),
        ("mssql",
         "MS SQL Server (.sql)",
         ["sql", "txt"]),
    ]

    tagged_files = []  # [{tech, file, divisions}]
    cols = st.columns(3)
    for idx, (tech, label, exts) in enumerate(tech_zones):
        with cols[idx % 3]:
            files_for_tech = st.file_uploader(
                label,
                type=exts,
                accept_multiple_files=True,
                key=f"rev_upl_{tech}",
            )
            if files_for_tech:
                for uf in files_for_tech:
                    sel_divs = (
                        st.multiselect(
                            f"Divisions for `{uf.name}`",
                            options=division_options,
                            default=[],
                            key=f"rev_div_{tech}_{uf.name}",
                        )
                        if division_options else []
                    )
                    tagged_files.append({
                        "tech": tech,
                        "file": uf,
                        "divisions": sel_divs,
                    })

    # ── 2 reference uploaders: STTM and attribute-mapping CSVs.
    st.markdown("###### Reference inputs (optional, used as ground truth)")
    ref_col_l, ref_col_r = st.columns(2)
    with ref_col_l:
        sttm_seed_file = st.file_uploader(
            "Existing STTM CSV (seeds analysis)",
            type=["csv"],
            accept_multiple_files=False,
            key="rev_sttm_seed",
        )
    with ref_col_r:
        mapping_seed_file = st.file_uploader(
            "Attribute mapping CSV (seeds catalog)",
            type=["csv"],
            accept_multiple_files=False,
            key="rev_mapping_seed",
        )

    # Stash seed CSVs in session state so the four artifact buttons
    # (Lineage, STTM, Catalog, Raw Vault) can each consume them.
    if sttm_seed_file is not None:
        try:
            seed_text = sttm_seed_file.read().decode(
                "utf-8", errors="ignore",
            )
            st.session_state["rev_sttm_seed_parsed"] = parse_sttm_csv(
                seed_text,
            )
            st.success(
                f"✓ STTM seed loaded: "
                f"{st.session_state['rev_sttm_seed_parsed']['row_count']} "
                f"rows. The LLM will treat this as ground truth and "
                f"extend rather than override."
            )
        except Exception as e:
            st.warning(f"STTM seed failed to parse: {e}")
    if mapping_seed_file is not None:
        try:
            seed_text = mapping_seed_file.read().decode(
                "utf-8", errors="ignore",
            )
            st.session_state["rev_mapping_seed_parsed"] = (
                parse_metadata_csv(seed_text)
            )
            st.success(
                f"✓ Mapping seed loaded: "
                f"{st.session_state['rev_mapping_seed_parsed']['row_count']} "
                f"rows."
            )
        except Exception as e:
            st.warning(f"Mapping seed failed to parse: {e}")

    if tagged_files:
        # Entities are keyed by COMPOSITE id ({filename}::{job_name}::{stage_name})
        # so that stages with the same name in different jobs/files don't
        # collapse into each other. The human-readable name lives in
        # `display_name` and is what surfaces in shared-column detection
        # and the LLM summary. This preserves cross-file business-key
        # detection (columns match on value, not on entity key) while
        # stopping unrelated stages from silently merging.
        all_parsed = []   # [{filename, kind, tech, divisions, parsed}]
        all_entities = {} # entity_key -> {display_name, columns, source_file,
                          #                type, job_name, tech, divisions}

        # Metadata-ignore regex — extracted once, reused everywhere below
        META_COL_RE = re.compile(
            r'(LOAD_|CREATED_|UPDATED_|LAST_|DW_|ETL_|EFFECTIVE_|'
            r'DTS$|_DTS|_TS$|TIMESTAMP|FILLER)'
        )

        # Tech → parser dispatch table. Each parser returns the same
        # {jobs, stages, links, columns} shape, so the entity-extraction
        # logic below is uniform.
        TECH_PARSERS = {
            "datastage": parse_datastage_dsx,
            "bods":      parse_bods_xml,
            "legacy_sql": parse_legacy_sql,
            "netezza":   parse_netezza_sql,
            "mssql":     parse_mssql_sql,
            "controlm":  parse_control_m,
            "shell":     parse_shell_script,
            "ssis":      parse_ssis_dtsx,
            "denodo":    parse_denodo_vql,
        }

        with st.spinner(
            f"Parsing {len(tagged_files)} file(s) across "
            f"{len({t['tech'] for t in tagged_files})} tech stack(s)…"
        ):
            for tagged in tagged_files:
                uf = tagged["file"]
                tech = tagged["tech"]
                divisions = tagged["divisions"]
                try:
                    content = uf.read().decode("utf-8", errors="ignore")
                except Exception:
                    continue

                parser = TECH_PARSERS.get(tech)
                if not parser:
                    continue
                parsed = parser(content)
                all_parsed.append({
                    "filename":  uf.name,
                    "kind":      tech,
                    "tech":      tech,
                    "divisions": divisions,
                    "parsed":    parsed,
                })

                # Build a (job, stage) -> entity_key map for this file so
                # columns attach to the right composite entity. Key is
                # unique per (file, job, stage).
                stage_key_map = {}
                for s in parsed.get("stages", []):
                    key = f"{uf.name}::{s['job_name']}::{s['stage_name']}"
                    stage_key_map[
                        (s["job_name"], s["stage_name"])
                    ] = key
                    all_entities.setdefault(key, {
                        "display_name":  s["stage_name"],
                        "columns":       set(),
                        "columns_typed": {},
                        "source_file":   uf.name,
                        "type":          s["stage_type"],
                        "job_name":      s["job_name"],
                        "tech":          tech,
                        "divisions":     divisions,
                    })

                for c in parsed.get("columns", []):
                    key = stage_key_map.get(
                        (c["job_name"], c["stage_name"])
                    )
                    if key and key in all_entities:
                        col_name = c["column_name"].upper()
                        all_entities[key]["columns"].add(col_name)
                        resolved = resolve_sql_type(
                            c.get("sql_type", ""),
                            c.get("precision", ""),
                            c.get("scale", ""),
                        )
                        if resolved and not all_entities[key][
                            "columns_typed"
                        ].get(col_name):
                            all_entities[key]["columns_typed"][
                                col_name
                            ] = resolved

            # Stash for the four artifact buttons (Lineage / STTM /
            # Catalog / Raw Vault) which will read this and the seed
            # CSVs from session state.
            st.session_state["rev_tagged_files"] = tagged_files
            st.session_state["rev_all_parsed"] = all_parsed
            st.session_state["rev_all_entities"] = all_entities

            # ── Detect relationships at THREE levels ──────────────────
            # 1. Pairwise shared columns between entities
            relationships = []
            entity_keys = list(all_entities.keys())
            for i, a in enumerate(entity_keys):
                for b in entity_keys[i + 1:]:
                    shared = (all_entities[a]["columns"]
                              & all_entities[b]["columns"])
                    shared = {c for c in shared if not META_COL_RE.match(c)}
                    if shared:
                        relationships.append({
                            "entity_a":       all_entities[a]["display_name"],
                            "entity_b":       all_entities[b]["display_name"],
                            "entity_a_key":   a,
                            "entity_b_key":   b,
                            "shared_columns": sorted(shared),
                            "file_a":         all_entities[a]["source_file"],
                            "file_b":         all_entities[b]["source_file"],
                        })

            # 2. Column → entity clusters (TRANSITIVE relationship detection)
            # If a column appears in 2+ entities, it's likely a business
            # key. We use DISPLAY names here so the cluster reads
            # meaningfully in the LLM summary (same logical table across
            # files still shows up as one name).
            column_to_entities = {}  # col_name -> set of display_names
            for ent_key, meta in all_entities.items():
                for col in meta["columns"]:
                    if META_COL_RE.match(col):
                        continue
                    column_to_entities.setdefault(col, set()).add(
                        meta["display_name"]
                    )
            shared_column_clusters = {
                col: sorted(names)
                for col, names in column_to_entities.items()
                if len(names) >= 2
            }

            # 3. Cross-job dataflows:
            #    (a) within-file: stage A → stage B where A and B live in
            #        different jobs in the same file
            #    (b) cross-file: terminal stage in file X shares columns
            #        with a stage in file Y — implicit dataflow via a
            #        shared landing table
            cross_job_flows = []

            # (a) within-file cross-job links
            for f in all_parsed:
                if f["kind"] != "datastage":
                    continue
                p = f["parsed"]
                # Stage name -> job. If a file has two stages with the
                # same name in different jobs, this keeps the LAST one,
                # which is acceptable for this within-file heuristic.
                stage_to_job = {s["stage_name"]: s["job_name"]
                                for s in p["stages"]}
                for l in p["links"]:
                    fj = stage_to_job.get(l["from_stage"])
                    tj = stage_to_job.get(l["to_stage"])
                    if fj and tj and fj != tj:
                        cross_job_flows.append({
                            "from_stage": l["from_stage"],
                            "from_job":   fj,
                            "to_stage":   l["to_stage"],
                            "to_job":     tj,
                            "file":       f["filename"],
                            "kind":       "within-file",
                        })

            # (b) cross-file dataflows: if stage A in file1 shares ≥1
            # non-metadata column with stage B in file2, and the two
            # entities have high column overlap (≥30% of the smaller
            # side), mark it as a likely cross-file flow. This catches
            # the "job 1 writes to CUSTOMER landing, job 2 reads from
            # CUSTOMER landing" pattern where no explicit link exists.
            for r in relationships:
                if r["file_a"] == r["file_b"]:
                    continue
                ka, kb = r["entity_a_key"], r["entity_b_key"]
                cols_a = all_entities[ka]["columns"]
                cols_b = all_entities[kb]["columns"]
                smaller = min(len(cols_a), len(cols_b)) or 1
                overlap_ratio = len(r["shared_columns"]) / smaller
                if overlap_ratio >= 0.3 and len(r["shared_columns"]) >= 2:
                    cross_job_flows.append({
                        "from_stage": all_entities[ka]["display_name"],
                        "from_job":   all_entities[ka]["job_name"] or "—",
                        "to_stage":   all_entities[kb]["display_name"],
                        "to_job":     all_entities[kb]["job_name"] or "—",
                        "file":       f"{r['file_a']} ↔ {r['file_b']}",
                        "kind":       "cross-file (shared columns)",
                        "shared_columns": r["shared_columns"],
                    })

            st.session_state.parsed_metadata = {
                "files": all_parsed,
                "entities": all_entities,
                "relationships": relationships,
                "shared_column_clusters": shared_column_clusters,
                "cross_job_flows": cross_job_flows,
            }
            st.session_state.source_filename = ", ".join(
                t["file"].name for t in tagged_files
            )

            # ── Build the LLM summary ─────────────────────────────────
            # Structure it to make deduplication OBVIOUS. Lead with shared
            # business keys so the model anchors on cross-file unification
            # rather than treating each job as an island.
            summary = [
                "# METADATA ANALYSIS",
                f"Files uploaded: {len(tagged_files)} "
                f"({', '.join(t['file'].name for t in tagged_files)})",
                f"Tech stacks: "
                f"{', '.join(sorted({t['tech'] for t in tagged_files}))}",
                f"Total entities: {len(all_entities)} across all files",
                f"Shared business-key candidates: "
                f"{len(shared_column_clusters)}",
                f"Cross-job dataflows: {len(cross_job_flows)}",
                "",
                "## 🔑 CRITICAL: SHARED BUSINESS KEY CANDIDATES",
                "The following columns appear in MULTIPLE entities. Each "
                "such column represents ONE shared business concept and "
                "must map to ONE Hub (do NOT create duplicate Hubs per job).",
                "",
            ]
            if shared_column_clusters:
                # Sort so most-shared columns come first
                sorted_clusters = sorted(
                    shared_column_clusters.items(),
                    key=lambda x: (-len(x[1]), x[0])
                )
                for col, entities in sorted_clusters[:100]:
                    summary.append(
                        f"- **`{col}`** appears in {len(entities)} entities: "
                        f"{', '.join(entities)}  →  "
                        f"SHOULD MAP TO A SINGLE SHARED HUB"
                    )
            else:
                summary.append("(None detected — entities are independent)")

            summary.extend([
                "",
                "## 🔗 PAIRWISE ENTITY RELATIONSHIPS",
                "Entities linked by shared columns (candidates for Links):",
                "",
            ])
            if relationships:
                # Sort by number of shared columns
                rels_sorted = sorted(
                    relationships,
                    key=lambda r: -len(r["shared_columns"])
                )
                for r in rels_sorted[:150]:
                    summary.append(
                        f"- {r['entity_a']} ↔ {r['entity_b']}: "
                        f"shared columns = [{', '.join(r['shared_columns'])}]"
                    )
            else:
                summary.append("(None detected)")

            if cross_job_flows:
                summary.extend([
                    "",
                    "## 🔀 CROSS-JOB DATA FLOWS",
                    "Links that cross job boundaries — entities at the "
                    "downstream end CONSUME from the upstream end and are "
                    "part of the same business process. Includes both "
                    "explicit within-file links AND cross-file flows "
                    "inferred from shared columns.",
                    "",
                ])
                for cf in cross_job_flows[:80]:
                    kind = cf.get("kind", "within-file")
                    via = ""
                    if cf.get("shared_columns"):
                        via = (f"  (via shared columns: "
                               f"{', '.join(cf['shared_columns'][:6])})")
                    summary.append(
                        f"- [{kind}] "
                        f"`{cf['from_job']}::{cf['from_stage']}` → "
                        f"`{cf['to_job']}::{cf['to_stage']}`{via}"
                    )

            summary.extend([
                "",
                "## 📋 COMPLETE ENTITY INVENTORY",
                "Every entity with ALL its columns. Use the 'SHARED BUSINESS "
                "KEY CANDIDATES' section above to decide which entities "
                "should share Hubs.",
                "",
            ])
            for ent_key, meta in all_entities.items():
                cols = sorted(meta["columns"])
                typed = meta.get("columns_typed", {})
                # Qualify the heading with job when we have one so the LLM
                # can distinguish same-named stages across jobs, but still
                # sees the business-level display name for Hub unification.
                heading = meta["display_name"]
                if meta.get("job_name"):
                    heading = f"{meta['display_name']}  ({meta['job_name']})"
                summary.append(
                    f"### {heading}"
                )
                summary.append(
                    f"- Source file: `{meta['source_file']}`"
                )
                summary.append(
                    f"- Type: {meta['type']}"
                )
                # Columns with types, formatted: NAME TYPE, NAME TYPE, ...
                # The LLM uses these to fill the "Source Data Type" column
                # in the STTM. Without types here, those cells come out
                # blank/"Unknown" because the model has nothing to infer
                # from.
                col_parts = []
                for c in cols:
                    t = typed.get(c)
                    if t:
                        col_parts.append(f"{c} {t}")
                    else:
                        col_parts.append(c)
                summary.append(
                    f"- Columns ({len(cols)}): {', '.join(col_parts)}"
                )
                summary.append("")

            # Include per-file job/stage/link details for DataStage
            for f in all_parsed:
                if f["kind"] != "datastage":
                    continue
                p = f["parsed"]
                summary.extend([
                    "",
                    f"## 📄 DataStage file: {f['filename']}",
                ])
                if p["jobs"]:
                    summary.append(f"**Jobs** ({len(p['jobs'])}):")
                    for j in p["jobs"]:
                        summary.append(
                            f"- `{j['job_name']}`: {j['description']}"
                        )
                if p["stages"]:
                    summary.append(f"**Stages** ({len(p['stages'])}):")
                    for s in p["stages"][:60]:
                        summary.append(
                            f"- `{s['stage_name']}` (type={s['stage_type']}, "
                            f"job={s['job_name']})"
                        )
                if p["links"]:
                    summary.append(
                        f"**Stage-to-stage links** ({len(p['links'])}):"
                    )
                    for l in p["links"][:60]:
                        summary.append(
                            f"- `{l['from_stage']}` → `{l['to_stage']}`"
                        )

            # Append seed CSVs (STTM and attribute mapping) as
            # GROUND TRUTH context if uploaded. Per user direction the
            # LLM should treat these as authoritative anchors and extend
            # rather than override. Both seeds are stored in session
            # state by the upload handler; we re-emit them here so all
            # downstream artifact prompts pick them up automatically.
            sttm_seed = st.session_state.get("rev_sttm_seed_parsed")
            if sttm_seed and sttm_seed.get("row_count", 0) > 0:
                summary.append("")
                summary.append("# GROUND-TRUTH STTM (seed — extend, do not override)")
                summary.append(
                    f"User uploaded an existing STTM with "
                    f"{sttm_seed['row_count']} rows. The mappings below "
                    f"are authoritative; new STTM/Lineage entries you "
                    f"derive from the parsed source code MUST be "
                    f"consistent with these. Do not contradict the "
                    f"source/target table or column names. If you find "
                    f"additional mappings the seed doesn't cover, ADD "
                    f"them (don't delete or rename existing ones)."
                )
                seed_df = sttm_seed["sttm_df"]
                # Cap at 200 rows to keep prompt size sane
                preview_df = seed_df.head(200)
                summary.append(preview_df.to_csv(index=False))
                if len(seed_df) > 200:
                    summary.append(
                        f"...({len(seed_df) - 200} additional rows "
                        f"truncated from prompt; full file is on stage)"
                    )

            mapping_seed = st.session_state.get("rev_mapping_seed_parsed")
            if mapping_seed and mapping_seed.get("row_count", 0) > 0:
                summary.append("")
                summary.append("# GROUND-TRUTH attribute mapping (seed)")
                summary.append(
                    f"User uploaded an attribute mapping CSV with "
                    f"{mapping_seed['row_count']} rows. Use these "
                    f"definitions when building the Data Catalog and "
                    f"resolving column types — do not infer types that "
                    f"contradict this CSV."
                )
                seed_df2 = mapping_seed["columns_df"]
                preview2 = seed_df2.head(200)
                summary.append(preview2.to_csv(index=False))
                if len(seed_df2) > 200:
                    summary.append(
                        f"...({len(seed_df2) - 200} additional rows "
                        f"truncated)"
                    )

            # Per-division grouping: surface which files belong to which
            # division so the LLM can group artifacts accordingly.
            div_to_files: dict = {}
            for t in tagged_files:
                for d in (t.get("divisions") or ["(unassigned)"]):
                    div_to_files.setdefault(d, []).append(
                        f"{t['file'].name} [{t['tech']}]"
                    )
            if div_to_files and any(
                d != "(unassigned)" for d in div_to_files
            ):
                summary.append("")
                summary.append("# DIVISION ASSIGNMENTS")
                for d, files_in_div in sorted(div_to_files.items()):
                    summary.append(
                        f"- **{d}**: {', '.join(files_in_div)}"
                    )

            st.session_state.metadata_summary = "\n".join(summary)

        st.success(f"✓ Parsed {len(tagged_files)} file(s): "
                   f"{len(all_entities)} entities, "
                   f"{len(shared_column_clusters)} shared keys, "
                   f"{len(cross_job_flows)} cross-job flows")

        # Metadata preview
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Files", len(tagged_files))
        c2.metric("Entities", len(all_entities))
        c3.metric("Shared Keys", len(shared_column_clusters))
        c4.metric("Cross-Job Flows", len(cross_job_flows))

        with st.expander("Preview entities"):
            ent_rows = [
                {"Entity": m["display_name"],
                 "Job": m.get("job_name", "") or "—",
                 "Type": m["type"],
                 "Source File": m["source_file"],
                 "Columns": ", ".join(sorted(m["columns"])[:20]) +
                            (f" (+{len(m['columns']) - 20} more)"
                             if len(m["columns"]) > 20 else ""),
                 "Column Count": len(m["columns"])}
                for n, m in all_entities.items()
            ]
            st.dataframe(pd.DataFrame(ent_rows),
                         use_container_width=True, hide_index=True)

        if shared_column_clusters:
            with st.expander(
                f"🔑 Preview shared business-key candidates "
                f"({len(shared_column_clusters)})"
            ):
                cluster_rows = [
                    {"Shared Column": col,
                     "# Entities": len(entities),
                     "Entities": ", ".join(entities)}
                    for col, entities in sorted(
                        shared_column_clusters.items(),
                        key=lambda x: (-len(x[1]), x[0])
                    )
                ]
                st.dataframe(pd.DataFrame(cluster_rows),
                             use_container_width=True, hide_index=True)

        if cross_job_flows:
            with st.expander(
                f"🔀 Preview cross-job dataflows ({len(cross_job_flows)})"
            ):
                flow_rows = [
                    {"Kind": cf.get("kind", "within-file"),
                     "From Job": cf["from_job"],
                     "From Stage": cf["from_stage"],
                     "To Job": cf["to_job"],
                     "To Stage": cf["to_stage"],
                     "Via": ", ".join(cf.get("shared_columns", [])) or "—",
                     "File": cf["file"]}
                    for cf in cross_job_flows
                ]
                st.dataframe(pd.DataFrame(flow_rows),
                             use_container_width=True, hide_index=True)

        if relationships:
            with st.expander(
                f"🔗 Preview pairwise relationships ({len(relationships)})"
            ):
                rel_rows = [
                    {"Entity A": r["entity_a"],
                     "Entity B": r["entity_b"],
                     "Shared Columns": ", ".join(r["shared_columns"]),
                     "File A": r["file_a"],
                     "File B": r["file_b"]}
                    for r in relationships
                ]
                st.dataframe(pd.DataFrame(rel_rows),
                             use_container_width=True, hide_index=True)

    # Generation controls
    if st.session_state.metadata_summary:
        st.markdown("---")
        st.markdown("#### Generate artifacts")
        st.caption("Each runs through your selected Cortex model. "
                   "Artifacts appear below and in the sidebar.")

        g1, g2, g3, g4, g5 = st.columns(5)

        # Each artifact has a kind that drives how it's rendered:
        #   raw_vault → {narrative_md, mermaid, sql} (three separate LLM calls)
        #   table     → pd.DataFrame (parsed from Markdown pipe table)
        #   lineage   → {mermaid, source_to_hub_df} (deterministic graph
        #               built from parsed metadata + LLM source→Hub mapping)
        ARTIFACT_SPECS = {
            "Raw Vault Model": {
                "label": "Raw Vault Data Model",
                "kind":  "raw_vault",
            },
            "STTM": {
                "label":  "Source-to-Target Mapping",
                "kind":   "table",
                "prompt": build_sttm_prompt,
            },
            "Data Catalog": {
                "label":  "Data Catalog",
                "kind":   "table",
                "prompt": build_data_catalog_prompt,
            },
            "Data Domain": {
                "label":  "Data Domain Model",
                "kind":   "table",
                "prompt": build_data_domain_prompt,
            },
            "Data Lineage": {
                "label": "Data Lineage Graph",
                "kind":  "lineage",
            },
        }

        def _call(model_id: str, prompt: str, max_tokens: int = 16000) -> str:
            return call_cortex(model_id, prompt,
                               temperature=0.1, max_tokens=max_tokens)

        def _ensure_raw_vault_sql(model_id: str, meta: str) -> str:
            """
            Return the Raw Vault DDL, generating it if not already present.
            STTM uses this as target-side ground truth so target columns
            and types don't come out as "Unknown".
            """
            rv = st.session_state.artifacts.get("Raw Vault Model")
            if rv and rv.get("content", {}).get("sql"):
                return rv["content"]["sql"]

            # Generate JUST the DDL piece (not the narrative/Mermaid) so
            # STTM isn't gated on a full Raw Vault build if the user
            # clicked STTM first.
            with st.spinner(
                "Generating Raw Vault DDL (needed for STTM target "
                "columns/types)…"
            ):
                sql_raw = _call(model_id, build_raw_vault_sql_prompt(meta))
                sql = extract_sql_ddl(sql_raw)
            return sql

        def generate(key: str):
            spec = ARTIFACT_SPECS[key]
            model_id = MODELS[st.session_state.selected_model]
            meta = st.session_state.metadata_summary

            if spec["kind"] == "raw_vault":
                # Three separate LLM calls — far more reliable than one JSON blob
                with st.spinner(
                    f"Generating Raw Vault narrative with "
                    f"{st.session_state.selected_model}…"
                ):
                    narrative = _unwrap_json_string(
                        _call(model_id,
                              build_raw_vault_narrative_prompt(meta))
                    )
                with st.spinner("Generating Mermaid ER diagram…"):
                    mermaid_raw = _call(
                        model_id, build_raw_vault_mermaid_prompt(meta)
                    )
                    mermaid = extract_mermaid_script(mermaid_raw)
                with st.spinner("Generating Snowflake DDL…"):
                    sql_raw = _call(
                        model_id, build_raw_vault_sql_prompt(meta)
                    )
                    sql = extract_sql_ddl(sql_raw)

                content = {
                    "narrative_md": narrative,
                    "mermaid":      mermaid,
                    "mermaid_raw":  mermaid_raw,
                    "sql":          sql,
                    "sql_raw":      sql_raw,
                }

            elif spec["kind"] == "table":
                # STTM and Data Catalog emit one row per source column, so
                # for wide multi-job inputs the response can easily exceed
                # the 16k default and get truncated mid-CSV. Give them room.
                table_max_tokens = 32000 if key in ("STTM", "Data Catalog") \
                                          else 16000

                # STTM is anchored on the Raw Vault DDL — that's what
                # gives it real target table/column names and types.
                # Without this, targets come out as "Unknown" or
                # hallucinated.
                def _prompt_for(mk: str) -> str:
                    if key == "STTM":
                        rv_sql = _ensure_raw_vault_sql(model_id, mk)
                        return spec["prompt"](mk, rv_sql)
                    return spec["prompt"](mk)

                with st.spinner(
                    f"Generating {spec['label']} with "
                    f"{st.session_state.selected_model}…"
                ):
                    raw = _call(model_id, _prompt_for(meta),
                                max_tokens=table_max_tokens)
                df = parse_table_response(raw)

                # Retry with an even stricter prompt if parsing failed
                if df is None or df.empty:
                    with st.spinner(
                        f"Retrying {spec['label']} with stricter format…"
                    ):
                        retry_prompt = (
                            "Your previous response could not be parsed.\n"
                            "Output ONLY CSV with double-quoted cells.\n"
                            "First line must be the header. Each subsequent "
                            "line is one record.\n"
                            "No prose, no code fences, no markdown.\n\n"
                            f"Original task:\n{_prompt_for(meta)}\n\n"
                            f"Your broken previous response:\n{raw[:2000]}"
                        )
                        raw2 = _call(model_id, retry_prompt,
                                     max_tokens=table_max_tokens)
                        df2 = parse_table_response(raw2)
                        if df2 is not None and not df2.empty:
                            df = df2
                            raw = raw2

                content = {"df": df, "raw": raw}

            elif spec["kind"] == "lineage":
                # Data Lineage = deterministic graph from parsed metadata +
                # LLM source-to-Hub mapping grounded in the Raw Vault DDL.
                # The LLM only decides which Hub each source entity feeds;
                # the rest of the graph (file→stage, stage→stage within a
                # job, cross-file via shared columns, Hub→Sat, Hub→Link)
                # is structural and comes from parsed metadata or the DDL.
                parsed_meta = st.session_state.parsed_metadata or {}

                # Step 1: ensure we have Raw Vault DDL as ground truth
                rv_sql = _ensure_raw_vault_sql(model_id, meta)
                rv_tables = extract_raw_vault_tables(rv_sql)

                # Step 2: ask the LLM to map source entities to Hubs
                source_to_hub_df = None
                raw_mapping = ""
                if parsed_meta.get("entities") and rv_tables["hubs"]:
                    with st.spinner(
                        "Mapping source entities to Hubs via Raw Vault DDL…"
                    ):
                        raw_mapping = _call(
                            model_id,
                            build_source_to_hub_prompt(parsed_meta, rv_sql),
                            max_tokens=8000,
                        )
                        source_to_hub_df = parse_table_response(raw_mapping)

                # Step 3: build the deterministic Mermaid flowchart
                with st.spinner("Building lineage graph…"):
                    mermaid = build_lineage_mermaid(
                        parsed_meta, rv_tables, source_to_hub_df
                    )
                    # Also build a structured graph for the interactive
                    # SVG renderer — same edges, different format.
                    graph = build_lineage_graph(
                        parsed_meta, rv_tables, source_to_hub_df
                    )

                content = {
                    "mermaid":       mermaid,
                    "graph":         graph,
                    "source_to_hub": source_to_hub_df,
                    "mapping_raw":   raw_mapping,
                    "rv_tables":     rv_tables,
                }

            st.session_state.artifacts[key] = {
                "kind":    spec["kind"],
                "label":   spec["label"],
                "content": content,
            }
            st.success(f"✓ {spec['label']} generated")

        if g1.button("🔀 Data Lineage", use_container_width=True,
                     type="primary"):
            generate("Data Lineage")
        if g2.button("🗺 STTM", use_container_width=True, type="primary"):
            generate("STTM")
        if g3.button("📚 Data Catalog", use_container_width=True, type="primary"):
            generate("Data Catalog")
        if g4.button("🌐 Data Domain", use_container_width=True, type="primary"):
            generate("Data Domain")
        if g5.button("🏛 Raw Vault Model", use_container_width=True,
                     type="primary"):
            generate("Raw Vault Model")

        if st.button("⚡ Generate ALL five artifacts",
                     use_container_width=True):
            # Run in the requested pipeline order: Lineage → STTM →
            # Catalog → Domain → Raw Vault. Raw Vault is LAST because
            # it consumes everything upstream and is the most expensive
            # artifact to generate.
            for key in ("Data Lineage", "STTM", "Data Catalog",
                        "Data Domain", "Raw Vault Model"):
                generate(key)

        # Bundle-download for everything at once — shown only when at
        # least one artifact exists so the button isn't dangling empty.
        if st.session_state.artifacts:
            try:
                bundle_bytes = build_artifacts_bundle(
                    st.session_state.artifacts,
                    source_filename=st.session_state.source_filename or "",
                    metadata_summary=(
                        st.session_state.metadata_summary or ""
                    ),
                )
                stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                st.download_button(
                    f"📦 Download ALL resources "
                    f"({len(st.session_state.artifacts)} artifact"
                    f"{'s' if len(st.session_state.artifacts) != 1 else ''}"
                    f", .zip)",
                    data=bundle_bytes,
                    file_name=f"data_engineering_artifacts_{stamp}.zip",
                    mime="application/zip",
                    use_container_width=True,
                    key="dl_bundle_all",
                )
            except Exception as e:
                st.warning(f"Bundle export error: {e}")

        # ── Vector-store indexing ────────────────────────────────────────
        # Artifacts → internal stage (versioned) → chunker → Cortex embed
        # → VECTOR column in Snowflake → semantic search.
        if st.session_state.artifacts:
            with st.expander(
                "🧠  Store as vector embeddings  "
                "(versioned, by data domain)",
                expanded=False,
            ):
                st.caption(
                    f"Artifacts land in stage "
                    f"`@{VECTOR_DB}.{VECTOR_SCHEMA}.{VECTOR_STAGE}/v<version>/` "
                    f"then get chunked, embedded with Snowflake Cortex, "
                    f"and stored by `(version, data_domain, "
                    f"artifact_type)`. Re-running with the same version "
                    f"replaces that version's rows (idempotent)."
                )

                vcol1, vcol2 = st.columns([1, 2])
                version_in = vcol1.text_input(
                    "Version number",
                    value=st.session_state.get(
                        "vectorize_version", "1.0.0"
                    ),
                    placeholder="e.g. 1.0.0 or 2026-04-20-rc1",
                    key="vectorize_version_in",
                    help="Used to tag every row and as the stage folder "
                         "name (special chars collapsed).",
                )
                embed_label = vcol2.selectbox(
                    "Embedding model",
                    list(EMBED_MODELS.keys()),
                    index=0,
                    key="vectorize_model_in",
                    help="Model determines the VECTOR dimension and "
                         "which table the vectors land in.",
                )
                if not embed_label:
                    embed_label = list(EMBED_MODELS.keys())[0]
                embed_model, embed_dim, embed_table = \
                    EMBED_MODELS[embed_label]

                # Preview how many chunks would be produced — cheap, no
                # Cortex calls — so the user knows what they're committing
                # to before clicking.
                try:
                    domain_map = extract_domain_map(
                        st.session_state.artifacts
                    )
                    preview_chunks = chunk_artifacts(
                        st.session_state.artifacts, domain_map
                    )
                    from collections import Counter
                    by_domain = Counter(
                        c["data_domain"] for c in preview_chunks
                    )
                    by_type = Counter(
                        c["artifact_type"] for c in preview_chunks
                    )
                    preview_rows = pd.DataFrame([
                        {"Data Domain": d, "Chunks": n}
                        for d, n in sorted(
                            by_domain.items(), key=lambda x: -x[1]
                        )
                    ])
                    st.markdown(
                        f"**Preview:** {len(preview_chunks)} chunks "
                        f"across {len(by_domain)} domain(s) "
                        f"and {len(by_type)} artifact type(s). "
                        f"Domain map resolved {len(domain_map)} entities "
                        f"from the Data Domain artifact."
                    )
                    if not preview_rows.empty:
                        st.dataframe(
                            preview_rows,
                            use_container_width=True,
                            hide_index=True,
                        )
                    if by_domain.get("UNCLASSIFIED"):
                        st.caption(
                            f"ℹ️ {by_domain['UNCLASSIFIED']} chunks could "
                            f"not be mapped to a domain — generate the "
                            f"Data Domain artifact first to minimize this."
                        )
                except Exception as e:
                    st.warning(f"Chunk preview failed: {e}")
                    preview_chunks = []

                # Run the pipeline
                if st.button(
                    "🚀 Store vectors in Snowflake",
                    type="primary",
                    use_container_width=True,
                    key="vectorize_run",
                    disabled=not version_in.strip()
                             or not preview_chunks,
                ):
                    try:
                        with st.spinner(
                            "Creating stage and vector tables if needed…"
                        ):
                            ensure_vector_infrastructure(session)
                        with st.spinner(
                            f"Uploading bundle to internal stage "
                            f"v{version_in}…"
                        ):
                            up = upload_artifacts_to_stage(
                                session,
                                st.session_state.artifacts,
                                version=version_in,
                                source_filename=(
                                    st.session_state.source_filename or ""
                                ),
                                metadata_summary=(
                                    st.session_state.metadata_summary or ""
                                ),
                            )
                        st.caption(
                            f"Staged at `{up['path']}` "
                            f"({up['bytes']:,} bytes)"
                        )

                        progress = st.progress(0.0,
                            text=f"Embedding 0 / {len(preview_chunks)}…")
                        def _cb(done, total):
                            progress.progress(
                                min(1.0, done / max(1, total)),
                                text=(
                                    f"Embedding {done} / {total} "
                                    f"with {embed_model}…"
                                ),
                            )
                        inserted = embed_and_store(
                            session, preview_chunks,
                            version=version_in,
                            embed_model=embed_model,
                            dim=embed_dim,
                            table=embed_table,
                            progress_cb=_cb,
                        )
                        progress.progress(1.0,
                            text=f"Inserted {inserted} rows.")
                        st.success(
                            f"✓ Vectorized version **{version_in}**: "
                            f"{inserted} rows into "
                            f"`{VECTOR_DB}.{VECTOR_SCHEMA}.{embed_table}` "
                            f"with model `{embed_model}`"
                        )
                        # Persist so the search panel below can default to
                        # the same version/model the user just loaded
                        st.session_state["vectorize_version"] = version_in
                        st.session_state["vectorize_last_model"] = (
                            embed_label
                        )
                    except Exception as e:
                        st.error(f"Vectorization failed: {e}")

            # ── Semantic search ─────────────────────────────────────────
            with st.expander(
                "🔎  Semantic search over stored vectors",
                expanded=False,
            ):
                search_model_label = st.selectbox(
                    "Search model (must match how vectors were stored)",
                    list(EMBED_MODELS.keys()),
                    index=list(EMBED_MODELS.keys()).index(
                        st.session_state.get(
                            "vectorize_last_model",
                            list(EMBED_MODELS.keys())[0]
                        )
                    ) if st.session_state.get("vectorize_last_model")
                         in EMBED_MODELS else 0,
                    key="search_model_in",
                )
                if not search_model_label:
                    search_model_label = list(EMBED_MODELS.keys())[0]
                s_model, s_dim, s_table = EMBED_MODELS[search_model_label]

                # Populate filter dropdowns by querying the table for
                # distinct values — only when the table already exists.
                try:
                    versions = [r["VERSION"] for r in session.sql(
                        f"SELECT DISTINCT VERSION "
                        f"FROM {_fqn(s_table)} "
                        f"WHERE EMBEDDING_MODEL = ? "
                        f"ORDER BY VERSION DESC",
                        params=[s_model],
                    ).collect()]
                    domains = ["(any)"] + [r["DATA_DOMAIN"] for r in
                        session.sql(
                            f"SELECT DISTINCT DATA_DOMAIN "
                            f"FROM {_fqn(s_table)} "
                            f"WHERE EMBEDDING_MODEL = ? "
                            f"ORDER BY DATA_DOMAIN",
                            params=[s_model],
                        ).collect()]
                    types_ = ["(any)"] + [r["ARTIFACT_TYPE"] for r in
                        session.sql(
                            f"SELECT DISTINCT ARTIFACT_TYPE "
                            f"FROM {_fqn(s_table)} "
                            f"WHERE EMBEDDING_MODEL = ? "
                            f"ORDER BY ARTIFACT_TYPE",
                            params=[s_model],
                        ).collect()]
                except Exception:
                    versions, domains, types_ = [], ["(any)"], ["(any)"]

                if not versions:
                    st.info(
                        "No vectors stored yet for this model. Run "
                        "vectorization above first."
                    )
                else:
                    fcol1, fcol2, fcol3 = st.columns(3)
                    search_version = fcol1.selectbox(
                        "Version", ["(latest)"] + versions, index=0,
                        key="search_version",
                    )
                    search_domain = fcol2.selectbox(
                        "Data Domain", domains, index=0,
                        key="search_domain",
                    )
                    search_type = fcol3.selectbox(
                        "Artifact Type", types_, index=0,
                        key="search_type",
                    )
                    top_k = st.slider(
                        "Number of results", 1, 50, 10,
                        key="search_topk",
                    )
                    query_text = st.text_input(
                        "Search query",
                        placeholder=(
                            "e.g. customer PII attributes, "
                            "or: where does loan amount live"
                        ),
                        key="search_query",
                    )
                    if st.button("🔎 Search",
                                 type="primary",
                                 use_container_width=True,
                                 key="search_run",
                                 disabled=not query_text.strip()):
                        try:
                            with st.spinner(f"Embedding query + "
                                            f"ranking vectors…"):
                                hits = semantic_search(
                                    session,
                                    query=query_text,
                                    table=s_table,
                                    dim=s_dim,
                                    embed_model=s_model,
                                    version=(
                                        None
                                        if search_version == "(latest)"
                                        else search_version
                                    ),
                                    data_domain=search_domain,
                                    artifact_type=search_type,
                                    top_k=top_k,
                                )
                            if hits.empty:
                                st.info("No matches.")
                            else:
                                st.caption(
                                    f"{len(hits)} match"
                                    f"{'es' if len(hits) != 1 else ''} — "
                                    f"higher similarity is closer."
                                )
                                # Show the table
                                st.dataframe(
                                    hits[[
                                        "SIMILARITY", "DATA_DOMAIN",
                                        "ARTIFACT_TYPE", "ENTITY",
                                        "CONTENT", "VERSION",
                                    ]],
                                    use_container_width=True,
                                    hide_index=True,
                                )
                                # Let the user download results
                                st.download_button(
                                    "⬇ Download results (.csv)",
                                    data=hits.to_csv(index=False),
                                    file_name=(
                                        f"semantic_search_"
                                        f"{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
                                        f".csv"
                                    ),
                                    mime="text/csv",
                                    key="dl_search",
                                )
                        except Exception as e:
                            st.error(f"Search failed: {e}")

        # ── Render artifacts (shared helper used by View Artifacts tab too)
        if st.session_state.artifacts:
            st.markdown("---")
            st.markdown("#### Artifacts")
            render_artifacts(st.session_state.artifacts, key_prefix="rev")

    # Chat history rendered inside the tab; chat_input lives at app-level
    # (outside tabs) because Streamlit disallows chat_input inside st.tabs.
    if st.session_state.chat_history:
        st.markdown("---")
        st.markdown("#### Conversation")
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])


# ═════════════════════════════════════════════════════════════════════════════
# FORWARD ENGINEERING TAB
# ═════════════════════════════════════════════════════════════════════════════
def _render_tab_forward():
    st.markdown("#### Forward Engineering")
    st.caption(
        "Generate Semantic Model, Business Vault, STTM, Catalog, "
        "Domains, and dbt projects - grounded on an existing Reverse "
        "Engineering artifact set plus a dashboard spec and banking "
        "rules doc."
    )

    # ── STEP 1: Pick Reverse Engineering artifact version ─────────
    st.markdown("##### 1. Select Reverse Engineering artifacts")
    try:
        versions_df = list_stored_versions(session)
    except Exception as e:
        versions_df = pd.DataFrame()
        st.error(f"Could not list stored versions: {e}")

    if versions_df.empty:
        st.info(
            "No Reverse Engineering versions stored yet. Run the "
            "Reverse Engineering tab first and use 'Store as vector "
            "embeddings' to publish a version."
        )
        st.stop()

    def _fwd_opt_label(row):
        last = str(row["last_loaded"])[:19] if row["last_loaded"] else ""
        return (
            f"v{row['version']}  ·  {row['model_label']}  ·  "
            f"{row['chunk_count']} chunks, {row['domain_count']} "
            f"domains"
            + (f"  ·  {last}" if last else "")
        )
    fwd_opts = list(versions_df.apply(_fwd_opt_label, axis=1))
    fwd_chosen_labels = st.multiselect(
        "Reverse artifacts version(s)",
        fwd_opts,
        default=(fwd_opts[:1] if fwd_opts else []),
        key="fwd_version_select_multi",
        help=(
            "Pick one or more reverse-engineering versions. Artifacts "
            "from all selected versions are merged for forward generation."
        ),
    )
    if not fwd_chosen_labels and fwd_opts:
        fwd_chosen_labels = [fwd_opts[0]]

    fwd_chosen_versions = [
        str(versions_df.iloc[fwd_opts.index(lbl)]["version"])
        for lbl in fwd_chosen_labels
    ]
    if not fwd_chosen_versions and not versions_df.empty:
        fwd_chosen_versions = [str(versions_df.iloc[0]["version"])]
    fwd_versions_label = ", ".join(f"v{v}" for v in fwd_chosen_versions)
    first_fwd_version = fwd_chosen_versions[0] if fwd_chosen_versions else "0"

    # Load bundle(s) from stage (cached by selected version list)
    fwd_cache_key = f"fwd_loaded::{'|'.join(fwd_chosen_versions)}"
    bundles = list_stage_bundles(session)

    if st.session_state.get("fwd_loaded_key") != fwd_cache_key:
        try:
            with st.spinner(
                f"Loading reverse artifacts ({fwd_versions_label})…"
            ):
                merged = {}
                loaded_count = 0
                missing_versions = []
                for ver in fwd_chosen_versions:
                    matching = [
                        b for b in bundles if b["version_slug"] == ver
                    ]
                    if not matching:
                        missing_versions.append(ver)
                        continue
                    bundle = load_artifacts_from_stage(
                        session, matching[0]["path"]
                    )
                    arts = (bundle.get("artifacts", {}) or {})
                    for k, v in arts.items():
                        kk = k if k not in merged else f"{k} [v{ver}]"
                        merged[kk] = v
                    loaded_count += 1

                if not merged:
                    miss = ", ".join(f"v{v}" for v in missing_versions) \
                        if missing_versions else fwd_versions_label
                    st.warning(
                        f"No ZIP bundle found for selected version(s): {miss}. "
                        "Re-upload from Reverse Engineering."
                    )
                    st.stop()

                st.session_state[fwd_cache_key] = {
                    "artifacts": merged,
                    "selected_versions": fwd_chosen_versions,
                    "loaded_bundle_count": loaded_count,
                    "missing_versions": missing_versions,
                }
                st.session_state.fwd_loaded_key = fwd_cache_key
                st.session_state.fwd_source_version = first_fwd_version
        except Exception as e:
            st.error(f"Failed to load reverse artifacts: {e}")
            st.stop()
    fwd_loaded = st.session_state.get(fwd_cache_key, {})
    fwd_source_arts = fwd_loaded.get("artifacts", {}) or {}
    st.session_state.fwd_source_artifacts = fwd_source_arts
    st.caption(
        f"Loaded **{len(fwd_source_arts)} reverse artifacts** from "
        f"{fwd_versions_label}"
    )

    # Reverse summary - prepared once, reused across all forward prompts
    reverse_summary = summarize_reverse_artifacts(fwd_source_arts)

    # ── STEP 2: Dashboard type + upload ───────────────────────────
    st.markdown("---")
    st.markdown("##### 2. Target dashboard")
    d_col1, d_col2 = st.columns([1, 1])
    dashboard_type = d_col1.selectbox(
        "Dashboard category",
        list(DASHBOARD_TYPES.keys()),
        index=0,
        key="fwd_dashboard_type",
    )
    # Streamlit's selectbox can return None in some re-execution races
    # (especially across multiple tabs) even when the options list is
    # non-empty. Fall back to the first key so downstream code never
    # dereferences None.
    if not dashboard_type:
        dashboard_type = list(DASHBOARD_TYPES.keys())[0]
    d_col2.caption(f"Scope: {DASHBOARD_TYPES.get(dashboard_type, '—')}")

    dashboard_files = st.file_uploader(
        "Dashboard specification (PDF / Excel / Image) - optional",
        type=["pdf", "xlsx", "xls", "png", "jpg", "jpeg", "csv",
              "txt", "md"],
        accept_multiple_files=True,
        key="fwd_dashboard_uploader",
    )
    dashboard_description = st.text_area(
        "Additional dashboard description",
        placeholder=(
            "e.g. 'Daily cash position by business unit with drill-"
            "down to transaction type. Include 30-day rolling "
            "average and variance vs forecast.'"
        ),
        height=120,
        key="fwd_dashboard_desc",
    )
    # Combine file-extracted text with free-form description
    dashboard_parts = []
    if dashboard_description.strip():
        dashboard_parts.append(f"USER DESCRIPTION:\n{dashboard_description}")
    for uf in (dashboard_files or []):
        try:
            txt = extract_text_from_upload(uf)
            if txt:
                dashboard_parts.append(
                    f"--- FILE: {uf.name} ---\n{txt[:10000]}"
                )
        except Exception as e:
            st.warning(f"Could not read {uf.name}: {e}")
    dashboard_text = "\n\n".join(dashboard_parts)
    st.session_state.fwd_dashboard_text = dashboard_text

    # ── STEP 3: Rules / knowledge doc ─────────────────────────────
    st.markdown("---")
    st.markdown("##### 3. Banking rules / internal knowledge / do's & don'ts")
    rules_files = st.file_uploader(
        "Rules or knowledge documents - optional",
        type=["pdf", "xlsx", "xls", "txt", "md", "csv"],
        accept_multiple_files=True,
        key="fwd_rules_uploader",
    )
    rules_parts = []
    for uf in (rules_files or []):
        try:
            txt = extract_text_from_upload(uf)
            if txt:
                rules_parts.append(
                    f"--- FILE: {uf.name} ---\n{txt[:12000]}"
                )
        except Exception as e:
            st.warning(f"Could not read {uf.name}: {e}")
    rules_text = "\n\n".join(rules_parts)
    st.session_state.fwd_rules_text = rules_text

    # ── STEP 4-8: Generation ──────────────────────────────────────
    st.markdown("---")
    st.markdown("##### 4-8. Generate forward artifacts")

    model_id = MODELS[st.session_state.selected_model]

    def _fwd_call(prompt, opts=None):
        """Call Cortex with normal or code-generation options."""
        if opts:
            return call_cortex(
                model_id, prompt,
                temperature=opts.get("temperature", 0.2),
                max_tokens=opts.get("max_tokens", 8000),
                top_p=opts.get("top_p"),
                guardrails=opts.get("guardrails"),
            )
        return call_cortex(model_id, prompt,
                           temperature=0.1, max_tokens=16000)

    def _generate_forward(key: str):
        """
        Dispatch on which forward artifact to generate. Stores into
        st.session_state.fwd_artifacts in the same shape as reverse
        artifacts, so render_artifacts() can display it.
        """
        fwd = st.session_state.fwd_artifacts

        if key == "Semantic Model":
            with st.spinner("Generating Semantic Model narrative…"):
                nar = _fwd_call(build_semantic_model_prompt(
                    dashboard_type, dashboard_text, rules_text,
                    reverse_summary,
                ))
            with st.spinner("Generating Semantic Model ER diagram…"):
                mer_raw = _fwd_call(build_semantic_model_mermaid_prompt(
                    nar, dashboard_type,
                ))
                mer = extract_mermaid_script(mer_raw)
            with st.spinner("Generating Semantic Model DDL…"):
                sql_raw = _fwd_call(build_semantic_model_sql_prompt(
                    nar, dashboard_type,
                ))
                sql = extract_sql_ddl(sql_raw)
            fwd["Semantic Model"] = {
                "kind": "raw_vault",  # reuse raw_vault renderer
                "label": "Semantic (Dimensional) Model",
                "content": {"narrative_md": nar, "mermaid": mer,
                            "sql": sql, "mermaid_raw": mer_raw,
                            "sql_raw": sql_raw},
            }

        elif key == "Business Vault":
            sem = fwd.get("Semantic Model", {}).get("content", {}) \
                     .get("narrative_md", "") or ""
            if not sem:
                st.warning("Generate the Semantic Model first.")
                return
            with st.spinner("Generating Business Vault narrative…"):
                nar = _fwd_call(build_business_vault_narrative_prompt(
                    dashboard_type, dashboard_text, rules_text,
                    reverse_summary, sem,
                ))
            with st.spinner("Generating Business Vault ER diagram…"):
                mer_raw = _fwd_call(build_business_vault_mermaid_prompt(
                    nar, reverse_summary,
                ))
                mer = extract_mermaid_script(mer_raw)
            with st.spinner("Generating Business Vault DDL…"):
                sql_raw = _fwd_call(
                    build_business_vault_sql_prompt(nar, reverse_summary)
                )
                sql = extract_sql_ddl(sql_raw)
            fwd["Business Vault"] = {
                "kind": "raw_vault",
                "label": "Business Vault Data Model",
                "content": {"narrative_md": nar, "mermaid": mer,
                            "sql": sql, "mermaid_raw": mer_raw,
                            "sql_raw": sql_raw},
            }

        elif key == "Forward STTM":
            bv = fwd.get("Business Vault", {}).get("content", {}) \
                    .get("narrative_md", "") or ""
            if not bv:
                st.warning("Generate the Business Vault first.")
                return
            with st.spinner("Generating Forward STTM…"):
                raw = _fwd_call(build_forward_sttm_prompt(
                    bv, reverse_summary, dashboard_type
                ))
                df = parse_table_response(raw)
            fwd["Forward STTM"] = {
                "kind": "table",
                "label": "STTM (Raw Vault -> Business Vault / Semantic)",
                "content": {"df": df, "raw": raw},
            }

        elif key == "Forward Catalog":
            sem = fwd.get("Semantic Model", {}).get("content", {}) \
                     .get("narrative_md", "") or ""
            bv = fwd.get("Business Vault", {}).get("content", {}) \
                    .get("narrative_md", "") or ""
            if not sem or not bv:
                st.warning("Generate Semantic Model and Business "
                           "Vault first.")
                return
            with st.spinner("Generating Data Catalog…"):
                raw = _fwd_call(build_forward_catalog_prompt(
                    sem, bv, dashboard_type
                ))
                df = parse_table_response(raw)
            fwd["Forward Catalog"] = {
                "kind": "table",
                "label": "Data Catalog (BV + Semantic)",
                "content": {"df": df, "raw": raw},
            }

        elif key == "Forward Domains":
            sem = fwd.get("Semantic Model", {}).get("content", {}) \
                     .get("narrative_md", "") or ""
            bv = fwd.get("Business Vault", {}).get("content", {}) \
                    .get("narrative_md", "") or ""
            if not sem or not bv:
                st.warning("Generate Semantic Model and Business "
                           "Vault first.")
                return
            with st.spinner("Generating Data Domains…"):
                raw = _fwd_call(build_forward_domains_prompt(
                    sem, bv, dashboard_type, rules_text
                ))
                df = parse_table_response(raw)
            fwd["Forward Domains"] = {
                "kind": "table",
                "label": "Data Domain Model (BV + Semantic)",
                "content": {"df": df, "raw": raw},
            }

        elif key == "Raw Vault dbt":
            # Build strict context: Data Vault model + STTM only.
            rv_sttm_txt = ""
            try:
                rev_sttm = fwd_source_arts.get("STTM", {}) or {}
                rev_sttm_content = rev_sttm.get("content", {}) or {}
                rev_sttm_df = rev_sttm_content.get("df")
                rev_sttm_raw = rev_sttm_content.get("raw", "") or ""
                if rev_sttm_df is not None and not rev_sttm_df.empty:
                    rv_sttm_txt = rev_sttm_df.to_csv(index=False)
                elif rev_sttm_raw:
                    rv_sttm_txt = str(rev_sttm_raw)
            except Exception:
                rv_sttm_txt = ""
            if not rv_sttm_txt:
                fwd_sttm = fwd.get("Forward STTM", {}) or {}
                fwd_sttm_content = fwd_sttm.get("content", {}) or {}
                fwd_sttm_df = fwd_sttm_content.get("df")
                fwd_sttm_raw = fwd_sttm_content.get("raw", "") or ""
                if fwd_sttm_df is not None and not fwd_sttm_df.empty:
                    rv_sttm_txt = fwd_sttm_df.to_csv(index=False)
                elif fwd_sttm_raw:
                    rv_sttm_txt = str(fwd_sttm_raw)

            rv_codegen_ctx = forward_raw_vault_dbt_codegen_context(
                reverse_summary, rv_sttm_txt,
            )

            def _call_for_dbt(prompt, opts):
                return call_cortex(
                    model_id, prompt,
                    temperature=opts.get("temperature", 0.1),
                    max_tokens=opts.get("max_tokens", 2500),
                    top_p=opts.get("top_p"),
                    guardrails=opts.get("guardrails"),
                )

            progress = st.progress(0.0, text="Planning dbt files…")
            plan_paths_seen = [0]  # mutable for closure

            def _cb(done, total, current):
                plan_paths_seen[0] = total
                progress.progress(
                    min(1.0, done / max(1, total)),
                    text=f"Generating ({done}/{total}): {current}",
                )

            files, plan_raw, errors = generate_dbt_project_per_file(
                _call_for_dbt, rv_codegen_ctx, "raw_vault",
                progress_cb=_cb,
            )
            progress.progress(1.0, text=f"Done — {len(files)} files")

            if errors:
                with st.expander(
                    f"⚠ {len(errors)} file(s) had issues",
                    expanded=False,
                ):
                    for p, err in errors:
                        st.caption(f"`{p}` — {err[:200]}")

            fwd["Raw Vault dbt"] = {
                "kind": "dbt_project",
                "label": (
                    "Raw Vault dbt — Medallion (bronze→silver) + "
                    "AutomateDV, Jinja, macros"
                ),
                "content": {
                    "files": files,
                    "raw": (
                        "=== PLAN ===\n" + plan_raw +
                        "\n\n=== ERRORS ===\n" +
                        "\n".join(f"{p}: {e}" for p, e in errors)
                    ),
                },
            }

        elif key == "Business Vault dbt":
            forward_summary = (
                (fwd.get("Semantic Model", {}).get("content", {})
                     .get("narrative_md", "") or "")
                + "\n\n=== BUSINESS VAULT NARRATIVE ===\n\n"
                + (fwd.get("Business Vault", {}).get("content", {})
                       .get("narrative_md", "") or "")
                + "\n\n=== BUSINESS VAULT DDL ===\n\n"
                + (fwd.get("Business Vault", {}).get("content", {})
                       .get("sql", "") or "")
            )
            if "=== BUSINESS VAULT NARRATIVE ===" in forward_summary \
                    and len(forward_summary) < 200:
                st.warning("Generate Business Vault first.")
                return

            def _call_for_bv_dbt(prompt, opts):
                return call_cortex(
                    model_id, prompt,
                    temperature=opts.get("temperature", 0.1),
                    max_tokens=opts.get("max_tokens", 2500),
                    top_p=opts.get("top_p"),
                    guardrails=opts.get("guardrails"),
                )

            fwd_sttm_txt = ""
            fwd_sttm = fwd.get("Forward STTM", {}) or {}
            fwd_sttm_content = fwd_sttm.get("content", {}) or {}
            fwd_sttm_df = fwd_sttm_content.get("df")
            fwd_sttm_raw = fwd_sttm_content.get("raw", "") or ""
            if fwd_sttm_df is not None and not fwd_sttm_df.empty:
                fwd_sttm_txt = fwd_sttm_df.to_csv(index=False)
            elif fwd_sttm_raw:
                fwd_sttm_txt = str(fwd_sttm_raw)

            combined_ctx = forward_business_vault_dbt_codegen_context(
                forward_summary, reverse_summary, fwd_sttm_txt,
            )

            progress = st.progress(0.0, text="Planning BV dbt files…")

            def _cb2(done, total, current):
                progress.progress(
                    min(1.0, done / max(1, total)),
                    text=f"Generating ({done}/{total}): {current}",
                )

            files, plan_raw, errors = generate_dbt_project_per_file(
                _call_for_bv_dbt, combined_ctx, "business_vault",
                progress_cb=_cb2,
            )
            progress.progress(1.0, text=f"Done — {len(files)} files")

            if errors:
                with st.expander(
                    f"⚠ {len(errors)} file(s) had issues",
                    expanded=False,
                ):
                    for p, err in errors:
                        st.caption(f"`{p}` — {err[:200]}")

            fwd["Business Vault dbt"] = {
                "kind": "dbt_project",
                "label": (
                    "Business Vault dbt — Medallion (silver BV→gold) + "
                    "Jinja/macros"
                ),
                "content": {
                    "files": files,
                    "raw": (
                        "=== PLAN ===\n" + plan_raw +
                        "\n\n=== ERRORS ===\n" +
                        "\n".join(f"{p}: {e}" for p, e in errors)
                    ),
                },
            }

        elif key == "dbt Tests":
            rv_files = (fwd.get("Raw Vault dbt", {}).get("content", {})
                           .get("files", {}) or {})
            bv_files = (fwd.get("Business Vault dbt", {})
                           .get("content", {}).get("files", {}) or {})
            if not rv_files or not bv_files:
                st.warning("Generate Raw Vault and Business Vault dbt "
                           "projects first.")
                return

            # Build a compact context: just the list of models in each
            # project + a few representative file bodies (keeps prompts
            # small while still grounding tests on real model names)
            rv_model_list = "\n".join(f"- {p}" for p in rv_files
                                      if p.endswith(".sql"))
            bv_model_list = "\n".join(f"- {p}" for p in bv_files
                                      if p.endswith(".sql"))
            combined_ctx = (
                "RAW VAULT MODELS:\n" + rv_model_list[:2500]
                + "\n\nBUSINESS VAULT + GOLD MODELS:\n"
                + bv_model_list[:2500]
            )

            def _call_for_tests(prompt, opts):
                return call_cortex(
                    model_id, prompt,
                    temperature=opts.get("temperature", 0.3),
                    max_tokens=opts.get("max_tokens", 2500),
                    top_p=opts.get("top_p"),
                    guardrails=opts.get("guardrails"),
                )

            progress = st.progress(0.0, text="Planning dbt tests…")

            def _cb3(done, total, current):
                progress.progress(
                    min(1.0, done / max(1, total)),
                    text=f"Generating ({done}/{total}): {current}",
                )

            files, plan_raw, errors = generate_dbt_project_per_file(
                _call_for_tests, combined_ctx, "dbt_tests",
                progress_cb=_cb3,
            )
            progress.progress(1.0, text=f"Done — {len(files)} files")

            if errors:
                with st.expander(
                    f"⚠ {len(errors)} file(s) had issues",
                    expanded=False,
                ):
                    for p, err in errors:
                        st.caption(f"`{p}` — {err[:200]}")

            fwd["dbt Tests"] = {
                "kind": "dbt_project",
                "label": "dbt Test Project",
                "content": {
                    "files": files,
                    "raw": (
                        "=== PLAN ===\n" + plan_raw +
                        "\n\n=== ERRORS ===\n" +
                        "\n".join(f"{p}: {e}" for p, e in errors)
                    ),
                },
            }

        st.session_state.fwd_artifacts = fwd
        st.success(f"✓ Generated: {key}")

    # Generation buttons - two rows
    fg1, fg2, fg3, fg4 = st.columns(4)
    if fg1.button("① Semantic Model", use_container_width=True,
                  type="primary", key="fwd_gen_sem"):
        _generate_forward("Semantic Model")
    if fg2.button("② Business Vault", use_container_width=True,
                  type="primary", key="fwd_gen_bv"):
        _generate_forward("Business Vault")
    if fg3.button("③ Forward STTM", use_container_width=True,
                  type="primary", key="fwd_gen_sttm"):
        _generate_forward("Forward STTM")
    if fg4.button("④ Data Catalog", use_container_width=True,
                  type="primary", key="fwd_gen_cat"):
        _generate_forward("Forward Catalog")

    fg5, fg6, fg7, fg8 = st.columns(4)
    if fg5.button("⑤ Data Domains", use_container_width=True,
                  type="primary", key="fwd_gen_dom"):
        _generate_forward("Forward Domains")
    if fg6.button("⑥ Raw Vault dbt", use_container_width=True,
                  type="primary", key="fwd_gen_rvdbt"):
        _generate_forward("Raw Vault dbt")
    if fg7.button("⑦ Business Vault dbt", use_container_width=True,
                  type="primary", key="fwd_gen_bvdbt"):
        _generate_forward("Business Vault dbt")
    if fg8.button("⑧ dbt Tests", use_container_width=True,
                  type="primary", key="fwd_gen_tests"):
        _generate_forward("dbt Tests")

    if st.button("⚡ Generate ALL forward artifacts (in order)",
                 use_container_width=True, key="fwd_gen_all"):
        for step in ["Semantic Model", "Business Vault",
                     "Forward STTM", "Forward Catalog",
                     "Forward Domains", "Raw Vault dbt",
                     "Business Vault dbt", "dbt Tests"]:
            _generate_forward(step)

    # ── Render forward artifacts ──────────────────────────────────
    fwd_arts = st.session_state.fwd_artifacts
    if fwd_arts:
        st.markdown("---")
        st.markdown("##### Generated forward artifacts")
        # render_artifacts now handles all kinds including dbt_project
        render_artifacts(fwd_arts, key_prefix="fwd")

        # ── STEP 9-11: Download + persist ─────────────────────────
        st.markdown("---")
        st.markdown("##### Persist forward artifacts")

        fwd_version_in = st.text_input(
            "Version number for forward artifacts",
            value=f"fwd-{first_fwd_version}",
            key="fwd_persist_version",
            help="Saved to stage and vector tables under this version.",
        )
        fwd_vector_version_in = st.text_input(
            "Version number for vector storage",
            value=fwd_version_in,
            key="fwd_vector_version_in",
            help="Used by 'Store as vectors' in this Forward tab.",
        )

        fc1, fc2, fc3 = st.columns(3)

        # Download-all bundle
        try:
            bundle_bytes = build_artifacts_bundle(
                fwd_arts,
                source_filename=f"forward-from-{fwd_versions_label}",
                metadata_summary=(
                    f"# Forward Engineering\n\n"
                    f"Reverse artifacts version(s): {fwd_versions_label}\n"
                    f"Dashboard: {dashboard_type}\n"
                    f"\n## Dashboard Specification\n{dashboard_text}\n"
                    f"\n## Rules\n{rules_text}\n"
                ),
            )
            stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            fc1.download_button(
                f"📦 Download all ({len(fwd_arts)} artifacts)",
                data=bundle_bytes,
                file_name=f"forward_artifacts_{stamp}.zip",
                mime="application/zip",
                use_container_width=True,
                key="fwd_dl_all",
            )
        except Exception as e:
            fc1.warning(f"Bundle error: {e}")

        # Upload to stage
        if fc2.button("☁ Copy to Snowflake stage",
                      use_container_width=True,
                      key="fwd_upload_stage",
                      disabled=not fwd_version_in.strip()):
            try:
                with st.spinner("Ensuring infrastructure…"):
                    ensure_vector_infrastructure(session)
                with st.spinner(f"Uploading to "
                                f"v{fwd_version_in}…"):
                    up = upload_artifacts_to_stage(
                        session, fwd_arts,
                        version=fwd_version_in,
                        source_filename=(
                            f"forward-from-{fwd_versions_label}"
                        ),
                        metadata_summary=(
                            f"# Forward Engineering\n\n"
                            f"Source reverse: {fwd_versions_label}\n"
                            f"Dashboard: {dashboard_type}\n"
                        ),
                    )
                st.success(f"✓ Staged at {up['path']}")
            except Exception as e:
                st.error(f"Stage upload failed: {e}")

        # Vectorize into vector table
        embed_label = fc3.selectbox(
            "Embedding model",
            list(EMBED_MODELS.keys()),
            index=0,
            key="fwd_vec_model",
            label_visibility="collapsed",
        )
        if fc3.button("🧠 Store as vectors",
                      use_container_width=True,
                      key="fwd_vectorize",
                      disabled=not fwd_vector_version_in.strip()):
            if not embed_label:
                embed_label = list(EMBED_MODELS.keys())[0]
            em, ed, et = EMBED_MODELS[embed_label]
            try:
                with st.spinner("Ensuring vector infrastructure…"):
                    ensure_vector_infrastructure(session)
                # Use the Data Domain artifact from FORWARD side as
                # the domain map - it reflects the target architecture.
                dm = extract_domain_map(fwd_arts)
                chunks = chunk_artifacts(fwd_arts, dm)
                if not chunks:
                    st.warning("No chunks produced - generate "
                               "artifacts first.")
                else:
                    progress = st.progress(0.0,
                        text=f"Embedding 0 / {len(chunks)}…")
                    def _cb(done, total):
                        progress.progress(
                            min(1.0, done / max(1, total)),
                            text=f"Embedding {done} / {total}…",
                        )
                    inserted = embed_and_store(
                        session, chunks,
                        version=fwd_vector_version_in,
                        embed_model=em, dim=ed, table=et,
                        progress_cb=_cb,
                    )
                    progress.progress(1.0, text=f"Inserted {inserted} rows.")
                    st.success(
                        f"✓ {inserted} vectors stored in "
                        f"{VECTOR_DB}.{VECTOR_SCHEMA}.{et} "
                        f"under version `{fwd_vector_version_in}`"
                    )
            except Exception as e:
                st.error(f"Vectorization failed: {e}")

        # ── STEP 11.5: Publish dbt projects to GitHub ──────────────
        st.markdown("---")
        st.markdown("##### Publish dbt code to GitHub")
        st.caption(
            "Publishes generated dbt files to your GitHub repository "
            "using the GitHub Contents API."
        )
        gh_owner = "murthybackyard"
        gh_repo = "nexus"
        gh_branch = st.text_input(
            "GitHub branch",
            value="main",
            key="fwd_github_branch",
            help="Target branch in the repository.",
        )
        gh_token_default = resolve_github_token(
            st.session_state.get("github_token", "")
        )
        gh_token = st.text_input(
            "GitHub token (PAT)",
            type="password",
            value=gh_token_default,
            key="fwd_github_token",
            help=(
                "Requires repo content write access. "
                "Token is used only for this publish action."
            ),
        )
        st.session_state.github_token = gh_token or ""
        if not gh_token.strip():
            st.info(
                "No GitHub PAT detected. Add `GITHUB_TOKEN` (or "
                "`github.token` / `github.pat`) in `.streamlit/secrets.toml` "
                "or set env var `GITHUB_TOKEN`."
            )
        gh_auto_init = st.checkbox(
            "Auto-initialize GitHub repo if empty",
            value=True,
            key="fwd_github_auto_init",
            help=(
                "Creates an initial README commit/branch if repository is "
                "empty, then publishes dbt files."
            ),
        )

        raw_files = (
            (fwd_arts.get("Raw Vault dbt", {}) or {})
            .get("content", {})
            .get("files", {})
            or {}
        )
        bv_files = (
            (fwd_arts.get("Business Vault dbt", {}) or {})
            .get("content", {})
            .get("files", {})
            or {}
        )
        pc1, pc2 = st.columns(2)
        pub_raw = pc1.button(
            "⬆ Publish Raw Vault dbt → raw_vault/",
            use_container_width=True,
            key="fwd_publish_raw_github",
            disabled=(not gh_token.strip() or not raw_files),
        )
        pub_bv = pc2.button(
            "⬆ Publish Business Vault dbt → business_vault/",
            use_container_width=True,
            key="fwd_publish_bv_github",
            disabled=(not gh_token.strip() or not bv_files),
        )
        if pub_raw:
            try:
                with st.spinner("Publishing Raw Vault dbt files to GitHub…"):
                    res = publish_dbt_project_to_github(
                        files=raw_files,
                        token=gh_token,
                        owner=gh_owner,
                        repo=gh_repo,
                        subfolder="raw_vault",
                        branch=gh_branch.strip() or "main",
                        auto_init_repo=gh_auto_init,
                        commit_message_prefix="publish raw_vault dbt",
                    )
                st.success(
                    f"✓ Published {res['published']} files to "
                    f"`{res['subfolder']}/` in {gh_owner}/{gh_repo} "
                    f"(branch `{res['branch']}`)"
                )
                if res["errors"]:
                    with st.expander(
                        f"⚠ {len(res['errors'])} file(s) failed",
                        expanded=False,
                    ):
                        for p, err in res["errors"]:
                            st.caption(f"`{p}` — {err[:250]}")
            except Exception as e:
                st.error(f"GitHub publish failed: {e}")

        if pub_bv:
            try:
                with st.spinner(
                    "Publishing Business Vault dbt files to GitHub…"
                ):
                    res = publish_dbt_project_to_github(
                        files=bv_files,
                        token=gh_token,
                        owner=gh_owner,
                        repo=gh_repo,
                        subfolder="business_vault",
                        branch=gh_branch.strip() or "main",
                        auto_init_repo=gh_auto_init,
                        commit_message_prefix="publish business_vault dbt",
                    )
                st.success(
                    f"✓ Published {res['published']} files to "
                    f"`{res['subfolder']}/` in {gh_owner}/{gh_repo} "
                    f"(branch `{res['branch']}`)"
                )
                if res["errors"]:
                    with st.expander(
                        f"⚠ {len(res['errors'])} file(s) failed",
                        expanded=False,
                    ):
                        for p, err in res["errors"]:
                            st.caption(f"`{p}` — {err[:250]}")
            except Exception as e:
                st.error(f"GitHub publish failed: {e}")

        # ── STEP 12: Build & Deploy dbt Project ────────────────────
        # Rendered via the shared helper so the Quick GO tab can
        # offer the exact same flow without code duplication.
        render_dbt_deploy_section(fwd_arts, key_prefix='fwd')


# ═════════════════════════════════════════════════════════════════════════════
# VIEW ARTIFACTS TAB — browse previously-stored bundles
# ═════════════════════════════════════════════════════════════════════════════
def _render_tab_view():
    st.markdown("#### View stored artifacts")
    st.caption(
        f"Browse artifact bundles previously uploaded to stage "
        f"`@{VECTOR_DB}.{VECTOR_SCHEMA}.{VECTOR_STAGE}` and indexed "
        f"in the vector tables. Pick a version to reload the full "
        f"artifact set — displayed exactly as in Reverse Engineering."
    )

    # Discover what's been stored
    try:
        versions_df = list_stored_versions(session)
    except Exception as e:
        versions_df = pd.DataFrame()
        st.error(
            f"Could not list stored versions: {e}  "
            f"(have you run the Store vectors step in Reverse "
            f"Engineering at least once?)"
        )

    if versions_df.empty:
        st.info(
            "No versioned artifact bundles found yet. Generate "
            "artifacts in the **Reverse Engineering** tab, then use "
            "**Store as vector embeddings** to publish a version."
        )
    else:
        # Build a human-readable label per (version, model) combo
        def _opt_label(row):
            last = str(row["last_loaded"])[:19] if row["last_loaded"] \
                   else ""
            return (
                f"v{row['version']}  ·  "
                f"{row['model_label']}  ·  "
                f"{row['chunk_count']} chunk"
                f"{'s' if row['chunk_count'] != 1 else ''}, "
                f"{row['domain_count']} domain"
                f"{'s' if row['domain_count'] != 1 else ''}"
                + (f"  ·  {last}" if last else "")
            )
        opts = list(versions_df.apply(_opt_label, axis=1))

        col1, col2 = st.columns([3, 1])
        chosen_label = col1.selectbox(
            "Select a stored version",
            opts,
            index=0,
            key="view_version_select",
        )
        load_clicked = col2.button(
            "🔄 Reload",
            use_container_width=True,
            key="view_reload",
            help="Refresh the list of stored versions",
        )
        if load_clicked:
            st.rerun()
        if not chosen_label and opts:
            chosen_label = opts[0]
        chosen_row = versions_df.iloc[opts.index(chosen_label)]
        chosen_version = chosen_row["version"]

        # ── Metadata card ───────────────────────────────────────────
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Version", chosen_version)
        c2.metric("Chunks", int(chosen_row["chunk_count"]))
        c3.metric("Domains", int(chosen_row["domain_count"]))
        c4.metric("Artifact types", int(chosen_row["type_count"]))

        # Domain breakdown for this specific version
        try:
            dom_rows = session.sql(f"""
                SELECT DATA_DOMAIN,
                       COUNT(*)                      AS CHUNKS,
                       COUNT(DISTINCT ARTIFACT_TYPE) AS TYPES,
                       COUNT(DISTINCT ENTITY)        AS ENTITIES
                FROM {_fqn(chosen_row['table'])}
                WHERE VERSION = ? AND EMBEDDING_MODEL = ?
                GROUP BY DATA_DOMAIN
                ORDER BY CHUNKS DESC
            """, params=[chosen_version, chosen_row["model"]]).to_pandas()
        except Exception:
            dom_rows = pd.DataFrame()

        if not dom_rows.empty:
            with st.expander(
                f"Domain breakdown for v{chosen_version}",
                expanded=False,
            ):
                st.dataframe(
                    dom_rows, use_container_width=True, hide_index=True
                )

        # ── Locate the bundle on stage ──────────────────────────────
        bundles = list_stage_bundles(session)
        matching = [
            b for b in bundles
            if b["version_slug"] == chosen_version
        ]

        if not matching:
            st.warning(
                f"Vector rows exist for version `{chosen_version}` "
                f"but no matching ZIP bundle found on stage. "
                f"The stage folder may have been cleared — try "
                f"re-uploading from Reverse Engineering."
            )
            with st.expander("Show chunks directly from vector table"):
                try:
                    raw_chunks = session.sql(f"""
                        SELECT DATA_DOMAIN, ARTIFACT_TYPE, ENTITY,
                               CHUNK_ID, CONTENT
                        FROM {_fqn(chosen_row['table'])}
                        WHERE VERSION = ? AND EMBEDDING_MODEL = ?
                        ORDER BY DATA_DOMAIN, ARTIFACT_TYPE, CHUNK_ID
                        LIMIT 500
                    """, params=[chosen_version,
                                 chosen_row["model"]]).to_pandas()
                    st.dataframe(
                        raw_chunks, use_container_width=True,
                        hide_index=True
                    )
                except Exception as e:
                    st.error(f"Could not fetch chunks: {e}")
        else:
            bundle = matching[0]
            st.caption(
                f"Loading from stage: `{bundle['path']}` "
                f"({bundle['size']:,} bytes)"
            )

            # Load + cache in session state (keyed by version so
            # switching versions triggers a reload)
            cache_key = f"view_loaded::{chosen_version}::{chosen_row['model']}"
            if st.session_state.get("view_active_key") != cache_key:
                try:
                    with st.spinner(
                        f"Downloading and unpacking "
                        f"v{chosen_version}…"
                    ):
                        loaded = load_artifacts_from_stage(
                            session, bundle["path"]
                        )
                    st.session_state[cache_key]     = loaded
                    st.session_state.view_active_key = cache_key
                except Exception as e:
                    st.error(f"Failed to load bundle: {e}")
                    loaded = None
            else:
                loaded = st.session_state.get(cache_key)

            if loaded:
                # Summary row
                arts = loaded.get("artifacts") or {}
                src_fn = loaded.get("source_filename") or "—"
                st.markdown(
                    f"**Source files (at generation time):** `{src_fn}`  "
                    f"·  **Artifacts in bundle:** {len(arts)}"
                )

                # Let the user re-download the bundle zip too
                try:
                    stream = session.file.get_stream(bundle["path"])
                    zb = stream.read()
                    st.download_button(
                        f"📦 Download this bundle (v{chosen_version})",
                        data=zb,
                        file_name=bundle["filename"],
                        mime="application/zip",
                        use_container_width=True,
                        key=f"view_dl_bundle_{chosen_version}",
                    )
                except Exception:
                    pass

                # ── Render artifacts using the shared helper ────────
                if arts:
                    st.markdown("---")
                    st.markdown("#### Artifacts")
                    render_artifacts(
                        arts,
                        key_prefix=f"view_{chosen_version}"
                    )
                else:
                    st.info(
                        "Bundle was loaded but contained no artifacts "
                        "(empty manifest)."
                    )


_tab_vis = get_tab_visibility()
_tab_labels = build_tab_labels(_tab_vis)
if not _tab_labels:
    st.error(
        "All main tabs are hidden. Enable at least one tab in "
        "`nexus/app_tabs_config.py` or clear env "
        "`NEXUS_APP_TAB_VISIBILITY`."
    )
    st.stop()

_tab_widgets = st.tabs(_tab_labels)
_tab_i = 0
if _tab_vis.get("quick_go", True):
    with _tab_widgets[_tab_i]:
        _render_tab_quickgo()
    _tab_i += 1
if _tab_vis.get("reverse_engineering", True):
    with _tab_widgets[_tab_i]:
        _render_tab_reverse()
    _tab_i += 1
if _tab_vis.get("forward_engineering", True):
    with _tab_widgets[_tab_i]:
        _render_tab_forward()
    _tab_i += 1
if _tab_vis.get("view_artifacts", True):
    with _tab_widgets[_tab_i]:
        _render_tab_view()
    _tab_i += 1


# ═════════════════════════════════════════════════════════════════════════════
# MODEL SELECTOR + GLOBAL CHAT INPUT
# Must live OUTSIDE st.tabs — Streamlit API constraint.
# Model dropdown is right-aligned above the chat input, Claude-Code style.
# ═════════════════════════════════════════════════════════════════════════════

# Custom chat input that sits directly below the model picker (no gap).
# We intentionally DON'T use st.chat_input because it's forcibly pinned
# to a fixed bottom dock, creating unavoidable empty space above it.
# Instead: a text_input + send button in the page flow, styled as a pill.

# Row 1 — model picker, right-aligned
_sp1, _sp_mid, _model_col = st.columns([6, 2, 2])
with _model_col:
    st.session_state.selected_model = st.selectbox(
        "Model",
        list(MODELS.keys()),
        label_visibility="collapsed",
        index=list(MODELS.keys()).index(st.session_state.selected_model),
        key="model_picker_bottom",
    )

# Row 2 — custom chat input: text field + send button as a single pill
def _submit_chat():
    """Handle submit from either the text field or the send button."""
    txt = st.session_state.get("chat_input_text", "").strip()
    if not txt:
        return
    st.session_state.chat_history.append({"role": "user", "content": txt})
    ctx = st.session_state.metadata_summary or "(no metadata uploaded yet)"
    full_prompt = CHAT_ASSISTANT_SYSTEM.format(
        metadata=ctx[:6000],
        question=txt,
    )
    with st.spinner(f"Thinking with {st.session_state.selected_model}…"):
        reply = call_cortex(
            MODELS[st.session_state.selected_model],
            full_prompt,
        )
    st.session_state.chat_history.append(
        {"role": "assistant", "content": reply}
    )
    st.session_state["chat_input_text"] = ""  # clear the field

_in_col, _btn_col = st.columns([10, 1])
with _in_col:
    st.text_input(
        "Ask a follow-up",
        placeholder="Ask a follow-up about your metadata…",
        label_visibility="collapsed",
        key="chat_input_text",
        on_change=_submit_chat,
    )
with _btn_col:
    st.button("➤", key="chat_send_btn", on_click=_submit_chat,
              use_container_width=True)

# Chat-input styling — applied AFTER the widgets render so these rules win
st.markdown("""
<style>
/* Tight spacing between model picker and the custom chat input row */
html body div[class*="st-key-model_picker_bottom"] {
    margin: 0 !important;
    padding: 0 !important;
}

/* The custom chat input field — make it look like a pill */
html body div[class*="st-key-chat_input_text"] input,
html body div[data-testid="stTextInput"]:has(#chat_input_text) input {
    background: #FFFFFF !important;
    background-color: #FFFFFF !important;
    border: 1px solid #E8E6DC !important;
    border-radius: 20px !important;
    padding: 12px 20px !important;
    font-size: 15px !important;
    color: #141413 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04) !important;
    caret-color: #C96442 !important;
    height: 48px !important;
}
html body div[class*="st-key-chat_input_text"] input:focus,
html body div[data-testid="stTextInput"]:has(#chat_input_text) input:focus {
    border-color: #C96442 !important;
    outline: none !important;
    box-shadow: 0 0 0 3px rgba(201,100,66,0.15) !important;
}
html body div[class*="st-key-chat_input_text"] input::placeholder {
    color: #A39C8B !important;
    opacity: 1 !important;
}

/* The send button — Claude orange circle */
html body div[class*="st-key-chat_send_btn"] button {
    background: #C96442 !important;
    background-color: #C96442 !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 50% !important;
    width: 48px !important;
    height: 48px !important;
    min-width: 48px !important;
    font-size: 18px !important;
    padding: 0 !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.08) !important;
}
html body div[class*="st-key-chat_send_btn"] button:hover {
    background: #B85537 !important;
    background-color: #B85537 !important;
    color: #FFFFFF !important;
}
html body div[class*="st-key-chat_send_btn"] button p {
    color: #FFFFFF !important;
    margin: 0 !important;
}

/* Hide Streamlit's default bottom dock entirely since we're not using chat_input */
html body div[data-testid="stBottom"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

