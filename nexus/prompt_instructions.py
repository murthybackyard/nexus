"""
Prompt and codegen instruction blocks extracted from the main app.

Edit this file to tune LLM contracts and UI copy without touching
``streamlit_app.py`` logic.
"""
from __future__ import annotations

# ── Quick GO: dbt per-file planner contexts ────────────────────────────────

QUICK_GO_RV_DBTC_OUTPUT_CONTRACT = """
OUTPUT CONTRACT (STRICT):
- Return a dbt project grouped by folder.
- Required folders/files:
  models/silver/staging/stg_<entity>.sql
  models/silver/raw_vault/hubs/hub_<entity>.sql
  models/silver/raw_vault/links/lnk_<rel>.sql
  models/silver/raw_vault/sats/sat_<entity>_<context>.sql
  models/silver/raw_vault/schema.yml
- Use incremental materialization for staging/hub/link/sat.
- Follow Medallion architecture (silver raw vault layer).
- Ensure referential integrity in schema.yml tests.
- Use consistent lower_snake_case naming.
- In each staging model, list explicit columns
  in `SELECT ... FROM {{ source('raw', '<entity>') }}` so the deploy pipeline can
  infer RAW landing columns.
""".strip()

def quick_go_raw_vault_dbt_codegen_context(rv_sql: str, sttm_csv: str) -> str:
    return (
        "INPUT: DATA VAULT MODEL\n"
        + ((rv_sql or "")[:9000] or "(no Raw Vault DDL)")
        + "\n\nINPUT: STTM\n"
        + ((sttm_csv or "")[:9000] or "(no STTM)")
        + "\n\n"
        + QUICK_GO_RV_DBTC_OUTPUT_CONTRACT
    )


QUICK_GO_BV_DBTC_OUTPUT_CONTRACT = """
OUTPUT CONTRACT (STRICT):
- Return a dbt project grouped by folder.
- Required folders/files:
  models/silver/business_vault/hubs/hub_<entity>.sql
  models/silver/business_vault/links/lnk_<rel>.sql
  models/silver/business_vault/sats/sat_<entity>_<context>.sql
  models/silver/business_vault/schema.yml
- BV models reference upstream Raw Vault hubs/links/sats via ref().
- Use incremental materialization throughout.
- Use consistent lower_snake_case naming.
""".strip()


def quick_go_business_vault_dbt_codegen_context(
    bv_nar: str,
    bv_sql: str,
    rv_sql: str,
    fwd_sttm_csv: str,
) -> str:
    return (
        "INPUT: BUSINESS VAULT NARRATIVE\n"
        + ((bv_nar or "")[:6000] or "(no BV narrative)")
        + "\n\nINPUT: BUSINESS VAULT DDL\n"
        + ((bv_sql or "")[:9000] or "(no BV DDL)")
        + "\n\nINPUT: RAW VAULT DDL (upstream)\n"
        + ((rv_sql or "")[:6000] or "(no upstream RV DDL)")
        + "\n\nINPUT: FORWARD STTM (RV → BV mapping)\n"
        + ((fwd_sttm_csv or "")[:6000] or "(no Forward STTM)")
        + "\n\n"
        + QUICK_GO_BV_DBTC_OUTPUT_CONTRACT
    )


# ── Forward Engineering tab: Raw Vault dbt context ────────────────────────

FORWARD_RV_DBTC_OUTPUT_CONTRACT = """
OUTPUT CONTRACT (STRICT):
- Return a dbt project grouped by folder.
- Required folders/files:
  models/silver/staging/stg_<entity>.sql
  models/silver/raw_vault/hubs/hub_<entity>.sql
  models/silver/raw_vault/links/lnk_<relationship>.sql
  models/silver/raw_vault/sats/sat_<entity>_<context>.sql
  models/silver/raw_vault/schema.yml
- Use incremental materialization for staging/hub/link/sat.
- Follow Medallion architecture (silver raw vault layer).
- Ensure referential integrity in schema.yml tests.
- Use consistent lower_snake_case naming.
- In each staging model, list explicit columns (avoid SELECT *)
  in `SELECT ... FROM {{ source('raw', '<entity>') }}` so the app
  can infer RAW landing columns for `scripts/create_raw_landing_tables.sql`.
""".strip()


def forward_raw_vault_dbt_codegen_context(
    reverse_summary: str,
    rv_sttm_txt: str,
) -> str:
    return (
        "INPUT: DATA VAULT MODEL\n"
        + (reverse_summary or "")[:9000]
        + "\n\nINPUT: STTM\n"
        + ((rv_sttm_txt or "")[:9000] if rv_sttm_txt else "(not available)")
        + "\n\n"
        + FORWARD_RV_DBTC_OUTPUT_CONTRACT
    )


FORWARD_BV_DBTC_OUTPUT_CONTRACT = """
OUTPUT CONTRACT (STRICT):
- Return grouped-by-folder dbt output.
- Include silver business-vault models and gold semantic refs.
- Required core folders:
  models/silver/business_vault/staging/
  models/silver/business_vault/hubs/
  models/silver/business_vault/links/
  models/silver/business_vault/sats/
  models/silver/business_vault/schema.yml
- Use incremental materialization for staging/hub/link/sat.
- Enforce referential integrity via relationship tests.
- Consistent lower_snake_case naming.
""".strip()


def forward_business_vault_dbt_codegen_context(
    forward_summary: str,
    reverse_summary: str,
    fwd_sttm_txt: str,
) -> str:
    return (
        "INPUT: DATA VAULT MODEL (RAW + BUSINESS)\n"
        + (forward_summary or "")[:6000]
        + "\n\n=== RAW VAULT REFERENCE ===\n\n"
        + (reverse_summary or "")[:2500]
        + "\n\n=== INPUT: STTM ===\n\n"
        + ((fwd_sttm_txt or "")[:7000] if fwd_sttm_txt else "(not available)")
        + "\n\n"
        + FORWARD_BV_DBTC_OUTPUT_CONTRACT
    )


# ── Global chat (bottom of app) ───────────────────────────────────────────

CHAT_ASSISTANT_SYSTEM = (
    "You are a data engineering assistant. The user has uploaded "
    "this metadata:\n\n{metadata}\n\n"
    "User question: {question}\n\n"
    "Give a concise, technically precise answer."
)
