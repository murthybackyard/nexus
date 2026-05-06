"""nexus — Data Engineering Co-Pilot package.

Phase 1 split: pure modules with no Streamlit/Snowflake coupling.

The Streamlit-coupled code (vector_store, dbt_pipeline, artifacts, ui_tabs)
remains in streamlit_app.py for now and will be split in a later phase.
"""

# Re-export the most-used names for convenient `from nexus import X`
from nexus.utils import (
    extract_json, parse_table_response, parse_markdown_table,
    df_to_excel_bytes, resolve_sql_type, extract_text_from_upload,
)
from nexus.llm import call_cortex, MODELS, EMBED_MODELS
from nexus.parsers import (
    parse_datastage_dsx, parse_metadata_csv, parse_sttm_csv,
    parse_bods_xml, parse_legacy_sql, parse_netezza_sql,
    parse_mssql_sql, parse_control_m, parse_shell_script,
    parse_ssis_dtsx, parse_denodo_vql, parse_source_ddl_or_metadata,
    detect_tech_stack,
)
from nexus.prompts import (
    _dv_standards_block,
    build_sttm_prompt, build_data_catalog_prompt,
    build_data_domain_prompt, build_source_to_hub_prompt,
    build_raw_vault_narrative_prompt, build_raw_vault_sql_prompt,
    build_raw_vault_mermaid_prompt,
)
__version__ = "0.1.0-phase1"
