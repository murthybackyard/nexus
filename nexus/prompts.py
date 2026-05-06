"""
nexus.prompts — all LLM prompt builders + Data Vault 2.0 standards.

This module is pure-string: it has no side effects, no Streamlit, no
Snowflake. It builds prompt text from input metadata. Callers feed
the returned string to nexus.llm.call_cortex().

Contents:
    Data Vault 2.0 standards (the bank's authoritative rules):
      _DV_STD_ABBREVIATIONS    — approved entity/column abbreviations
      _DV_STD_NAMING           — entity/column naming conventions
      _DV_STD_HASH             — hash key algorithm + null/preprocessing rules
      _DV_STD_METADATA_COLS    — required metadata columns per entity type
      _DV_STD_LINK_RULES       — link design + hub-reuse rules
      _DV_STD_SATELLITE_RULES  — source-specific satellite rules + change freq
      _DV_STANDARDS_BY_ARTIFACT — selector dict
      _dv_standards_block(art) — formatter helper

    Forward Engineering prompts:
      build_raw_vault_narrative_prompt
      build_raw_vault_sql_prompt
      build_raw_vault_mermaid_prompt
      build_raw_vault_validation_prompt   (independent GenAI validator)
      build_business_vault_narrative_prompt
      build_business_vault_sql_prompt
      build_business_vault_mermaid_prompt
      build_semantic_model_*  (3 prompts)
      build_forward_*         (3 prompts: STTM, catalog, domain)

    Reverse Engineering prompts:
      build_sttm_prompt
      build_data_catalog_prompt
      build_data_domain_prompt
      build_transformation_rules_prompt   (Markdown rules report)
      build_source_to_hub_prompt   (lineage source→hub mapping)

    DBT planner:
      build_dbt_planner_prompt

    Medallion + DV full specs:
      MEDALLION_DV_FULL_SPEC
      MEDALLION_DV_FILE_CONSTRAINTS
      BUSINESS_VAULT_GUIDE
"""


import re
from typing import Dict, List, Optional


def _mermaid_safe_id(s: str) -> str:
    """Make a string safe to use as a Mermaid node id."""
    # Mermaid node ids must be alphanumeric + underscores. Collapse
    # everything else; prefix if the result starts with a digit.
    out = re.sub(r'[^A-Za-z0-9_]', '_', s)
    out = re.sub(r'_+', '_', out).strip('_')
    if not out:
        out = "n"
    if out[0].isdigit():
        out = "n_" + out
    return out[:80]


DASHBOARD_TYPES = {
    "Business dashboards": "cash flow, payments, receivables, settlements",
    "Lending dashboards":  "loans, credit risk, delinquency, collections",
    "Treasury dashboards": "liquidity, AP/AR, funding, cash positioning",
    "Internal analytics":  "enterprise KPIs, operational metrics",
}


def build_semantic_model_prompt(dashboard_type, dashboard_text,
                                rules_text, reverse_summary,
                                business_vault_md: str = "",
                                forward_catalog_csv: str = ""):
    """Build the Semantic (Dimensional) Model narrative prompt.

    Optional ``business_vault_md`` and ``forward_catalog_csv`` arguments
    let callers supply the just-built Business Vault narrative and
    Forward Catalog so the dimensional model can be sourced explicitly
    from those upstream artifacts (instead of being inferred from the
    Raw Vault alone).
    """
    bv_block = ""
    if business_vault_md and business_vault_md.strip():
        bv_block = f"""

UPSTREAM BUSINESS VAULT (narrative just produced for this dashboard):
{business_vault_md[:10000]}

When building facts and dimensions, prefer sourcing measures from
Business Vault PIT/Bridge/Computed-Sat objects rather than going back
to the Raw Vault directly. Reference Business Vault tables by name in
the "Sourced from" notes."""
    cat_block = ""
    if forward_catalog_csv and forward_catalog_csv.strip():
        cat_block = f"""

FORWARD DATA CATALOG (CSV just produced for the target):
{forward_catalog_csv[:6000]}

Use the catalog to align dimension attributes (display names, business
descriptions, PII flags, units) with what consumers will see."""
    return f"""You are a senior data architect designing a dimensional
semantic model (Kimball star-schema) to serve a specific dashboard need.

DASHBOARD CATEGORY: {dashboard_type}
(scope hint: {DASHBOARD_TYPES.get(dashboard_type, '')})

DASHBOARD SPECIFICATION (from uploaded PDF/Excel/image + description):
{dashboard_text or '(no dashboard spec provided)'}

BANKING RULES / INTERNAL KNOWLEDGE / DO'S AND DON'TS:
{rules_text or '(no rules doc provided)'}

UPSTREAM REVERSE-ENGINEERING CONTEXT (existing Raw Vault, entities, lineage):
{reverse_summary or '(no reverse artifacts selected)'}
{bv_block}{cat_block}

TASK: Design a semantic model (Dimensional Data Model) sized to answer
this dashboard's questions. Produce ONE Markdown document organized as:

## Overview
2-3 sentences on the business purpose and the star-schema approach.
State which upstream Raw Vault Hubs/Links/Sats this consumes.

## Grain Statement
For each fact table, state the grain in one sentence.

## Fact Tables
For each fact:
- **FCT_<n>**
  - **Grain:** (one sentence)
  - **Measures:** each measure with type and calculation
  - **Foreign keys:** each dimension FK (DIM_<n>_SK)
  - **Sourced from:** which Raw Vault Hubs/Links/Sats feed it

## Dimension Tables
For each dim:
- **DIM_<n>**
  - **Grain:** one row per ...
  - **Natural key:** <column>
  - **Attributes:** descriptive columns with types
  - **SCD Type:** (1 / 2 / hybrid) and why
  - **Sourced from:** which Raw Vault objects

## Conformed Dimensions
Dimensions shared across facts and why conformance matters.

## Key KPIs & Measures
For each dashboard metric, state the formula referencing fact/dim columns.

## Regulatory / Banking Considerations
Call out PII, materiality, reporting, or audit concerns.

Return clean Markdown. No outer code fences.
"""

def build_business_vault_narrative_prompt(dashboard_type, dashboard_text,
                                          rules_text, reverse_summary,
                                          semantic_model_md,
                                          dm_standards_text: str = ""):
    return f"""You are a senior Data Vault 2.0 architect designing a
Business Vault layer that sits on top of a Raw Vault and powers the
semantic model specified below.

DASHBOARD: {dashboard_type}
SEMANTIC MODEL (already designed):
{semantic_model_md[:8000]}

REVERSE (Raw Vault) CONTEXT:
{reverse_summary[:8000]}

BANKING RULES:
{rules_text[:3000] or '(none)'}
{_dm_standards_block(dm_standards_text)}

TASK: Design Business Vault objects (BV Hubs, BV Links, Effectivity
Sats, Point-in-Time tables, Bridge tables, Computed Sats) that
transform Raw Vault data into business-ready form feeding the semantic
model.

Return clean Markdown:

## Overview
One paragraph on what Business Vault objects you propose and why each
is needed.

## Business Vault Objects
For each object:
- **BV_<n>** (kind: PIT / Bridge / Computed Sat / BV Hub / BV Link)
  - **Purpose:** what business logic it applies
  - **Sources (Raw Vault):** which Hubs/Links/Sats it reads from
  - **Grain:** one sentence
  - **Key columns:** hash keys, business keys, measures with types
  - **Dependency on other BV objects:** explicit chain

## Mapping to Semantic Model
For each Fact/Dim in the semantic model, state which BV object(s)
feed it and the grain transformation.

## Load Order
BV objects in the order they must be built each run.

Return clean Markdown. No outer code fences.
"""

def build_business_vault_sql_prompt(bv_narrative_md, reverse_summary,
                                    dm_standards_text: str = ""):
    return f"""You are a Snowflake DDL generator.

Given this Business Vault narrative:
{bv_narrative_md[:12000]}

And the Raw Vault context (already exists):
{reverse_summary[:8000]}
{_dm_standards_block(dm_standards_text)}

Produce Snowflake CREATE TABLE statements for every Business Vault
object described in the narrative - PIT tables, Bridge tables,
Computed Satellites, BV Hubs, BV Links.

DDL RULES:
- Use Snowflake types (VARCHAR, NUMBER, TIMESTAMP_NTZ, BINARY, VARIANT).
- PIT: PARENT_HK BINARY(20), SNAPSHOT_DATE DATE, <FK_SAT_N_HK> BINARY(20),
  <FK_SAT_N_LDTS> TIMESTAMP_NTZ, LOAD_DTS TIMESTAMP_NTZ.
- Bridge: BRIDGE_HK BINARY(20), <FK_HUB_N_HK cols>, LOAD_DTS, LOAD_END_DTS.
- Computed Sats: <PARENT>_HK BINARY(20), LOAD_DTS, HASH_DIFF BINARY(20),
  <computed attributes>.
- Section comments: `-- === PIT ===`, `-- === BRIDGE ===`,
  `-- === COMPUTED SATS ===`, `-- === BV HUBS ===`, `-- === BV LINKS ===`.
- Every table COMMENT references the BV object purpose.

Return ONLY SQL. No prose. First chars must be `-- ===`.
"""

def build_business_vault_mermaid_prompt(bv_narrative_md, reverse_summary,
                                        dm_standards_text: str = ""):
    """Produce a Mermaid ER diagram for the Business Vault."""
    return f"""You are a Data Vault 2.0 architect producing a Mermaid
erDiagram for a Business Vault layer.

BV NARRATIVE:
{bv_narrative_md[:10000]}
{_dm_standards_block(dm_standards_text, max_chars=8000)}

RAW VAULT CONTEXT (already exists - draw as dashed-relationship
parents where a BV object reads from them):
{reverse_summary[:6000]}

MERMAID RULES:
- First line MUST be: erDiagram
- Naming: PIT_<Entity>, BRIDGE_<Rel>, CSAT_<Entity>_<Ctx>, BVHUB_<X>,
  BVLNK_<X>
- Show each BV object with its key columns inside {{ ... }}
- Relationships: BVHUB_CUSTOMER ||--o{{ CSAT_CUSTOMER_SEGMENT : "enriches"
- Include relationships back to Raw Vault Hubs where relevant:
  HUB_CUSTOMER ||--o{{ PIT_CUSTOMER : "snapshot of"

Return ONLY the raw Mermaid script — no fences, no prose.
First non-whitespace chars must be: erDiagram
"""

def build_semantic_model_mermaid_prompt(semantic_narrative_md,
                                        dashboard_type):
    """Produce a star-schema Mermaid diagram for the semantic model."""
    return f"""You are a dimensional-modeling specialist producing a
Mermaid erDiagram for a {dashboard_type} star/snowflake schema.

SEMANTIC MODEL NARRATIVE:
{semantic_narrative_md[:10000]}

MERMAID RULES:
- First line MUST be: erDiagram
- Naming: FACT_<measure>, DIM_<entity>
- Facts reference dims via relationships:
  FACT_SALES }}o--|| DIM_CUSTOMER : "customer_key"
  FACT_SALES }}o--|| DIM_DATE : "order_date_key"
- Inside each entity block, include the surrogate key first, then
  foreign keys, then measures/attributes with types:
  FACT_SALES {{
      number SALES_KEY PK
      number CUSTOMER_KEY FK
      date ORDER_DATE_KEY FK
      number QUANTITY
      decimal AMOUNT
  }}
- Show conformed/shared dimensions when multiple facts exist.

Return ONLY the raw Mermaid script. No fences, no prose.
First non-whitespace chars must be: erDiagram
"""

def build_semantic_model_sql_prompt(semantic_narrative_md, dashboard_type):
    """Produce Snowflake DDL for the semantic model facts and dims."""
    return f"""You are a Snowflake DDL generator for a dimensional model.

SEMANTIC MODEL NARRATIVE:
{semantic_narrative_md[:12000]}

Produce CREATE TABLE statements for every Fact and Dimension in the
narrative, targeting Snowflake.

DDL RULES:
- Dimension: <DIM>_KEY NUMBER PRIMARY KEY (surrogate), natural key,
  descriptive attrs, SCD2 columns (VALID_FROM TIMESTAMP_NTZ,
  VALID_TO TIMESTAMP_NTZ, IS_CURRENT BOOLEAN) where appropriate,
  LOAD_DTS TIMESTAMP_NTZ.
- Fact: <FACT>_KEY NUMBER PRIMARY KEY, foreign-key NUMBER columns to
  each DIM referenced, measures as NUMBER(18,2) or appropriate
  numeric types, DATE_KEY NUMBER (join to DIM_DATE).
- Every fact should have a DATE_KEY.
- Section comments: `-- === DIMENSIONS ===`, `-- === FACTS ===`.
- Every table COMMENT names the business process it supports.
- Use Snowflake types (NUMBER, VARCHAR, DATE, TIMESTAMP_NTZ, BOOLEAN).

Return ONLY SQL. No prose, no fences.
First chars must be `-- === DIMENSIONS ===`.
"""

def build_forward_sttm_prompt(bv_narrative_md, reverse_summary,
                              dashboard_type,
                              sttm_template_text: str = ""):
    # Optional user-supplied template wins over the default header.
    tmpl_header = ""
    tmpl_samples = ""
    if sttm_template_text and sttm_template_text.strip():
        _lines = [ln for ln in sttm_template_text.splitlines()
                  if ln.strip()]
        if _lines:
            tmpl_header = _lines[0].strip()
            tmpl_samples = "\n".join(_lines[1:6])[:4000]

    if tmpl_header:
        header_block = (
            "═══════════════════════════════════════════════════════════════════════\n"
            "STTM TEMPLATE — provided by the user. Use this as the\n"
            "AUTHORITATIVE output shape. The header below OVERRIDES the\n"
            "default schema. Match it exactly, column order included.\n"
            "═══════════════════════════════════════════════════════════════════════\n\n"
            "HEADER (first line, exactly — copy verbatim):\n"
            f"{tmpl_header}\n"
        )
        if tmpl_samples:
            header_block += (
                "\nEXAMPLE ROWS from the template (mirror the shape of "
                "these cells):\n"
                f"{tmpl_samples}\n"
            )
        header_block += (
            "\nIf a column from the template has no obvious value, leave "
            "it as an empty quoted cell (\"\") rather than omitting it — "
            "every output row must have the same number of columns as "
            "the template header.\n"
        )
    else:
        header_block = (
            "HEADER:\n"
            '"Source Layer","Source Table","Source Column","Target Layer",'
            '"Target Table","Target Column","Target Data Type",'
            '"Transform Kind","Transform Logic","Grain Notes",'
            '"Business Rule"\n'
        )

    return f"""You are a data mapping specialist. Output ONLY machine-
parseable CSV.

TASK: Build Source-to-Target Mapping from Raw Vault (source) to
Business Vault + Semantic Model (target) for a {dashboard_type} use
case. Produce ONE ROW PER TARGET COLUMN.

BUSINESS VAULT / SEMANTIC MODEL (target):
{bv_narrative_md[:10000]}

RAW VAULT (source):
{reverse_summary[:8000]}

OUTPUT FORMAT:
1. Output ONLY CSV. No fences, no Markdown.
2. First line the EXACT header below.
3. Quote every cell. Escape internal quotes as "".
4. Target Layer in {{BusinessVault, SemanticFact, SemanticDim}}.
5. Transform Kind in {{DirectCopy, Lookup, PIT, Bridge, Computed, Aggregate}}.

{header_block}

Begin now. Response must start with `"`.
"""

def build_forward_catalog_prompt(semantic_md, bv_narrative_md,
                                 dashboard_type):
    return f"""You are a data catalog curator. Output ONLY machine-
parseable CSV.

TASK: Catalog every column across Business Vault and Semantic Model.
Produce ONE ROW PER COLUMN.

SEMANTIC MODEL:
{semantic_md[:8000]}

BUSINESS VAULT:
{bv_narrative_md[:8000]}

RULES:
1. CSV only. Quote every cell.
2. Classification in {{Public, Internal, Confidential, Restricted}}.
3. Sensitivity in {{PII, PCI, PHI, Financial, None}}.
4. Layer in {{BusinessVault, SemanticFact, SemanticDim}}.

HEADER:
"Layer","Entity","Column","Data Type","Business Definition","Classification","Sensitivity","Nullable","Key Type","DQ Rule","Sample Value","Owner Role"

Begin now. First char must be `"`.
"""

def build_forward_domains_prompt(semantic_md, bv_narrative_md,
                                 dashboard_type, rules_text):
    return f"""You are a data governance lead. Output ONLY CSV.

TASK: Map every entity from Business Vault and Semantic Model to a Data
Domain. Produce ONE ROW PER (domain, entity) pair. Every entity must
appear.

DASHBOARD CONTEXT: {dashboard_type}

SEMANTIC MODEL:
{semantic_md[:8000]}

BUSINESS VAULT:
{bv_narrative_md[:6000]}

BANKING RULES (for domain ownership hints):
{rules_text[:2000] or '(none)'}

RULES:
1. CSV only. Quote every cell.
2. Upstream/Downstream comma-separated inside a single quoted cell.
   Use "None" when none.

HEADER:
"Domain","Subdomain","Business Capability","Entity","Entity Layer","Domain Owner","Upstream Domains","Downstream Domains","Critical Data Elements","Governance Notes"

Begin now. First char must be `"`.
"""

def build_raw_vault_dbt_prompt(reverse_summary: str) -> str:
    return f"""You are a senior analytics engineer generating a dbt
project for a Data Vault 2.0 Raw Vault using AutomateDV macros and
medallion architecture (bronze -> silver).

RAW VAULT CONTEXT:
{reverse_summary[:7000]}

OUTPUT FORMAT — CRITICAL:
Respond with ONLY a JSON array. No prose before or after. No Markdown
fences around the JSON. Each element is an object with two fields:

  {{"path": "relative/path/to/file.ext", "content": "file body"}}

Within each "content" value, use `\\n` for newlines and escape
double-quotes as `\\"`. Ensure the JSON is valid and parseable.

Example of the exact shape required (abbreviated):

[
  {{"path": "dbt_project.yml", "content": "name: 'fs_datavault'\\nversion: '1.0.0'\\nprofile: 'snowflake'\\n..."}},
  {{"path": "models/bronze/br_customer.sql", "content": "{{{{ config(materialized='view', tags=['bronze']) }}}}\\nSELECT * FROM {{{{ source('raw', 'customer') }}}}"}}
]

REQUIRED FILES (include ALL of them — keep each "content" concise,
≤ 50 lines of actual code):

1. dbt_project.yml
   - name: 'fs_datavault', version: '1.0.0', profile: 'snowflake'
   - require-dbt-version: [">=1.5.0"]
   - models config with materialization per layer:
     bronze → view, silver → incremental
   - DO NOT add on-run-start / on-run-end hooks. The deploy pipeline
     pre-creates RAW, RAW_VAULT, BUSINESS_VAULT, and GOLD schemas
     before EXECUTE DBT PROJECT runs. Any hook that issues SHOW
     PARAMETER(S) (directly or via a macro like adapter.dispatch) will
     fail in the Snowflake native runtime.

2. packages.yml
   - automate_dv pinned: [">=0.11.0", "<1.0.0"]

3. profiles.yml
   - Snowflake target with account/user/role/warehouse/database
     placeholders and schema_bronze, schema_silver

4. models/bronze/_sources.yml
   - ONE source named 'raw' with every source entity as a table

5. models/bronze/br_<entity>.sql  (one per source entity)
   - select ALL columns from {{{{ source('raw', '<entity>') }}}}
   - add LOAD_DTS, RECORD_SOURCE
   - config(materialized='view', tags=['bronze'])

6. models/silver/raw_vault/stage/stg_<entity>.sql
   - use automate_dv.stage macro
   - hashed_columns for hash keys, derived_columns for LOAD_DTS/RECORD_SOURCE
   - reference br_<entity>

7. models/silver/raw_vault/hubs/hub_<entity>.sql  (one per Hub)
   - use automate_dv.hub macro, incremental
   - config(materialized='incremental', tags=['silver','raw_vault'])

8. models/silver/raw_vault/links/lnk_<rel>.sql
   - use automate_dv.link macro, incremental

9. models/silver/raw_vault/sats/sat_<entity>_<ctx>.sql
   - use automate_dv.sat macro, incremental
   - include src_hashdiff and src_payload

10. models/silver/raw_vault/schema.yml
    - unique + not_null on every HK
    - relationships tests from Link FKs to parent Hub HKs

11. macros/get_record_source.sql
    - Jinja macro returning a namespaced record-source string

AutomateDV parameter names: src_pk, src_nk, src_ldts, src_source,
source_model, src_hashdiff, src_payload.

Every hub/link/sat must use {{{{ config(materialized='incremental',
tags=['silver','raw_vault']) }}}}.
Every bronze must use {{{{ config(materialized='view', tags=['bronze']) }}}}.

Respond NOW with the JSON array only. First character must be `[`.
"""

def build_business_vault_dbt_prompt(forward_summary, reverse_summary):
    return f"""You are a senior analytics engineer generating a dbt
project that extends a Raw Vault with a Business Vault and Gold
semantic layer, using AutomateDV macros, Jinja templating, and
medallion architecture.

BUSINESS VAULT / SEMANTIC MODEL (target):
{forward_summary[:6000]}

RAW VAULT (upstream - already built):
{reverse_summary[:4000]}

OUTPUT FORMAT - CRITICAL:
Respond with ONLY a JSON array. No prose before or after. No Markdown
fences around the JSON. Each element has two fields:

  {{"path": "relative/path/to/file.ext", "content": "file body"}}

Within each "content" value, use `\\n` for newlines and escape
double-quotes as `\\"`. Keep each file CONCISE (<= 50 lines).

Example shape:
[
  {{"path": "models/silver/business_vault/pit/pit_customer.sql", "content": "{{{{ config(materialized='incremental', tags=['silver','bv']) }}}}\\n{{{{ automate_dv.pit(...) }}}}"}},
  {{"path": "models/gold/semantic/dim_customer.sql", "content": "{{{{ config(materialized='table', tags=['gold','semantic']) }}}}\\nSELECT ..."}}
]

REQUIRED FILES (produce ALL of them):

1. models/silver/business_vault/pit/pit_<entity>.sql
   (PIT tables using automate_dv.pit macro)

2. models/silver/business_vault/bridge/bridge_<rel>.sql
   (Bridge tables using automate_dv.bridge)

3. models/silver/business_vault/computed_sats/csat_<n>.sql
   (Computed Satellites - incremental, derive business metrics)

4. models/silver/business_vault/bv_hubs/bv_hub_<x>.sql
   (BV Hubs for new business concepts)

5. models/silver/business_vault/bv_links/bv_lnk_<x>.sql
   (BV Links for new relationships)

6. models/gold/semantic/dim_<n>.sql  (one per Dimension)
   - materialized='table', SCD2 where semantic model specifies it
   - surrogate key _SK
   - effective_from, effective_to, is_current columns for SCD2

7. models/gold/semantic/fct_<n>.sql  (one per Fact)
   - materialized='table'
   - foreign-key _SK columns to relevant dims
   - DATE_KEY, measure columns, grain statement in comment

8. models/silver/business_vault/schema.yml
   - unique + not_null on keys

9. models/gold/semantic/schema.yml
   - relationships tests from fact FKs to dim keys

CRITICAL:
- Dim/fact models reference ref('pit_...'), ref('bridge_...'),
  ref('csat_...') - NEVER Raw Vault directly.
- Every BV file: config(materialized='incremental',
  tags=['silver','bv']).
- Every gold file: config(materialized='table',
  tags=['gold','semantic']).
- Every fact has a DATE_KEY and a surrogate SK.

Respond NOW with the JSON array only. First character must be `[`.
"""

def build_dbt_tests_prompt(raw_vault_dbt_md, business_vault_dbt_md):
    return f"""You are a data quality engineer writing dbt tests for a
production Data Vault 2.0 + Dimensional pipeline.

RAW VAULT DBT (excerpt):
{raw_vault_dbt_md[:3500]}

BUSINESS VAULT + SEMANTIC DBT (excerpt):
{business_vault_dbt_md[:3500]}

OUTPUT FORMAT - CRITICAL:
Respond with ONLY a JSON array. No prose. No Markdown fences.
Each element has two fields:

  {{"path": "tests/generic/my_test.sql", "content": "SELECT ..."}}

Within each "content" value, use `\\n` for newlines and escape
double-quotes as `\\"`. Keep each file CONCISE (<= 40 lines).

REQUIRED FILES:

1. tests/generic/test_hashkey_no_collision.sql
   (custom generic test ensuring no hash key maps to multiple natural keys)

2. tests/generic/test_sat_references_parent_hub.sql
   (every satellite hash key exists in its parent Hub)

3. tests/generic/test_pit_consistency.sql
   (PIT FK hash+ldts pairs exist in the referenced satellite)

4. tests/singular/test_row_count_sanity.sql
   (example singular test: bronze row count >= corresponding Hub count)

5. models/silver/raw_vault/schema.yml  (ADDITIONAL data_tests using
   the generic tests above)

6. models/silver/business_vault/schema.yml  (same pattern for BV)

7. models/gold/semantic/schema.yml  (dim SK uniqueness, NK not_null,
   fact grain test via compound unique on FK combinations)

Respond NOW with the JSON array only. First character must be `[`.
"""

MEDALLION_DV_FULL_SPEC = """\
You are a senior analytics engineer producing a complete, scalable,
maintainable, and production-ready dbt implementation that transforms data
from a Business Vault Data Model (Data Vault 2.0) into a Medallion
Architecture (Bronze → Silver → Gold).

The solution must:
- Follow dbt best practices
- Use reusable macros
- Leverage Jinja templating
- Be modular, testable, and extensible
- Be optimized for performance, maintainability, and scalability
- Support enterprise-grade deployment standards

INPUTS YOU WILL BE GIVEN
1. Business Vault Data Model — Hubs, Links, Satellites, BV entities,
   Point-in-Time (PIT) tables, Bridge tables, Reference tables.
2. Source-to-Target Mapping (STTM) — Source systems and tables, source
   columns and data types, target models and columns, transformation
   rules, business rules, join conditions, filtering conditions, derived
   column logic, data quality rules, incremental loading logic, surrogate
   key generation rules.

ARCHITECTURE
- Bronze Layer: Raw ingestion models, source-aligned, minimal
  transformation. Materialize as `view`.
- Silver Layer: Cleansed, standardized, conformed Business Vault models
  (staging, hubs, links, satellites, PITs, bridges). Materialize as
  `incremental`.
- Gold Layer: Business-ready dimensional, fact, aggregate, and
  consumption models. Materialize as `table`.

DBT REQUIREMENTS
- Use incremental models wherever applicable.
- Implement snapshots for slowly changing dimensions when required.
- Use ephemeral models for reusable intermediate logic.
- Create reusable macros for: hash key generation, hash diff generation,
  surrogate key generation, audit column creation, standardized null
  handling, data type casting, incremental filtering, soft delete
  handling, record deduplication, data quality validations.

JINJA REQUIREMENTS
- Parameterize all repetitive logic.
- Use loops for repetitive column generation.
- Use conditional logic for environment-specific behavior.
- Support configurable schema/database naming.
- Enable reusable model templates.

PROJECT STRUCTURE
models/
  bronze/
  silver/
  gold/
macros/
tests/
snapshots/
seeds/
analyses/

For each layer, provide: dbt model SQL files, YAML schema files, tests,
documentation. Required generated files: source definitions
(sources.yml), bronze/silver/gold models, reusable macros, generic
tests, custom tests, snapshots (if applicable), documentation
(schema.yml), dbt_project.yml, packages.yml.

CODING STANDARDS
- SQL style best practices, lowercase SQL keywords, meaningful aliases.
- Inline comments for complex logic, model-level documentation.
- Idempotent and deterministic transformations.

PERFORMANCE
- Optimize joins and filters. Cluster where supported.
- Incremental strategies: merge / insert_overwrite / delete+insert as
  appropriate.
- Minimize full table scans. Push predicates down. Avoid unnecessary
  CTE materialization.

DATA QUALITY
Implement tests for: not_null, unique, relationships, accepted_values,
freshness, custom business rule validations, referential integrity,
duplicate detection, hash key uniqueness, hash diff change detection.

AUDIT AND GOVERNANCE
All models must include the audit columns: record_source,
load_datetime, effective_datetime, expiry_datetime, current_flag,
created_by, updated_by, batch_id. Include lineage documentation,
business definitions, technical metadata, column-level descriptions.

ENVIRONMENT & DEPLOYMENT
Support dev, test, prod environments with environment-specific
configurations. CI/CD compatible. Include variables, target-specific
configurations, hooks (pre/post), run-operation compatibility.

IMPLEMENTATION RULES
- Strictly align implementation with STTM specifications.
- Preserve all business rules from the Business Vault model.
- Use Data Vault 2.0 modeling standards.
- Ensure historical tracking where required.
- Implement SCD Type 2 logic where applicable.
- Generate reusable and parameterized code; avoid hardcoded values.
- Externalize configurable parameters.

ADVANCED REQUIREMENTS
- Create metadata-driven patterns where possible.
- Support automated model generation. Enable extensibility for future
  entities. Implement reusable framework components. Provide error
  handling and logging patterns.

NAMING CASE — HARD RULE (two separate rules, do not confuse)

RULE A — dbt node references match the FILENAME (lowercase per dbt
convention). The argument inside ``ref(...)`` and the value of
``source_model=`` MUST be the filename without extension. Files are
``stg_customer.sql``, ``hub_customer.sql``, ``br_customer.sql`` — so:
  ✓ ref('stg_customer')             ✗ ref('STG_CUSTOMER')
  ✓ source_model=['stg_customer']    ✗ source_model=['STG_CUSTOMER']
  ✓ source_model='br_customer'       ✗ source_model='BR_CUSTOMER'
This is mandatory. dbt resolves these by filename — uppercase here
produces "depends on a node named 'STG_CUSTOMER' which was not found"
and the build fails.

RULE B — SQL identifiers and physical Snowflake names are UPPER_SNAKE_CASE.
This applies to the SECOND arg of ``source()`` (which is the physical
Snowflake table name) and to every column / hash key / payload column
arg passed to automate_dv macros:
  ✓ source('raw', 'CUSTOMER')        ✗ source('raw', 'customer')
  ✓ src_pk='CUSTOMER_HK'              ✗ src_pk='customer_hk'
  ✓ src_nk=['ACCOUNT_NUM','TENANT_ID']  ✗ src_nk=['account_num']
  ✓ HASH_DIFF, RECORD_SOURCE          ✗ hash_diff, record_source
  ✓ SELECT CUSTOMER_ID, EMAIL ...     ✗ select customer_id, email ...

Mnemonic: dbt `ref()` matches files (lowercase). Anything Snowflake
will see as an identifier (column, table NAME inside source(), macro
args) is UPPER_SNAKE_CASE.

NEVER emit ``<entity>``, ``<context>``, ``<bk_col>``, or any other
``<...>``-bracketed placeholder in generated files. Use the real
entity name from the project context. Files with placeholders fail
to compile.

PER-LAYER SCHEMA MAPPING — HARD RULE
Each Medallion layer materializes into a DIFFERENT Snowflake schema.
This separation is enforced via dbt ``+schema:`` configs in
``dbt_project.yml`` and a custom ``generate_schema_name`` macro that
the deploy pipeline injects. The macro returns the ``+schema`` value
verbatim (uppercased) — NOT dbt's default
``{target.schema}_{custom_schema}`` concatenation.

Layer → schema mapping (set these in ``dbt_project.yml``):
  - bronze/                         → {target.schema}    (no override)
  - silver/staging/                 → +schema: raw_vault
  - silver/raw_vault/               → +schema: raw_vault
  - silver/business_vault/          → +schema: business_vault
  - gold/                           → +schema: gold
  - snapshots/                      → +schema: snapshots (optional)

The deploy pipeline pre-creates RAW, RAW_VAULT, and BUSINESS_VAULT
schemas before ``dbt build`` runs, so models will find their target
schemas ready. DO NOT use absolute database.schema.table references in
SQL — let dbt resolve via ``ref()`` and ``source()`` and the +schema
config above will route models to the right physical schema.

SNOWFLAKE NATIVE DBT RUNTIME CONSTRAINTS (HARD RULES — DO NOT VIOLATE)
This project will execute on Snowflake's native dbt runtime via
EXECUTE DBT PROJECT. The following constraints are non-negotiable —
violating any will cause the build to fail at runtime with no
in-project remedy:
- DO NOT set `query_tag` anywhere (not in dbt_project.yml, not in
  profiles.yml, not in `{{ config(...) }}` blocks, not in pre/post
  hooks). Native dbt rejects `SHOW PARAMETER` which dbt-snowflake runs
  whenever query_tag is configured.
- DO NOT set `persist_docs`. The native runtime cannot run the
  underlying `COMMENT ON ...` privilege escalations.
- DO NOT use `query-comment:`. Same reason as query_tag.
- DO NOT use `env_var(...)` calls — env vars are not available in
  EXECUTE DBT PROJECT. Use dbt `var(...)` or hardcode for the target
  environment.
- DO NOT add `on-run-start` / `on-run-end` hooks that run SHOW or
  ALTER SESSION statements.
- External packages (automate_dv, dbt_utils, dbt_expectations) MAY be
  referenced in code; the deploy pipeline replaces `automate_dv.X(...)`
  calls with local fallback macros named `dv_X(...)` that have
  identical signatures, so generated code can be written either way.
"""

MEDALLION_DV_FILE_CONSTRAINTS = """\
MEDALLION + DATA VAULT 2.0 — KEY CONSTRAINTS FOR THIS FILE

Audit columns (every Silver/Gold model that holds business records):
  record_source, load_datetime, effective_datetime, expiry_datetime,
  current_flag, created_by, updated_by, batch_id.

Materialization by layer:
  - bronze/  → view
  - silver/  → incremental
  - gold/    → table
  - snapshots/ → snapshot
  - seeds → CSV only

Reusable macros are mandatory (don't inline logic that could be a macro):
hash key generation, hash diff, surrogate key, audit columns, null
handling, data type casting, incremental filtering, soft delete, dedup,
data quality validations.

Jinja: parameterize, use loops for repetitive columns, use conditionals
for env-specific behaviour, support configurable schema/database.

SQL: lowercase keywords, meaningful aliases, inline comments for
complex logic.

SNOWFLAKE NATIVE RUNTIME — HARD RULES (violate = build fails):
- NO `query_tag` anywhere. NO `persist_docs`. NO `query-comment:`.
- NO `env_var(...)` — use `var(...)` instead.
- NO `on-run-start`/`on-run-end` with SHOW or ALTER SESSION statements.
- automate_dv.* and dbt_utils.* macro calls ARE allowed — the deploy
  pipeline shims them to local fallbacks. Stick to the documented
  signatures of automate_dv.stage / .hub / .link / .sat / .pit /
  .bridge.

NAMING CASE — HARD RULE (two sub-rules)

(A) dbt node refs match the FILENAME (lowercase). The argument inside
``ref(...)`` and the value of ``source_model=`` MUST equal the dbt
filename without extension — files are lowercase per dbt convention:
  ✓ ref('stg_customer'),  source_model=['stg_customer']
  ✗ ref('STG_CUSTOMER'),  source_model=['STG_CUSTOMER']
Uppercase here causes "depends on a node named '...' which was not
found" — dbt won't fall back to lowercase matching.

(B) SQL identifiers and physical Snowflake names are UPPER_SNAKE_CASE:
the second arg of ``source()`` (physical table name), all column names,
aliases, src_pk/src_nk/src_payload/src_hashdiff/src_ldts/src_source
values, hashed_columns keys/values, SELECT-list columns:
  ✓ source('raw', 'CUSTOMER'),  src_nk=['ACCOUNT_NUM','TENANT_ID']
  ✓ HASH_DIFF, RECORD_SOURCE,  CUSTOMER_HK
  ✗ source('raw', 'customer'),  src_nk=['account_num']

NEVER emit ``<entity>``, ``<context>``, or any ``<...>``-bracketed
placeholder. Use real names from the project context — placeholder
text causes "depends on a node named 'STG_<entity>'" errors.

PER-LAYER SCHEMA — HARD RULE
Each Medallion layer goes in a different Snowflake schema, set via
dbt ``+schema:`` in ``dbt_project.yml`` (resolved via the custom
generate_schema_name macro injected by the deploy pipeline):
  - bronze/                  → {target.schema}    (no override)
  - silver/staging,raw_vault → +schema: raw_vault
  - silver/business_vault    → +schema: business_vault
  - gold/                    → +schema: gold
RAW, RAW_VAULT, BUSINESS_VAULT, and GOLD schemas are pre-created by the
deploy pipeline. Do not write absolute database.schema.table refs.

DO NOT emit a `generate_schema_name` macro in any file (e.g.
`macros/native_schema_overrides.sql`, `macros/get_schema_name.sql`,
`macros/schema.sql`). The deploy pipeline injects the canonical
override. Two definitions in the same project triggers a dbt
compilation error: ``dbt found two macros named "generate_schema_name"``.
Same applies to `generate_database_name` and `generate_alias_name` —
do not override those either.
"""

_DV_STD_ABBREVIATIONS = """\
APPROVED ABBREVIATION TABLE — use these abbreviations in entity and column
names. Format: (ABBR, EXPANSION, DOMAIN, ABBR_TO_USE).

-- PARTY DOMAIN
('CUST', 'Customer', 'PARTY', 'CUST')
('CLNT', 'Client', 'PARTY', 'CLNT')
('PRTY', 'Party', 'PARTY', 'PRTY')
('PRSN', 'Person', 'PARTY', 'PRSN')
('EMPL', 'Employee', 'PARTY', 'EMPL')
('VNDR', 'Vendor', 'PARTY', 'VNDR')
('BRWR', 'Borrower', 'PARTY', 'BRWR')
('BENE', 'Beneficiary', 'PARTY', 'BENE')
('CNTR', 'Counterparty', 'PARTY', 'CNTR')
('AGNT', 'Agent', 'PARTY', 'AGNT')

-- ACCOUNT DOMAIN
('ACCT', 'Account', 'ACCOUNT', 'ACCT')
('DPST', 'Deposit', 'ACCOUNT', 'DPST')
('OVRDR', 'Overdraft', 'ACCOUNT', 'OVRDR')
('LDGR', 'Ledger', 'ACCOUNT', 'LDGR')
('GL', 'General Ledger', 'ACCOUNT', 'GL')
('SUBL', 'Sub-Ledger', 'ACCOUNT', 'SUBL')
('PORT', 'Portfolio', 'ACCOUNT', 'PORT')
('SUBSID', 'Subsidiary', 'ACCOUNT', 'SUBSID')

-- FINANCE / TRANSACTION DOMAIN
('TXN', 'Transaction', 'FINANCE', 'TXN')
('TRN', 'Transaction', 'FINANCE', 'TRN')
('PYMT', 'Payment', 'FINANCE', 'PYMT')
('AMT', 'Amount', 'FINANCE', 'AMT')
('BAL', 'Balance', 'FINANCE', 'BAL')
('INTRST', 'Interest', 'FINANCE', 'INTRST')
('RATE', 'Rate', 'FINANCE', 'RATE')
('MRGN', 'Margin', 'FINANCE', 'MRGN')
('FEE', 'Fee', 'FINANCE', 'FEE')
('CHRG', 'Charge', 'FINANCE', 'CHRG')
('XFER', 'Transfer', 'FINANCE', 'XFER')
('CRDTL', 'Credit Limit', 'FINANCE', 'CRDTL')
('LMT', 'Limit', 'FINANCE', 'LMT')
('EXCH', 'Exchange', 'FINANCE', 'EXCH')

-- LOAN / CREDIT DOMAIN
('LN', 'Loan', 'CREDIT', 'LN')
('MTG', 'Mortgage', 'CREDIT', 'MTG')
('COLL', 'Collateral', 'CREDIT', 'COLL')
('INVST', 'Investment', 'CREDIT', 'INVST')
('FACLT', 'Facility', 'CREDIT', 'FACLT')
('EXPSR', 'Exposure', 'CREDIT', 'EXPSR')

-- PRODUCT DOMAIN
('PROD', 'Product', 'PRODUCT', 'PROD')
('PRDCT', 'Product', 'PRODUCT', 'PRDCT')
('SRVC', 'Service', 'PRODUCT', 'SRVC')
('OFFR', 'Offer', 'PRODUCT', 'OFFR')
('CNTRCT', 'Contract', 'PRODUCT', 'CNTRCT')
('AGRMT', 'Agreement', 'PRODUCT', 'AGRMT')
('BNDLE', 'Bundle', 'PRODUCT', 'BNDLE')

-- REFERENCE / CODE DOMAIN
('CURR', 'Currency', 'REFERENCE', 'CURR')
('CNTRY', 'Country', 'REFERENCE', 'CNTRY')
('CTRY', 'Country', 'REFERENCE', 'CTRY')
('BR', 'Branch', 'REFERENCE', 'BR')
('BRN', 'Branch', 'REFERENCE', 'BRN')
('INST', 'Institution', 'REFERENCE', 'INST')
('RGN', 'Region', 'REFERENCE', 'RGN')
('DEPT', 'Department', 'REFERENCE', 'DEPT')
('GRP', 'Group', 'REFERENCE', 'GRP')
('CD', 'Code', 'REFERENCE', 'CD')
('TYP', 'Type', 'REFERENCE', 'TYP')
('STAT', 'Status', 'REFERENCE', 'STAT')
('KYC', 'Know Your Customer', 'REFERENCE', 'KYC')

-- GENERIC / COMMON
('ID', 'Identifier'), ('NBR', 'Number'), ('NUM', 'Number'),
('NM', 'Name'), ('FRST', 'First'), ('LST', 'Last'), ('MDL', 'Middle'),
('ADDR', 'Address'), ('EMAIL', 'Email Address'), ('PHNE', 'Phone'),
('DT', 'Date'), ('DTS', 'Date Timestamp'), ('YR', 'Year'),
('EFF', 'Effective'), ('EXPRY', 'Expiry'), ('DESCR', 'Description'),
('DEFN', 'Definition'), ('FLG', 'Flag'), ('REF', 'Reference'),
('SRC', 'Source'), ('SYS', 'System'), ('REC', 'Record'),
('ORG', 'Organization'), ('BSN', 'Business'), ('MGMT', 'Management'),
('TIN', 'Tax Identification Number'), ('SSN', 'Social Security Number'),
('RLS', 'Relationship')

USAGE: When constructing entity names or identifying domains, prefer
the abbreviation column over the full word. Domains (PARTY, ACCOUNT,
FINANCE, CREDIT, PRODUCT, REFERENCE) are the canonical Data Domain
classifications.
"""

_DV_STD_NAMING = """\
NAMING CONVENTIONS — applies to all Raw Vault entities and columns.

ENTITY NAMING:
  Hub              → HUB_<BUSINESS_NOUN>                  (e.g. HUB_CUSTOMER)
  Link             → LNK_<NOUN1>_<NOUN2>  alphabetical    (e.g. LNK_ACCOUNT_CUSTOMER)
  Satellite        → SAT_<SOURCE_SYSTEM>_<PARENT_NOUN>_<DESCRIPTOR>
                                                          (e.g. SAT_ACCTS_CUSTOMER_DETAILS)
  Multi-Active Sat → MSAT_<SOURCE_SYSTEM>_<PARENT_NOUN>_<DESCRIPTOR>
  Effectivity Sat  → ESAT_<SOURCE_SYSTEM>_<LINK_NAME>
  PIT Table        → PIT_<HUB_NAME>
  Bridge Table     → BRG_<BUSINESS_CONCEPT>

RULES:
- UPPER_SNAKE_CASE for all physical entity names
- Use the approved abbreviation table — do NOT spell out words that
  have an abbreviation
- Link nouns must be in alphabetical order: LNK_ACCOUNT_CUSTOMER not
  LNK_CUSTOMER_ACCOUNT
- Satellite source-system suffix uses double underscore: __<SOURCE_SYSTEM>
- Business nouns are singular: HUB_CUSTOMER not HUB_CUSTOMERS

COLUMN NAMING:
  Hash Key         → <NOUN>_HK              BINARY(32)
  Business Key     → exact source name      source-dependent type
  Hashdiff         → <ENTITY_SHORT>_HASHDIFF BINARY(32)
  Load Timestamp   → LOAD_DTS               TIMESTAMP_NTZ
  Record Source    → REC_SRC                VARCHAR(100)
  Effectivity From → EFF_FROM_DTS           TIMESTAMP_NTZ
  Effectivity To   → EFF_TO_DTS             TIMESTAMP_NTZ
  Multi-Active Key → exact source name      source-dependent type
  Attribute        → exact source name      source-dependent type

COLUMN RULES:
- Vault metadata columns (HK, HASHDIFF, LOAD_DTS, REC_SRC) use
  UPPER_SNAKE_CASE
- ATTRIBUTE and BUSINESS KEY columns must use the EXACT source column
  name — do not abbreviate or rename them
- Apply UPPER(TRIM()) before hashing
- logical_name may be a readable/expanded form; column_name must
  match the source
"""

_DV_STD_HASH = """\
HASH KEY STANDARDS

ALGORITHM: SHA2_BINARY(256), output type BINARY(32). All hash keys are
BINARY(32).

NULL HANDLING: COALESCE(CAST(column AS VARCHAR), '-1') — replace nulls
with the string '-1' before hashing.

MULTI-COLUMN HASH KEYS:
- Concatenate columns with '||' as delimiter
- Sort columns alphabetically by column name before concatenation
- Example: SHA2_BINARY(COALESCE(ACCT_ID, '-1') || '||' ||
  COALESCE(CUST_ID, '-1'), 256)

PRE-PROCESSING:
- Apply UPPER(TRIM(value)) to all character columns before hashing
- Numeric columns: CAST to VARCHAR with no leading/trailing spaces

HASHDIFF (for Satellites):
- Computed over ALL descriptive attribute columns in the satellite
- EXCLUDE metadata columns: LOAD_DTS, REC_SRC, <PARENT>_HK,
  <SAT>_HASHDIFF itself
- Sort columns alphabetically before concatenation
- Changes in hashdiff indicate a new record version is needed

NAMING:
- Hub hash key: <HUB_NOUN>_HK (e.g. CUSTOMER_HK)
- Link hash key: <LINK_NOUN1>_<LINK_NOUN2>_HK alphabetical
  (e.g. ACCOUNT_CUSTOMER_HK)
- Hashdiff: <ABBREVIATED_SAT_NAME>_HASHDIFF
"""

_DV_STD_METADATA_COLS = """\
STANDARD METADATA COLUMNS — every vault entity must include these in
the order shown.

HUB COLUMNS:
  1. <NOUN>_HK            BINARY(32)      NOT NULL    PK / hash key
  2. <BK_COLUMN(S)>       source type     NOT NULL    business key(s)
  3. LOAD_DTS             TIMESTAMP_NTZ   NOT NULL    load timestamp
  4. REC_SRC              VARCHAR(100)    NOT NULL    record source

LINK COLUMNS:
  1. <LNK_NAME>_HK        BINARY(32)      NOT NULL    PK / link hash key
  2. <HUB1_NOUN>_HK       BINARY(32)      NOT NULL    FK to hub 1
  3. <HUB2_NOUN>_HK       BINARY(32)      NOT NULL    FK to hub 2
  4. (additional hub HKs if n-ary link)
  5. LOAD_DTS             TIMESTAMP_NTZ   NOT NULL
  6. REC_SRC              VARCHAR(100)    NOT NULL

SATELLITE COLUMNS:
  1. <PARENT>_HK          BINARY(32)      NOT NULL    FK to parent hub or link
  2. LOAD_DTS             TIMESTAMP_NTZ   NOT NULL
  3. <SAT_SHORT>_HASHDIFF BINARY(32)      NOT NULL    hashdiff for change detection
  4. REC_SRC              VARCHAR(100)    NOT NULL
  5. ... descriptive attribute columns ...

MULTI-ACTIVE SATELLITE — additional column after LOAD_DTS:
  <MULTI_ACTIVE_KEY_COLUMN>  the key that distinguishes rows within
  the same snapshot. PK is composite: parent_HK + LOAD_DTS +
  multi_active_key.

EFFECTIVITY SATELLITE COLUMNS:
  1. <LINK>_HK            BINARY(32)      NOT NULL    FK to link
  2. LOAD_DTS             TIMESTAMP_NTZ   NOT NULL
  3. <ESAT_SHORT>_HASHDIFF BINARY(32)     NOT NULL
  4. REC_SRC              VARCHAR(100)    NOT NULL
  5. EFF_FROM_DTS         TIMESTAMP_NTZ   NOT NULL    effective from
  6. EFF_TO_DTS           TIMESTAMP_NTZ               effective to (NULL = current)
"""

_DV_STD_LINK_RULES = """\
LINK DESIGN RULES

WHEN TO CREATE A LINK:
- Two or more business keys appear together in the same source record
- The link represents the relationship between those business entities
- Example: ACCT_ID + CUST_ID in an account record → LNK_ACCOUNT_CUSTOMER

LINK HASH KEY:
- Concatenation of ALL participating hub hash keys, sorted
  alphabetically, separated by '||'
- Example: SHA2_BINARY(ACCOUNT_HK || '||' || CUSTOMER_HK, 256)

DEGENERATE LINKS:
- If a source has only ONE business key (no other BKs to link to),
  do NOT create a link — create a hub and satellites only.
- Degenerate keys (e.g. transaction number that's not reused) go into
  the link as a degenerate attribute, not as a separate hub.

LINK NAMING:
- Alphabetical noun order: LNK_ACCOUNT_CUSTOMER not LNK_CUSTOMER_ACCOUNT
- For n-ary links (3+ hubs): list all nouns alphabetically, abbreviated

HUB REUSE DETECTION:
- Before creating a new hub, check the registry / context for an
  existing hub with the same business key
- If HUB_CUSTOMER already exists and the source contains CUST_ID,
  REUSE that hub. State the reuse decision in your rationale.
- Only create a new hub if no matching business key exists in the
  approved registry.

REFERENCE HUBS — pre-seeded, always reuse:
  HUB_CURRENCY, HUB_COUNTRY, HUB_BRANCH, HUB_GL_ACCOUNT
- When a source column maps to one of these (e.g. CURR_CD → currency),
  reference the existing hub. Do NOT create a new hub for these.
"""

_DV_STD_SATELLITE_RULES = """\
SATELLITE DESIGN RULES

SOURCE-SPECIFIC SATELLITES (CRITICAL):
- The Raw Vault preserves data EXACTLY as delivered by each source system
- Create ONE satellite per source system per hub or link — ALWAYS
- If two source systems deliver customer name and address, create TWO
  satellites:
    SAT_CUSTOMER_DETAILS__ACCT_SYS  (from account system)
    SAT_CUSTOMER_DETAILS__CRM_SYS   (from CRM system)
- NEVER merge attributes from different source systems into one Raw
  Vault satellite. Merging/reconciliation is the Business Vault's job.

CHOOSING SATELLITE TYPE:
  Standard descriptive attributes              → SAT
  Multiple rows per snapshot (e.g. phone list) → MSAT (multi-active)
  Tracks when a link relationship starts/ends  → ESAT (effectivity)

SATELLITE SPLITTING BY CHANGE FREQUENCY:
- Consider splitting one physical source into multiple satellites if:
  - Some columns change FAST (balances, status) and others change SLOW
    (names, addresses)
  - Splitting reduces hashdiff churn and storage bloat
- Faster-changing satellite: name with FAST or VOLATILE descriptor
- Slower-changing satellite: name with STATIC or DETAILS descriptor

CHANGE FREQUENCY CLASSIFICATION:
- FAST:   >20% of values change between snapshots, or columns like
          amounts, balances, rates, status flags
- SLOW:   1-20% change, or names, addresses, type codes
- STATIC: <1% change, or IDs, birth dates, SSN, account-open dates

CONFIDENCE FLAGS — tag each satellite with the confidence level of
its type decision:
- HIGH:     profiling data confirms change frequency and key structure
- MEDIUM:   metadata/definitions provide semantic evidence
- LOW:      inferred from column names + abbreviation table only
- INFERRED: no metadata available, purely AI inference from data patterns
"""

_DV_STANDARDS_BY_ARTIFACT = {
    "lineage": [
        ("Approved Abbreviations (for consistent node labeling)",
         _DV_STD_ABBREVIATIONS),
    ],
    "sttm": [
        ("Naming Conventions", _DV_STD_NAMING),
        ("Standard Metadata Columns", _DV_STD_METADATA_COLS),
        ("Hash Key Standards", _DV_STD_HASH),
        ("Approved Abbreviations", _DV_STD_ABBREVIATIONS),
    ],
    "catalog": [
        ("Approved Abbreviations (use the DOMAIN column to classify entities)",
         _DV_STD_ABBREVIATIONS),
        ("Naming Conventions", _DV_STD_NAMING),
    ],
    "domain": [
        ("Approved Abbreviations — the DOMAIN column is the canonical "
         "Data Domain classification (PARTY, ACCOUNT, FINANCE, CREDIT, "
         "PRODUCT, REFERENCE)",
         _DV_STD_ABBREVIATIONS),
    ],
    "raw_vault": [
        ("Naming Conventions", _DV_STD_NAMING),
        ("Standard Metadata Columns", _DV_STD_METADATA_COLS),
        ("Hash Key Standards", _DV_STD_HASH),
        ("Link Design Rules", _DV_STD_LINK_RULES),
        ("Satellite Design Rules", _DV_STD_SATELLITE_RULES),
        ("Approved Abbreviations", _DV_STD_ABBREVIATIONS),
    ],
}

def _dv_standards_block(artifact: str) -> str:
    """
    Return a single formatted text block of DV 2.0 standards relevant
    to the given artifact. Returns empty string if the artifact key is
    unknown (defensive). The block is meant to be appended to the
    Reverse Engineering prompts so generated artifacts conform to the
    bank's authoritative standards.
    """
    sections = _DV_STANDARDS_BY_ARTIFACT.get(artifact, [])
    if not sections:
        return ""
    parts = [
        "",
        "═" * 72,
        "DATA VAULT 2.0 STANDARDS — these are the bank's authoritative",
        "rules. Generated output MUST conform. Treat as ground truth.",
        "═" * 72,
    ]
    for title, body in sections:
        parts.append("")
        parts.append(f"### {title}")
        parts.append("")
        parts.append(body.rstrip())
    parts.append("")
    parts.append("═" * 72)
    parts.append(
        "END DATA VAULT STANDARDS. The above rules override any "
        "conflicting instinct from the model."
    )
    parts.append("═" * 72)
    return "\n".join(parts)

def build_dbt_planner_prompt(context: str, layer: str) -> str:
    """
    Ask the LLM to plan the file list for a dbt project. Response is
    a plain list of relative paths, one per line. Simple format that
    any LLM gets right.

    The planner leads with the full Medallion + Data Vault spec
    (MEDALLION_DV_FULL_SPEC), then layers on the per-call layer scope
    (raw_vault / business_vault / dbt_tests). The layer scope is what
    constrains which deliverables this particular planner call should
    produce — the rest of the app dispatches the layers separately.
    """
    if layer == "raw_vault":
        guidance = """
LAYER SCOPE FOR THIS PLANNER CALL: Bronze + Silver Raw Vault.

Required deliverables (use real entity names from the context, not
placeholders):

- dbt_project.yml
- packages.yml
- models/_sources.yml          (defines source 'raw' with all upstream tables)
- models/bronze/br_<entity>.sql           (one per source entity, view materialization)
- models/bronze/_bronze.yml              (bronze model docs + tests)
- models/silver/staging/stg_<entity>.sql  (one per entity, incremental, with hashed_columns)
- models/silver/raw_vault/hubs/hub_<entity>.sql           (incremental)
- models/silver/raw_vault/links/lnk_<relationship>.sql    (incremental, one per relationship)
- models/silver/raw_vault/sats/sat_<entity>_<context>.sql (incremental, one per descriptive context)
- models/silver/raw_vault/_raw_vault.yml (schema tests: unique+not_null on every HK,
                                          relationships from sat→hub and lnk→hub, freshness)
- macros/dv/dv_record_source.sql   (record_source macro)
- macros/dv/dv_audit_columns.sql   (returns the audit-column SQL block per spec)
- tests/generic/test_hash_key_uniqueness.sql (custom generic test)

Rules:
- bronze must materialize as `view`; silver staging/hub/link/sat as `incremental`.
- Use automate_dv.stage / automate_dv.hub / automate_dv.link / automate_dv.sat
  in models — the deploy pipeline shims them to local fallbacks.
- Tags: bronze → ['bronze','medallion']; silver staging → ['silver','medallion','staging'];
  silver raw_vault hubs → ['silver','medallion','raw_vault','hub']; analogous for link/sat.
- _raw_vault.yml MUST cover every silver model with at least the HK uniqueness
  and not_null tests, plus relationships from sats/links to their parent hubs.
"""
    elif layer == "business_vault":
        guidance = """
LAYER SCOPE FOR THIS PLANNER CALL: Silver BV + Gold semantic layer +
shared macros + snapshots + analyses.

Required deliverables (use real entity names from the context):

- dbt_project.yml
- packages.yml
- models/silver/business_vault/staging/stg_<entity>.sql   (incremental)
- models/silver/business_vault/hubs/hub_<entity>.sql      (incremental)
- models/silver/business_vault/links/lnk_<relationship>.sql (incremental)
- models/silver/business_vault/sats/sat_<entity>_<context>.sql (incremental)
- models/silver/business_vault/pit/pit_<entity>.sql       (incremental, optional but include
                                                           if PIT is in the model)
- models/silver/business_vault/bridge/bridge_<rel>.sql    (incremental, if bridge is in model)
- models/silver/business_vault/_business_vault.yml       (schema + tests)
- models/gold/semantic/dim_<entity>.sql                  (materialized=table, SCD2-ready)
- models/gold/semantic/fct_<process>.sql                 (materialized=table)
- models/gold/semantic/_semantic.yml                     (schema + tests with relationships
                                                          fct→dim)
- macros/bv/dv_audit_columns.sql                         (audit-column block macro)
- macros/bv/semantic_sk.sql                              (surrogate-key macro)
- snapshots/snap_<scd2_entity>.sql                       (one snapshot per SCD2 entity)
- tests/generic/test_hash_diff_change_detection.sql      (custom generic test)
- analyses/<example_analysis>.sql                        (one example analysis)

Rules:
- Silver BV uses ref() to Raw Vault silver models only — never source().
- Gold uses ref() to PIT/bridge/csat/BV hubs.
- All Silver/Gold models include the audit columns block (use
  dv_audit_columns macro).
- Tags: silver BV → ['silver','medallion','business_vault', <hub|link|sat|pit|bridge>];
  gold → ['gold','medallion','semantic', <dim|fact>].
- SCD2 columns on dims: effective_datetime, expiry_datetime, current_flag.
"""
    else:  # dbt_tests
        guidance = """
LAYER SCOPE FOR THIS PLANNER CALL: Project-wide test layer.

Required deliverables:

- tests/generic/test_hash_key_uniqueness.sql      (custom generic test)
- tests/generic/test_referential_integrity.sql   (custom generic test)
- tests/generic/test_hash_diff_change_detection.sql (custom generic test)
- tests/generic/test_audit_columns_present.sql   (custom generic test)
- tests/singular/test_no_orphan_satellites.sql   (singular test)
- tests/singular/test_dim_keys_unique.sql        (singular test)
- models/silver/raw_vault/_raw_vault_tests.yml   (schema overlay applying tests)
- models/silver/business_vault/_business_vault_tests.yml (schema overlay)
- models/gold/semantic/_semantic_tests.yml       (schema overlay)

Rules:
- Generic tests accept (model, column_name=None) and return failing rows.
- Singular tests are plain SELECT statements that return rows iff they fail.
- Schema overlays attach tests via `data_tests:` (dbt 1.5+ key name).
"""

    return f"""{MEDALLION_DV_FULL_SPEC}

═══════════════════════════════════════════════════════════════════════
PLANNER TASK
═══════════════════════════════════════════════════════════════════════
You are now planning the FILE LIST for the dbt project described above.
Output ONLY a plain list of relative file paths — one per line. No
prose, no numbering, no Markdown, no JSON.

{guidance}

CONTEXT (source / target data model):
{context[:5000]}

Based on the context, list the actual files you would create. Use real
entity/table names from the context — not placeholders like "<entity>".
Paths must be relative (no leading slash). Between 12 and 30 files
total. You MUST include every YAML config file listed in the layout
(_sources, _<layer>, schema) — missing YAML is a common failure mode.

Example first lines of a valid response:
dbt_project.yml
packages.yml
models/_sources.yml
models/bronze/br_customer.sql
models/silver/raw_vault/hubs/hub_customer.sql

Respond now with ONLY the file paths, one per line.

CRITICAL: Do NOT wrap the list in JSON, quotes, or one big string with
``\\n`` escapes. The first character of your response must be ``d`` from
``dbt_project.yml`` — not ``"`` or ``[``.
"""

def build_dbt_file_prompt(file_path: str, context: str,
                          layer: str) -> str:
    """
    Ask the LLM to produce the raw body of ONE dbt file. Instruct it
    to output just the file content — no fences, no FILE markers, no
    JSON, no prose. Maximally simple so the response is usable as-is.
    """
    lp = file_path.lower()

    # File-specific guidance to steer the LLM
    if lp.endswith("dbt_project.yml"):
        if layer == "business_vault":
            specifics = """
Produce a production **dbt_project.yml** for a **Medallion** pipeline:
**Silver** = business_vault (incremental) + **Gold** = semantic (table).

Required top-level keys: name ('fs_datavault' or similar), version,
profile ('snowflake'), config-version: 2, require-dbt-version,
model-paths: ['models'], macro-paths: ['macros'], seed-paths: ['seeds'],
test-paths: ['tests'], target-path: 'target'.

Under `models:` use a single project package (e.g. `fs_datavault:`) with
nested folders matching the repo:
  - `silver/business_vault/` → +materialized incremental, +tags
    ['silver','medallion','bv']
  - `gold/semantic/` → +materialized table, +tags ['gold','medallion',
    'semantic']

Use **Jinja** in YAML where helpful (e.g. `+schema` from vars). Add
`vars:` with optional `semantic_schema`, `bv_schema`. Do NOT invent
custom macros in YAML that are not defined in this repo."""
        else:
            specifics = """
Produce a production **dbt_project.yml** for **Medallion** +
**Data Vault Raw Vault**:
  - **Bronze** (`models/bronze/`): views, tags ['bronze','medallion']
  - **Silver** staging + raw_vault: incremental, tags ['silver',
    'medallion','staging'] or ['silver','medallion','raw_vault']
  - **Gold**: reserved (+tags ['gold','medallion']) for future marts

Required: name, version, profile 'snowflake', config-version 2,
require-dbt-version [">=1.5.0","<2.0.0"], model-paths, macro-paths,
seed-paths, test-paths, target-path.

Nested `models:` tree MUST mirror folders: `bronze`, `silver/staging`,
`silver/raw_vault` with sub-keys `hubs`, `links`, `sats` as needed.
Set +materialized per layer (view / incremental). Do NOT use
+persist_docs, +query_tag, query-comment, or any on-run hooks that
issue SHOW or ALTER SESSION — these break the Snowflake native dbt
runtime."""
    elif lp.endswith("packages.yml"):
        specifics = """
Pin Datavault-UK/automate_dv at [">=0.11.0", "<1.0.0"]. Include
dbt_utils and dbt_expectations if useful."""
    elif lp.endswith("profiles.yml"):
        specifics = """
Define a 'snowflake' profile with dev and prod targets. The project
runs under EXECUTE DBT PROJECT, which provides session context — so
hardcode `account: ''` and `user: ''` (empty strings; Snowflake ignores
them in native runtime). For role, warehouse, database, schema use dbt
`var(...)` references like
{{ var('snowflake_role', 'ACCOUNTADMIN') }},
{{ var('snowflake_warehouse', 'COMPUTE_WH') }},
{{ var('snowflake_database', 'ANALYTICS') }},
{{ var('snowflake_schema', 'PUBLIC') }}.
Threads: 4. Type: snowflake.
NEVER use env_var() — env vars are not available in EXECUTE DBT PROJECT."""
    elif "_sources.yml" in lp:
        specifics = """
Define ONE source named 'raw' whose tables match the upstream entities
in the context. Every table gets a description. Include columns with
data_tests (not_null on keys) where obvious from the context."""
    elif "__models.yml" in lp or "_models.yml" in lp:
        specifics = """
YAML file listing **models:** entries for every .sql model in THIS
folder only (not child folders). For each model: name, description,
config with tags aligned to **Medallion** layer (bronze / silver /
gold). Use `meta:` for owner / medallion_layer. This file is the dbt
best-practice pattern for co-located YAML — no SQL inside."""
    elif lp.endswith("schema.yml"):
        specifics = """
Define models: entries for every model in this folder (based on the
context). For hubs/links/sats: unique+not_null on the hash key.
For gold facts/dims: unique+not_null on surrogate keys, relationships
from fact FKs to dim keys. Add brief descriptions to each model and
column."""
    elif "/bronze/br_" in lp:
        specifics = """
**Medallion Bronze** thin view (1:1 with landing / external table).
Use **Jinja** only — no hard-coded database names.

Structure:
{{ config(materialized='view', tags=['bronze','medallion']) }}

SELECT
    *,
    CURRENT_TIMESTAMP() AS LOAD_DTS,
    {{ dv_record_source('<SOURCE_SYSTEM>') }} AS RECORD_SOURCE
FROM {{ source('raw', '<entity>') }}

Use the `dv_record_source` macro from `macros/dv/record_source.sql`, or
inline `{{ var('default_record_source', 'LANDING') }}` if simpler."""
    elif "/stage/stg_" in lp or "/staging/stg_" in lp:
        stage_tag_block = ("['silver','business_vault','staging']"
                           if ("business_vault" in lp
                               or layer == "business_vault")
                           else "['silver','raw_vault','staging']")
        specifics = """
Use automate_dv.stage macro. Reference the br_<entity> model via
{{ ref('br_<entity>') }}. Define hashed_columns (e.g.
'<ENTITY>_HK' from the business key columns, and for links combined
HKs). Define derived_columns for LOAD_DATETIME, RECORD_SOURCE, and
the rest of the audit-column block (CREATED_BY, UPDATED_BY, BATCH_ID,
EFFECTIVE_DATETIME, EXPIRY_DATETIME, CURRENT_FLAG).

CRITICAL — derived_columns values are SQL EXPRESSIONS, not Jinja.
They are captured inside `{%- set yaml_metadata -%}` and parsed by
`fromyaml`, then emitted by automate_dv as raw SQL. Jinja expressions
like `{{ var('x') }}` will NOT re-render and Snowflake will reject
them as syntax errors. Use Snowflake SQL functions instead:
  ✓ RECORD_SOURCE:       "'CRM'"               (literal string)
  ✓ LOAD_DATETIME:       "CURRENT_TIMESTAMP()" (SQL fn)
  ✓ EFFECTIVE_DATETIME:  "CURRENT_TIMESTAMP()"
  ✓ EXPIRY_DATETIME:     "CAST(NULL AS TIMESTAMP_NTZ)"
  ✓ CURRENT_FLAG:        "TRUE"
  ✓ CREATED_BY:          "CURRENT_USER()"
  ✓ UPDATED_BY:          "CURRENT_USER()"
  ✓ BATCH_ID:            "TO_VARCHAR(CURRENT_TIMESTAMP())"
  ✗ CREATED_BY:          "'{{ var(''dbt_user'', ''dbt'') }}'"  ← FAILS

Structure:
{{ config(materialized='incremental', tags=__TAG_BLOCK__) }}

{%- set yaml_metadata -%}
source_model: '<br_entity>'
derived_columns:
  RECORD_SOURCE: "'<SOURCE>'"
  LOAD_DATETIME: "CURRENT_TIMESTAMP()"
  EFFECTIVE_DATETIME: "CURRENT_TIMESTAMP()"
  EXPIRY_DATETIME: "CAST(NULL AS TIMESTAMP_NTZ)"
  CURRENT_FLAG: "TRUE"
  CREATED_BY: "CURRENT_USER()"
  UPDATED_BY: "CURRENT_USER()"
  BATCH_ID: "TO_VARCHAR(CURRENT_TIMESTAMP())"
hashed_columns:
  <ENTITY>_HK:
    - '<BK_COL>'
  HASH_DIFF:
    is_hashdiff: true
    columns:
      - '<attr1>'
      - '<attr2>'
{%- endset -%}
{% set metadata_dict = fromyaml(yaml_metadata) %}
{{ automate_dv.stage(
    include_source_columns=true,
    source_model=metadata_dict['source_model'],
    derived_columns=metadata_dict['derived_columns'],
    hashed_columns=metadata_dict['hashed_columns']
) }}""".replace("__TAG_BLOCK__", stage_tag_block)
    elif "/hubs/hub_" in lp:
        hub_tag_block = ("['silver','business_vault','hub']"
                         if ("business_vault" in lp
                             or layer == "business_vault")
                         else "['silver','raw_vault','hub']")
        specifics = """
Use automate_dv.hub. Reference the stg_<entity> model. Parameters:
src_pk (e.g. '<ENTITY>_HK'), src_nk (business key), src_ldts, src_source,
source_model (list with the stage model name).

Structure:
{{ config(materialized='incremental', tags=__TAG_BLOCK__) }}

{{ automate_dv.hub(
    src_pk='<ENTITY>_HK',
    src_nk='<ENTITY>_BK',
    src_ldts='LOAD_DATETIME',
    src_source='RECORD_SOURCE',
    source_model=['stg_<entity>']
) }}""".replace("__TAG_BLOCK__", hub_tag_block)
    elif "/links/lnk_" in lp:
        link_tag_block = ("['silver','business_vault','link']"
                          if ("business_vault" in lp
                              or layer == "business_vault")
                          else "['silver','raw_vault','link']")
        specifics = """
Use automate_dv.link. Parameters: src_pk (link hash key), src_fk (list
of parent hub HKs), src_ldts, src_source, source_model.

Structure:
{{ config(materialized='incremental', tags=__TAG_BLOCK__) }}

{{ automate_dv.link(
    src_pk='LNK_<REL>_HK',
    src_fk=['<HUB_A>_HK', '<HUB_B>_HK'],
    src_ldts='LOAD_DATETIME',
    src_source='RECORD_SOURCE',
    source_model=['stg_<entity>']
) }}""".replace("__TAG_BLOCK__", link_tag_block)
    elif "/sats/sat_" in lp or "/satellites/sat_" in lp:
        sat_tag_block = ("['silver','business_vault','sat']"
                         if ("business_vault" in lp
                             or layer == "business_vault")
                         else "['silver','raw_vault','sat']")
        specifics = """
Use automate_dv.sat. Parameters: src_pk (parent hash key), src_hashdiff,
src_payload (list of descriptive columns), src_eff, src_ldts,
src_source, source_model.

Structure:
{{ config(materialized='incremental', tags=__TAG_BLOCK__) }}

{{ automate_dv.sat(
    src_pk='<ENTITY>_HK',
    src_hashdiff='HASH_DIFF',
    src_payload=['<attr1>', '<attr2>'],
    src_eff='EFFECTIVE_FROM',
    src_ldts='LOAD_DATETIME',
    src_source='RECORD_SOURCE',
    source_model=['stg_<entity>']
) }}""".replace("__TAG_BLOCK__", sat_tag_block)
    elif "/pit/pit_" in lp:
        specifics = """
Use automate_dv.pit. Parameters: source_model (parent hub ref), src_pk,
as_of_dates_table, satellites (dict of sat_name → {pk, ldts}).

Structure:
{{ config(materialized='incremental', tags=['silver','bv','pit']) }}

{{ automate_dv.pit(
    source_model='hub_<entity>',
    src_pk='<ENTITY>_HK',
    as_of_dates_table='as_of_date',
    satellites={
        'sat_<entity>_detail': {'pk': {'PK': '<ENTITY>_HK', 'Name': '<ENTITY>_HK'},
                                'ldts': {'LDTS': 'LOAD_DATETIME', 'Name': 'LOAD_DATETIME'}}
    },
    stage_tables_ldts={'stg_<entity>': 'LOAD_DATETIME'},
    src_ldts='LOAD_DATETIME'
) }}"""
    elif "/bridge/bridge_" in lp:
        specifics = """
Use automate_dv.bridge. Parameters: source_model, src_pk, src_ldts,
bridge_walk (dict describing hub-link-hub path), as_of_dates_table,
stage_tables_ldts.

Structure:
{{ config(materialized='incremental', tags=['silver','bv','bridge']) }}

{{ automate_dv.bridge(
    source_model='hub_<anchor>',
    src_pk='<ANCHOR>_HK',
    src_ldts='LOAD_DATETIME',
    bridge_walk={...},
    as_of_dates_table='as_of_date',
    stage_tables_ldts={...}
) }}"""
    elif "/csat_" in lp or "/computed_sats/" in lp:
        specifics = """
A Computed Satellite. Materialize incremental. Derive business metrics
by joining Raw Vault satellites and applying business rules. Keep the
parent hash key and add LOAD_DATETIME, HASH_DIFF, plus computed
attributes.

Structure:
{{ config(materialized='incremental', tags=['silver','bv','csat']) }}

WITH src AS (
  SELECT ... FROM {{ ref('sat_<entity>_detail') }}
)
SELECT
    <ENTITY>_HK,
    <computed_attrs>,
    CURRENT_TIMESTAMP() AS LOAD_DATETIME,
    MD5_BINARY(CONCAT_WS('|', ...)) AS HASH_DIFF
FROM src"""
    elif "/gold/semantic/dim_" in lp:
        specifics = """
Dimension table per Medallion + Data Vault 2.0 spec. materialized='table'.
Include: surrogate key _SK, natural business key (_BK), parent hash key
(_HK), descriptive attributes, FULL audit-column block per spec
(record_source, load_datetime, effective_datetime, expiry_datetime,
current_flag, created_by, updated_by, batch_id), and SCD2 columns
where appropriate. Source from PIT + satellite refs only — no source().

Structure:
{{ config(materialized='table', tags=['gold','medallion','semantic','dim']) }}

SELECT
    ROW_NUMBER() OVER (ORDER BY <ENTITY>_HK) AS <ENTITY>_SK,
    <ENTITY>_HK,
    <ENTITY>_BK,
    <descriptive_attrs>,
    -- audit columns (per Medallion + DV 2.0 spec)
    record_source,
    load_datetime,
    load_datetime                  AS effective_datetime,
    LEAD(load_datetime) OVER (PARTITION BY <ENTITY>_HK
                              ORDER BY load_datetime) AS expiry_datetime,
    CASE WHEN LEAD(load_datetime) OVER (PARTITION BY <ENTITY>_HK
                                        ORDER BY load_datetime) IS NULL
         THEN TRUE ELSE FALSE END AS current_flag,
    CURRENT_USER()                 AS created_by,
    CURRENT_USER()                 AS updated_by,
    {{ var('batch_id', 'CURRENT_TIMESTAMP()::STRING') }} AS batch_id
FROM {{ ref('pit_<entity>') }}"""
    elif "/gold/semantic/fct_" in lp:
        specifics = """
Fact table per Medallion + Data Vault 2.0 spec. materialized='table'.
Include: surrogate key _SK, foreign-key _SK columns to every referenced
dim, DATE_KEY (NUMBER, YYYYMMDD), measures, plus the FULL audit-column
block per spec (record_source, load_datetime, effective_datetime,
expiry_datetime, current_flag, created_by, updated_by, batch_id).

Structure:
{{ config(materialized='table', tags=['gold','medallion','semantic','fact']) }}

SELECT
    ROW_NUMBER() OVER (ORDER BY <LNK>_HK) AS <FCT>_SK,
    <dim_a>_SK AS DIM_A_SK,
    TO_NUMBER(TO_CHAR(<date_col>, 'YYYYMMDD')) AS DATE_KEY,
    <measures>,
    -- audit columns (per Medallion + DV 2.0 spec)
    record_source,
    load_datetime,
    load_datetime AS effective_datetime,
    NULL          AS expiry_datetime,
    TRUE          AS current_flag,
    CURRENT_USER() AS created_by,
    CURRENT_USER() AS updated_by,
    {{ var('batch_id', 'CURRENT_TIMESTAMP()::STRING') }} AS batch_id
FROM {{ ref('bridge_<rel>') }}
JOIN {{ ref('dim_<a>') }} ON ...
"""
    elif "/seeds/" in lp and lp.endswith(".csv"):
        specifics = """
CSV **seed** for dbt `ref('as_of_dates')` used by AutomateDV PIT/bridge
macros. Single column header `AS_OF_DATE` (DATE or ISO text). Include
~10 representative calendar dates spanning the analytic window. No
Markdown — raw CSV only."""
    elif "macros/dv/record_source" in lp:
        specifics = """
Implement `dv_record_source(system_name)` as a **Jinja macro** in SQL:

{% macro dv_record_source(system_name) %}
  '{{ system_name | upper | replace("'", "''") }}'
{% endmacro %}

No YAML — only macro definition(s). Used by bronze models."""
    elif "macros/dv/hashdiff_columns" in lp:
        specifics = """
Jinja macro that accepts a list of column names and returns a YAML-
friendly list block for AutomateDV `hashed_columns` / HASH_DIFF
payloads, e.g. `{% macro dv_hashdiff_list(cols) %}...{% endmacro %}`.
Use `{% for c in cols %}` loops."""
    elif "macros/medallion/schema_suffix" in lp:
        specifics = """
Jinja macro `medallion_schema_suffix(layer)` where layer is one of
'bronze','silver','gold' — returns a short suffix STRING (e.g. 'bronze',
'silver', 'gold') used for documentation, tagging, or building model
descriptions. Do NOT define `generate_schema_name`, `generate_database_name`,
or `generate_alias_name` in this file or anywhere — those are reserved
overrides handled by the deploy pipeline."""
    elif "macros/bv/pit_satellites" in lp:
        specifics = """
Jinja macro that **returns** a Python-dict literal (as Jinja text) for
AutomateDV `pit` `satellites=` parameter, built with `{% set sats = {}
%}` or dict literals keyed by satellite model names. Document expected
keys (pk, ldts) in comments."""
    elif "macros/bv/semantic_sk" in lp:
        specifics = """
Jinja macro `semantic_sk(pk_col)` wrapping `{{ dbt_utils.generate_surrogate_key([pk_col]) }}` or Snowflake `HASH` pattern — one macro file,
multiple macros allowed."""
    elif "tests/generic/" in lp:
        specifics = """
A custom generic test. Accepts `model` (and optional columns)
parameters. Returns any failing rows — an empty result = test passes.

Structure:
{% test <test_name>(model, column_name=None) %}
SELECT <problematic rows>
FROM {{ model }}
WHERE <failure condition>
{% endtest %}"""
    elif "tests/singular/" in lp:
        specifics = """
A singular test. Plain SELECT that returns rows only when the test
fails.

Structure:
-- Test purpose: <what it checks>
SELECT <columns>
FROM {{ ref('<model>') }}
WHERE <failure condition>"""
    elif "macros/" in lp:
        specifics = """
A reusable **Jinja macro** file under `macros/`. Use
`{{% macro name(args) %}} ... {{% endmacro %}}` with clear argument
names. Prefer composition: call other project macros with
`{{ return(...) }}` only when needed (dbt 1.2+). Top comment `{# ... #}`
describes medallion usage (bronze vs silver vs gold)."""
    else:
        if layer == "business_vault":
            specifics = """
**Medallion Silver + Gold** dbt SQL or YAML. Use `{{ config(...) }}`
as the first statement for SQL models. Silver BV must use `ref()` to
Raw Vault **silver** models (hub_/lnk_/sat_/stg_) only — never
`source()`. Gold semantic models use `ref()` to PIT/bridge/csat/BV
hubs. Use Jinja for joins, effective dates, and SK generation; factor
repeated expressions into `macros/bv/`."""
        elif layer == "raw_vault":
            specifics = """
**Medallion Bronze + Silver Raw Vault** dbt SQL or YAML. Bronze uses
`{{ source('raw', ...) }}`; silver uses `{{ ref('br_...') }}` and
`{{ ref('stg_...') }}` then AutomateDV hub/link/sat macros. Start SQL
with `{{ config(materialized=..., tags=[...]) }}` matching the folder
(medallion layer)."""
        else:
            specifics = """Follow dbt/AutomateDV conventions.
Start with a {{ config(...) }} block. Use {{ ref(...) }} and
{{ source(...) }} for references. Document purpose at the top."""

    layer_reminder = ""
    if layer == "raw_vault":
        layer_reminder = (
            "\nLAYER: Medallion bronze (views) → silver staging → "
            "silver raw_vault (incremental hubs/links/sats). "
            "AutomateDV + Jinja + `macros/dv/`.\n"
        )
    elif layer == "business_vault":
        layer_reminder = (
            "\nLAYER: Medallion silver (BV) → gold (semantic). "
            "refs to Raw Vault silver only; macros in `macros/bv/`.\n"
        )

    # Short context slice to conserve tokens
    return f"""{MEDALLION_DV_FILE_CONSTRAINTS}

═══════════════════════════════════════════════════════════════════════
FILE GENERATION TASK
═══════════════════════════════════════════════════════════════════════
You are generating ONE file of a dbt project. Output ONLY the raw file
content — no Markdown fences, no explanations, no FILE: markers, no
JSON wrapping. Your entire response becomes the file body verbatim.

FILE TO GENERATE: {file_path}
{layer_reminder}
{specifics}

PROJECT CONTEXT:
{context[:4000]}

CRITICAL: Respond with ONLY the file contents. First line IS the
first line of the file. No preamble like "Here is" or "```sql".
"""

def _dm_standards_block(dm_standards_text: str, max_chars: int = 14000) -> str:
    """Build a "Data Modeling Standards (uploaded)" block that the RV
    and BV generators must honor. Returns "" when no standards are
    supplied (so the prompt is unchanged for legacy callers)."""
    if not dm_standards_text or not dm_standards_text.strip():
        return ""
    body = dm_standards_text[:max_chars]
    return f"""

═══════════════════════════════════════════════════════════════════════
ENTERPRISE DATA MODELING STANDARDS — uploaded by the user
═══════════════════════════════════════════════════════════════════════
The following text is the bank's / enterprise's own data modeling
standards document. THESE STANDARDS OVERRIDE any defaults you may
have learned. When the standards conflict with built-in conventions
in this prompt, the standards win.

Apply these rules CONCRETELY in the generated artifact:
  • Naming conventions for tables, columns, hash keys, satellites
  • Hash-key data types and lengths
  • Required metadata columns (load timestamps, record source, hashdiff)
  • Satellite-splitting rules (by source, rate of change, sensitivity)
  • Audit, PII, classification columns
  • Comment / DESCRIPTION conventions on every object
  • Any prohibitions or required patterns

If the standards leave a topic unaddressed, fall back to standard
Data Vault 2.0 conventions.

{body}
═══════════════════════════════════════════════════════════════════════
"""


def build_raw_vault_narrative_prompt(metadata_summary: str,
                                     transformation_rules_md: str = "",
                                     dm_standards_text: str = "") -> str:
    # Inject the transformation-rules Markdown as additional context if
    # provided. Source-driven only — kept out of the metadata blob so the
    # model can clearly see "this is the same source's business logic,
    # already extracted by an earlier pass".
    rules_block = ""
    if transformation_rules_md and transformation_rules_md.strip():
        rules_block = f"""

═══════════════════════════════════════════════════════════════════════
TRANSFORMATION RULES & BUSINESS LOGIC (extracted from the same source)
═══════════════════════════════════════════════════════════════════════
The following Markdown report enumerates every transformation rule,
derivation, filter, lookup, surrogate-key strategy, type conversion, and
business rule observed in the parsed source. USE THIS to:
  • Identify candidate Hubs (entities with a stable, unique business key
    surfaced by key-handling rules)
  • Identify candidate Links (relationships implied by join logic)
  • Identify candidate Satellites (descriptive attributes plus their
    rate-of-change hints from filter/SCD/dedup rules)
  • Group attributes correctly by source system and change frequency
  • PRESERVE the legacy business semantics — the Raw Vault must be
    loadable from the existing logic without redesigning rules

{transformation_rules_md[:18000]}
═══════════════════════════════════════════════════════════════════════
"""
    return f"""You are a senior Data Vault 2.0 architect.

Given this source metadata (which may span multiple DataStage jobs and CSV
files), design ONE unified Raw Vault data model (Hubs, Links, Satellites)
following Dan Linstedt's Data Vault 2.0 standards.

SOURCE METADATA:
{metadata_summary}
{rules_block}{_dm_standards_block(dm_standards_text)}

═══════════════════════════════════════════════════════════════════════
CRITICAL MULTI-JOB / MULTI-FILE RULES — READ CAREFULLY
═══════════════════════════════════════════════════════════════════════
1. DEDUPLICATE HUBS. If a business key (e.g. CUSTOMER_ID) appears in
   entities across multiple jobs or files, produce EXACTLY ONE Hub for
   that key — never one Hub per source entity. The metadata section
   "SHARED BUSINESS KEY CANDIDATES" lists these explicitly.
2. NAME HUBS BY BUSINESS CONCEPT, not source table. Hub names are
   HUB_CUSTOMER, HUB_ORDER, HUB_PRODUCT — NOT HUB_JOB1_CUSTOMER.
3. CONSOLIDATE SATELLITES by source system and rate-of-change — not by
   job. Multiple jobs may feed attributes into the SAME Satellite.
4. LINKS USE DEDUPLICATED HUBS. If Job A has Customer↔Order and Job B
   has Customer↔Payment, produce LNK_CUSTOMER_ORDER and
   LNK_CUSTOMER_PAYMENT — both referencing ONE shared HUB_CUSTOMER.
5. HONOR CROSS-JOB DATAFLOWS. When Job B consumes Job A's output, the
   downstream entities DO NOT introduce new Hubs.
═══════════════════════════════════════════════════════════════════════

Return clean Markdown with these sections:

## Overview
2-3 sentences on the proposed model. Explicitly state how many UNIQUE
Hubs you identified AFTER deduplicating across files/jobs.

## Hubs
For each Hub: **HUB_<n>** - business key, grain.
  - **Source entities consolidated into this Hub:** list every source
    entity (with its source file/job) that feeds this Hub.

## Links
For each Link: **LNK_<n>** - connected Hubs, cardinality, driving key,
contributing jobs/files.

## Satellites
For each Satellite: **SAT_<entity>_<context>** - parent Hub/Link,
attributes, rate of change, contributing jobs/files.

## Design Notes
Bullet list covering:
- How many Hubs were consolidated across jobs (before vs after dedup)
- Cross-job data lineage observations
- PII handling, naming conventions, assumptions

Use proper Markdown syntax - headings, bold, bullet lists. No code blocks,
no diagrams - just narrative Markdown.
""" + _dv_standards_block("raw_vault")

def build_raw_vault_mermaid_prompt(metadata_summary: str,
                                   transformation_rules_md: str = "",
                                   dm_standards_text: str = "") -> str:
    rules_block = ""
    if transformation_rules_md and transformation_rules_md.strip():
        rules_block = f"""

═══════════════════════════════════════════════════════════════════════
TRANSFORMATION RULES & BUSINESS LOGIC (use as additional design context)
═══════════════════════════════════════════════════════════════════════
{transformation_rules_md[:8000]}
═══════════════════════════════════════════════════════════════════════
"""
    return f"""You are a Data Vault 2.0 architect.

For this source metadata (which may span MULTIPLE DataStage jobs / CSV files),
produce ONE Mermaid erDiagram script that shows every UNIQUE Hub, Link, and
Satellite — deduplicated across all files.

SOURCE METADATA:
{metadata_summary}
{rules_block}{_dm_standards_block(dm_standards_text, max_chars=8000)}

CRITICAL RULES:
- If the same business key appears in multiple source entities, produce ONE
  shared Hub (e.g. HUB_CUSTOMER) that all related entities relate to.
  See "SHARED BUSINESS KEY CANDIDATES" in the metadata above.
- Do NOT create HUB_JOB1_X and HUB_JOB2_X for the same concept X.
- Links connect deduplicated Hubs.
- Satellites hang off their Hub/Link parent.

MERMAID RULES:
- First line must be: erDiagram
- Entity names: HUB_CUSTOMER, LNK_ACCOUNT_CUSTOMER (alphabetical),
  SAT_<SOURCE_SYSTEM>_<PARENT>_<DESCRIPTOR>
- Relationships: HUB_X ||--o{{ LNK_Y : "has"
- Satellite attachment: HUB_X ||--o{{ SAT_X_DETAIL : "describes"
- Include key columns inside each entity block — use REC_SRC (not
  RECORD_SOURCE) and BINARY hash keys per the bank's DV 2.0 standards:
  HUB_CUSTOMER {{
      binary CUSTOMER_HK
      varchar CUSTOMER_ID
      timestamp LOAD_DTS
      varchar REC_SRC
  }}

Return ONLY the raw Mermaid script. No ```mermaid fences. No prose.
First non-whitespace characters must be the word "erDiagram".
"""

def build_raw_vault_sql_prompt(metadata_summary: str,
                               transformation_rules_md: str = "",
                               dm_standards_text: str = "") -> str:
    rules_block = ""
    if transformation_rules_md and transformation_rules_md.strip():
        rules_block = f"""

═══════════════════════════════════════════════════════════════════════
TRANSFORMATION RULES & BUSINESS LOGIC (use as additional design context)
═══════════════════════════════════════════════════════════════════════
The following rules were extracted by an earlier pass over the SAME
source metadata. They describe legacy business semantics that the Raw
Vault must preserve. Use them to guide:
  • Hub selection (which business keys are stable / unique)
  • Satellite splitting (group attributes by the rate-of-change hints
    surfaced in dedup / SCD / filter rules)
  • Link identification (joins → Links between deduplicated Hubs)

{transformation_rules_md[:12000]}
═══════════════════════════════════════════════════════════════════════
"""
    return f"""You are a Snowflake DDL generator.

For this source metadata (which may span MULTIPLE DataStage jobs / CSV
files), produce ONLY Snowflake CREATE TABLE statements for ONE unified
Data Vault 2.0 Raw Vault — every UNIQUE Hub, Link, and Satellite after
deduplicating across all files.

SOURCE METADATA:
{metadata_summary}
{rules_block}{_dm_standards_block(dm_standards_text)}

CRITICAL MULTI-JOB RULES:
- If a business key appears in entities from multiple jobs/files, emit
  ONE Hub table — do NOT create one Hub per job.
  See "SHARED BUSINESS KEY CANDIDATES" in the metadata.
- Consolidate Satellites by source system per the SATELLITE RULES below.
- Every table's COMMENT should mention which source jobs/files contribute.

DDL FORMATTING:
- Use Snowflake data types (VARCHAR, NUMBER, TIMESTAMP_NTZ, BINARY, VARIANT).
- Apply ALL naming, hashing, and metadata-column rules from the
  DATA VAULT 2.0 STANDARDS appendix below — types, lengths, column
  ordering, and entity prefixes come from there.
- Include PRIMARY KEY on HK columns.
- Group with section comments: `-- === HUBS ===`, `-- === LINKS ===`,
  `-- === SATELLITES ===`.

Return ONLY the SQL. No prose before or after. No ```sql fences.
First non-whitespace characters must be `-- === HUBS ===`.
""" + _dv_standards_block("raw_vault")

def build_sttm_prompt(metadata_summary: str,
                      raw_vault_sql: str = "",
                      pre_raw_vault: bool = False,
                      sttm_template_text: str = "") -> str:
    """
    Build the STTM prompt.

    Three modes are supported:

    1. ``pre_raw_vault=True`` — the STTM is generated BEFORE the Raw Vault
       data model exists. The output MUST NOT mention Hub / Link /
       Satellite / Raw Vault / Data Vault concepts at all. Targets are
       proposed STAGING tables that mirror the source structure (e.g.
       ``STG_<source>``). This keeps STTM as a pure source-to-staging
       mapping artifact that can be reviewed independently of any vault
       design decisions.

    2. ``raw_vault_sql`` provided — the STTM uses the exact table and
       column names from the supplied Raw Vault DDL as ground truth.

    3. Neither — the legacy "guess the Data Vault target" behavior, kept
       only for backward compatibility with callers that still rely on
       it. New callers should pick mode 1 or 2 explicitly.

    ``sttm_template_text`` (optional): a sample STTM file uploaded by the
    user. Its FIRST non-empty line becomes the authoritative output
    header (overriding the default schema), and remaining lines are
    treated as example rows the model should mirror in shape. The
    template wins over the built-in HEADER in either mode.
    """
    # Parse the optional STTM template — its first non-empty line is the
    # authoritative header; subsequent non-empty lines are sample rows.
    tmpl_header = ""
    tmpl_samples = ""
    if sttm_template_text and sttm_template_text.strip():
        _lines = [ln for ln in sttm_template_text.splitlines()
                  if ln.strip()]
        if _lines:
            tmpl_header = _lines[0].strip()
            # Keep up to 5 sample rows, total truncated to 4000 chars
            sample_lines = _lines[1:6]
            tmpl_samples = "\n".join(sample_lines)[:4000]

    def _template_block(default_header: str, default_example: str) -> str:
        """Build the HEADER + EXAMPLE section, swapping in the user's
        template when provided. Returns the full block as a string."""
        if tmpl_header:
            block = (
                "═══════════════════════════════════════════════════════════════════════\n"
                "STTM TEMPLATE — provided by the user. Use this as the\n"
                "AUTHORITATIVE output shape. The header below OVERRIDES any\n"
                "default schema. Match it exactly, column order included.\n"
                "═══════════════════════════════════════════════════════════════════════\n\n"
                "HEADER (first line, exactly — copy verbatim):\n"
                f"{tmpl_header}\n"
            )
            if tmpl_samples:
                block += (
                    "\nEXAMPLE ROWS from the template (mirror the shape, "
                    "tone, and level of detail of these cells):\n"
                    f"{tmpl_samples}\n"
                )
            block += (
                "\nIf a column from the template has no obvious source-driven\n"
                "value, leave it as an empty quoted cell (\"\") rather than\n"
                "omitting it — every output row must have the same number of\n"
                "columns as the template header.\n"
            )
            return block
        # Fallback: built-in default
        return (
            f"HEADER (first line, exactly):\n{default_header}\n\n"
            f"{default_example}"
        )

    # ── Mode 1: source → staging, NO Raw Vault concepts at all ──────────
    if pre_raw_vault:
        default_header = (
            '"Source System","Source Table","Source Column","Source Data Type",'
            '"Target Table","Target Column","Target Data Type","Transformation",'
            '"Business Key","PII Flag","Notes"'
        )
        default_example = (
            'EXAMPLE — same CUSTOMER_ID from two different jobs, mapped to staging:\n'
            '"DataStage-JobA","CUSTOMER_FEED","CUSTOMER_ID","VARCHAR(20)",'
            '"STG_CUSTOMER_FEED","CUSTOMER_ID","VARCHAR(20)",'
            '"UPPER(TRIM(CUSTOMER_ID))","Y","N","Source primary key"\n'
            '"DataStage-JobB","LOAN_APPLICATIONS","CUSTOMER_ID","VARCHAR(20)",'
            '"STG_LOAN_APPLICATIONS","CUSTOMER_ID","VARCHAR(20)",'
            '"passthrough","Y","N","Foreign key into CUSTOMER_FEED"'
        )
        return f"""You are a data mapping specialist. Output must be
machine-parseable CSV — not prose, not Markdown.

TASK: Build a Source-to-Target Mapping (STTM) from this source metadata
(which may span MULTIPLE jobs / files / tech stacks). The TARGET in this
STTM is a clean STAGING layer that mirrors the source structure — one
staging table per distinct source entity.

STRICT RULES — these override anything else you may have been trained on:
1. DO NOT mention "Raw Vault", "Data Vault", "Hub", "Link", "Satellite",
   "HK", "HASHDIFF", "LOAD_DTS", "REC_SRC", or any vault-modeling
   terminology in any cell. This STTM is produced BEFORE the vault
   model is designed — vault concepts are out of scope.
2. Target tables MUST be named ``STG_<SOURCE_ENTITY>`` (uppercase, with
   non-alphanumeric characters replaced by underscore).
3. Target columns MUST mirror the source column names verbatim, with
   one row per source column. No renaming, no prefixing.
4. Target data types should be Snowflake-friendly equivalents of the
   source types (VARCHAR, NUMBER, TIMESTAMP_NTZ, DATE, BOOLEAN, etc.).
5. The "Transformation" column captures the legacy logic that produces
   each column (UPPER, TRIM, CAST, CASE, lookup, etc.) — copy verbatim
   from the source where possible. Use "passthrough" if the value is
   moved as-is. Never write "Hub key" / "Sat attribute" — those are
   vault concepts.
6. "Business Key" flag (Y/N) marks columns that uniquely identify a
   business entity in the source — that's a SOURCE observation, not a
   vault decision.
7. "PII Flag" (Y/N) flags personally-identifiable columns (name, email,
   SSN, phone, address, etc.).
8. Every source column from every entity in every file MUST appear as a
   row.

SOURCE METADATA:
{metadata_summary}

CRITICAL RULES FOR THE "Source Data Type" COLUMN:
- Every source entity in the COMPLETE ENTITY INVENTORY lists columns
  in the form ``COLUMN_NAME TYPE`` (e.g. ``CUSTOMER_ID VARCHAR(20)``).
- Copy the type VERBATIM into the "Source Data Type" cell.
- If a column is listed without a type (bare name), write "VARCHAR" as
  a safe default AND add "(type unknown in source metadata)" to Notes.
- NEVER write "Unknown" in the Source Data Type cell.

OUTPUT FORMAT:
1. Output ONLY CSV. No prose, no fences, no headings, no Markdown.
2. First line is the EXACT header shown below.
3. Quote every cell with double quotes. Escape internal quotes as "".
4. No newlines inside cells.

{_template_block(default_header, default_example)}

Begin now. Your response must start with the character `"`.
"""

    # ── Modes 2 & 3: legacy Data-Vault-aware STTM ──────────────────────
    rv_block = ""
    if raw_vault_sql and raw_vault_sql.strip():
        # Truncate only if absurdly long — at 32k max_tokens we have room
        truncated = raw_vault_sql[:40000]
        rv_block = f"""

═══════════════════════════════════════════════════════════════════════
RAW VAULT TARGET MODEL — GROUND TRUTH
═══════════════════════════════════════════════════════════════════════
The following Snowflake DDL IS the target model. Your job is to map
source columns to columns that ACTUALLY EXIST in these tables.

- "Target Table" MUST be a table name from the DDL below.
- "Target Column" MUST be a column name that exists in that target table.
- "Target Data Type" MUST match the type declared in the DDL for that
  column (copy verbatim — VARCHAR, BINARY(20), TIMESTAMP_NTZ, NUMBER(18,2)
  etc.).
- Do NOT invent target columns. If a source column has no matching target
  column, map it to the appropriate Satellite's descriptive attribute or
  flag it in Notes as "no target — candidate for new Satellite attr".

```sql
{truncated}
```
═══════════════════════════════════════════════════════════════════════
"""

    # NOTE: these are computed OUTSIDE the f-string below because older
    # Python (≤ 3.11) doesn't allow backslashes inside f-string {}
    # expressions, and the example rows contain `\'` (escaped single
    # quotes inside the SQL fragments).
    _legacy_default_header = (
        '"Source System","Source Table","Source Column","Source Data Type",'
        '"Target Object Type","Target Table","Target Column",'
        '"Target Data Type","Transformation","Business Key","Hash Key",'
        '"PII Flag","Notes"'
    )
    _legacy_default_example = (
        'EXAMPLE — same CUSTOMER_ID from two different jobs mapping to '
        'ONE Hub:\n'
        '"DataStage-JobA","CUSTOMER_FEED","CUSTOMER_ID","VARCHAR(20)","Hub",'
        '"HUB_CUSTOMER","CUSTOMER_ID","VARCHAR(20)",'
        '"UPPER(TRIM(CUSTOMER_ID))","Y",'
        + '"SHA2_BINARY(COALESCE(CAST(CUSTOMER_ID AS VARCHAR),'
        + "'-1'),256)" + '","N","Primary key; shared across JobA and '
        'JobB; CUSTOMER_HK is BINARY(32)"\n'
        '"DataStage-JobB","LOAN_APPLICATIONS","CUSTOMER_ID","VARCHAR(20)",'
        '"Hub","HUB_CUSTOMER","CUSTOMER_ID","VARCHAR(20)",'
        '"UPPER(TRIM(CUSTOMER_ID))","Y",'
        + '"SHA2_BINARY(COALESCE(CAST(CUSTOMER_ID AS VARCHAR),'
        + "'-1'),256)" + '","N","Same customer; routed to the shared '
        'HUB_CUSTOMER"'
    )
    _legacy_template_block = _template_block(
        _legacy_default_header, _legacy_default_example,
    )

    return f"""You are a data mapping specialist. Output must be machine-
parseable CSV — not prose, not Markdown.

TASK: Build a Source-to-Target Mapping (STTM) from this metadata (which
may span MULTIPLE DataStage jobs / CSV files), targeting a Data Vault 2.0
Raw Vault in Snowflake. Produce ONE ROW PER SOURCE COLUMN across ALL
source entities in ALL files.

SOURCE METADATA:
{metadata_summary}
{rv_block}
CRITICAL RULES FOR THE "Source Data Type" COLUMN:
- Every source entity in the COMPLETE ENTITY INVENTORY above lists
  columns in the form `COLUMN_NAME TYPE` (e.g. `CUSTOMER_ID VARCHAR(20)`).
- Copy the type VERBATIM into the "Source Data Type" cell.
- If a column is listed without a type (bare name), write "VARCHAR" as
  a safe default AND add "(type unknown in source metadata)" to Notes.
- NEVER write "Unknown" in the Source Data Type cell.

CRITICAL RULES FOR "Target Table", "Target Column", "Target Data Type":
- If a RAW VAULT TARGET MODEL block is present above, draw all three
  values from that DDL. Do not invent table or column names.
- If no Raw Vault DDL is present, follow the bank's DATA VAULT 2.0
  STANDARDS appendix below for naming, hash-key types, metadata
  columns, and abbreviations.
- NEVER write "Unknown" in any Target cell.

CRITICAL MULTI-JOB RULES:
1. If a business key (e.g. CUSTOMER_ID) appears in multiple source
   entities across different jobs, ALL of those source columns map to
   the SAME target Hub (e.g. HUB_CUSTOMER). The Target Table value must
   be IDENTICAL across all such rows.
2. See the "SHARED BUSINESS KEY CANDIDATES" section of the metadata —
   every column listed there must map to one shared Hub.
3. Descriptive attributes from different jobs describing the same Hub
   map to SOURCE-SPECIFIC satellites (one satellite per source system,
   per the SATELLITE RULES in the standards appendix). Note the source
   system in the satellite name.
4. Every source column from every file MUST appear as a row.

OUTPUT FORMAT:
1. Output ONLY CSV. No prose, no fences, no headings, no Markdown.
2. First line is the EXACT header shown below.
3. Quote every cell with double quotes. Escape internal quotes as "".
4. No newlines inside cells.
5. Target Object Type in {{Hub, Link, Satellite}}.
6. Business Key in {{Y, N}}. PII Flag in {{Y, N}}.
7. Hash Key column: per the bank's hash standards, use
   SHA2_BINARY(COALESCE(CAST(...AS VARCHAR),'-1'), 256). For
   multi-column hash keys, sort columns alphabetically and concatenate
   with '||'. Apply UPPER(TRIM()) to character columns.

{_legacy_template_block}

Begin now. Your response must start with the character `"`.
""" + _dv_standards_block("sttm")

def build_data_catalog_prompt(metadata_summary: str) -> str:
    return f"""You are a data catalog curator. Output must be machine-
parseable CSV — not prose, not Markdown.

TASK: Build an enterprise Data Catalog from this metadata. Produce ONE ROW
PER COLUMN across all entities.

SOURCE METADATA:
{metadata_summary}

OUTPUT FORMAT — critical rules:
1. Output ONLY CSV. No prose, no fences, no headings, no Markdown.
2. First line is the EXACT header shown below.
3. Quote every cell with double quotes. Escape internal quotes as "".
4. One row per column. No newlines inside cells.
5. Classification ∈ {{Public, Internal, Confidential, Restricted}}.
6. Sensitivity Tag ∈ {{PII, PCI, PHI, Financial, None}}.
7. Key Type ∈ {{PK, FK, None}}. Nullable ∈ {{Y, N}}.

HEADER (first line, exactly):
"Entity","Column","Data Type","Business Definition","System of Record","Data Steward","Classification","Sensitivity Tag","Nullable","Key Type","DQ Rule","Sample Value"

EXAMPLE row:
"CUSTOMERS","EMAIL","VARCHAR(255)","Primary contact email for the customer","Salesforce","Customer Domain Steward","Confidential","PII","N","None","Must match RFC 5322 email pattern","john.doe@example.com"

Begin now. Your response must start with the character `"`.
""" + _dv_standards_block("catalog")

def build_data_domain_prompt(metadata_summary: str) -> str:
    return f"""You are a data governance lead. Output must be machine-
parseable CSV — not prose, not Markdown.

TASK: Propose a Data Domain model following data-mesh principles. Produce
ONE ROW PER DOMAIN-ENTITY pair. Every source entity must appear.

SOURCE METADATA:
{metadata_summary}

OUTPUT FORMAT — critical rules:
1. Output ONLY CSV. No prose, no fences, no headings, no Markdown.
2. First line is the EXACT header shown below.
3. Quote every cell with double quotes. Escape internal quotes as "".
4. One row per (domain, entity) pair. No newlines inside cells.
5. Upstream / Downstream Domains: comma-separated values WITHIN the quoted
   cell. If none, use "None".

HEADER (first line, exactly):
"Domain","Subdomain","Business Capability","Entity","Domain Owner","Upstream Domains","Downstream Domains","Critical Data Elements","Notes"

EXAMPLE row:
"Customer","Customer Identity","Know Your Customer","CUSTOMERS","Customer Domain Team","None","Sales, Support","CUST_ID, EMAIL, NAME, COUNTRY_CODE","Source of truth for customer identity"

Begin now. Your response must start with the character `"`.
""" + _dv_standards_block("domain")


def build_transformation_rules_prompt(metadata_summary: str) -> str:
    """
    Ask the LLM to enumerate every transformation rule, derivation, filter,
    lookup, surrogate-key strategy, type cast, code/value conversion, and
    business rule observed in the parsed Reverse Engineering Inputs.

    The output is structured Markdown — designed to be:
      (a) readable by a human reviewer in the Quick GO panel, and
      (b) re-injected as ADDITIONAL CONTEXT into the Raw Vault Model
          prompts so the generator preserves the same business semantics.

    Input is strictly the parsed source metadata (DataStage stages, SQL
    DDL/DML, BODS XML, Control-M, shell, SSIS, Denodo, etc.). NO Raw Vault
    information is fed in — this artifact is source-driven only.
    """
    return f"""You are a senior data integration analyst reverse-engineering
a legacy ETL/ELT estate. Your job is to extract EVERY transformation rule,
derivation, filter, lookup, business rule, and value conversion present in
the parsed source metadata below — and document them in a clear,
implementer-ready Markdown report.

SOURCE METADATA (parsed from uploaded ETL / SQL / shell / scheduler files):
{metadata_summary}

═══════════════════════════════════════════════════════════════════════
WHAT TO EXTRACT — be exhaustive
═══════════════════════════════════════════════════════════════════════
Look for and document the following kinds of logic, wherever they appear:

1. COLUMN-LEVEL DERIVATIONS
   • CASE / IF-THEN-ELSE expressions → state the conditions and outputs
   • Concatenations, substring extractions, padding, trimming
   • Arithmetic and aggregate calculations (SUM, AVG, ratio formulas)
   • Date arithmetic (date diffs, period buckets, fiscal calendar logic)
   • String parsing / regex / split-on-delimiter rules

2. FILTERS & PREDICATES
   • WHERE clauses, HAVING clauses, DataStage Transformer constraints
   • Soft-delete or active-record filters (WHERE active = 'Y')
   • Date-window filters (only last 90 days, current quarter, etc.)
   • Reject conditions (rows routed to error/reject outputs)

3. LOOKUPS & ENRICHMENTS
   • Reference-data lookups (code → description, ID → name)
   • Cross-reference / mapping tables
   • Default values when lookup misses

4. KEY HANDLING
   • Surrogate-key generation strategy (sequence, hash, max+1)
   • Natural-key construction (concatenated business keys)
   • Null handling for keys (default sentinel values like -1, 'UNK')

5. TYPE & FORMAT CONVERSIONS
   • Implicit and explicit casts (VARCHAR → DATE, NUMBER → CHAR, etc.)
   • Date / timestamp format conversions (YYYYMMDD → DATE)
   • Currency, unit, or precision conversions
   • Code-page / character-set conversions

6. DEDUPLICATION & UNIQUENESS RULES
   • DISTINCT, GROUP BY collapsing
   • Window functions (ROW_NUMBER over partition)
   • "Latest record wins" / SCD-style logic

7. AGGREGATIONS & ROLLUPS
   • GROUP BY level and grain
   • Aggregate measures (SUM/COUNT/AVG/MIN/MAX) and what they roll up

8. JOIN LOGIC
   • Join keys (which columns, which join type — INNER/LEFT/etc.)
   • Driving table / driven table (DataStage join stage convention)
   • Cartesian / cross-join intent (rare but call it out if seen)

9. BUSINESS RULES (named, where possible)
   • Validation rules (e.g. "email must contain @")
   • Eligibility rules (e.g. "customer is delinquent if balance > 90 days")
   • Risk / scoring rules
   • Regulatory or PII handling rules

10. CONTROL FLOW / ORCHESTRATION
   • Pre/post-processing shell or SQL steps
   • Conditional branching in Control-M / job sequences
   • Parameterization (e.g. business date passed in)
   • Error-handling / restart strategy if explicit

═══════════════════════════════════════════════════════════════════════
RULES FOR THE REPORT
═══════════════════════════════════════════════════════════════════════
• ATTRIBUTE every rule to its source: filename, job (if applicable), and
  stage / SQL object. Exact format: `(file: <name>, job: <name>, stage: <name>)`.
• If a rule is REPEATED across files/jobs, dedupe but list ALL contributing
  sources in the attribution.
• If the source code suggests a rule but does not state it explicitly,
  mark it as `(inferred)` — never invent rules out of thin air.
• Use exact column names from the metadata. Quote SQL fragments verbatim
  in fenced code blocks when they help clarity.
• Group by the 10 categories above. If a category has no rules, write
  "_None observed in source._" — do not skip the heading.
• Preserve the original business intent — DO NOT redesign or "clean up"
  the rules. Downstream consumers need to see legacy logic faithfully.

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — strict Markdown
═══════════════════════════════════════════════════════════════════════
Return ONLY Markdown. No code fences around the whole document, no
preamble, no closing remarks. Use this structure verbatim:

# Transformation Rules & Business Logic

## Overview
2-3 sentences on the scope of legacy logic uncovered. Mention the number
of files / jobs / stages reviewed and the total count of distinct rules.

## 1. Column-Level Derivations
Bullet list. Each bullet:
- **<target column>** — <plain-English description of the derivation>
  - **Source(s):** <attribution>
  - **Logic:** `<verbatim SQL/expression if available>`

## 2. Filters & Predicates
Same bullet shape, attributed to source.

## 3. Lookups & Enrichments
## 4. Key Handling
## 5. Type & Format Conversions
## 6. Deduplication & Uniqueness Rules
## 7. Aggregations & Rollups
## 8. Join Logic
## 9. Business Rules
## 10. Control Flow & Orchestration

## Cross-Cutting Observations
Brief bullet list of patterns that span multiple categories (e.g. "all
date columns use YYYYMMDD strings and are cast in target stages",
"SCD2 is implemented via end-dating in 4 jobs").

## Open Questions / Ambiguities
Bullet list of rules that are unclear from the source alone — items a
data architect would need to clarify with the business before building
the target model.

Begin now. Your first non-whitespace character must be `#`.
""" + _dv_standards_block("raw_vault")


def build_source_to_hub_prompt(parsed_metadata: dict,
                               raw_vault_sql: str) -> str:
    """
    Ask the LLM to map each source entity to the Hub(s) it feeds, by
    reading the Raw Vault DDL. Output is simple CSV: one row per
    (source_entity, hub_name) pair.
    """
    # Compact inventory: entity display name, job, columns (names only
    # — types aren't needed for mapping)
    lines = []
    for ent_key, meta in parsed_metadata.get("entities", {}).items():
        cols = sorted(meta["columns"])
        heading = meta["display_name"]
        if meta.get("job_name"):
            heading = f"{meta['display_name']} ({meta['job_name']})"
        lines.append(
            f"- {heading}  [file: {meta['source_file']}]  "
            f"cols: {', '.join(cols[:15])}"
            f"{' …' if len(cols) > 15 else ''}"
        )
    inventory = "\n".join(lines) if lines else "(no entities)"

    rv_truncated = (raw_vault_sql or "")[:30000]

    return f"""You resolve source-to-target lineage for a Data Vault
migration. Output ONLY CSV — no prose, no fences.

SOURCE ENTITIES (parsed from uploaded DataStage/CSV files):
{inventory}

RAW VAULT TARGET MODEL (Snowflake DDL — ground truth):
```sql
{rv_truncated}
```

TASK: For every source entity above, decide which Hub(s) it FEEDS in the
Raw Vault, based on which business key columns it carries. A single
source entity often feeds multiple Hubs — e.g. an ORDERS feed carries
both CUSTOMER_ID and PRODUCT_ID, so it feeds HUB_CUSTOMER AND HUB_PRODUCT.

RULES:
- "Hub Name" MUST be an exact Hub table name from the DDL above
  (starts with HUB_). Do NOT invent names.
- Emit one row per (source_entity, hub) pair. If a source feeds 3 Hubs,
  produce 3 rows for that source.
- If a source doesn't feed any Hub (pure pass-through / staging with no
  business key), emit a single row with Hub Name = "NONE".
- Confidence ∈ {{HIGH, MEDIUM, LOW}}. HIGH = business key explicitly
  present in source columns. MEDIUM = strong name overlap. LOW = inferred.

HEADER (first line, exactly):
"Source Entity","Source File","Hub Name","Business Key Column","Confidence","Reason"

EXAMPLE rows:
"CustomerFeed","customer.dsx","HUB_CUSTOMER","CUSTOMER_ID","HIGH","Source carries CUSTOMER_ID which matches HUB_CUSTOMER business key"
"OrderSource","orders.dsx","HUB_CUSTOMER","CUSTOMER_ID","HIGH","Orders reference customers via CUSTOMER_ID"
"OrderSource","orders.dsx","HUB_ORDER","ORDER_ID","HIGH","Primary business key for orders"

Begin now. Your response must start with the character `"`.
""" + _dv_standards_block("lineage")

def build_lineage_mermaid(parsed_metadata: dict, raw_vault_tables: dict,
                          source_to_hub_df) -> str:
    """
    Construct a Mermaid flowchart LR showing end-to-end data lineage:

        File[📄 file.dsx] --> Stage[CustomerSource]
        Stage --> Hub[HUB_CUSTOMER]
        Hub --> Sat[SAT_CUSTOMER_DETAIL]

    Edges come from three sources:
      1. Within-file stage-to-stage links (parsed from DSX)
      2. Cross-file shared-column relationships (inferred)
      3. LLM-provided source_entity → Hub mapping (from the DDL)
    """
    if not parsed_metadata:
        return ""

    entities = parsed_metadata.get("entities", {})
    cross_flows = parsed_metadata.get("cross_job_flows", [])
    files = parsed_metadata.get("files", [])

    # Collect file nodes
    file_nodes = {}   # filename -> mermaid id
    for f in files:
        fid = "F_" + _mermaid_safe_id(f["filename"])
        file_nodes[f["filename"]] = fid

    # Collect stage nodes (one per composite entity key). We want the
    # display_name shown on the node but the composite key as the id so
    # two stages named "Target" in different files don't collide.
    stage_nodes = {}  # entity_key -> mermaid id
    for ent_key, meta in entities.items():
        sid = "S_" + _mermaid_safe_id(ent_key)
        stage_nodes[ent_key] = sid

    # Hub/Link/Sat nodes from the Raw Vault DDL
    hub_nodes = {name: "H_" + _mermaid_safe_id(name)
                 for name in raw_vault_tables.get("hubs", [])}
    link_nodes = {name: "K_" + _mermaid_safe_id(name)
                  for name in raw_vault_tables.get("links", [])}
    sat_nodes = {name: "T_" + _mermaid_safe_id(name)
                 for name in raw_vault_tables.get("sats", [])}

    mm = ["flowchart LR"]

    # ── STYLING ──
    mm.append("    classDef fileStyle fill:#F0EEE6,stroke:#6B6456,"
              "color:#141413,stroke-width:1.5px")
    mm.append("    classDef stageStyle fill:#E8F0E8,stroke:#3D6B3D,"
              "color:#141413,stroke-width:1.5px")
    mm.append("    classDef hubStyle fill:#F9E4D4,stroke:#C96442,"
              "color:#141413,stroke-width:2px")
    mm.append("    classDef linkStyle fill:#E4E0F9,stroke:#6B4CC9,"
              "color:#141413,stroke-width:2px")
    mm.append("    classDef satStyle fill:#E4EEF9,stroke:#4C80C9,"
              "color:#141413,stroke-width:1.5px")

    # ── FILE SUBGRAPHS: one per file, containing its stages ──
    for f in files:
        fname = f["filename"]
        fid = file_nodes[fname]
        safe_fname = fname.replace('"', "'")
        sub_id = "SG_" + _mermaid_safe_id(fname)
        mm.append(f'    subgraph {sub_id}["📄 {safe_fname}"]')
        mm.append(f'        {fid}["{safe_fname}"]:::fileStyle')

        # Stages that live in this file
        stages_in_file = [
            (k, m) for k, m in entities.items()
            if m["source_file"] == fname
        ]
        for ent_key, meta in stages_in_file:
            sid = stage_nodes[ent_key]
            label = meta["display_name"]
            if meta.get("job_name"):
                label = f"{label}\\n({meta['job_name']})"
            safe_label = label.replace('"', "'")
            mm.append(f'        {sid}["{safe_label}"]:::stageStyle')
            # file → stage
            mm.append(f"        {fid} --> {sid}")
        mm.append("    end")

    # ── RAW VAULT SUBGRAPH ──
    if hub_nodes or link_nodes or sat_nodes:
        mm.append('    subgraph RV["🏛 Raw Vault"]')
        for name, nid in hub_nodes.items():
            mm.append(f'        {nid}["{name}"]:::hubStyle')
        for name, nid in link_nodes.items():
            mm.append(f'        {nid}["{name}"]:::linkStyle')
        for name, nid in sat_nodes.items():
            mm.append(f'        {nid}["{name}"]:::satStyle')
        mm.append("    end")

        # Hub → Link edges (based on LNK_X_Y naming where X and Y are
        # Hub stems — best-effort, but the DDL also has FK columns we
        # could parse if needed)
        for lname, lid in link_nodes.items():
            stem = lname[4:] if lname.startswith("LNK_") else lname[5:]
            for hname, hid in hub_nodes.items():
                hstem = hname[4:]
                if hstem and hstem in stem:
                    mm.append(f"    {hid} -.-> {lid}")

        # Hub/Link → Sat edges (based on SAT_ENTITY naming)
        for sname, sid in sat_nodes.items():
            core = sname[4:] if sname.startswith("SAT_") else sname
            # Look for the parent Hub whose stem is a prefix of the Sat name
            for hname, hid in hub_nodes.items():
                hstem = hname[4:]
                if hstem and core.startswith(hstem):
                    mm.append(f"    {hid} --> {sid}")
                    break
            else:
                for lname, lid in link_nodes.items():
                    lstem = lname[4:] if lname.startswith("LNK_") else lname[5:]
                    if lstem and core.startswith(lstem):
                        mm.append(f"    {lid} --> {sid}")
                        break

    # ── STAGE → HUB edges from LLM-produced mapping ──
    # Keyed by (display_name, file) → hub_name for O(1) lookup
    added_source_hub = set()
    if source_to_hub_df is not None and not source_to_hub_df.empty:
        cols = {c.lower(): c for c in source_to_hub_df.columns}
        src_col  = cols.get("source entity") or cols.get("source_entity")
        file_col = cols.get("source file") or cols.get("source_file")
        hub_col  = cols.get("hub name") or cols.get("hub_name")

        if src_col and hub_col:
            for _, row in source_to_hub_df.iterrows():
                src = str(row[src_col]).strip()
                hub = str(row[hub_col]).strip().upper()
                file_hint = (str(row[file_col]).strip()
                             if file_col else "")
                if not src or hub in ("", "NONE", "NAN"):
                    continue
                if hub not in hub_nodes:
                    continue  # LLM hallucinated a non-existent Hub
                # Resolve source display_name (+ optional file) to an
                # entity_key. Prefer an exact (name, file) match, fall
                # back to name-only.
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
                edge = (matched, hub)
                if edge in added_source_hub:
                    continue
                added_source_hub.add(edge)
                sid = stage_nodes[matched]
                hid = hub_nodes[hub]
                mm.append(f"    {sid} ==> {hid}")

    # ── WITHIN-FILE STAGE-TO-STAGE LINKS (explicit DSX links) ──
    # Build (filename, stage_name) -> entity_key reverse index
    file_stage_to_key = {}
    for k, m in entities.items():
        file_stage_to_key[(m["source_file"], m["display_name"])] = k

    for f in files:
        if f["kind"] != "datastage":
            continue
        for l in f["parsed"].get("links", []):
            fk = file_stage_to_key.get((f["filename"], l["from_stage"]))
            tk = file_stage_to_key.get((f["filename"], l["to_stage"]))
            if fk and tk and fk != tk:
                mm.append(f"    {stage_nodes[fk]} --> {stage_nodes[tk]}")

    # ── CROSS-FILE FLOWS (inferred from shared columns) ──
    for cf in cross_flows:
        if cf.get("kind", "").startswith("cross-file"):
            # Resolve back to entity keys by display_name + the file
            # hint embedded in the flow's "file" field
            from_file = cf.get("file", "").split("↔")[0].strip()
            to_file = cf.get("file", "").split("↔")[-1].strip()
            fk = file_stage_to_key.get((from_file, cf["from_stage"]))
            tk = file_stage_to_key.get((to_file, cf["to_stage"]))
            if fk and tk and fk != tk:
                mm.append(
                    f"    {stage_nodes[fk]} -. shared keys .-> "
                    f"{stage_nodes[tk]}"
                )

    return "\n".join(mm)


# ═════════════════════════════════════════════════════════════════════════════
# RAW VAULT VALIDATION (GenAI as independent validator)
# ═════════════════════════════════════════════════════════════════════════════
#
# Implements the multi-layer validation framework described in the
# accompanying spec:
#   1. Source Understanding Validation
#   2. Structural Data Vault Validation
#   3. Semantic and Business Validation
#   4. Lineage and Traceability Validation
#   5. Operational Readiness Validation
#
# The prompt asks the model — a SECOND pass, independent of the model that
# produced the Raw Vault — to act as a Data Vault 2.0 reviewer and emit a
# strict-JSON validation report. JSON (instead of free-form Markdown) keeps
# the output machine-parseable so the UI can render score gauges, a rule
# violations register, business-key confidence matrix, and a final
# production-readiness scorecard.
#
def build_raw_vault_validation_prompt(metadata_summary: str,
                                      raw_vault_sql: str,
                                      raw_vault_narrative: str = "",
                                      raw_vault_mermaid: str = "",
                                      sttm_text: str = "",
                                      data_catalog_text: str = "",
                                      source_to_hub_text: str = "") -> str:
    """Build the Raw Vault validation prompt.

    The model is positioned as an INDEPENDENT validator — not the same role
    that generated the model. It must score 7 dimensions, emit a violations
    register, a business-key confidence matrix, and a final readiness
    verdict, all as machine-parseable JSON.

    Parameters
    ----------
    metadata_summary
        The same source-metadata summary that was fed to the Raw Vault
        generator. Needed so the validator can verify source interpretation
        (Layer 1) and lineage completeness (Layer 4).
    raw_vault_sql
        The generated Snowflake DDL — the primary structural artifact under
        validation.
    raw_vault_narrative
        Optional Markdown narrative that accompanies the DDL. Helps the
        validator cross-check semantic intent against the DDL.
    raw_vault_mermaid
        Optional Mermaid erDiagram. Helps the validator see relationship
        cardinality at a glance.
    sttm_text
        Optional Source-to-Target Mapping (raw text or rendered table).
        Used for lineage / traceability validation.
    data_catalog_text
        Optional Data Catalog (CSV-form). Used for semantic validation
        (every Raw Vault attribute should trace to a catalog entry).
    source_to_hub_text
        Optional source→Hub mapping (CSV-form). Used to cross-check Hub
        coverage of source entities.
    """
    # Truncate over-long inputs to keep within model context. The Raw Vault
    # DDL is the most important — give it the most room.
    rv_sql_blob = (raw_vault_sql or "")[:30000]
    rv_nar_blob = (raw_vault_narrative or "")[:8000]
    rv_mer_blob = (raw_vault_mermaid or "")[:6000]
    sttm_blob = (sttm_text or "")[:6000]
    cat_blob = (data_catalog_text or "")[:6000]
    s2h_blob = (source_to_hub_text or "")[:4000]
    meta_blob = (metadata_summary or "")[:18000]

    return f"""You are an INDEPENDENT Data Vault 2.0 validator. You did not
build this Raw Vault — your role is to validate it as a second, adversarial
reviewer before implementation.

Your job is to assess whether the Raw Vault is:
  • Structurally compliant with Data Vault 2.0 standards
  • Semantically aligned with business meaning
  • Technically implementable
  • Fully traceable to source systems
  • Ready for automated deployment and testing

═══════════════════════════════════════════════════════════════════════
INPUTS UNDER REVIEW
═══════════════════════════════════════════════════════════════════════

# 1. SOURCE METADATA (what the generator was given)
{meta_blob}

# 2. RAW VAULT DDL (the primary artifact under validation)
{rv_sql_blob}

# 3. RAW VAULT NARRATIVE (designer's intent — optional)
{rv_nar_blob if rv_nar_blob else "(none provided)"}

# 4. RAW VAULT ER DIAGRAM (Mermaid — optional)
{rv_mer_blob if rv_mer_blob else "(none provided)"}

# 5. SOURCE-TO-TARGET MAPPING (lineage signal — optional)
{sttm_blob if sttm_blob else "(none provided)"}

# 6. DATA CATALOG (semantic signal — optional)
{cat_blob if cat_blob else "(none provided)"}

# 7. SOURCE → HUB MAPPING (Hub coverage signal — optional)
{s2h_blob if s2h_blob else "(none provided)"}

═══════════════════════════════════════════════════════════════════════
FIVE-LAYER VALIDATION FRAMEWORK — apply each layer
═══════════════════════════════════════════════════════════════════════

LAYER 1 — Source Understanding Validation
  • Complete source table inventory captured
  • Primary keys correctly identified
  • Candidate business keys accurately inferred
  • Foreign key relationships properly recognized
  • Source data grain correctly interpreted
  • Change data capture behavior understood
  • Flag low-confidence inferences and missing entities

LAYER 2 — Structural Data Vault Validation
  HUBS must:
    - represent ONE unique business concept
    - contain ONE stable, immutable business key
    - exclude descriptive attributes
    - include required metadata cols (HUB_HASH_KEY / HK, BUSINESS_KEY,
      LOAD_DATE / LOAD_DTS, RECORD_SOURCE / REC_SRC)
    - use deterministic hash key generation
  LINKS must:
    - represent a legitimate relationship/transaction
    - connect ONLY Hub keys
    - include ALL participating Hub hash keys
    - contain NO descriptive attributes
    - include LINK_HASH_KEY, LOAD_DATE, RECORD_SOURCE
  SATELLITES must:
    - attach to EXACTLY ONE parent Hub or Link
    - contain ONLY descriptive/contextual attributes
    - include HASHDIFF, parent HASH_KEY, LOAD_DATE, RECORD_SOURCE
    - be grouped by source AND rate of change (one-record-source rule)
  Forbidden anti-patterns: null business keys, duplicate business keys,
  orphan Links, orphan Satellites, mixed-source Satellites, business
  transformations in Raw Vault, mutable keys, non-insert-only loads,
  inconsistent hash key generation.

LAYER 3 — Semantic and Business Validation
  • Business keys are stable, unique, cross-system consistent, non-null,
    business-relevant, independent of operational implementation
  • Satellite splitting is optimal (by source / change rate / sensitivity)
  • Detect anti-patterns: overloaded Satellites, mixed-rate attributes,
    mixed-source attributes, derived/transformed columns

LAYER 4 — Lineage and Traceability Validation
  • Every Hub maps to one or more source entities
  • Every Link maps to a source relationship or transaction
  • Every Satellite attribute maps to a source attribute
  • All transformations are transparent and documented
  • No orphaned lineage nodes
  • Column-level lineage is complete

LAYER 5 — Operational Readiness Validation
  • Hash algorithm standardized (SHA-256 expected)
  • Deterministic attribute ordering for hashing
  • Standard null replacement, consistent delimiters
  • Data type normalization before hashing
  • Identical inputs always produce identical outputs
  • DDL is technically implementable on Snowflake

═══════════════════════════════════════════════════════════════════════
SCORECARD — these weights are FIXED. Use them verbatim.
═══════════════════════════════════════════════════════════════════════
  Structural Compliance  : 20%
  Business Key Accuracy  : 20%
  Relationship Integrity : 15%
  Satellite Design       : 15%
  Lineage Completeness   : 15%
  Hash Standardization   :  5%
  Data Quality Readiness : 10%

Readiness levels:
  90–100: Production Ready
  75–89 : Minor Refinements Required
  60–74 : Significant Review Required
  <60   : Re-modeling Recommended

═══════════════════════════════════════════════════════════════════════
OUTPUT FORMAT — STRICT JSON. NOTHING ELSE.
═══════════════════════════════════════════════════════════════════════

Return ONE valid JSON object — no prose, no Markdown, no ```json fences.
The first non-whitespace character must be `{{` and the last must be `}}`.

Use this EXACT schema (every key required; arrays may be empty if nothing
applies):

{{
  "summary": "2-4 sentence executive summary of the Raw Vault's quality.",
  "overall_score": 87,
  "readiness_level": "Minor Refinements Required",
  "scorecard": [
    {{"area": "Structural Compliance",   "weight": 20, "score": 0,
     "weighted": 0, "comment": "..."}},
    {{"area": "Business Key Accuracy",   "weight": 20, "score": 0,
     "weighted": 0, "comment": "..."}},
    {{"area": "Relationship Integrity",  "weight": 15, "score": 0,
     "weighted": 0, "comment": "..."}},
    {{"area": "Satellite Design",        "weight": 15, "score": 0,
     "weighted": 0, "comment": "..."}},
    {{"area": "Lineage Completeness",    "weight": 15, "score": 0,
     "weighted": 0, "comment": "..."}},
    {{"area": "Hash Standardization",    "weight":  5, "score": 0,
     "weighted": 0, "comment": "..."}},
    {{"area": "Data Quality Readiness",  "weight": 10, "score": 0,
     "weighted": 0, "comment": "..."}}
  ],
  "source_interpretation": {{
    "tables_captured": true,
    "primary_keys_identified": true,
    "business_keys_inferred": true,
    "fk_relationships_recognized": true,
    "grain_understood": true,
    "cdc_behavior_understood": true,
    "low_confidence_items": [
      "list each ambiguous interpretation with a 1-sentence rationale"
    ],
    "missing_entities": [
      "list source entities NOT represented downstream, if any"
    ]
  }},
  "structural_compliance": {{
    "hubs_pass":       true,
    "links_pass":      true,
    "satellites_pass": true,
    "missing_metadata_columns": [
      {{"entity": "HUB_X", "missing": ["LOAD_DTS"]}}
    ],
    "anti_patterns": [
      "describe any anti-pattern observed (orphan Sat, mixed-source Sat, etc.)"
    ]
  }},
  "violations": [
    {{
      "severity":    "High|Medium|Low",
      "rule":        "short rule id (e.g. HUB_NO_DESC_ATTRS, SAT_MIXED_SOURCE)",
      "entity":      "table name involved (or '*' if global)",
      "description": "1-2 sentence explanation of what's wrong",
      "remediation": "1 sentence on the recommended fix"
    }}
  ],
  "business_keys": [
    {{
      "hub":              "HUB_CUSTOMER",
      "business_key":     "CUSTOMER_ID",
      "confidence":       "High|Medium|Low",
      "stability":        "High|Medium|Low",
      "uniqueness":       "High|Medium|Low",
      "cross_system":     "High|Medium|Low",
      "non_nullable":     true,
      "alternatives":     ["EMAIL", "TAX_ID"],
      "risk":             "Low|Medium|High",
      "notes":            "1-sentence rationale"
    }}
  ],
  "satellite_quality": [
    {{
      "satellite":      "SAT_CUSTOMER_PROFILE",
      "grouped_by_source":     true,
      "grouped_by_rate":       true,
      "single_record_source":  true,
      "issues":         ["overloaded with 40+ cols", "mixes daily + monthly attrs"],
      "recommendation": "split into SAT_CUSTOMER_DEMOG and SAT_CUSTOMER_CONTACT"
    }}
  ],
  "relationship_integrity": {{
    "all_links_reference_valid_hubs": true,
    "cardinality_modeled":            true,
    "many_to_many_via_links":         true,
    "recursive_links_modeled":        true,
    "transactional_vs_static_distinguished": true,
    "issues": ["..."]
  }},
  "lineage_completeness": {{
    "hubs_traceable_to_source":         "Full|Partial|Missing",
    "links_traceable_to_relationships": "Full|Partial|Missing",
    "satellite_attrs_traceable":        "Full|Partial|Missing",
    "transformations_documented":       true,
    "orphan_lineage_nodes":             [],
    "column_level_lineage":             "Full|Partial|Missing",
    "completeness_pct":                  92
  }},
  "hash_standardization": {{
    "algorithm":              "SHA-256",
    "deterministic_ordering": true,
    "null_replacement":       "documented (e.g. NVL(col,'^^'))",
    "delimiter":              "consistent",
    "type_normalization":     true,
    "issues": ["..."]
  }},
  "remediation_recommendations": [
    {{"priority": "High|Medium|Low",
      "area":     "Structural|Business Key|Lineage|Hash|...",
      "action":   "1-2 sentence specific action item"}}
  ],
  "deliverables_checklist": {{
    "validation_report":              true,
    "rule_violations_register":       true,
    "business_key_confidence_matrix": true,
    "lineage_completeness_matrix":    true,
    "remediation_recommendations":    true,
    "production_readiness_scorecard": true
  }}
}}

CRITICAL RULES for the JSON:
  • `weighted = round(weight * score / 100)` for each scorecard row.
  • `overall_score` MUST equal the sum of `weighted` values.
  • `readiness_level` MUST be derived from `overall_score` per the bands above.
  • Be honest. If something is missing or wrong, mark it. A 100/100 score
    is unrealistic — only award it if every layer truly passes.
  • Cite specific entity / column / rule names from the inputs above —
    do not speak in generalities.
  • Do not output anything outside the JSON object. No prose preamble, no
    closing remarks, no Markdown fences.
  • Use ONLY ASCII double-quotes ("), never smart-quotes (\u201c \u201d).
  • Do NOT include trailing commas before `}}` or `]`.
  • Do NOT include comments (no `//`, no `/* */`).
  • If you cannot answer a key, use an empty string "" or empty array [] or
    null — never omit the key.

Begin. Your response must start with the character `{{` and end with `}}`.
"""
