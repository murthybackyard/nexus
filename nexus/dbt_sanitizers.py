"""
nexus.dbt_sanitizers — file-level sanitizers used by the dbt deploy
pipeline.

Each function takes a ``files`` dict (path → string content) and
returns a transformed dict, applying ONE specific transformation:

    _strip_query_tag_from_sql_model       — remove ALTER SESSION SET
                                            QUERY_TAG (forbidden in
                                            Snowflake native dbt)
    _strip_disallowed_show_parameter_sql  — gut macros containing
                                            SHOW PARAMETER (forbidden)
    _strip_hooks_from_sql_model           — remove pre_hook/post_hook
                                            kwargs from {{ config(...) }}
    _sanitize_dbt_project_yml_for_snowflake_native
                                          — strip on-run-start/end +
                                            +pre_hook/+post_hook from
                                            dbt_project.yml
    _sanitize_jinja_in_yaml_metadata      — fix YAML in dv_stage blocks
    _normalize_source_raw_table_casing    — uppercase RAW source tables
    _normalize_dbt_node_refs              — lowercase ref('...') names
    _repair_missing_stage_references      — fuzzy match OR create stub
                                            stg_*.sql for missing refs
    _ensure_bronze_shims_for_missing_refs — auto-create br_* models
    _pin_dbt_project_object_version       — ALTER DBT PROJECT ... SET DBT_VERSION
    _verify_dbt_project_version           — confirm pin via SHOW DBT PROJECTS
    _packages_yml_requires_dbt_deps       — detect external package deps

These all live here because they are pure file-content transforms —
no Streamlit, no Snowflake calls. The orchestration that calls them
in the right order lives in nexus.dbt_pipeline.
"""

import json
import re
from typing import Dict, List, Optional, Tuple



# ── Snowflake / native-dbt configuration constants ──────────────────────────
# These were originally module-level constants in streamlit_app.py.
# Moved here because the helpers below depend on them.

VECTOR_DB     = "INNOVATION_DB"

VECTOR_SCHEMA = "INNOVATION_DEV"

VECTOR_STAGE  = "ARTIFACTS_STAGE"

NATIVE_DBT_PROFILE_ACCOUNT = ""

NATIVE_DBT_PROFILE_USER = ""

NATIVE_DBT_PROFILE_ROLE = "ACCOUNTADMIN"

NATIVE_DBT_PROFILE_WAREHOUSE = "COMPUTE_WH"

RAW_SOURCE_SCHEMA = "RAW"

RAW_VAULT_SCHEMA = "RAW_VAULT"

BUSINESS_VAULT_SCHEMA = "BUSINESS_VAULT"

SNOWFLAKE_NATIVE_DBT_VERSION = "1.9.4"

# ── Phase-specific vector table names ────────────────────────────────────────
# Three separate 1024-dimensional tables, one per engineering phase.
# All use APPLICATION_NAME + VERSION as the deduplication namespace so the
# same pipeline can be run multiple times without re-embedding unchanged data.
VECTORS_REVERSE_ENG = "VECTORS_REVERSE_ENG"
VECTORS_FORWARD_ENG = "VECTORS_FORWARD_ENG"
VECTORS_RESULTS     = "VECTORS_RESULTS"
APP_RUN_REGISTRY    = "APP_RUN_REGISTRY"

# Dimension fixed for all three phase tables.
PHASE_TABLE_DIM = 1024

# ── Scaffold templates used by the dbt deploy pipeline ───────────────────────
_SCAFFOLD_PACKAGES_YML = """packages:
  - package: Datavault-UK/automate_dv
    version: [">=0.11.0", "<1.0.0"]
  - package: dbt-labs/dbt_utils
    version: [">=1.1.0", "<2.0.0"]
"""

_SCAFFOLD_DBT_PROJECT_RV = """name: 'fs_datavault'
version: '1.0.0'
config-version: 2
require-dbt-version: [">=1.5.0", "<2.0.0"]
profile: 'snowflake'

model-paths: ['models']
macro-paths: ['macros']
analysis-paths: ['analyses']
seed-paths: ['seeds']
test-paths: ['tests']
target-path: 'target'

# nexus: per-layer schema separation. Uses the custom generate_schema_name
# macro injected by _prepare_files_for_snowflake_native_dbt, which returns
# the +schema value verbatim (upper-cased) instead of concatenating it
# with {target.schema}.
models:
  fs_datavault:
    +persist_docs:
      relation: true
      columns: true
    bronze:
      +materialized: view
      +tags: ['bronze', 'medallion']
    silver:
      +materialized: incremental
      +tags: ['silver', 'medallion']
      staging:
        +schema: raw_vault
        +tags: ['staging', 'silver']
      raw_vault:
        +schema: raw_vault
        +tags: ['raw_vault', 'silver']
        hubs:
          +tags: ['hub', 'raw_vault']
        links:
          +tags: ['link', 'raw_vault']
        sats:
          +tags: ['sat', 'raw_vault']
    gold:
      +materialized: table
      +schema: gold
      +tags: ['gold', 'medallion', 'reserved']
"""

_SCAFFOLD_DBT_PROJECT_BV = """name: 'fs_datavault'
version: '1.0.0'
config-version: 2
require-dbt-version: [">=1.5.0", "<2.0.0"]
profile: 'snowflake'

model-paths: ['models']
macro-paths: ['macros']
seed-paths: ['seeds']
test-paths: ['tests']
target-path: 'target'

# nexus: per-layer schema separation (see notes in _SCAFFOLD_DBT_PROJECT_RV).
models:
  fs_datavault:
    +persist_docs:
      relation: true
      columns: true
    silver:
      +materialized: incremental
      +tags: ['silver', 'medallion']
      business_vault:
        +schema: business_vault
        +tags: ['business_vault', 'silver']
    gold:
      +materialized: table
      +schema: gold
      +tags: ['gold', 'medallion', 'semantic']
"""

_RECORD_SOURCE_MACRO = """{# Medallion bronze: shared RECORD_SOURCE literal #}
{% macro dv_record_source(system_name) %}
  '{{ system_name | upper }}'
{% endmacro %}
"""

def _native_snowflake_profiles_yml() -> str:
    """Canonical ``profiles.yml`` for in-account Snowflake native dbt runs."""
    return (
        "snowflake:\n"
        "  target: dev\n"
        "  outputs:\n"
        "    dev:\n"
        "      type: snowflake\n"
        f"      account: {json.dumps(NATIVE_DBT_PROFILE_ACCOUNT)}\n"
        f"      user: {json.dumps(NATIVE_DBT_PROFILE_USER)}\n"
        f"      role: {NATIVE_DBT_PROFILE_ROLE}\n"
        f"      database: {VECTOR_DB}\n"
        f"      schema: {VECTOR_SCHEMA}\n"
        f"      warehouse: {NATIVE_DBT_PROFILE_WAREHOUSE}\n"
        "      threads: 4\n"
    )

def _validated_native_dbt_version(dbt_version: Optional[str]) -> str:
    """Return a ``x.y.z`` string safe for Snowflake ``DBT_VERSION`` literals."""
    raw = (dbt_version or SNOWFLAKE_NATIVE_DBT_VERSION or "").strip()
    if not re.fullmatch(r"\d+\.\d+\.\d+", raw):
        raw = SNOWFLAKE_NATIVE_DBT_VERSION
    return raw

def _sql_dbt_version_suffix(dbt_version: Optional[str]) -> str:
    """Return a safe `` DBT_VERSION = 'x.y.z'`` clause for Snowflake SQL."""
    raw = _validated_native_dbt_version(dbt_version)
    return f" DBT_VERSION = '{raw}'"

def _pin_dbt_project_object_version(session, fq_project: str,
                                    dbt_version: Optional[str]) -> Optional[str]:
    """
    Belt-and-suspenders: ensure the project object pins DBT_VERSION.
    Some accounts ignore trailing CREATE clauses; ALTER is explicit.

    Returns the version string we attempted to pin, or None if the
    ALTER failed (caller can decide whether to surface this).
    Previously this swallowed all errors silently; now we surface to
    stderr so the user can see version-pin failures in the deploy log.
    """
    raw = _validated_native_dbt_version(dbt_version)
    try:
        session.sql(
            f"ALTER DBT PROJECT {fq_project} SET DBT_VERSION = '{raw}'"
        ).collect()
        return raw
    except Exception as e:
        # Don't crash the deploy, but make the failure visible. Use
        # st.warning if Streamlit is in scope, else fall back to print.
        try:
            import streamlit as _st
            _st.warning(
                f"⚠ ALTER DBT PROJECT SET DBT_VERSION failed for "
                f"{fq_project}: {str(e)[:200]}. The project may run on "
                f"the wrong dbt version (currently pinned only via "
                f"CREATE OR REPLACE clause, which some Snowflake "
                f"versions silently ignore)."
            )
        except Exception:
            print(
                f"WARN: could not pin DBT_VERSION on {fq_project}: {e}",
                flush=True,
            )
        return None

def _verify_dbt_project_version(session, fq_project: str) -> Optional[str]:
    """
    Query the project object's actual ``DBT_VERSION`` after CREATE/ALTER.
    Returns the version string or None if unavailable. Snowflake exposes
    project metadata via ``SHOW DBT PROJECTS LIKE ...`` and via
    ``DESCRIBE DBT PROJECT``. We try both.
    """
    parts = fq_project.split(".")
    obj_name = parts[-1]
    try:
        rows = session.sql(
            f"SHOW DBT PROJECTS LIKE '{obj_name}'"
        ).collect()
        for r in rows:
            d = dict(r.asDict()) if hasattr(r, "asDict") else dict(r)
            for k, v in d.items():
                if "dbt_version" in k.lower() or "version" == k.lower():
                    if v:
                        return str(v)
    except Exception:
        pass
    try:
        rows = session.sql(f"DESCRIBE DBT PROJECT {fq_project}").collect()
        for r in rows:
            d = dict(r.asDict()) if hasattr(r, "asDict") else dict(r)
            prop = (d.get("property") or d.get("PROPERTY") or "").lower()
            val = d.get("value") or d.get("VALUE")
            if "dbt_version" in prop and val:
                return str(val)
    except Exception:
        pass
    return None

def _packages_yml_requires_dbt_deps(packages_yml: Optional[str]) -> bool:
    """True when ``packages.yml`` lists remote packages (dbt deps is useful)."""
    s = (packages_yml or "").strip()
    if not s:
        return False
    lines = [ln.split("#", 1)[0] for ln in s.splitlines()]
    body = "\n".join(lines).lower()
    if re.search(r"^\s*-\s*package\s*:", body, re.MULTILINE):
        return True
    if "packages:" in body and re.search(r"\bgit\s*:", body):
        return True
    return False

def _strip_disallowed_show_parameter_sql(files: dict) -> dict:
    """
    Snowflake native dbt cannot execute SHOW PARAMETER(S) inside model jobs.

    Strategy:
      1. For SQL files (typically macros): if any line contains
         ``SHOW PARAMETER(S)``, gut the entire ENCLOSING ``{% macro %}``
         body — replace the body with a no-op ``{{ return('') }}``. This
         is the only safe transformation, because macros that issue
         SHOW PARAMETER usually do so via:
             {% set q = "SHOW PARAMETERS LIKE 'X'" %}
             {% set rs = run_query(q) %}
         Stripping just the first line leaves ``run_query(q)`` with an
         undefined ``q`` — the macro is broken in a way that may still
         issue the original statement at runtime depending on context.

      2. For non-macro SQL contexts (raw SQL outside any ``{% macro %}``
         block) and for YAML files, just comment out the offending line
         like before — there's no enclosing scope to gut.

    Macros that contained SHOW PARAMETER are now safe no-ops and won't
    issue the disallowed statement.
    """
    out = dict(files or {})
    macro_re = re.compile(
        r"(?P<opener>\{%-?\s*macro\s+(?P<name>\w+)\s*\([^)]*\)\s*-?%\})"
        r"(?P<body>.*?)"
        r"(?P<closer>\{%-?\s*endmacro\s*-?%\})",
        re.DOTALL | re.IGNORECASE,
    )
    show_param_re = re.compile(r"(?is)\bSHOW\s+PARAMETERS?\b")

    for p, body in list(out.items()):
        lp = p.lower()
        if not isinstance(body, str):
            continue
        if not (lp.endswith(".sql") or lp.endswith(".yml")
                or lp.endswith(".yaml")):
            continue
        if not show_param_re.search(body):
            continue

        # YAML / non-macro path: line-by-line comment-out
        if lp.endswith(".yml") or lp.endswith(".yaml"):
            new_lines = []
            for line in body.splitlines(True):
                if show_param_re.search(line):
                    new_lines.append(
                        "# nexus: stripped SHOW PARAMETER(S) "
                        "(unsupported in Snowflake native dbt)\n"
                    )
                else:
                    new_lines.append(line)
            out[p] = "".join(new_lines)
            continue

        # SQL path: gut whole macros that contain SHOW PARAMETER.
        def _gut_if_contains_show(m):
            macro_body = m.group("body")
            if show_param_re.search(macro_body):
                # Use groupdict to avoid named-group mismatch issues that
                # cropped up in earlier edits.
                gd = m.groupdict()
                opener = gd.get("opener") or m.group(1)
                closer = gd.get("closer") or m.group(4) or m.group(3)
                name = (
                    gd.get("name") or gd.get("n")
                    or (m.group(2) if m.lastindex and m.lastindex >= 2 else "anon")
                )
                return (
                    opener
                    + (
                        f"\n  {{# nexus: gutted body of macro `{name}` "
                        f"because it contained SHOW PARAMETER(S), "
                        f"unsupported in Snowflake native dbt. "
                        f"Macro is now a no-op. #}}\n"
                        f"  {{{{ return('') }}}}\n"
                    )
                    + closer
                )
            return m.group(0)

        gutted = macro_re.sub(_gut_if_contains_show, body)

        # If after gutting there are still SHOW PARAMETER occurrences
        # (i.e. raw SQL outside any macro), fall back to line-strip.
        if show_param_re.search(gutted):
            new_lines = []
            for line in gutted.splitlines(True):
                if show_param_re.search(line):
                    new_lines.append(
                        "-- [nexus] stripped SHOW PARAMETER(S) "
                        "(unsupported in Snowflake native dbt)\n"
                    )
                else:
                    new_lines.append(line)
            gutted = "".join(new_lines)

        out[p] = gutted
    return out

def _sanitize_dbt_project_yml_for_snowflake_native(body: str) -> str:
    """
    Remove patterns that often compile to unsupported SQL in Snowflake's
    native dbt job runtime (on-run hooks, persist_docs metadata DDL).

    Strips ``query_tag`` anywhere in the file: dbt-snowflake's
    ``get_current_query_tag()`` runs ``show parameters like 'query_tag' in
    session``, which native dbt rejects as ``Unsupported statement type
    'SHOW PARAMETER'``.
    """
    if not body or not isinstance(body, str):
        return body
    lines = body.splitlines(keepends=True)
    out: List[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = re.match(r"^(\s*)on-run-(start|end)(\s*:.*)$", line, re.I)
        if m:
            base_indent = len(m.group(1))
            i += 1
            while i < n:
                nxt = lines[i]
                stripped = nxt.strip()
                if not stripped or stripped.startswith("#"):
                    i += 1
                    continue
                cur_indent = len(nxt) - len(nxt.lstrip(" \t"))
                if cur_indent > base_indent:
                    i += 1
                    continue
                break
            out.append(
                f"{m.group(1)}# nexus: stripped on-run-{m.group(2).lower()} "
                f"(native dbt / SHOW PARAMETER risk)\n"
            )
            continue
        # Strip +pre_hook / +post_hook / pre-hook / post-hook (and the
        # automate_dv-specific +pre-hook variants). These cascade to all
        # models in the tree and the LLM frequently writes hooks that
        # invoke macros emitting SHOW PARAMETERS or ALTER SESSION SET
        # QUERY_TAG — both forbidden in the native runtime.
        mh = re.match(
            r"^(\s*)\+?(pre[-_]hook|post[-_]hook)(\s*:.*)$",
            line, re.I,
        )
        if mh:
            base_indent = len(mh.group(1))
            i += 1
            while i < n:
                nxt = lines[i]
                stripped = nxt.strip()
                if not stripped or stripped.startswith("#"):
                    i += 1
                    continue
                cur_indent = len(nxt) - len(nxt.lstrip(" \t"))
                if cur_indent > base_indent:
                    i += 1
                    continue
                break
            out.append(
                f"{mh.group(1)}# nexus: stripped {mh.group(2).lower()} "
                f"(native dbt / SHOW PARAMETER and ALTER SESSION risk)\n"
            )
            continue
        mq = re.match(r"^(\s*)\+?query_tag\s*:\s*(.*)$", line, re.I)
        if mq:
            remainder = (mq.group(2) or "").split("#", 1)[0].strip()
            if remainder:
                out.append(
                    f"{mq.group(1)}# nexus: stripped query_tag (native dbt)\n"
                )
                i += 1
                continue
            base_indent = len(mq.group(1))
            i += 1
            while i < n:
                nxt = lines[i]
                stripped = nxt.strip()
                if not stripped or stripped.startswith("#"):
                    i += 1
                    continue
                cur_indent = len(nxt) - len(nxt.lstrip(" \t"))
                if cur_indent > base_indent:
                    i += 1
                    continue
                break
            out.append(
                f"{mq.group(1)}# nexus: stripped query_tag block (native dbt)\n"
            )
            continue
        out.append(line)
        i += 1
    s = "".join(out)
    # One-line mapping form: +persist_docs: {relation: true, columns: true}
    s = re.sub(
        r"(?m)^(\s*)\+?persist_docs\s*:\s*\{[^}]*\}\s*$",
        r"\1# nexus: stripped persist_docs (native dbt)\n",
        s,
    )
    # Block form under models (+persist_docs: then relation/columns)
    s = re.sub(
        r"(?ms)^(\s*)\+?persist_docs\s*:\s*\n"
        r"(?:\s+(relation|columns)\s*:\s*[^\n]+\n)+",
        r"\1# nexus: stripped persist_docs block (native dbt)\n",
        s,
    )
    # Catch-all for any remaining query_tag YAML lines (edge-case layouts).
    s = re.sub(
        r"(?m)^(\s*)\+?query_tag\s*:.*$",
        r"\1# nexus: stripped query_tag line (native dbt)\n",
        s,
    )
    # Strip top-level query-comment: block. dbt-snowflake's query-comment
    # wrapper can also cause unsupported statements in the native runtime.
    s = re.sub(
        r"(?ms)^query-comment\s*:\s*\n(?:[ \t]+[^\n]+\n)+",
        "# nexus: stripped query-comment block (native dbt)\n",
        s,
    )
    s = re.sub(
        r"(?m)^query-comment\s*:.*$",
        "# nexus: stripped query-comment (native dbt)",
        s,
    )

    # Ensure a `dispatch:` block exists that puts the root project's macros
    # AHEAD of the `dbt` and `dbt_snowflake` namespaces. This makes our
    # macros/native_query_tag_overrides.sql win deterministically — without
    # it, Jinja name resolution alone is not always enough in dbt-snowflake
    # 1.10.x for namespaced calls like snowflake__set_query_tag.
    #
    # Only inject `dispatch:` into a real dbt_project.yml — detect by the
    # presence of a top-level `name:` key. profiles.yml has no such key and
    # `dispatch:` would be a syntax error there.
    name_match = re.search(
        r"(?m)^\s*name\s*:\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?\s*$",
        s,
    )
    if name_match and not re.search(r"(?m)^\s*dispatch\s*:", s):
        proj_name = name_match.group(1)
        if not s.endswith("\n"):
            s += "\n"
        s += (
            "\n# nexus: dispatch block forces project macros ahead of dbt-snowflake\n"
            "dispatch:\n"
            "  - macro_namespace: dbt\n"
            f"    search_order: ['{proj_name}', 'dbt']\n"
            "  - macro_namespace: dbt_snowflake\n"
            f"    search_order: ['{proj_name}', 'dbt_snowflake']\n"
        )
    return s

_AUDIT_COL_SAFE_SQL = {
    "RECORD_SOURCE":      "'GENERATED'",
    "LOAD_DATETIME":      "CURRENT_TIMESTAMP()",
    "LOAD_DTS":           "CURRENT_TIMESTAMP()",
    "LOADED_AT":          "CURRENT_TIMESTAMP()",
    "EFFECTIVE_DATETIME": "CURRENT_TIMESTAMP()",
    "EFFECTIVE_FROM":     "CURRENT_TIMESTAMP()",
    "EFFECTIVE_AT":       "CURRENT_TIMESTAMP()",
    "EXPIRY_DATETIME":    "CAST(NULL AS TIMESTAMP_NTZ)",
    "EXPIRY_AT":          "CAST(NULL AS TIMESTAMP_NTZ)",
    "CURRENT_FLAG":       "TRUE",
    "IS_CURRENT":         "TRUE",
    "CREATED_BY":         "CURRENT_USER()",
    "UPDATED_BY":         "CURRENT_USER()",
    "MODIFIED_BY":        "CURRENT_USER()",
    "CREATED_AT":         "CURRENT_TIMESTAMP()",
    "UPDATED_AT":         "CURRENT_TIMESTAMP()",
    "BATCH_ID":           "TO_VARCHAR(CURRENT_TIMESTAMP())",
}

def _sanitize_jinja_in_yaml_metadata(body: str) -> str:
    """
    Inside ``{%- set yaml_metadata -%}...{%- endset -%}`` blocks (used by
    automate_dv staging models to declare derived/hashed columns), find
    YAML scalar values that contain Jinja expressions like
    ``{{ var('foo') }}`` and replace them with a safe SQL equivalent
    keyed off the column name (CURRENT_USER, CURRENT_TIMESTAMP, etc.).

    Why: those YAML values are captured verbatim by ``set...endset``,
    parsed by ``fromyaml``, then emitted by ``automate_dv.stage`` as
    raw SQL fragments. Jinja inside them does NOT re-render, so a
    line like ``CREATED_BY: "'{{ var(''dbt_user'', ''dbt'') }}'"`` ends
    up as literal text in the generated SQL and Snowflake rejects it
    with errors like ``expected token ',', got 'dbt_user'``.

    For any unrecognized column name with a Jinja expression, the
    value is replaced with ``'GENERATED'`` (a safe literal) so the
    project still compiles. Non-Jinja values are untouched.
    """
    if not body or "{%- set yaml_metadata" not in body and \
            "{% set yaml_metadata" not in body:
        return body
    if "{{" not in body:
        return body  # no Jinja anywhere → fast path

    # Find every set yaml_metadata...endset block. Allow both `{%- set %}`
    # and `{% set %}` openers and the matching closers.
    block_re = re.compile(
        r"(\{%-?\s*set\s+yaml_metadata\s*-?%\})"
        r"(?P<body>.*?)"
        r"(\{%-?\s*endset\s*-?%\})",
        flags=re.DOTALL,
    )

    def _fix_block(match):
        opener, inner, closer = match.group(1), match.group("body"), match.group(3)
        new_inner_lines = []
        for line in inner.splitlines(keepends=True):
            if "{{" not in line or "}}" not in line:
                new_inner_lines.append(line)
                continue
            # Match `  KEY: "..."` or `  KEY: '...'` lines.
            m = re.match(
                r"^(\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:\s*(.+?)\s*$",
                line.rstrip("\n").rstrip("\r"),
            )
            if not m:
                new_inner_lines.append(line)
                continue
            indent, key, value = m.group(1), m.group(2), m.group(3)
            # Only intervene if the value itself contains Jinja
            if "{{" not in value:
                new_inner_lines.append(line)
                continue
            safe_sql = _AUDIT_COL_SAFE_SQL.get(key.upper(), "'GENERATED'")
            replacement = (
                f'{indent}{key}: "{safe_sql}"  '
                f"# nexus: replaced Jinja {{...}} with safe SQL\n"
            )
            new_inner_lines.append(replacement)
        return opener + "".join(new_inner_lines) + closer

    return block_re.sub(_fix_block, body)

_DBT_NODE_PREFIX_RE = re.compile(
    r"^(stg|hub|lnk|sat|br|pit|bridge|dim|fct|snap|csat)_",
    flags=re.I,
)

def _normalize_source_raw_table_casing(body: str) -> str:
    """
    Upper-case the second argument of every ``source('raw', '<X>')``
    call in a model body. The physical Snowflake landing tables created
    by the deploy pipeline are upper-case (Snowflake's default for
    unquoted identifiers), and ``_sources.yml`` is generated with
    upper-case ``name:`` entries — so model SQL must call
    ``source('raw', 'CUSTOMER')`` not ``source('raw', 'customer')`` for
    dbt's source resolution to match. dbt does exact-string-match here.
    """
    if not body or "source(" not in body:
        return body

    def _repl(m):
        return (
            f"source({m.group('q1')}{m.group('schema')}{m.group('q1')}, "
            f"{m.group('q2')}{m.group('table').upper()}{m.group('q2')})"
        )

    return re.sub(
        r"source\(\s*"
        r"(?P<q1>['\"])(?P<schema>raw)(?P=q1)\s*,\s*"
        r"(?P<q2>['\"])(?P<table>[A-Za-z0-9_]+)(?P=q2)\s*\)",
        _repl, body, flags=re.I,
    )

def _normalize_dbt_node_refs(body: str) -> str:
    """
    Lowercase dbt node references inside ``ref('...')`` and
    ``source_model=...`` calls. dbt resolves these by FILENAME (lowercase
    per dbt convention) — uppercase causes "depends on a node named
    'STG_FOO' which was not found".

    Only names matching dbt-naming-convention prefixes are touched
    (``stg_``, ``hub_``, ``lnk_``, ``sat_``, ``br_``, ``pit_``, ``bridge_``,
    ``dim_``, ``fct_``, ``snap_``, ``csat_``). Other strings — including
    the second arg of ``source('raw', 'CUSTOMER')`` which IS a physical
    Snowflake table name and should stay UPPER — are left untouched.

    Also drops obvious placeholder leakage like ``ref('stg_<entity>')``
    by replacing the entire reference with ``ref('placeholder_dropped')``
    and adding a clearly visible comment, so the failure mode shifts
    from a confusing dbt error to a visible warning the user can fix.
    """
    if not body or "ref(" not in body and "source_model" not in body:
        return body

    def _fix_arg(name: str) -> str:
        # Drop placeholder forms like 'stg_<entity>' or 'hub_<x>_<y>'
        if "<" in name or ">" in name:
            return None  # signal: drop / replace with comment
        if _DBT_NODE_PREFIX_RE.match(name):
            return name.lower()
        return name  # unrecognized prefix — leave as-is

    def _ref_repl(m):
        quote = m.group(1)
        name = m.group(2)
        fixed = _fix_arg(name)
        if fixed is None:
            return (
                "ref('PLACEHOLDER_DROPPED__FIX_THIS')"
                "  /* nexus: original name had unfilled <...> placeholder */"
            )
        return f"ref({quote}{fixed}{quote})"

    body = re.sub(
        r"ref\(\s*(['\"])([A-Za-z_][A-Za-z0-9_<>]*)\1\s*\)",
        _ref_repl, body,
    )

    # source_model='stg_X'  or  source_model="stg_X"
    def _sm_str_repl(m):
        prefix, quote, name, suffix = (
            m.group(1), m.group(2), m.group(3), m.group(4),
        )
        fixed = _fix_arg(name)
        if fixed is None:
            return (
                f"{prefix}{quote}PLACEHOLDER_DROPPED__FIX_THIS{quote}{suffix}"
            )
        return f"{prefix}{quote}{fixed}{quote}{suffix}"

    body = re.sub(
        r"(source_model\s*=\s*)(['\"])([A-Za-z_][A-Za-z0-9_<>]*)\2(\s*[,)])",
        _sm_str_repl, body,
    )

    # source_model=['stg_X', 'stg_Y']  — list form
    def _sm_list_repl(m):
        items = m.group(1)

        def _list_item(im):
            q, n = im.group(1), im.group(2)
            fixed = _fix_arg(n)
            if fixed is None:
                return f"{q}PLACEHOLDER_DROPPED__FIX_THIS{q}"
            return f"{q}{fixed}{q}"

        items = re.sub(
            r"(['\"])([A-Za-z_][A-Za-z0-9_<>]*)\1",
            _list_item, items,
        )
        return f"source_model=[{items}]"

    body = re.sub(
        r"source_model\s*=\s*\[([^\]]*)\]",
        _sm_list_repl, body, flags=re.S,
    )

    return body

def _strip_hooks_from_sql_model(body: str) -> str:
    """
    Remove ``pre_hook=...`` / ``post_hook=...`` keyword args from
    ``{{ config(...) }}`` calls in model files. These hooks frequently
    invoke macros that emit ``SHOW PARAMETERS`` or ``ALTER SESSION SET
    QUERY_TAG`` — both forbidden by Snowflake's native dbt runtime
    (``EXECUTE DBT PROJECT``).

    Hook values can be: a single string, a list of strings, or a Jinja
    expression — and any of them may transitively call a macro that
    issues SHOW PARAMETER. Rather than parse them, we simply drop the
    whole kwarg. Schemas are pre-created by the deploy pipeline; query
    tags are unnecessary in the native runtime; if a project genuinely
    needs side effects, that side effect needs to live elsewhere.
    """
    if not body or not isinstance(body, str):
        return body
    if "_hook" not in body.lower() and "-hook" not in body.lower():
        return body
    s = body
    for hook in ("pre_hook", "post_hook", "pre-hook", "post-hook"):
        # Hook value may be: 'string', "string", a list [...], a dict {...},
        # or a bare Jinja expression. Match any of these as a kwarg inside
        # config(...). The patterns handle: ", hook=..." | "hook=..., " |
        # "(hook=...)" | "(hook=..., ...".
        # Value-pattern: quoted string, list, dict, or bareword identifier.
        val = (
            r"(?:"
            r"'(?:[^'\\]|\\.)*'"          # 'single' quoted
            r"|\"(?:[^\"\\]|\\.)*\""      # "double" quoted
            r"|\[[^\]]*\]"                 # [list]
            r"|\{[^}]*\}"                  # {dict}
            r"|[A-Za-z_][\w.]*"            # bareword identifier
            r")"
        )
        h = re.escape(hook)
        # Trailing: ", hook=value"
        s = re.sub(rf",\s*{h}\s*=\s*{val}", "", s, flags=re.I | re.S)
        # Leading: "hook=value, "
        s = re.sub(rf"{h}\s*=\s*{val}\s*,", "", s, flags=re.I | re.S)
        # Sole arg with closing paren: "(hook=value)"
        s = re.sub(rf"\(\s*{h}\s*=\s*{val}\s*\)", "()", s, flags=re.I | re.S)
        # Trailing before close paren: ", hook=value)"
        s = re.sub(rf",\s*{h}\s*=\s*{val}\s*\)", ")", s, flags=re.I | re.S)
    return s

def _strip_query_tag_from_sql_model(body: str) -> str:
    """
    Remove ``query_tag=...`` from ``{{ config(...) }}`` so Snowflake native
    dbt never dispatches ``snowflake__set_query_tag`` (which calls
    ``show parameters like 'query_tag' in session``).
    """
    if not body or "query_tag" not in body.lower():
        return body
    s = body
    # Trailing: ..., query_tag='x')
    s = re.sub(
        r",\s*query_tag\s*=\s*(?:'[^']*'|\"[^\"]*\"|`[^`]*`|[^\s,)]+)",
        "",
        s,
        flags=re.I,
    )
    # Leading: query_tag='x', ...
    s = re.sub(
        r"query_tag\s*=\s*(?:'[^']*'|\"[^\"]*\"|`[^`]*`|[^\s,)]+)\s*,",
        "",
        s,
        flags=re.I,
    )
    # Sole or last named arg before closing paren
    s = re.sub(
        r",\s*query_tag\s*=\s*(?:'[^']*'|\"[^\"]*\"|`[^`]*`|[^\s,)]+)\s*\)",
        ")",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"\(\s*query_tag\s*=\s*(?:'[^']*'|\"[^\"]*\"|`[^`]*`|[^\s,)]+)\s*\)",
        "()",
        s,
        flags=re.I,
    )
    s = re.sub(
        r"\(\s*query_tag\s*=\s*(?:'[^']*'|\"[^\"]*\"|`[^`]*`|[^\s,)]+)\s*,",
        "(",
        s,
        flags=re.I,
    )
    return s

def _final_native_dbt_safety_sanitize(files: dict) -> dict:
    """
    Last-pass sanitizer before stage upload.
    Removes any residual query_tag / SHOW PARAMETER(S) lines from text files.
    """
    out = dict(files or {})
    for p, body in list(out.items()):
        if not isinstance(body, str):
            continue
        lp = p.replace("\\", "/").lower()
        if not (
            lp.endswith(".sql")
            or lp.endswith(".yml")
            or lp.endswith(".yaml")
            or lp.endswith(".md")
            or lp.endswith(".txt")
        ):
            continue
        text = body
        if "query_tag" in text.lower():
            if lp.endswith(".sql"):
                text = _strip_query_tag_from_sql_model(text)
            text = re.sub(
                r"(?im)^\s*.*\bquery_tag\b.*\n?",
                (
                    "-- nexus: stripped residual query_tag\n"
                    if lp.endswith(".sql")
                    else "# nexus: stripped residual query_tag\n"
                ),
                text,
            )
        if re.search(r"(?im)\bshow\s+parameters?\b", text):
            text = re.sub(
                r"(?im)^\s*.*\bshow\s+parameters?\b.*\n?",
                (
                    "-- nexus: stripped residual show parameter(s)\n"
                    if lp.endswith(".sql")
                    else "# nexus: stripped residual show parameter(s)\n"
                ),
                text,
            )
        out[p] = text
    return out

def _ensure_forward_dbt_scaffold(files: dict, layer: str) -> None:
    """Back-fill minimal medallion / dbt layout if the planner skipped files."""
    if layer not in ("raw_vault", "business_vault"):
        return
    if "packages.yml" not in files:
        files["packages.yml"] = _SCAFFOLD_PACKAGES_YML
    if "dbt_project.yml" not in files:
        files["dbt_project.yml"] = (
            _SCAFFOLD_DBT_PROJECT_BV if layer == "business_vault"
            else _SCAFFOLD_DBT_PROJECT_RV
        )
    if "profiles.yml" not in files:
        files["profiles.yml"] = _SCAFFOLD_PROFILES_YML
    if layer == "raw_vault":
        rs = "macros/dv/record_source.sql"
        if rs not in files:
            files[rs] = _RECORD_SOURCE_MACRO
    if layer == "business_vault":
        seed = "seeds/as_of_dates.csv"
        if seed not in files:
            files[seed] = (
                "AS_OF_DATE\n"
                "2024-01-01\n"
                "2024-06-30\n"
                "2025-01-01\n"
            )

def _ensure_dbt_runtime_profiles(files: dict) -> dict:
    """
    Ensure dbt runtime has both dbt_project.yml and profiles.yml aligned.
    Fixes 'Could not find profile named snowflake' during EXECUTE DBT PROJECT.
    """
    out = dict(files or {})
    if "dbt_project.yml" not in out:
        out["dbt_project.yml"] = _SCAFFOLD_DBT_PROJECT_RV
    if "profiles.yml" not in out:
        out["profiles.yml"] = _SCAFFOLD_PROFILES_YML

    dp = str(out.get("dbt_project.yml", ""))
    if "profile:" not in dp:
        out["dbt_project.yml"] = dp.rstrip() + "\nprofile: 'snowflake'\n"
    elif re.search(r"^\s*profile\s*:\s*['\"]?snowflake['\"]?\s*$",
                   dp, flags=re.MULTILINE) is None:
        dp = re.sub(
            r"^\s*profile\s*:\s*.*$",
            "profile: 'snowflake'",
            dp,
            count=1,
            flags=re.MULTILINE,
        )
        out["dbt_project.yml"] = dp
    return out

def _ensure_bronze_shims_for_missing_refs(files: dict) -> dict:
    """
    Make sure every silver model has a bronze view above it.

    Triggers on TWO signals (either is enough):

    1. Any model contains ``ref('br_<X>')`` but the file
       ``models/bronze/br_<X>.sql`` is missing — the LLM forgot to emit it,
       or the planner listed it but generation failed.

    2. Any model contains ``source('raw', '<entity>')`` and there is no
       ``models/bronze/br_<entity>.sql`` — the LLM skipped the bronze layer
       entirely and went straight from silver to source. This is the more
       common failure mode in raw_vault generation.

    For each missing entity we create:
      - ``models/bronze/br_<entity>.sql`` — a thin view over
        ``source('raw', '<entity>')`` with the audit columns the spec
        requires.
      - ``models/bronze/_sources.yml`` — minimal source definition so dbt
        can resolve ``source('raw', ...)``.
    """
    out = dict(files or {})
    model_names = set()
    for p in out:
        pp = p.replace("\\", "/")
        if pp.lower().endswith(".sql"):
            model_names.add(pp.rsplit("/", 1)[-1].rsplit(".", 1)[0])

    missing_br: set = set()
    raw_entities_in_use: set = set()

    for p, body in out.items():
        pp = p.replace("\\", "/").lower()
        if not pp.endswith(".sql") or not isinstance(body, str):
            continue
        # Signal 1: ref('br_<X>') with no matching file
        for m in re.finditer(r"ref\(\s*['\"](br_[A-Za-z0-9_]+)['\"]\s*\)", body):
            nm = m.group(1)
            if nm not in model_names:
                missing_br.add(nm)
        # Signal 1b: any string literal 'br_X' (refs constructed via
        # variables — be conservative)
        for m in re.finditer(r"['\"](br_[A-Za-z0-9_]+)['\"]", body):
            nm = m.group(1)
            if nm not in model_names:
                missing_br.add(nm)
        # Signal 2: source('raw', '<entity>') with no bronze file at all
        for m in re.finditer(
            r"source\(\s*['\"]raw['\"]\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)",
            body, flags=re.I,
        ):
            ent = m.group(1)
            raw_entities_in_use.add(ent)
            br_name = f"br_{ent.lower()}"
            if br_name not in model_names:
                missing_br.add(br_name)

    if not missing_br and not raw_entities_in_use:
        return out

    # Union of (entities derived from missing br_* names) + (entities pulled
    # straight from source('raw',...) calls). Always create source defs for
    # every raw entity in use, not just the missing ones.
    src_tables = sorted(
        {nm.replace("br_", "", 1) for nm in missing_br}
        | raw_entities_in_use
    )

    out["models/bronze/_sources.yml"] = (
        "version: 2\n"
        "sources:\n"
        "  - name: raw\n"
        f"    database: {VECTOR_DB}\n"
        f"    schema: {RAW_SOURCE_SCHEMA}\n"
        "    tables:\n"
        + "".join([f"      - name: {t}\n" for t in src_tables])
    )

    # Generate a bronze view per missing entity. File path stays lowercase
    # (dbt convention), but the source('raw', ...) identifier is
    # UPPER_SNAKE_CASE to match the physical table created by the deploy
    # pipeline.
    #
    # IMPORTANT: bronze is a passthrough of the raw source. We do NOT
    # add audit columns here — those belong on staging via dv_stage's
    # derived_columns. Adding LOAD_DATETIME / RECORD_SOURCE here causes
    # duplicate-column errors at staging when the staging model defines
    # the same names in its derived_columns block (which it should, per
    # the Medallion spec).
    for br in sorted(missing_br):
        ent = br.replace("br_", "", 1)
        ent_upper = ent.upper()
        out[f"models/bronze/{br}.sql"] = (
            "{{ config(materialized='view', "
            "tags=['bronze','medallion','shim']) }}\n"
            "-- nexus: auto-generated bronze shim. The LLM did not emit a\n"
            "-- bronze model for this entity (or generation failed); this\n"
            "-- thin view passes the raw landing table through unchanged.\n"
            "-- Audit columns are added at the staging layer via dv_stage.\n"
            "select *\n"
            f"from {{{{ source('raw', '{ent_upper}') }}}}\n"
        )
    return out

def _normalize_sources_yml_table_casing(files: dict) -> dict:
    """
    Upper-case every ``- name: <X>`` entry under ``sources: - name: raw``
    → ``tables:`` in EVERY ``*_sources.yml`` file in the project, no
    matter where it lives in the tree.

    Why: the LLM often emits ``models/_sources.yml`` (top-level) with
    lowercase entries, while our auto-generated catalog and bronze shim
    write ``models/bronze/_sources.yml`` with upper-cased entries. dbt
    sees both files. When a model calls ``source('raw', 'CUSTOMER')``
    (upper, post-normalization), dbt's exact-string source resolution
    fails against the lowercase entries in the LLM's YAML and errors
    "depends on a source named 'raw.CUSTOMER' which was not found".

    This helper walks every file path ending in ``_sources.yml`` (or
    ``sources.yml``) and rewrites the ``- name:`` lines under the raw
    source's ``tables:`` block to UPPER_SNAKE_CASE. Other source
    definitions (e.g. for source schemas other than ``raw``) are left
    alone in case they intentionally need different casing.
    """
    out = dict(files or {})
    for p in list(out.keys()):
        lp = p.replace("\\", "/").lower()
        if not (lp.endswith("_sources.yml") or lp.endswith("/sources.yml")
                or lp.endswith("\\sources.yml")):
            continue
        body = out.get(p)
        if not isinstance(body, str) or not body.strip():
            continue

        lines = body.splitlines(keepends=True)
        new_lines: List[str] = []
        in_raw_source = False
        in_tables_block = False
        raw_source_indent = -1
        tables_indent = -1

        for line in lines:
            stripped = line.lstrip()
            indent = len(line) - len(stripped)

            # Detect entering/leaving the raw source block.
            # `- name: raw` indicates we're inside the raw source.
            m_name = re.match(
                r"-\s+name\s*:\s*['\"]?(\w+)['\"]?\s*$",
                stripped,
            )
            if m_name and indent <= raw_source_indent and in_raw_source:
                # Sibling source — leaving raw block
                in_raw_source = False
                in_tables_block = False
            if m_name and m_name.group(1).lower() == "raw":
                in_raw_source = True
                raw_source_indent = indent
                in_tables_block = False
                new_lines.append(line)
                continue

            # Detect tables: block under the raw source
            if in_raw_source and re.match(r"tables\s*:\s*$", stripped):
                in_tables_block = True
                tables_indent = indent
                new_lines.append(line)
                continue

            # Detect leaving the tables: block — sibling key at same
            # indent as `tables:` or shallower.
            if in_tables_block and stripped and indent <= tables_indent and \
                    not stripped.startswith("-"):
                in_tables_block = False

            # Inside tables: block — rewrite `- name: X` to upper.
            if in_tables_block:
                m_table = re.match(
                    r"(\s*-\s+name\s*:\s*)(['\"]?)([A-Za-z_][A-Za-z0-9_]*)(['\"]?)(\s*)$",
                    line.rstrip("\n").rstrip("\r"),
                )
                if m_table:
                    prefix, q1, name, q2, trailing = m_table.groups()
                    eol = "\n" if line.endswith("\n") else ""
                    new_lines.append(
                        f"{prefix}{q1}{name.upper()}{q2}{trailing}{eol}"
                    )
                    continue

            new_lines.append(line)

        out[p] = "".join(new_lines)

    return out

def _ensure_raw_source_catalog(files: dict) -> dict:
    """
    Ensure source('raw', '<table>') resolves in Snowflake by generating
    models/bronze/_sources.yml with database/schema/table entries discovered
    from SQL models.

    Table names are always emitted UPPER_SNAKE_CASE — they must match the
    physical Snowflake landing tables (which the deploy pipeline creates
    UPPER) AND any source('raw', '<X>') calls in models (which we
    normalize to UPPER via _normalize_source_raw_table_casing). dbt's
    source() resolution is exact-string-match, case-sensitive, so any
    casing mismatch fails compilation.
    """
    out = dict(files or {})
    raw_tables = set()
    for _, body in out.items():
        if not isinstance(body, str):
            continue
        for m in re.finditer(
            r"source\(\s*['\"]raw['\"]\s*,\s*['\"]([A-Za-z0-9_]+)['\"]\s*\)",
            body,
            flags=re.I,
        ):
            raw_tables.add(m.group(1).upper())
    if not raw_tables:
        return out
    out["models/bronze/_sources.yml"] = (
        "version: 2\n"
        "sources:\n"
        "  - name: raw\n"
        f"    database: {VECTOR_DB}\n"
        f"    schema: {RAW_SOURCE_SCHEMA}\n"
        "    tables:\n"
        + "".join([f"      - name: {t}\n" for t in sorted(raw_tables)])
    )
    return out

def _infer_raw_columns_from_dv_metadata(files: dict,
                                        raw_table: str) -> Optional[List[str]]:
    """
    When ``_parse_select_columns_for_raw_source`` returns None (which is
    common when staging uses ``automate_dv.stage`` macros instead of an
    explicit SELECT list), look into Data Vault metadata for the same
    entity to recover the source column names.

    Sources we mine, in priority order:

    1. ``stg_<entity>.sql`` — extract column names referenced in
       ``hashed_columns:`` blocks under the embedded YAML metadata. Both
       the natural-key columns under ``<HK>:`` and the descriptive payload
       under ``HASH_DIFF.columns:`` are real source columns.
    2. ``hub_<entity>.sql`` — extract ``src_nk='<COL>'`` or
       ``src_nk=['<A>','<B>']``. These are business-key columns that exist
       in the source table.
    3. ``sat_<entity>_<ctx>.sql`` — extract ``src_payload=['<C>','<D>']``.
       These are descriptive columns that exist in the source.

    Returns a deduped list of column names (preserving first-seen order),
    or ``None`` if no metadata could be found. Hash-key columns
    (``*_HK``, ``HASH_DIFF``) are filtered out because they're computed,
    not source-resident.
    """
    if not raw_table:
        return None
    target = raw_table.strip().lower()

    # Helper: dedupe while preserving order, drop computed columns
    def _normalize(seq):
        seen = set()
        out: List[str] = []
        for c in seq:
            if not c:
                continue
            cu = c.strip().strip("'\"").upper()
            if not re.fullmatch(r"[A-Z_][A-Z0-9_]*", cu):
                continue
            # Skip computed/output columns — these don't exist in the source.
            if cu == "HASH_DIFF" or cu.endswith("_HK") or cu.endswith("_SK"):
                continue
            if cu in ("RECORD_SOURCE", "LOAD_DATETIME", "LOAD_DTS",
                      "EFFECTIVE_DATETIME", "EXPIRY_DATETIME",
                      "CURRENT_FLAG", "CREATED_BY", "UPDATED_BY",
                      "BATCH_ID"):
                continue
            if cu in seen:
                continue
            seen.add(cu)
            out.append(cu)
        return out

    cols_collected: List[str] = []

    # Match files whose basename contains stg_<target>, hub_<target>,
    # or sat_<target>_*. Case-insensitive.
    for p, body in (files or {}).items():
        if not isinstance(body, str):
            continue
        pp = p.replace("\\", "/").lower()
        if not pp.endswith(".sql"):
            continue
        base = pp.rsplit("/", 1)[-1].rsplit(".", 1)[0]

        related = (
            base == f"stg_{target}"
            or base == f"hub_{target}"
            or base.startswith(f"sat_{target}_")
            or base == f"sat_{target}"
            or base == f"br_{target}"
        )
        if not related:
            continue

        # 1) Pull column names from `hashed_columns:` blocks. Match both
        #    the per-HK natural-key list and the HASH_DIFF.columns list.
        #    The text inside `{%- set yaml_metadata -%}...{%- endset -%}`
        #    is plain YAML — but we don't need a YAML parser, just regex
        #    on the obvious patterns.
        for m in re.finditer(
            r"hashed_columns\s*:\s*\n((?:[ \t]+[^\n]+\n?)+)",
            body, flags=re.I,
        ):
            block = m.group(1)
            # Each indented line that's a list item with a quoted column.
            for cm in re.finditer(
                r"^[ \t]*-[ \t]*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?[ \t]*$",
                block, flags=re.M,
            ):
                cols_collected.append(cm.group(1))

        # 2) Pull `src_nk='<COL>'` or `src_nk=['<A>','<B>']` from hubs.
        for m in re.finditer(
            r"src_nk\s*=\s*(['\"][^'\"]+['\"]|\[[^\]]+\])",
            body, flags=re.I,
        ):
            raw = m.group(1)
            if raw.startswith("["):
                for cm in re.finditer(
                    r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", raw,
                ):
                    cols_collected.append(cm.group(1))
            else:
                cols_collected.append(raw.strip("'\""))

        # 3) Pull `src_payload=[...]` from sats.
        for m in re.finditer(
            r"src_payload\s*=\s*\[([^\]]+)\]",
            body, flags=re.I | re.S,
        ):
            for cm in re.finditer(
                r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", m.group(1),
            ):
                cols_collected.append(cm.group(1))

    cols = _normalize(cols_collected)
    return cols if cols else None

def _infer_raw_tables_from_model_paths(files: dict) -> set:
    """
    Heuristic: ``stg_<entity>.sql`` / ``br_<entity>.sql`` imply a RAW entity
    ``<entity>`` used by Raw Vault / bronze shims.
    """
    names = set()
    for p in (files or {}):
        pp = p.replace("\\", "/").lower()
        if not pp.endswith(".sql"):
            continue
        base = pp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        if base.startswith("stg_") and len(base) > 4:
            names.add(base[4:].upper())
        if base.startswith("br_") and len(base) > 3:
            names.add(base[3:].upper())
    return names

def _normalize_generated_file_body(body: str) -> str:
    """
    Normalize LLM-emitted file content before Snowflake deployment.
    Handles common failure mode where SQL is one line containing literal
    escape sequences like "\\n", which dbt parser treats as raw chars.
    """
    if not isinstance(body, str):
        return body
    s = body.strip()

    # If content is wrapped as a JSON string literal, decode it.
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        try:
            dec = json.loads(s)
            if isinstance(dec, str):
                s = dec
        except Exception:
            pass

    # Decode escaped newlines/tabs when they dominate the content.
    esc_n = s.count("\\n")
    real_n = s.count("\n")
    if esc_n and (real_n == 0 or esc_n > (real_n * 2)):
        s = (
            s.replace("\\r\\n", "\n")
             .replace("\\n", "\n")
             .replace("\\t", "\t")
             .replace('\\"', '"')
        )
    return s if s.endswith("\n") else (s + "\n")

def _best_stage_model_match(missing_name: str,
                            stage_models: List[str]) -> Optional[str]:
    """
    Pick the closest generated ``stg_*`` model for a missing stage ref —
    but ONLY if the match is good enough. Returns ``None`` when nothing
    plausibly fits, so the caller can create a stub instead of silently
    rerouting to an unrelated table (which would compile but produce
    wrong data).

    Match levels (best → worst):
      1. Exact match on the missing name
      2. Prefix match on the first token after ``stg_``
      3. Token overlap of at least half the tokens in the missing name
    Anything below level 3 returns ``None``.
    """
    if not stage_models:
        return None
    if missing_name in stage_models:
        return missing_name
    m = missing_name.lower()
    tokens = [t for t in m.replace("stg_", "").split("_") if t]
    if not tokens:
        return None
    # 1) Prefix match on base token
    base = f"stg_{tokens[0]}"
    pref = [s for s in stage_models if s.startswith(base)]
    if pref:
        pref.sort(key=lambda s: (len(s), s))
        return pref[0]
    # 2) Token-overlap threshold: require ≥ ceil(N/2) tokens in common.
    threshold = max(1, (len(tokens) + 1) // 2)
    best = None
    best_score = 0
    for cand in stage_models:
        ctoks = set(cand.replace("stg_", "").split("_"))
        score = sum(1 for t in tokens if t in ctoks)
        if score > best_score:
            best = cand
            best_score = score
    return best if best_score >= threshold else None

def _infer_staging_metadata_from_callers(missing_stage: str,
                                         files: dict) -> dict:
    """
    Look at every model in ``files`` that references the missing
    ``stg_<X>`` and harvest hints about how to build a stub for it:
      - src_pk (HK column name) — from hub or sat ``src_pk='...'``
      - src_nk (NK columns) — from hub ``src_nk=...`` (string or list)
      - hashdiff payload columns — from sat ``src_payload=[...]``
    Returns a dict with whatever was found. Empty dict if nothing.
    """
    pk_re = re.compile(
        r"src_pk\s*=\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    )
    nk_str_re = re.compile(
        r"src_nk\s*=\s*['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]",
    )
    nk_list_re = re.compile(r"src_nk\s*=\s*\[([^\]]+)\]")
    payload_list_re = re.compile(r"src_payload\s*=\s*\[([^\]]+)\]")
    referencing = []
    for p, body in (files or {}).items():
        if not isinstance(body, str):
            continue
        if not p.lower().endswith(".sql"):
            continue
        if missing_stage not in body:
            continue
        referencing.append(body)
    if not referencing:
        return {}
    pks: List[str] = []
    nks: List[str] = []
    payload: List[str] = []
    for body in referencing:
        for m in pk_re.finditer(body):
            pks.append(m.group(1).upper())
        for m in nk_str_re.finditer(body):
            nks.append(m.group(1).upper())
        for m in nk_list_re.finditer(body):
            for n in re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", m.group(1)):
                nks.append(n.upper())
        for m in payload_list_re.finditer(body):
            for n in re.findall(r"['\"]([A-Za-z_][A-Za-z0-9_]*)['\"]", m.group(1)):
                payload.append(n.upper())
    out: dict = {}
    if pks:
        # pick most common
        from collections import Counter
        out["src_pk"] = Counter(pks).most_common(1)[0][0]
    if nks:
        out["src_nk"] = sorted(set(nks))
    if payload:
        out["payload"] = sorted(set(payload))
    return out

def _build_stub_staging_model(stg_name: str,
                              files: dict,
                              business_vault: bool = False) -> str:
    """
    Build the body of a stub ``stg_<entity>.sql`` model. The stub calls
    ``dv_stage`` on ``br_<entity>`` (which will be auto-created later if
    missing), with hashed_columns inferred from any parent hub/sat in
    ``files``. If we can't infer anything, the stub still compiles —
    it just emits a passthrough with audit columns.
    """
    if stg_name.startswith("stg_"):
        ent = stg_name[4:]
    else:
        ent = stg_name
    ent_upper = ent.upper()
    br_name = f"br_{ent.lower()}"
    meta = _infer_staging_metadata_from_callers(stg_name, files)
    src_pk = meta.get("src_pk") or f"{ent_upper}_HK"
    src_nk = meta.get("src_nk") or [f"{ent_upper}_ID"]
    payload = meta.get("payload") or []
    nk_yaml = "".join(f"      - '{n}'\n" for n in src_nk)
    payload_yaml = "".join(f"      - '{p}'\n" for p in payload)
    payload_block = ""
    if payload:
        payload_block = (
            "  HASH_DIFF:\n"
            "    is_hashdiff: true\n"
            "    columns:\n"
            f"{payload_yaml}"
        )
    tags = (
        "['silver', 'business_vault', 'staging', 'shim']"
        if business_vault
        else "['silver', 'raw_vault', 'staging', 'shim']"
    )
    return (
        f"{{{{ config(materialized='incremental', tags={tags}) }}}}\n"
        f"-- nexus: auto-generated staging shim. Caller models referenced\n"
        f"-- ref('{stg_name}') but no stg model existed; the deploy\n"
        f"-- pipeline created this from inferred hub/sat metadata so the\n"
        f"-- project compiles. Edit to refine derived columns / payload.\n"
        f"\n"
        f"{{%- set yaml_metadata -%}}\n"
        f"source_model: '{br_name}'\n"
        f"derived_columns:\n"
        f"  RECORD_SOURCE: \"'{ent_upper}'\"\n"
        f"  LOAD_DATETIME: \"CURRENT_TIMESTAMP()\"\n"
        f"  EFFECTIVE_DATETIME: \"CURRENT_TIMESTAMP()\"\n"
        f"  EXPIRY_DATETIME: \"CAST(NULL AS TIMESTAMP_NTZ)\"\n"
        f"  CURRENT_FLAG: \"TRUE\"\n"
        f"  CREATED_BY: \"CURRENT_USER()\"\n"
        f"  UPDATED_BY: \"CURRENT_USER()\"\n"
        f"  BATCH_ID: \"TO_VARCHAR(CURRENT_TIMESTAMP())\"\n"
        f"hashed_columns:\n"
        f"  {src_pk}:\n"
        f"{nk_yaml}"
        f"{payload_block}"
        f"{{%- endset -%}}\n"
        f"{{% set metadata_dict = fromyaml(yaml_metadata) %}}\n"
        f"{{{{ dv_stage(\n"
        f"    include_source_columns=true,\n"
        f"    source_model=metadata_dict['source_model'],\n"
        f"    derived_columns=metadata_dict['derived_columns'],\n"
        f"    hashed_columns=metadata_dict['hashed_columns']\n"
        f") }}}}\n"
    )

def _repair_missing_stage_references(files: dict) -> dict:
    """
    Repair common LLM mismatch: hubs/links/sats reference ``stg_*`` names
    that don't exist in the generated staging folder.

    Strategy:
      1. Try to fuzzy-match the missing name to an existing stg model
         (only if the match is good enough — see ``_best_stage_model_match``).
         If found, rewrite the ref to point at the existing model.
      2. Otherwise, CREATE a stub ``models/silver/<vault>/staging/<stg>.sql``
         that calls ``dv_stage`` on ``br_<entity>``, inferring hashed_columns
         from the requesting hub/sat's automate_dv metadata. This way the
         project compiles and the user has a clear shim file to edit.
    """
    out = dict(files or {})
    model_names: set = set()
    stage_models: List[str] = []
    has_business_vault_layer = False
    for p in out:
        pp = p.replace("\\", "/")
        if not pp.lower().endswith(".sql"):
            continue
        name = pp.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        model_names.add(name)
        if name.startswith("stg_"):
            stage_models.append(name)
        if "/business_vault/" in pp.lower():
            has_business_vault_layer = True
    stage_models = sorted(set(stage_models))

    # Pass 1: collect every missing stg_* reference across the project.
    missing_globally: set = set()
    for p, body in out.items():
        if not isinstance(body, str) or not p.lower().endswith(".sql"):
            continue
        for m in re.finditer(r"ref\(\s*['\"](stg_[A-Za-z0-9_]+)['\"]\s*\)", body):
            if m.group(1) not in model_names:
                missing_globally.add(m.group(1))
        for m in re.finditer(r"['\"](stg_[A-Za-z0-9_]+)['\"]", body):
            if m.group(1) not in model_names:
                missing_globally.add(m.group(1))

    # Pass 2: for each missing stg, decide between rewrite-to-fuzzy-match
    # vs create-a-stub. Track which mode each missing name uses.
    rewrite_map: dict = {}
    create_set: set = set()
    for miss in sorted(missing_globally):
        repl = _best_stage_model_match(miss, stage_models)
        if repl and repl != miss:
            rewrite_map[miss] = repl
        else:
            create_set.add(miss)

    # Apply rewrites in body text.
    if rewrite_map:
        for p, body in list(out.items()):
            if not isinstance(body, str) or not p.lower().endswith(".sql"):
                continue
            text = body
            for miss, repl in rewrite_map.items():
                text = re.sub(
                    rf"(ref\(\s*['\"])({re.escape(miss)})(['\"]\s*\))",
                    rf"\1{repl}\3", text,
                )
                text = re.sub(
                    rf"(['\"])({re.escape(miss)})(['\"])",
                    rf"\1{repl}\3", text,
                )
            out[p] = text

    # Create stubs for any name that didn't have a good fuzzy match.
    for stg_name in sorted(create_set):
        # Determine the right folder based on whether the project has a
        # business_vault layer at all.
        if has_business_vault_layer:
            stub_path = (
                f"models/silver/business_vault/staging/{stg_name}.sql"
            )
            # If the project ALSO has raw_vault and the requesting model is
            # a raw_vault hub/sat, prefer raw_vault staging.
            for p, body in out.items():
                if not isinstance(body, str):
                    continue
                if stg_name not in body:
                    continue
                if "/raw_vault/" in p.replace("\\", "/").lower():
                    stub_path = (
                        f"models/silver/raw_vault/staging/{stg_name}.sql"
                    )
                    break
        else:
            stub_path = f"models/silver/raw_vault/staging/{stg_name}.sql"
        out[stub_path] = _build_stub_staging_model(
            stg_name, out,
            business_vault=("business_vault" in stub_path),
        )

    return out

def _split_sql_script_into_statements(script: str) -> List[str]:
    """
    Split a SQL script into individual statements at top-level semicolons,
    correctly skipping semicolons that appear inside:

      - single-quoted string literals  ('...'), with '' as escaped quote
      - double-quoted identifiers      ("...")
      - line comments                  (-- to end of line)
      - block comments                 (/* ... */)

    The previous naive ``script.split(";")`` broke whenever a generated
    ``COMMENT = 'Raw row (JSON/CSV); cast in staging as needed.'`` clause
    contained a ``;`` inside the literal — every CREATE TABLE then split
    into 3 malformed fragments and Snowflake errored.

    Returns a list of statements, each stripped of leading/trailing
    whitespace and the trailing semicolon. Empty fragments are dropped.
    """
    statements: List[str] = []
    buf: List[str] = []
    i = 0
    n = len(script)
    while i < n:
        c = script[i]
        nxt = script[i + 1] if i + 1 < n else ""

        # Line comment: consume to end of line (preserve in buf so the
        # statement still parses if a comment immediately precedes DDL)
        if c == "-" and nxt == "-":
            while i < n and script[i] != "\n":
                buf.append(script[i])
                i += 1
            continue
        # Block comment: consume to closing */
        if c == "/" and nxt == "*":
            buf.append(c); buf.append(nxt); i += 2
            while i < n and not (script[i] == "*" and i + 1 < n and script[i + 1] == "/"):
                buf.append(script[i]); i += 1
            if i < n:
                buf.append("*"); buf.append("/"); i += 2
            continue
        # Single-quoted literal: consume until matching ' (handles '' escape)
        if c == "'":
            buf.append(c); i += 1
            while i < n:
                if script[i] == "'" and i + 1 < n and script[i + 1] == "'":
                    buf.append("''"); i += 2; continue
                if script[i] == "'":
                    buf.append("'"); i += 1; break
                buf.append(script[i]); i += 1
            continue
        # Double-quoted identifier: consume until matching "
        if c == '"':
            buf.append(c); i += 1
            while i < n:
                if script[i] == '"' and i + 1 < n and script[i + 1] == '"':
                    buf.append('""'); i += 2; continue
                if script[i] == '"':
                    buf.append('"'); i += 1; break
                buf.append(script[i]); i += 1
            continue
        # Top-level statement terminator
        if c == ";":
            stmt = "".join(buf).strip()
            if stmt:
                statements.append(stmt)
            buf = []
            i += 1
            continue
        buf.append(c)
        i += 1

    # Trailing statement without ;
    tail = "".join(buf).strip()
    if tail:
        statements.append(tail)
    return statements

_SCAFFOLD_PROFILES_YML = _native_snowflake_profiles_yml()
