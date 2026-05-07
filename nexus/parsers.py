"""
nexus.parsers — legacy tech-stack and metadata parsers.

Each parser takes a raw text/XML string (the file content) and returns
a normalized dict shape:

    {"jobs": [...], "stages": [...], "links": [...], "columns": [...]}

so the upstream entity-extraction loop in the Reverse Engineering tab
can consume them generically. The shape is consistent across
DataStage, BODS, SSIS, Denodo, SQL dialects, Control-M, and shell.

Public functions:
    parse_datastage_dsx     — IBM DataStage .dsx / .xml exports
    parse_bods_xml          — SAP Business Objects Data Services
    parse_legacy_sql        — Oracle/DB2/Teradata-style SQL
    parse_netezza_sql       — Netezza SQL with DISTRIBUTE ON
    parse_mssql_sql         — MS SQL Server with [bracketed] identifiers
    parse_control_m         — Control-M jobs (XML or flat)
    parse_shell_script      — bash/ksh script function/invocation extraction
    parse_ssis_dtsx         — SSIS .dtsx packages
    parse_denodo_vql        — Denodo VQL views/wrappers/datasources
    parse_metadata_csv      — generic table/column inventory CSV
    parse_sttm_csv          — Source-to-Target Mapping CSV (alias-tolerant)
    parse_source_ddl_or_metadata — auto-dispatch SQL DDL vs metadata CSV
    detect_tech_stack       — best-guess classifier from filename + head
"""

import json
import re
from io import StringIO
from typing import Dict, List, Optional, Tuple

import pandas as pd


def parse_dbt_plan(text: str) -> list:
    """
    Parse a planner response into a list of unique relative paths.
    Tolerates numbered lists, bullets, backticks, prose, and JSON-wrapped
    newline-separated blobs (common Cortex planner failure mode).
    """
    if not text:
        return []

    work = text.strip()
    work = re.sub(r"^=+\s*PLAN\s*=+\s*", "", work, count=1, flags=re.I).strip()

    decoded = _decode_dbt_planner_path_blob(work)
    if decoded:
        work = decoded

    path_line_candidates: List[str] = []
    for raw_line in work.splitlines():
        ln = raw_line.strip()
        if not ln:
            continue
        if re.match(r"^=+$", ln) or re.match(r"^===.*===\s*$", ln):
            continue
        if ln.upper().startswith("=== ERR"):
            break
        blob = _decode_dbt_planner_path_blob(ln)
        if blob:
            for sub in blob.splitlines():
                sub = sub.strip()
                if sub:
                    path_line_candidates.append(sub)
        else:
            path_line_candidates.append(ln)

    paths: List[str] = []
    seen = set()
    for ln in path_line_candidates:
        ln = re.sub(r"^[\s#\-*•]+", "", ln)
        ln = re.sub(r"^\d+[.)]\s*", "", ln)
        ln = ln.strip('`"\' ')
        if not re.match(
            r"^[A-Za-z0-9_./-]+\.(sql|yml|yaml|md|json|py|csv)$",
            ln,
        ):
            continue
        if ln in seen:
            continue
        seen.add(ln)
        paths.append(ln)
    return paths

def parse_dbt_project_from_response(text: str) -> dict:
    """
    Parse the LLM's response into {filepath: content}. Handles many
    formats because different Cortex models emit different shapes.

    Strategy ORDER (first match wins):
      0. JSON array of {path, content} objects  (preferred — most reliable
         since LLMs handle JSON far better than strict Markdown)
      1. `### FILE: path` + fenced code block
      2. `## FILE: path` + fenced code block
      3. bare `FILE: path` + fenced code block
      4. `### path.ext` (path-in-header, extension-detected)
      5. fenced ```lang filename="path"
      6. `**path**` bold path + fenced code block

    If all structural strategies fail but the response DOES have code
    blocks, a last-ditch harvester infers filenames from the first line
    of each block (yaml headers, dbt materialization comments, etc).

    Unterminated final code blocks (LLM hit token limit) still yield
    partial files so the user sees what was produced.
    """
    if not text:
        return {}

    s = text.strip()

    # Unwrap JSON-string double-encoding — Cortex sometimes returns the
    # entire response as an escaped JSON string literal.
    if s.startswith('"') and s.endswith('"') and '\\n' in s:
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                s = decoded
        except (json.JSONDecodeError, ValueError):
            pass

    # ── Strategy 0: JSON array of {path, content} ────────────────────
    # The prompt now requests this format. It's the most reliable.
    files = _parse_dbt_json(s)
    if files:
        return files

    # ── Strategies 1-3: FILE: marker with varying header style ───────
    file_marker = re.compile(
        r'(?:^|\n)'
        r'(?:#{0,3}\s*)?'           # 0-3 hashes (optional)
        r'(?:FILE|File|file):\s*'   # FILE: label
        r'`?'                       # optional backtick around path
        r'([A-Za-z0-9_./\\-]+)'     # the path itself
        r'`?\s*\n+'
        r'```(?:[a-zA-Z0-9_+.-]*)?\s*\n'
        r'(.*?)'
        r'(?:\n```|\Z)',
        re.DOTALL,
    )
    files = {}
    for m in file_marker.finditer(s):
        path = m.group(1).strip()
        content = m.group(2)
        if path and content is not None:
            files[path] = content.rstrip() + "\n"

    if files:
        return files

    # ── Strategy 4: header is the path (detected by extension) ───────
    ext_marker = re.compile(
        r'(?:^|\n)#{1,4}\s+'
        r'([A-Za-z0-9_./\\-]+\.(?:sql|yml|yaml|md|json|py))\s*\n+'
        r'```(?:[a-zA-Z0-9_+.-]*)?\s*\n'
        r'(.*?)'
        r'(?:\n```|\Z)',
        re.DOTALL,
    )
    for m in ext_marker.finditer(s):
        path = m.group(1).strip()
        content = m.group(2)
        if path and content is not None:
            files[path] = content.rstrip() + "\n"

    if files:
        return files

    # ── Strategy 5: fence attribute ──────────────────────────────────
    attr_marker = re.compile(
        r'```[a-zA-Z0-9_+.-]*\s+'
        r'(?:filename|file|path|name)\s*=\s*["\']?'
        r'([A-Za-z0-9_./\\-]+)["\']?\s*\n'
        r'(.*?)'
        r'(?:\n```|\Z)',
        re.DOTALL,
    )
    for m in attr_marker.finditer(s):
        path = m.group(1).strip()
        content = m.group(2)
        if path and content is not None:
            files[path] = content.rstrip() + "\n"

    if files:
        return files

    # ── Strategy 6: **path** bold ────────────────────────────────────
    bold_marker = re.compile(
        r'(?:^|\n)\*\*'
        r'([A-Za-z0-9_./\\-]+\.(?:sql|yml|yaml|md|json|py))'
        r'\*\*\s*\n+'
        r'```(?:[a-zA-Z0-9_+.-]*)?\s*\n'
        r'(.*?)'
        r'(?:\n```|\Z)',
        re.DOTALL,
    )
    for m in bold_marker.finditer(s):
        path = m.group(1).strip()
        content = m.group(2)
        if path and content is not None:
            files[path] = content.rstrip() + "\n"

    if files:
        return files

    # ── Last resort: harvest naked fenced blocks and infer filenames ─
    # When the LLM emits code blocks but no filename markers at all,
    # infer paths from content signatures (dbt_project.yml, profiles.yml,
    # models starting with 'stg_', 'hub_', 'sat_', 'lnk_', etc.).
    return _harvest_naked_blocks(s)

def _scan_balanced_blocks(text: str, tag: str):
    """
    Walk BEGIN <tag> ... END <tag> blocks, correctly handling NESTED blocks
    of the same type (which defeats a simple non-greedy regex).
    Yields (identifier, body) for each top-level block.
    """
    begin_re = re.compile(rf'BEGIN\s+{tag}\b')
    end_re   = re.compile(rf'END\s+{tag}\b')
    pos = 0
    while pos < len(text):
        bm = begin_re.search(text, pos)
        if not bm:
            break
        depth = 1
        scan = bm.end()
        while depth > 0 and scan < len(text):
            next_begin = begin_re.search(text, scan)
            next_end   = end_re.search(text, scan)
            if not next_end:
                return
            if next_begin and next_begin.start() < next_end.start():
                depth += 1
                scan = next_begin.end()
            else:
                depth -= 1
                if depth == 0:
                    body = text[bm.end():next_end.start()]
                    ident_m = re.search(r'\bIdentifier\s+"([^"]*)"', body)
                    identifier = ident_m.group(1) if ident_m else ""
                    yield (identifier, body)
                    pos = next_end.end()
                    break
                scan = next_end.end()
        else:
            return
        if pos <= bm.end():
            pos = bm.end()

def _find_field(body: str, field: str) -> str:
    """Find `FieldName "value"` anywhere in `body`. Word-boundary anchored."""
    m = re.search(rf'\b{field}\s+"([^"]*)"', body)
    return m.group(1) if m else ""

def _parse_dsx_flat(content: str) -> dict:
    """Parse the classic flat-text DSX format."""
    jobs, stages, links, columns = [], [], [], []

    for job_id, job_body in _scan_balanced_blocks(content, "DSJOB"):
        if not job_id:
            continue
        desc = _find_field(job_body, "Description")
        jobs.append({"job_name": job_id, "description": desc})

        # Stages — Name > Identifier for display, since Identifier is often
        # a synthetic like "V0S3" while Name is the human-readable stage name.
        for stage_id, stage_body in _scan_balanced_blocks(job_body, "DSSTAGE"):
            stage_name = _find_field(stage_body, "Name") or stage_id
            stage_type = _find_field(stage_body, "StageType") or "Unknown"
            stages.append({
                "job_name":   job_id,
                "stage_name": stage_name,
                "stage_id":   stage_id,
                "stage_type": stage_type,
            })

            # Columns — each DSSUBRECORD inside a stage is one column.
            # Fields inside a DSSUBRECORD appear in arbitrary order and
            # may include Description, Scale, Nullable, KeyPosition, etc.
            # between Name and SqlType — so we search each field
            # independently rather than with a tight positional regex.
            sub_re = re.compile(
                r'BEGIN\s+DSSUBRECORD\b(.*?)\bEND\s+DSSUBRECORD\b',
                re.DOTALL
            )
            for sub_m in sub_re.finditer(stage_body):
                sub_body = sub_m.group(1)
                col_name = _find_field(sub_body, "Name")
                if not col_name:
                    continue
                columns.append({
                    "job_name":     job_id,
                    "stage_name":   stage_name,
                    "stage_id":     stage_id,
                    "column_name":  col_name,
                    "sql_type":     _find_field(sub_body, "SqlType"),
                    "precision":    _find_field(sub_body, "Precision"),
                    "scale":        _find_field(sub_body, "Scale"),
                    "nullable":     _find_field(sub_body, "Nullable"),
                    "key_position": _find_field(sub_body, "KeyPosition"),
                })

        # Links between stages
        for link_id, link_body in _scan_balanced_blocks(job_body, "DSLINK"):
            link_name = _find_field(link_body, "Name") or link_id
            links.append({
                "job_name":   job_id,
                "link_name":  link_name,
                "link_id":    link_id,
                "from_stage": _find_field(link_body, "FromStage"),
                "to_stage":   _find_field(link_body, "ToStage"),
            })

    return {"jobs": jobs, "stages": stages, "links": links, "columns": columns}

def _parse_dsx_xml(content: str) -> dict:
    """
    Parse XML-format DataStage exports. Some DataStage versions export as
    XML (<DSExport><Job>...<Stage>...<Column>...</Column></Stage></Job>).
    We keep this deliberately lenient — we don't assume strict XML.
    """
    jobs, stages, links, columns = [], [], [], []

    # Job blocks: <Job Name="..."> ... </Job>  (also <DSJob>)
    job_re = re.compile(
        r'<(?:DS)?Job\b[^>]*\bName\s*=\s*"([^"]+)"[^>]*>(.*?)</(?:DS)?Job>',
        re.DOTALL | re.IGNORECASE
    )
    for jm in job_re.finditer(content):
        job_name, job_body = jm.group(1), jm.group(2)
        desc_m = re.search(
            r'<Description\b[^>]*>(.*?)</Description>',
            job_body, re.DOTALL | re.IGNORECASE
        )
        jobs.append({
            "job_name": job_name,
            "description": (desc_m.group(1).strip() if desc_m else ""),
        })

        # Stages: <Stage Name="..." Type="..."> ... </Stage>
        stage_re = re.compile(
            r'<(?:DS)?Stage\b([^>]*)>(.*?)</(?:DS)?Stage>',
            re.DOTALL | re.IGNORECASE
        )
        for sm in stage_re.finditer(job_body):
            stage_attrs, stage_body = sm.group(1), sm.group(2)
            name_m  = re.search(r'\bName\s*=\s*"([^"]+)"', stage_attrs)
            type_m  = re.search(r'\b(?:Type|StageType)\s*=\s*"([^"]+)"',
                                stage_attrs)
            stage_name = name_m.group(1) if name_m else ""
            if not stage_name:
                continue
            stage_type = type_m.group(1) if type_m else "Unknown"
            stages.append({
                "job_name":   job_name,
                "stage_name": stage_name,
                "stage_id":   stage_name,
                "stage_type": stage_type,
            })

            # Columns: <Column Name="..." SqlType="..." Precision="..."/>
            # and also <Field Name="..."/>
            col_re = re.compile(
                r'<(?:Column|Field)\b([^/>]*?)/?>', re.IGNORECASE
            )
            for cm in col_re.finditer(stage_body):
                attrs = cm.group(1)
                cname_m = re.search(r'\bName\s*=\s*"([^"]+)"', attrs)
                if not cname_m:
                    continue
                col_name = cname_m.group(1)
                sqlt = re.search(r'\bSqlType\s*=\s*"([^"]*)"', attrs)
                prec = re.search(r'\bPrecision\s*=\s*"([^"]*)"', attrs)
                columns.append({
                    "job_name":    job_name,
                    "stage_name":  stage_name,
                    "stage_id":    stage_name,
                    "column_name": col_name,
                    "sql_type":    sqlt.group(1) if sqlt else "",
                    "precision":   prec.group(1) if prec else "",
                    "scale":       "",
                    "nullable":    "",
                    "key_position":"",
                })

        # Links: <Link From="..." To="..."/>
        link_re = re.compile(r'<(?:DS)?Link\b([^/>]*?)/?>', re.IGNORECASE)
        for lm in link_re.finditer(job_body):
            attrs = lm.group(1)
            f = re.search(r'\b(?:From|FromStage)\s*=\s*"([^"]+)"', attrs)
            t = re.search(r'\b(?:To|ToStage)\s*=\s*"([^"]+)"', attrs)
            n = re.search(r'\bName\s*=\s*"([^"]+)"', attrs)
            if f and t:
                links.append({
                    "job_name":   job_name,
                    "link_name":  n.group(1) if n else "",
                    "link_id":    n.group(1) if n else "",
                    "from_stage": f.group(1),
                    "to_stage":   t.group(1),
                })

    return {"jobs": jobs, "stages": stages, "links": links, "columns": columns}

def parse_datastage_dsx(content: str) -> dict:
    """
    Extract jobs, stages, links, and column metadata from an IBM DataStage
    export (.dsx or .xml). Handles both the classic flat keyword-based
    DSX format AND XML-format exports.

    Key points:
    - Column definitions live inside BEGIN DSSUBRECORD...END DSSUBRECORD
      blocks in flat DSX. Fields inside a subrecord appear in arbitrary
      order (Name, Description, SourceColumn, SqlType, Precision, Scale,
      Nullable, KeyPosition...) so we scan each field independently
      rather than assume a fixed positional layout.
    - Nested BEGIN/END blocks of the same tag (DSSTAGE can contain
      DSRECORD etc.) require a depth-aware scanner, not plain regex.
    - The stage's `Name` field is the user-facing identifier (e.g.
      "CustomerSource"); `Identifier` is often a synthetic id like "V0S0".
    """
    if not content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    # Decide flat vs XML by looking at the first ~2 KB — XML exports start
    # with <?xml or have obvious tags; flat DSX starts with BEGIN HEADER
    # or BEGIN DSJOB.
    head = content[:4096].lstrip()
    looks_xml = (
        head.startswith("<?xml")
        or re.search(r'<(?:DS)?Job\b[^>]*>', head, re.IGNORECASE)
        is not None
    )

    if looks_xml:
        result = _parse_dsx_xml(content)
        # Fall back to flat if XML parse produced nothing
        if result["jobs"] or result["stages"] or result["columns"]:
            return result

    return _parse_dsx_flat(content)

def parse_metadata_csv(content: str) -> dict:
    """Parse a generic metadata CSV (table/column inventory)."""
    df = pd.read_csv(StringIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    return {"columns_df": df, "row_count": len(df)}

def parse_sttm_csv(content: str) -> dict:
    """
    Parse an existing Source-to-Target Mapping CSV (used as ground-truth
    seed for the LLM analysis). Tolerant of column-naming variants.

    Recognized column aliases (case-insensitive, spaces/underscores
    normalized):
      - source_system, src_system, system
      - source_object, source_table, src_table, source_entity
      - source_column, source_field, src_column
      - source_type, source_data_type, src_type
      - target_system, tgt_system
      - target_object, target_table, tgt_table, target_entity
      - target_column, target_field, tgt_column
      - target_type, target_data_type, tgt_type
      - transformation, transformation_logic, mapping_rule, rule
    Returns ``{"sttm_df": df_normalized, "row_count": int}``.
    """
    df = pd.read_csv(StringIO(content))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    aliases = {
        "source_system":  ("source_system", "src_system", "system",
                           "source_app"),
        "source_object":  ("source_object", "source_table", "src_table",
                           "source_entity", "src_entity", "src_object"),
        "source_column":  ("source_column", "source_field", "src_column",
                           "src_field"),
        "source_type":    ("source_type", "source_data_type", "src_type",
                           "src_data_type"),
        "target_system":  ("target_system", "tgt_system", "target_app"),
        "target_object":  ("target_object", "target_table", "tgt_table",
                           "target_entity", "tgt_entity", "tgt_object"),
        "target_column":  ("target_column", "target_field", "tgt_column",
                           "tgt_field"),
        "target_type":    ("target_type", "target_data_type", "tgt_type",
                           "tgt_data_type"),
        "transformation": ("transformation", "transformation_logic",
                           "mapping_rule", "rule", "logic"),
    }
    rename_map = {}
    for canonical, options in aliases.items():
        for opt in options:
            if opt in df.columns:
                rename_map[opt] = canonical
                break
    df = df.rename(columns=rename_map)
    return {"sttm_df": df, "row_count": len(df)}

def parse_bods_xml(content: str) -> dict:
    """
    Parse an SAP BusinessObjects Data Services (BODS) XML export. BODS
    stores datastores, dataflows, sources, targets, and column mappings
    in a deeply-nested XML — we extract just enough to populate the
    {jobs, stages, links, columns} shape:
      - jobs: <DIBatchJob>, <DIRealtimeJob>, or <Job> elements
      - stages: <DISource>, <DITarget>, <DITransform>, <DIQuery>,
                <DataflowCallStep> elements (each becomes a "stage")
      - columns: <DIElement> children inside each stage's
                 <DIInputSchema>/<DIOutputSchema>
      - links: pairs of (source stage, target stage) inferred from
               <DIInputSchema> ref attributes
    Returns ``{jobs, stages, links, columns}`` (same shape as
    ``parse_datastage_dsx``).
    """
    if not content or "<" not in content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
    except Exception:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    def lname(elem):
        # Strip namespace prefix to keep matching simple
        return elem.tag.rsplit("}", 1)[-1]

    jobs, stages, links, columns = [], [], [], []
    job_tags = {"DIBatchJob", "DIRealtimeJob", "Job"}
    stage_tags = {
        "DISource", "DITarget", "DITransform", "DIQuery",
        "DataflowCallStep", "DIDataflow", "DIWorkflow",
    }

    # Walk: each Job element gets a job_name; each stage-tag descendant
    # becomes a stage in that job. Columns inside the stage's schema
    # children are emitted as column rows.
    def walk(elem, current_job=""):
        tag = lname(elem)
        if tag in job_tags:
            jname = (
                elem.attrib.get("name")
                or elem.attrib.get("Name")
                or "UnnamedJob"
            )
            jobs.append({"job_name": jname})
            current_job = jname
        if tag in stage_tags:
            sname = (
                elem.attrib.get("name")
                or elem.attrib.get("Name")
                or elem.attrib.get("typeId")
                or f"{tag}_{len(stages)}"
            )
            stype = elem.attrib.get("typeId") or tag
            stages.append({
                "job_name":   current_job or "BODS_DEFAULT",
                "stage_name": sname,
                "stage_type": stype,
            })
            # Columns from input/output schemas
            for schema_child in elem.iter():
                ctag = lname(schema_child)
                if ctag in ("DIElement", "DIColumn", "Element", "Column"):
                    cname = (
                        schema_child.attrib.get("name")
                        or schema_child.attrib.get("Name")
                    )
                    if not cname:
                        continue
                    columns.append({
                        "job_name":    current_job or "BODS_DEFAULT",
                        "stage_name":  sname,
                        "column_name": cname,
                        "sql_type":    (
                            schema_child.attrib.get("dataType")
                            or schema_child.attrib.get("DataType")
                            or ""
                        ),
                        "precision": (
                            schema_child.attrib.get("length")
                            or schema_child.attrib.get("precision") or ""
                        ),
                        "scale":     schema_child.attrib.get("scale", ""),
                    })
            # Link inference: <DIInputSchema fromObject="..."/>
            for child in elem:
                if lname(child) in ("DIInputSchema", "InputSchema"):
                    src = (
                        child.attrib.get("fromObject")
                        or child.attrib.get("FromObject")
                    )
                    if src:
                        links.append({
                            "job_name":   current_job or "BODS_DEFAULT",
                            "from_stage": src,
                            "to_stage":   sname,
                        })
        for child in elem:
            walk(child, current_job)

    walk(root)
    return {"jobs": jobs, "stages": stages, "links": links,
            "columns": columns}

def _sql_extract_objects(content: str, dialect: str = "generic") -> dict:
    """
    Shared extractor used by parse_legacy_sql / parse_netezza_sql /
    parse_mssql_sql. Pulls:
      - CREATE TABLE / CREATE VIEW definitions (each becomes a stage)
      - Column lists (each column inside a CREATE becomes a column row)
      - INSERT INTO ... SELECT FROM lineage (becomes a link)
    The dialect arg only changes ``stage_type`` labeling for downstream
    catalog grouping.

    Column-block extraction is paren-balanced: ``VARCHAR(100)`` and
    ``NUMERIC(18, 2)`` inside the CREATE TABLE clause must NOT terminate
    the column block. A non-greedy ``[^;]*?\\)`` regex (the obvious
    approach) stops at the first ``)`` it finds, mangling the column
    list. We walk the string forward from the opening paren and track
    paren depth so the close is found at depth 0.
    """
    if not content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    job_name = f"SQL_{dialect.upper()}"
    jobs = [{"job_name": job_name}]
    stages, columns, links = [], [], []

    # CREATE [OR REPLACE] [TEMP] (TABLE|VIEW|EXTERNAL TABLE) [IF NOT EXISTS] name
    head_re = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:TEMP\s+|TEMPORARY\s+)?"
        r"(TABLE|VIEW|EXTERNAL\s+TABLE)\s+"
        r"(?:IF\s+NOT\s+EXISTS\s+)?"
        r"((?:\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_][\w$]*)"
        r"(?:\.(?:\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_][\w$]*))*)",
        flags=re.IGNORECASE,
    )

    for hm in head_re.finditer(content):
        kind = hm.group(1).upper().replace("EXTERNAL ", "EXT_")
        full_name = hm.group(2).strip().strip('"').strip("[]")
        bare = full_name.rsplit(".", 1)[-1].strip('"').strip("[]")
        # Find the optional column block: paren-balanced
        col_block = ""
        # Skip whitespace after the matched name
        i = hm.end()
        while i < len(content) and content[i] in " \t\r\n":
            i += 1
        if i < len(content) and content[i] == "(":
            depth = 0
            start = i + 1
            j = i
            while j < len(content):
                ch = content[j]
                if ch == "(":
                    depth += 1
                elif ch == ")":
                    depth -= 1
                    if depth == 0:
                        col_block = content[start:j]
                        break
                j += 1

        stages.append({
            "job_name":   job_name,
            "stage_name": bare,
            "stage_type": f"{dialect.upper()}_{kind}",
        })

        for line in _split_top_level_commas(col_block):
            line = line.strip()
            if not line:
                continue
            head = line.split(None, 1)
            if not head:
                continue
            first = head[0].upper().strip(",").strip("(").strip()
            if first in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK",
                         "CONSTRAINT", "INDEX", "KEY"}:
                continue
            cm = re.match(
                r'^(\[[^\]]+\]|"[^"]+"|[A-Za-z_][\w$]*)'
                r'\s+([A-Za-z_][\w()\s,]*)',
                line,
            )
            if not cm:
                continue
            cname = cm.group(1).strip('"').strip("[]")
            ctype = cm.group(2).strip()
            tm = re.match(
                r"([A-Za-z_][\w]*)\s*(?:\(([^)]*)\))?", ctype,
            )
            base_type, params = ("", "")
            if tm:
                base_type = tm.group(1).upper()
                params = (tm.group(2) or "").strip()
            prec, scale = "", ""
            if params:
                bits = [p.strip() for p in params.split(",")]
                prec = bits[0] if len(bits) >= 1 else ""
                scale = bits[1] if len(bits) >= 2 else ""
            columns.append({
                "job_name":    job_name,
                "stage_name":  bare,
                "column_name": cname,
                "sql_type":    base_type,
                "precision":   prec,
                "scale":       scale,
            })

    # INSERT INTO target SELECT ... FROM source — basic lineage
    target_re = re.compile(
        r"INSERT\s+INTO\s+"
        r"((?:\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_][\w$]*)"
        r"(?:\.(?:\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_][\w$]*))*)",
        flags=re.IGNORECASE,
    )
    for tm in target_re.finditer(content):
        tgt = tm.group(1).rsplit(".", 1)[-1].strip('"').strip("[]")
        rest = content[tm.end():tm.end() + 4000]
        fm = re.search(
            r"\bFROM\s+"
            r"((?:\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_][\w$]*)"
            r"(?:\.(?:\[[^\]]+\]|\"[^\"]+\"|[A-Za-z_][\w$]*))*)",
            rest, flags=re.IGNORECASE,
        )
        if fm:
            src_tbl = fm.group(1).rsplit(".", 1)[-1].strip('"').strip("[]")
            links.append({"job_name": job_name,
                          "from_stage": src_tbl, "to_stage": tgt})
    return {"jobs": jobs, "stages": stages, "links": links,
            "columns": columns}

def _split_top_level_commas(s: str) -> List[str]:
    """Split a string on commas that are at parenthesis depth 0."""
    out, buf, depth = [], [], 0
    for ch in s:
        if ch == "(":
            depth += 1
            buf.append(ch)
        elif ch == ")":
            depth = max(0, depth - 1)
            buf.append(ch)
        elif ch == "," and depth == 0:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    if buf:
        out.append("".join(buf))
    return out

def parse_legacy_sql(content: str) -> dict:
    """Parse generic legacy SQL (Oracle/DB2/Teradata flavor)."""
    return _sql_extract_objects(content, dialect="legacy")

def parse_netezza_sql(content: str) -> dict:
    """Parse Netezza SQL — same general extractor, different dialect tag."""
    return _sql_extract_objects(content, dialect="netezza")

def parse_mssql_sql(content: str) -> dict:
    """Parse MS SQL Server scripts (.sql with [bracketed] identifiers)."""
    return _sql_extract_objects(content, dialect="mssql")

def parse_control_m(content: str) -> dict:
    """
    Parse Control-M job definitions (.ctm or .xml export). Captures:
      - JOB / JOBNAME entries → stages with stage_type='CONTROLM_JOB'
      - DAYS_AND_OR / SCHEDULE info as job-level metadata (ignored at
        column level)
      - DO_COND / IN_COND dependencies → links

    Control-M files come in three formats: legacy flat (KEY = VALUE),
    XML (<JOB ...>), and JSON. We try XML first then fall back to flat.
    """
    if not content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}
    stages, links = [], []
    parsed_xml = False
    if content.lstrip().startswith("<"):
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content)
            for elem in root.iter():
                tag = elem.tag.rsplit("}", 1)[-1]
                if tag.upper() in ("JOB", "SMART_FOLDER", "FOLDER"):
                    name = (
                        elem.attrib.get("JOBNAME")
                        or elem.attrib.get("FOLDER_NAME")
                        or elem.attrib.get("JOB_NAME")
                        or elem.attrib.get("name")
                    )
                    if name:
                        stages.append({
                            "job_name":   "CONTROLM",
                            "stage_name": name,
                            "stage_type": f"CONTROLM_{tag.upper()}",
                        })
                if tag.upper() == "INCOND" or tag.upper() == "OUTCOND":
                    nm = elem.attrib.get("NAME") or elem.attrib.get("name")
                    parent_name = None
                    # Walk up to find the enclosing JOB
                    # ElementTree doesn't track parents, so use attribute
                    # NAME proximity heuristic skipped — links inferred
                    # from co-occurrence of IN_COND/OUT_COND with same name
                    if nm:
                        # We collect both ends; resolve at end of iteration
                        if tag.upper() == "OUTCOND":
                            stages and links.append({
                                "job_name":   "CONTROLM",
                                "from_stage": stages[-1]["stage_name"],
                                "to_stage":   nm,
                            })
                        else:
                            stages and links.append({
                                "job_name":   "CONTROLM",
                                "from_stage": nm,
                                "to_stage":   stages[-1]["stage_name"],
                            })
            parsed_xml = True
        except Exception:
            parsed_xml = False

    if not parsed_xml:
        # Flat format: JOBNAME = X / IN_COND = Y / OUT_COND = Z, separated
        # by blank lines or repeated JOBNAME keys.
        current = None
        for raw_line in content.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                if current:
                    stages.append(current)
                    current = None
                continue
            m = re.match(
                r"^(JOBNAME|JOB_NAME|MEMNAME|TASKTYPE|"
                r"IN_COND|OUT_COND|DESCRIPTION|TABLE)"
                r"\s*[=:]\s*(.+?)\s*$",
                line, re.IGNORECASE,
            )
            if not m:
                continue
            key = m.group(1).upper()
            val = m.group(2).strip()
            if key in ("JOBNAME", "JOB_NAME", "MEMNAME"):
                if current:
                    stages.append(current)
                current = {
                    "job_name":   "CONTROLM",
                    "stage_name": val,
                    "stage_type": "CONTROLM_JOB",
                }
            elif key == "IN_COND" and current:
                links.append({
                    "job_name":   "CONTROLM",
                    "from_stage": val.split()[0],
                    "to_stage":   current["stage_name"],
                })
            elif key == "OUT_COND" and current:
                links.append({
                    "job_name":   "CONTROLM",
                    "from_stage": current["stage_name"],
                    "to_stage":   val.split()[0],
                })
        if current:
            stages.append(current)

    return {
        "jobs":    [{"job_name": "CONTROLM"}] if stages else [],
        "stages":  stages,
        "links":   links,
        "columns": [],
    }

def parse_shell_script(content: str) -> dict:
    """
    Parse shell scripts. We capture:
      - Each shell function definition as a stage
      - Top-level commands invoking other scripts/sql/dbt as a stage
      - File reads/writes (cat/awk + > redirection) as links
    Columns aren't meaningful here, so we return an empty list.
    """
    if not content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}
    job_name = "SHELL"
    stages, links = [], []
    # Functions: `name() {` or `function name {`
    for m in re.finditer(
        r"(?m)^\s*(?:function\s+)?([A-Za-z_][\w]*)\s*\(\)\s*\{",
        content,
    ):
        stages.append({
            "job_name":   job_name,
            "stage_name": m.group(1),
            "stage_type": "SHELL_FUNCTION",
        })
    # External commands (heuristic: invocations of another .sh or sql
    # tool). Each unique invocation becomes a stage.
    seen_invocations = set()
    for m in re.finditer(
        r"(?m)^\s*([./\w-]+\.(?:sh|ksh|bash|sql|py|pl))\b",
        content,
    ):
        inv = m.group(1).rsplit("/", 1)[-1]
        if inv not in seen_invocations:
            seen_invocations.add(inv)
            stages.append({
                "job_name":   job_name,
                "stage_name": inv,
                "stage_type": "SHELL_INVOCATION",
            })
    # Redirections: `cmd > file.txt` and `cmd < file.txt` → infer links
    for m in re.finditer(
        r"(?m)([\w./-]+)\s*>\s*([\w./-]+)",
        content,
    ):
        links.append({
            "job_name":   job_name,
            "from_stage": m.group(1).rsplit("/", 1)[-1],
            "to_stage":   m.group(2).rsplit("/", 1)[-1],
        })
    return {
        "jobs":    [{"job_name": job_name}] if stages else [],
        "stages":  stages,
        "links":   links,
        "columns": [],
    }

def parse_ssis_dtsx(content: str) -> dict:
    """
    Parse an SSIS package (.dtsx — XML). Extracts:
      - <DTS:Executable> / <DTS:Pipeline> elements as stages
      - <DTS:PrecedenceConstraint> as links (from→to executable refIds)
      - Column metadata from <inputColumn>/<outputColumn> children
        inside dataflow components.
    """
    if not content or "<" not in content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
    except Exception:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    def lname(elem):
        return elem.tag.rsplit("}", 1)[-1]

    # Find package name (root attribute or first Executable's ObjectName)
    pkg_name = (
        root.attrib.get("DTS:ObjectName")
        or root.attrib.get("ObjectName")
        or "SSIS_PACKAGE"
    )
    # Strip namespace from attrs by also checking simple keys
    for k in list(root.attrib.keys()):
        if k.endswith("}ObjectName") or k == "ObjectName":
            pkg_name = root.attrib[k]
            break

    stages, links, columns = [], [], []
    refid_to_name = {}

    for elem in root.iter():
        tag = lname(elem)
        attrs = elem.attrib
        # Resolve namespaced attrs
        def ga(*keys):
            for k in keys:
                for full in (k, f"{{www.microsoft.com/SqlServer/Dts}}{k}"):
                    if full in attrs:
                        return attrs[full]
            for full, v in attrs.items():
                if full.rsplit("}", 1)[-1] in keys:
                    return v
            return ""

        if tag == "Executable":
            name = ga("ObjectName") or ga("refId") or f"Exec_{len(stages)}"
            ref = ga("refId") or name
            stype = ga("ExecutableType") or "SSIS_EXECUTABLE"
            stages.append({
                "job_name":   pkg_name,
                "stage_name": name,
                "stage_type": stype.split(".")[-1].upper(),
            })
            refid_to_name[ref] = name
        elif tag == "PrecedenceConstraint":
            frm = ga("From")
            to  = ga("To")
            if frm and to:
                links.append({
                    "job_name":   pkg_name,
                    "from_stage": frm,
                    "to_stage":   to,
                })
        elif tag == "component":
            # Dataflow component
            name = (attrs.get("name")
                    or attrs.get("description")
                    or f"Comp_{len(stages)}")
            stages.append({
                "job_name":   pkg_name,
                "stage_name": name,
                "stage_type": "SSIS_DATAFLOW_COMPONENT",
            })
            for child in elem.iter():
                if lname(child) in ("inputColumn", "outputColumn",
                                    "externalMetadataColumn"):
                    cname = (
                        child.attrib.get("name")
                        or child.attrib.get("Name")
                    )
                    if not cname:
                        continue
                    columns.append({
                        "job_name":   pkg_name,
                        "stage_name": name,
                        "column_name": cname,
                        "sql_type":   (
                            child.attrib.get("dataType", "")
                            or child.attrib.get("DataType", "")
                        ),
                        "precision":  child.attrib.get("length", ""),
                        "scale":      child.attrib.get("scale", ""),
                    })

    # Resolve link endpoints from refId → name where possible
    resolved_links = []
    for lk in links:
        resolved_links.append({
            "job_name":   lk["job_name"],
            "from_stage": refid_to_name.get(lk["from_stage"],
                                            lk["from_stage"]),
            "to_stage":   refid_to_name.get(lk["to_stage"], lk["to_stage"]),
        })

    return {
        "jobs":    [{"job_name": pkg_name}],
        "stages":  stages,
        "links":   resolved_links,
        "columns": columns,
    }

def parse_denodo_vql(content: str) -> dict:
    """
    Parse Denodo VQL (.vql). Captures CREATE VIEW / CREATE BASE VIEW /
    CREATE WRAPPER / CREATE DATASOURCE statements as stages, their
    column lists as columns, and FROM clauses as lineage links.
    """
    if not content:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    job_name = "DENODO"
    stages, links, columns = [], [], []

    # CREATE [OR REPLACE] [BASE] VIEW name (col1 type, col2 type ...) AS ...
    view_re = re.compile(
        r"CREATE\s+(?:OR\s+REPLACE\s+)?(?:BASE\s+)?VIEW\s+"
        r"([A-Za-z_][\w]*)"
        r"(?:\s*\(([^)]*?)\))?"
        r"(?:\s+AS\b)?",
        flags=re.IGNORECASE,
    )
    for m in view_re.finditer(content):
        vname = m.group(1)
        col_block = m.group(2) or ""
        stages.append({
            "job_name":   job_name,
            "stage_name": vname,
            "stage_type": "DENODO_VIEW",
        })
        for line in _split_top_level_commas(col_block):
            line = line.strip()
            cm = re.match(
                r"^([A-Za-z_][\w]*)\s+([A-Za-z_][\w]*)"
                r"(?:\(([^)]*)\))?", line,
            )
            if cm:
                cname = cm.group(1)
                base_type = cm.group(2).upper()
                params = (cm.group(3) or "").strip()
                prec, scale = "", ""
                if params:
                    bits = [p.strip() for p in params.split(",")]
                    prec = bits[0] if bits else ""
                    scale = bits[1] if len(bits) > 1 else ""
                columns.append({
                    "job_name":    job_name,
                    "stage_name":  vname,
                    "column_name": cname,
                    "sql_type":    base_type,
                    "precision":   prec,
                    "scale":       scale,
                })

    # CREATE WRAPPER and CREATE DATASOURCE
    for keyword, label in (
        ("WRAPPER", "DENODO_WRAPPER"),
        ("DATASOURCE", "DENODO_DATASOURCE"),
    ):
        for m in re.finditer(
            rf"CREATE\s+(?:OR\s+REPLACE\s+)?{keyword}\s+"
            r"(?:\w+\s+)*([A-Za-z_][\w]*)",
            content, flags=re.IGNORECASE,
        ):
            stages.append({
                "job_name":   job_name,
                "stage_name": m.group(1),
                "stage_type": label,
            })

    # Lineage: FROM name → current view (best effort)
    for vm in view_re.finditer(content):
        vname = vm.group(1)
        # Look at the next 2 KB after AS ... for FROM clauses
        rest = content[vm.end():vm.end() + 2000]
        for fm in re.finditer(
            r"\bFROM\s+([A-Za-z_][\w]*)",
            rest, flags=re.IGNORECASE,
        ):
            links.append({
                "job_name":   job_name,
                "from_stage": fm.group(1),
                "to_stage":   vname,
            })

    return {"jobs": [{"job_name": job_name}] if stages else [],
            "stages": stages, "links": links, "columns": columns}

def detect_tech_stack(filename: str, content_head: str) -> str:
    """
    Best-guess classifier for content-based tech-stack detection.
    Used by the 'Auto-detect' uploader fallback. Returns one of:
      datastage, bods, ssis, denodo, controlm, shell, netezza, mssql,
      legacy_sql, sttm_csv, csv
    """
    fn = filename.lower()
    head = (content_head or "")[:4096]
    head_l = head.lower()
    # Extension-first quick wins
    if fn.endswith(".dtsx"):
        return "ssis"
    if fn.endswith(".vql"):
        return "denodo"
    if fn.endswith((".sh", ".ksh", ".bash")):
        return "shell"
    if fn.endswith(".dsx"):
        return "datastage"
    # Content sniffs for ambiguous extensions
    if "<DataIntegratorPackage" in head or "<DIBatchJob" in head:
        return "bods"
    if "<DTS:Executable" in head or "DTS:ObjectName" in head:
        return "ssis"
    if "BEGIN DSJOB" in head or "BEGIN HEADER" in head:
        return "datastage"
    if re.search(r"<\s*(?:DS)?Job\b", head, re.I):
        return "datastage"
    if "JOBNAME" in head and ("IN_COND" in head_l
                              or "out_cond" in head_l
                              or "schedule" in head_l):
        return "controlm"
    if fn.endswith(".vql") or "create wrapper" in head_l:
        return "denodo"
    if fn.endswith(".sql"):
        # Distinguish dialects via vendor-specific syntax
        if "[" in head and "]" in head and re.search(
            r"\[[A-Za-z_][\w]*\]\.\[[A-Za-z_][\w]*\]", head,
        ):
            return "mssql"
        if re.search(r"\bnzload\b|\bDISTRIBUTE\s+ON\b", head, re.I):
            return "netezza"
        return "legacy_sql"
    if fn.endswith(".csv"):
        # Sniff for STTM-shaped CSV (has both source_* and target_*)
        if re.search(
            r"source[_ ]?(column|table|object).*target[_ ]?(column|table|object)",
            head_l, re.S,
        ):
            return "sttm_csv"
        return "csv"
    if fn.endswith((".xml", ".txt")):
        # Could be DataStage XML, BODS XML, or Control-M XML
        if "<DataIntegratorPackage" in head:
            return "bods"
        if re.search(r"<\s*(?:DS)?Job\b", head, re.I):
            return "datastage"
        if "<JOB " in head.upper() or "<FOLDER " in head.upper():
            return "controlm"
        return "datastage"
    return "csv"


def _decode_dbt_planner_path_blob(s: str) -> Optional[str]:
    """
    Cortex often returns the whole plan as one JSON string, e.g.
    ``"dbt_project.yml\\npackages.yml\\n..."``. Decode to a newline-separated
    path list, or None if ``s`` is not such a blob.
    """
    s = s.strip().strip("`")
    if not s:
        return None
    if s.startswith('"') and s.endswith('"'):
        try:
            inner = json.loads(s)
            if isinstance(inner, str):
                return inner
        except (json.JSONDecodeError, TypeError, ValueError):
            inner2 = s[1:-1]
            if "\\n" in inner2 or "\\r" in inner2:
                return inner2.replace("\\r\\n", "\n").replace("\\n", "\n").replace(
                    "\\r", "\n"
                )
            return inner2
    if s.startswith("[") and s.endswith("]"):
        try:
            arr = json.loads(s)
            if isinstance(arr, list):
                return "\n".join(str(x).strip() for x in arr if str(x).strip())
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
    return None


def _harvest_naked_blocks(s: str) -> dict:
    """
    Last-ditch filename inference when the LLM produced code blocks
    without any filename markers. Inspects each block's content for
    dbt-specific signatures and assigns a reasonable path.
    """
    blocks = re.findall(
        r'```(?:[a-zA-Z0-9_+.-]*)?\s*\n(.*?)(?:\n```|\Z)',
        s, re.DOTALL,
    )
    if not blocks:
        return {}

    files = {}
    used_names = set()

    def _unique(name):
        if name not in used_names:
            used_names.add(name)
            return name
        i = 2
        while f"{name}_{i}" in used_names:
            i += 1
        used_names.add(f"{name}_{i}")
        return f"{name}_{i}"

    for i, body in enumerate(blocks):
        body = body.rstrip() + "\n"
        first = body.strip().split("\n", 1)[0].lower()
        path = None

        # YAML files
        if re.search(r"^\s*name:\s*['\"]?\w", body, re.MULTILINE) and \
           re.search(r"^\s*version:", body, re.MULTILINE) and \
           "profile:" in body:
            path = _unique("dbt_project.yml")
        elif "packages:" in body and "automate_dv" in body.lower():
            path = _unique("packages.yml")
        elif "target:" in body and "outputs:" in body and \
             "snowflake" in body.lower():
            path = _unique("profiles.yml")
        elif re.search(r"^\s*sources:", body, re.MULTILINE):
            path = _unique("models/bronze/_sources.yml")
        elif re.search(r"^\s*models:\s*$", body, re.MULTILINE) or \
             re.search(r"^\s*- name:\s+\w", body, re.MULTILINE):
            path = _unique(f"models/schema_{i:02d}.yml")

        # dbt SQL files — infer from automate_dv macro or model prefix
        elif "automate_dv.hub" in body:
            m = re.search(r"automate_dv\.hub.*?source_model\s*=\s*['\"]?"
                          r"(?:stg_)?(\w+)", body, re.DOTALL)
            stem = m.group(1) if m else f"{i}"
            # Avoid stems that already contain "hub_" (prevents "hub_hub_2")
            stem = re.sub(r"^(hub_|hub$)", "", stem) or str(i)
            path = _unique(f"models/silver/raw_vault/hubs/hub_{stem}.sql")
        elif "automate_dv.link" in body:
            path = _unique(
                f"models/silver/raw_vault/links/lnk_{i}.sql"
            )
        elif "automate_dv.sat" in body:
            path = _unique(
                f"models/silver/raw_vault/sats/sat_{i}.sql"
            )
        elif "automate_dv.stage" in body:
            path = _unique(
                f"models/silver/raw_vault/stage/stg_{i}.sql"
            )
        elif "automate_dv.pit" in body:
            path = _unique(
                f"models/silver/business_vault/pit/pit_{i}.sql"
            )
        elif "automate_dv.bridge" in body:
            path = _unique(
                f"models/silver/business_vault/bridge/bridge_{i}.sql"
            )
        elif "macro" in first and "{%" in body:
            path = _unique(f"macros/macro_{i:02d}.sql")
        elif re.search(r"\{\{\s*source\s*\(", body):
            # bronze model
            path = _unique(f"models/bronze/br_{i:02d}.sql")
        elif re.search(r"\{\{\s*ref\s*\(", body):
            path = _unique(f"models/gold/semantic/model_{i:02d}.sql")

        if not path:
            # Give it a neutral name so the user can still review it
            ext = "sql" if "SELECT" in body.upper() or "{{" in body \
                  else "txt"
            path = _unique(f"models/unclassified_{i:02d}.{ext}")

        files[path] = body

    return files


def _parse_dbt_json(s: str) -> dict:
    """
    Parse a JSON array response of the form:
      [{"path": "models/hub_customer.sql", "content": "..."}, ...]
    Tolerates leading/trailing prose around the JSON array.
    Returns {} if no valid JSON array found.
    """
    # Find the first `[` that starts a valid top-level array and the
    # matching closing `]`. Scan character by character respecting
    # string quoting so we don't match a `[` inside content.
    start = s.find("[")
    if start == -1:
        return {}

    # Try to parse progressively — LLM may have appended prose
    # after the array. Find the end of the array.
    depth = 0
    in_string = False
    escape = False
    end = -1
    for i in range(start, len(s)):
        c = s[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end == -1:
        # Truncated JSON — try to recover by trimming to the last
        # complete object before cut-off.
        candidate = s[start:]
    else:
        candidate = s[start:end]

    # Try parsing as-is first
    try:
        data = json.loads(candidate)
    except (json.JSONDecodeError, ValueError):
        # Try to recover a truncated array by closing it and removing
        # any trailing partial object.
        trimmed = candidate.rstrip(", \n\t")
        # Drop any trailing incomplete object
        last_obj_close = trimmed.rfind("}")
        if last_obj_close == -1:
            return {}
        recovered = trimmed[:last_obj_close + 1] + "]"
        try:
            data = json.loads(recovered)
        except (json.JSONDecodeError, ValueError):
            return {}

    if not isinstance(data, list):
        return {}

    files = {}
    for item in data:
        if not isinstance(item, dict):
            continue
        path = (item.get("path") or item.get("file")
                or item.get("filename") or item.get("name") or "")
        content = item.get("content") or item.get("body") or ""
        path = str(path).strip()
        content = str(content)
        if path and content:
            files[path] = content.rstrip() + "\n"
    return files


# ─────────────────────────────────────────────────────────────────────
# Source DDL / Metadata dispatcher — accepts both .sql and .csv content
# and routes to the right concrete parser, returning the standard
# {jobs, stages, links, columns} shape.
# ─────────────────────────────────────────────────────────────────────
def parse_source_ddl_or_metadata(content: str) -> dict:
    """Auto-dispatch parser for the "Source DDL / Metadata" tech zone.

    Sniffs the content shape:
      - Looks like SQL  → delegate to ``_sql_extract_objects`` (generic)
      - Looks like CSV  → convert the metadata-CSV row inventory into
        the standard parser shape so it merges into qg_all_entities and
        feeds Phase-1 artifacts (Lineage, STTM, Catalog, Raw Vault).

    The CSV is assumed to be a generic table/column inventory with at
    least these conceptual columns (case-insensitive, alias-tolerant):
        table  (or: table_name / source_table / object / src_table /
                source_object / entity / source_entity)
        column (or: column_name / field / source_column / src_column)
        type   (optional: data_type / sql_type / source_data_type)
        schema (optional: schema_name / src_schema / database)

    Returns ``{"jobs": [...], "stages": [...], "links": [...],
              "columns": [...]}`` — same shape as parse_legacy_sql so
    downstream code in streamlit_app.py can treat it identically.
    """
    if not content or not content.strip():
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    head = content.lstrip()[:2000].upper()
    sql_signals = ("CREATE TABLE", "CREATE OR REPLACE TABLE",
                   "CREATE VIEW", "CREATE OR REPLACE VIEW",
                   "INSERT INTO", "ALTER TABLE", "MERGE INTO")
    looks_like_sql = any(sig in head for sig in sql_signals)

    if looks_like_sql:
        return _sql_extract_objects(content, dialect="generic")

    # Otherwise treat as a metadata CSV.
    try:
        df = pd.read_csv(StringIO(content))
    except Exception:
        # Not parseable as CSV either — fall back to SQL extractor as a
        # last resort (it'll just return empty arrays for non-SQL).
        return _sql_extract_objects(content, dialect="generic")

    if df.empty:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    # Normalize headers: lowercase, replace spaces/dashes with underscore
    df.columns = [
        re.sub(r"[\s\-]+", "_", str(c).strip().lower())
        for c in df.columns
    ]

    def _first_match(cands):
        for c in cands:
            if c in df.columns:
                return c
        return None

    table_col = _first_match([
        "table", "table_name", "source_table", "src_table",
        "object", "source_object", "src_object",
        "entity", "source_entity", "src_entity",
    ])
    column_col = _first_match([
        "column", "column_name", "field",
        "source_column", "src_column", "source_field",
    ])
    type_col = _first_match([
        "type", "data_type", "sql_type",
        "source_data_type", "source_type", "src_type", "src_data_type",
    ])
    schema_col = _first_match([
        "schema", "schema_name", "src_schema", "source_schema",
        "database", "db",
    ])

    # If we can't identify a table column, give up gracefully.
    if not table_col:
        return {"jobs": [], "stages": [], "links": [], "columns": []}

    job_name = "SOURCE_METADATA"
    seen_tables = {}  # table_name → True
    stages = []
    columns = []

    for _, row in df.iterrows():
        tbl_raw = row.get(table_col)
        if tbl_raw is None or (isinstance(tbl_raw, float)
                               and pd.isna(tbl_raw)):
            continue
        tbl = str(tbl_raw).strip()
        if not tbl:
            continue
        # Strip schema prefix if it's mashed into the table name
        # (e.g. "SCHEMA.TABLE") for the bare entity name. Keep the
        # schema separately if a dedicated column exists.
        bare = tbl.rsplit(".", 1)[-1]
        if bare not in seen_tables:
            seen_tables[bare] = True
            stages.append({
                "job_name":   job_name,
                "stage_name": bare,
                "stage_type": "table",
            })

        if column_col:
            col_raw = row.get(column_col)
            if col_raw is not None and not (
                isinstance(col_raw, float) and pd.isna(col_raw)
            ):
                col = str(col_raw).strip()
                if col:
                    sql_type = ""
                    if type_col:
                        t_raw = row.get(type_col)
                        if t_raw is not None and not (
                            isinstance(t_raw, float) and pd.isna(t_raw)
                        ):
                            sql_type = str(t_raw).strip()
                    columns.append({
                        "job_name":    job_name,
                        "stage_name":  bare,
                        "column_name": col,
                        "sql_type":    sql_type,
                        "precision":   "",
                        "scale":       "",
                    })

    jobs = [{"job_name": job_name, "job_type": "metadata_csv"}] \
        if stages else []
    return {"jobs": jobs, "stages": stages, "links": [], "columns": columns}
