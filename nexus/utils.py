"""
nexus.utils — generic, framework-free helpers used everywhere.

This module has NO Streamlit, NO Snowflake, and NO LLM dependencies.
Anything in here can be imported by any other nexus module without
fear of circular imports.

Contents:
    extract_json                 — robust JSON extraction from LLM output
    _unwrap_json_string          — strip code fences from JSON-as-string
    extract_mermaid_script       — pull mermaid blocks out of markdown
    extract_sql_ddl              — pull SQL out of LLM responses
    parse_table_response         — flexible table parser (CSV/markdown/json)
    parse_markdown_table         — markdown table → DataFrame
    df_to_excel_bytes            — DataFrame → .xlsx bytes
    _slug                        — filename-safe slug
    resolve_sql_type             — combine type + precision + scale into a SQL type string
    _SQL_TYPE_CODES              — mapping of integer type codes (DataStage, etc.)
    _split_ddl_by_create_table   — split a DDL string into per-table chunks
    _split_narrative_by_section  — split markdown by ## headings
    extract_text_from_upload     — read text from an uploaded file
    clean_file_body              — strip code fences and Markdown noise
"""

import io
import json
import re
import zipfile
from io import BytesIO, StringIO
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def extract_json(raw: str):
    """
    Robustly extract JSON (object or array) from an LLM response.
    Handles: ```json fences, prose wrappers, trailing text, Markdown
    artifacts like ```sql inside values, minor truncation, single quotes.
    Returns parsed Python object, or None if parsing fails.
    """
    if not raw:
        return None
    s = raw.strip()

    # Strip any leading "Here is the JSON:" type prefixes
    s = re.sub(r'^[^{\[]*', '', s, count=1) if ('{' in s or '[' in s) else s

    # Strip ```json ... ``` or ``` ... ``` fences if they wrap the whole thing
    fence = re.search(r'^```(?:json)?\s*(.*?)```\s*$', s.strip(), re.DOTALL)
    if fence:
        s = fence.group(1).strip()

    def _try_parse(txt: str):
        try:
            return json.loads(txt)
        except json.JSONDecodeError:
            return None

    # Attempt 1: direct parse
    obj = _try_parse(s)
    if obj is not None:
        return obj

    # Attempt 2: scan for outermost JSON object or array via bracket depth
    for open_ch, close_ch in [('[', ']'), ('{', '}')]:
        start = s.find(open_ch)
        while start != -1:
            depth = 0
            in_str = False
            escape = False
            end = -1
            for i in range(start, len(s)):
                ch = s[i]
                if escape:
                    escape = False
                    continue
                if ch == '\\':
                    escape = True
                    continue
                if ch == '"' and not escape:
                    in_str = not in_str
                    continue
                if in_str:
                    continue
                if ch == open_ch:
                    depth += 1
                elif ch == close_ch:
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end > start:
                obj = _try_parse(s[start:end + 1])
                if obj is not None:
                    return obj
            start = s.find(open_ch, start + 1)

    # Attempt 3: fix common issues and retry
    # 3a: trim to last closing bracket
    for close_ch in (']', '}'):
        last = s.rfind(close_ch)
        first_open = s.find('[' if close_ch == ']' else '{')
        if last > 0 and first_open >= 0 and last > first_open:
            obj = _try_parse(s[first_open:last + 1])
            if obj is not None:
                return obj

    # 3b: salvage a JSON array by splitting on "},{" boundaries
    if '[' in s and '{' in s:
        rows = re.findall(r'\{[^{}]*\}', s, re.DOTALL)
        salvaged = []
        for r in rows:
            obj = _try_parse(r)
            if isinstance(obj, dict):
                salvaged.append(obj)
        if salvaged:
            return salvaged

    return None

def _unwrap_json_string(raw: str) -> str:
    """
    Snowflake Cortex sometimes returns content as a JSON-encoded string
    (literal \\n for newlines, outer double quotes, escaped inner quotes).
    Detect and unwrap that so downstream parsers see clean text.
    """
    if not raw:
        return ""
    s = raw.strip()
    has_escaped_newlines = "\\n" in s and "\n" not in s
    # Single-line JSON-encoded payloads: outer double-quotes wrap the
    # whole blob and the body contains escaped double-quotes (\").
    # Common Cortex shape when the inner content has no newlines.
    looks_like_quoted_json = (
        s.startswith('"') and s.endswith('"') and '\\"' in s
        and len(s) > 4
    )
    starts_with_json_string = s.startswith('"\\"') or s.startswith('"\\\\"') \
        or (s.startswith('"') and s.endswith('"') and "\\n" in s)
    if (has_escaped_newlines or starts_with_json_string
            or looks_like_quoted_json):
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                return decoded.strip()
        except (json.JSONDecodeError, ValueError):
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            s = (s.replace('\\"', '"')
                  .replace('\\n', '\n')
                  .replace('\\r', '\r')
                  .replace('\\t', '\t')
                  .replace('\\\\', '\\'))
            return s.strip()
    return s

def extract_mermaid_script(raw: str) -> str:
    """Pull a Mermaid script out of arbitrary LLM output."""
    if not raw:
        return ""
    s = _unwrap_json_string(raw)

    # 1. Look for a ```mermaid fenced block
    m = re.search(r'```mermaid\s*(.*?)```', s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Look for any fenced block and check if it contains 'erDiagram' / 'graph'
    m = re.search(r'```[a-zA-Z]*\s*(.*?)```', s, re.DOTALL)
    if m and re.search(r'\b(erDiagram|flowchart|graph|sequenceDiagram)\b',
                       m.group(1)):
        return m.group(1).strip()

    # 3. Find first occurrence of erDiagram / flowchart / graph keyword
    kw = re.search(
        r'\b(erDiagram|flowchart\s+[A-Z]{2}|graph\s+[A-Z]{2})\b', s
    )
    if kw:
        sub = s[kw.start():]
        # Strip any trailing fences or prose
        sub = re.sub(r'```.*$', '', sub, flags=re.DOTALL)
        return sub.strip()

    return s  # last resort — return as-is and let mermaid fail loudly

def extract_sql_ddl(raw: str) -> str:
    """Pull SQL DDL out of arbitrary LLM output."""
    if not raw:
        return ""
    s = _unwrap_json_string(raw)

    # 1. Look for ```sql fenced block
    m = re.search(r'```sql\s*(.*?)```', s, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 2. Any fenced block containing CREATE TABLE
    m = re.search(r'```[a-zA-Z]*\s*(.*?)```', s, re.DOTALL)
    if m and re.search(r'CREATE\s+(?:OR\s+REPLACE\s+)?TABLE',
                       m.group(1), re.IGNORECASE):
        return m.group(1).strip()

    # 3. Start from the first CREATE TABLE (or section header)
    kw = re.search(r'(--\s*═+\s*HUBS|CREATE\s+(?:OR\s+REPLACE\s+)?TABLE)',
                   s, re.IGNORECASE)
    if kw:
        sub = s[kw.start():]
        sub = re.sub(r'```.*$', '', sub, flags=re.DOTALL)
        return sub.strip()

    return s

def parse_table_response(raw: str) -> Optional[pd.DataFrame]:
    """
    Parse a tabular LLM response. Tries multiple strategies in order:
      1. JSON-string unwrapping (Cortex sometimes double-encodes)
      2. CSV (standard format, handles quoted cells with commas)
      3. Markdown pipe table (fallback)
      4. Semicolon-separated values
    Returns None if no strategy succeeds.
    """
    if not raw:
        return None

    s = raw.strip()

    # Strip any wrapping fences
    s = re.sub(r'^```[a-zA-Z]*\n?', '', s)
    s = re.sub(r'```\s*$', '', s)
    s = s.strip()

    # ── Strategy 0: JSON-string unwrap ────────────────────────────────
    # Some Cortex responses come back as a JSON-encoded string:
    #   "\"Entity\",\"Column\"\n\"CUSTOMERS\",\"ID\"..."
    # Signs: starts with `"\\"` or contains `\n` literals but no real newlines.
    has_escaped_newlines = "\\n" in s and "\n" not in s
    starts_with_json_string = s.startswith('"\\"') or s.startswith('"\\\\"')
    if has_escaped_newlines or starts_with_json_string:
        try:
            decoded = json.loads(s)
            if isinstance(decoded, str):
                s = decoded.strip()
        except (json.JSONDecodeError, ValueError):
            # Manual unescape as fallback
            if s.startswith('"') and s.endswith('"'):
                s = s[1:-1]
            s = (s.replace('\\"', '"')
                  .replace('\\n', '\n')
                  .replace('\\r', '\r')
                  .replace('\\t', '\t')
                  .replace('\\\\', '\\'))
            s = s.strip()

    # ── Strategy 1: CSV via pandas ────────────────────────────────────
    from io import StringIO

    # Find the first line that looks like CSV (starts with a quote or
    # contains commas) and trim everything before it
    lines = s.splitlines()
    csv_start = None
    for i, ln in enumerate(lines):
        stripped = ln.strip()
        if stripped.startswith('"') or (',' in stripped and
                                        not stripped.startswith("|")):
            csv_start = i
            break
    if csv_start is not None:
        csv_text = "\n".join(lines[csv_start:])
        # Also trim trailing prose
        csv_lines = []
        for ln in csv_text.splitlines():
            stripped = ln.strip()
            if not stripped:
                if csv_lines:  # blank line after csv ends it
                    break
                continue
            # Accept lines that look CSV-ish (comma present or starts with ")
            if ',' in stripped or stripped.startswith('"'):
                csv_lines.append(ln)
            else:
                if csv_lines:
                    break
        csv_text = "\n".join(csv_lines)

        for sep in (",", ";"):
            try:
                df = pd.read_csv(
                    StringIO(csv_text),
                    sep=sep,
                    quotechar='"',
                    skipinitialspace=True,
                    dtype=str,
                    keep_default_na=False,
                    on_bad_lines="skip",
                )
                if not df.empty and len(df.columns) >= 2:
                    # Strip whitespace from headers/values
                    df.columns = [str(c).strip().strip('"')
                                  for c in df.columns]
                    for c in df.columns:
                        df[c] = df[c].astype(str).str.strip().str.strip('"')
                    return df
            except Exception:
                continue

    # ── Strategy 2: markdown pipe table ───────────────────────────────
    mt = parse_markdown_table(s)
    if mt is not None and not mt.empty:
        return mt

    # ── Strategy 3: last-ditch CSV — treat any comma-rich block as CSV
    # This rescues cases where the LLM wrapped the CSV inside prose but
    # didn't fence it, OR the CSV has leading blank lines we skipped.
    try:
        all_lines = s.splitlines()
        csv_lines = [ln for ln in all_lines
                     if ln.count(",") >= 2 and not ln.strip().startswith("#")]
        if len(csv_lines) >= 2:  # at least header + one row
            df = pd.read_csv(
                StringIO("\n".join(csv_lines)),
                sep=",",
                quotechar='"',
                skipinitialspace=True,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="skip",
                engine="python",
            )
            if not df.empty and len(df.columns) >= 2:
                df.columns = [str(c).strip().strip('"')
                              for c in df.columns]
                for c in df.columns:
                    df[c] = df[c].astype(str).str.strip().str.strip('"')
                return df
    except Exception:
        pass

    return None

def parse_markdown_table(md: str) -> Optional[pd.DataFrame]:
    """
    Parse a Markdown pipe table into a DataFrame.
    Strategy: find the densest contiguous block of "pipe rows" with a
    consistent column count — that's the table. Ignores everything else.

    Returns None if no viable table found.
    """
    if not md:
        return None

    s = md.strip()

    # Remove ```lang ... ``` fences anywhere in the text (not just edges)
    s = re.sub(r'```[a-zA-Z]*\n?', '', s)
    s = s.replace('```', '')

    lines = s.splitlines()

    def _is_sep(row: str) -> bool:
        row = row.strip().strip("|")
        cells = [c.strip() for c in row.split("|")]
        non_empty = [c for c in cells if c]
        return bool(non_empty) and all(
            re.fullmatch(r':?-{2,}:?', c) for c in non_empty
        )

    def _split_cells(row: str):
        row = row.strip()
        if row.startswith("|"):
            row = row[1:]
        if row.endswith("|"):
            row = row[:-1]
        parts = re.split(r'(?<!\\)\|', row)
        return [p.strip().replace("\\|", "|") for p in parts]

    # Step 1: Find all lines that look like table rows
    # Definition: has ≥ 2 unescaped pipes, is not a heading, not a fence
    candidates = []  # list of (line_idx, cell_count, stripped_line, is_sep)
    for idx, ln in enumerate(lines):
        stripped = ln.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        # Count unescaped pipes
        pipes = len(re.findall(r'(?<!\\)\|', stripped))
        if pipes < 2:
            continue
        # Normalize: ensure it starts and ends with |
        norm = stripped
        if not norm.startswith("|"):
            norm = "|" + norm
        if not norm.endswith("|"):
            norm = norm + "|"
        cell_count = len(_split_cells(norm))
        candidates.append((idx, cell_count, norm, _is_sep(norm)))

    if len(candidates) < 2:
        return None

    # Step 2: Find the longest contiguous run of candidates with the same
    # cell count. This is almost always the actual table.
    best_start, best_end, best_count = 0, 0, 0
    i = 0
    while i < len(candidates):
        j = i
        # Gather a run of candidates that share a column count (ignoring
        # separator rows, which sometimes have a different count)
        counts_in_run = {}
        while j < len(candidates):
            if j > i:
                # Must be contiguous in original file (allow gap of 0 lines)
                prev_idx = candidates[j - 1][0]
                this_idx = candidates[j][0]
                if this_idx - prev_idx > 1:
                    break
            cc = candidates[j][1]
            counts_in_run[cc] = counts_in_run.get(cc, 0) + 1
            j += 1
        # Pick the dominant cell count in this run
        if counts_in_run:
            dominant = max(counts_in_run, key=counts_in_run.get)
            run_size = counts_in_run[dominant]
            if run_size > best_count:
                best_count = run_size
                best_start, best_end = i, j
        i = j if j > i else i + 1

    if best_count < 2:
        return None

    # Step 3: Pull the dominant-count rows from the best run
    run = candidates[best_start:best_end]
    cell_counts = [c[1] for c in run if not c[3]]  # non-separators
    if not cell_counts:
        return None
    from collections import Counter
    dominant_cc = Counter(cell_counts).most_common(1)[0][0]

    table_rows = [c[2] for c in run if c[1] == dominant_cc or c[3]]
    if len(table_rows) < 2:
        return None

    # Step 4: Identify header (first non-separator row) and body
    header_line = None
    body_lines = []
    for row in table_rows:
        if _is_sep(row):
            continue
        if header_line is None:
            header_line = row
        else:
            body_lines.append(row)

    if header_line is None:
        return None

    headers = _split_cells(header_line)
    if not headers:
        return None
    headers = [h if h else f"col_{i}" for i, h in enumerate(headers)]
    # De-duplicate
    seen = {}
    for i, h in enumerate(headers):
        if h in seen:
            seen[h] += 1
            headers[i] = f"{h}_{seen[h]}"
        else:
            seen[h] = 0

    rows = []
    for bl in body_lines:
        cells = _split_cells(bl)
        if len(cells) < len(headers):
            cells += [""] * (len(headers) - len(cells))
        elif len(cells) > len(headers):
            cells = cells[:len(headers)]
        rows.append(cells)

    if not rows:
        return None
    return pd.DataFrame(rows, columns=headers)

def df_to_excel_bytes(df: pd.DataFrame, sheet_name: str = "Sheet1") -> bytes:
    """Convert DataFrame to Excel bytes (for st.download_button)."""
    from io import BytesIO
    buf = BytesIO()
    try:
        with pd.ExcelWriter(buf, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    except (ImportError, ValueError):
        # Fallback if openpyxl isn't in the environment
        with pd.ExcelWriter(buf, engine="xlsxwriter") as writer:
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
    return buf.getvalue()

def _slug(s: str) -> str:
    """Make a filesystem-safe slug (used for zip member names)."""
    out = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s).strip()).strip("_")
    return out or "unnamed"

def _split_ddl_by_create_table(sql_ddl: str) -> list:
    """Split a DDL string into one chunk per CREATE [OR REPLACE] TABLE."""
    if not sql_ddl or not sql_ddl.strip():
        return []
    # Match each CREATE TABLE statement — body runs until the next CREATE
    # or end-of-string. Lenient on whitespace and OR REPLACE.
    pattern = re.compile(
        r'(CREATE\s+(?:OR\s+REPLACE\s+)?TABLE\s+[\s\S]*?;)',
        re.IGNORECASE
    )
    matches = pattern.findall(sql_ddl)
    if matches:
        return [m.strip() for m in matches if m.strip()]
    # Fallback: didn't find semicolon-terminated statements, return whole
    return [sql_ddl.strip()]

def _split_narrative_by_section(md: str) -> list:
    """Split markdown narrative into (section_title, section_body) pairs."""
    if not md or not md.strip():
        return []
    parts = re.split(r'(?m)^(#{1,3}\s+.+?)$', md)
    # re.split returns [preamble, heading1, body1, heading2, body2, ...]
    out = []
    if parts[0].strip():
        out.append(("(preamble)", parts[0].strip()))
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lstrip("#").strip()
        body = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if body:
            out.append((heading, f"{parts[i].strip()}\n\n{body}"))
    return out

def extract_text_from_upload(uf) -> str:
    """
    Pull usable text out of an uploaded file. Handles:
      - PDF via pypdf (text pages only; images are skipped)
      - Excel via pandas.read_excel (all sheets concatenated)
      - Images: filename+size hint (Cortex does not see pixels here)
      - text/markdown/csv/json: decode directly
    """
    if uf is None:
        return ""
    name = uf.name.lower()
    try:
        data = uf.read()
    except Exception:
        try:
            uf.seek(0); data = uf.read()
        except Exception:
            return ""

    if name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
            import io as _io
            reader = PdfReader(_io.BytesIO(data))
            pages = []
            for i, p in enumerate(reader.pages):
                try:
                    t = p.extract_text() or ""
                except Exception:
                    t = ""
                if t.strip():
                    pages.append(f"--- Page {i+1} ---\n{t.strip()}")
            return ("\n\n".join(pages)
                    or f"[PDF {uf.name}: no extractable text]")
        except ImportError:
            return (f"[PDF {uf.name}: pypdf not installed; paste "
                    f"the dashboard description as text]")
        except Exception as e:
            return f"[PDF {uf.name}: extraction failed - {e}]"

    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            import io as _io
            xls = pd.ExcelFile(_io.BytesIO(data))
            parts = []
            for sheet in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet, dtype=str,
                                   keep_default_na=False)
                parts.append(
                    f"--- Sheet: {sheet} ---\n"
                    f"{df.to_csv(index=False)}"
                )
            return "\n\n".join(parts)
        except Exception as e:
            return f"[Excel {uf.name}: extraction failed - {e}]"

    if name.endswith(".csv"):
        try:
            return data.decode("utf-8", errors="ignore")
        except Exception:
            return f"[CSV {uf.name}: could not decode]"

    if name.endswith((".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")):
        return (f"[Image attached: {uf.name} ({len(data)} bytes). "
                f"Dashboard visuals are described in the text input.]")

    try:
        return data.decode("utf-8", errors="ignore")
    except Exception:
        return f"[{uf.name}: could not decode]"

def clean_file_body(body: str) -> str:
    """
    Strip any LLM boilerplate that slipped past 'output raw content
    only' — leading fences, trailing fences, 'Here is...' prefaces.

    Runs two passes so combinations like `"Sure, here's:\\n\\n```yaml\\n...```"`
    get fully unwrapped — the preface strip exposes the fence, then
    the fence strip removes it.
    """
    if not body:
        return ""
    s = body.strip()
    for _ in range(2):  # two passes handles preface-then-fence
        # Strip leading "Here is..." / "Below is..." / "Sure,..." prefaces
        s = re.sub(
            r'^(?:here(?:\'s| is)|below is|sure[,.:]?|this is|okay[,.:]?'
            r'|certainly|of course)\b.{0,120}?\n+',
            '', s, count=1, flags=re.IGNORECASE,
        ).strip()
        # Strip leading markdown fence
        s = re.sub(r'^```[a-zA-Z0-9_+.-]*\s*\n', '', s).strip()
        # Strip trailing fence
        s = re.sub(r'\n?```\s*$', '', s).strip()
    return s + "\n"

_SQL_TYPE_CODES = {
    "1":   "CHAR",
    "12":  "VARCHAR",
    "-1":  "LONGVARCHAR",
    "-8":  "NCHAR",
    "-9":  "NVARCHAR",
    "-10": "LONGNVARCHAR",
    "-2":  "BINARY",
    "-3":  "VARBINARY",
    "-4":  "LONGVARBINARY",
    "2":   "NUMERIC",
    "3":   "DECIMAL",
    "4":   "INTEGER",
    "5":   "SMALLINT",
    "-5":  "BIGINT",
    "-6":  "TINYINT",
    "-7":  "BIT",
    "6":   "FLOAT",
    "7":   "REAL",
    "8":   "DOUBLE",
    "9":   "DATE",
    "10":  "TIME",
    "11":  "TIMESTAMP",
    "91":  "DATE",
    "92":  "TIME",
    "93":  "TIMESTAMP",
    "16":  "BOOLEAN",
    "2004": "BLOB",
    "2005": "CLOB",
}

def resolve_sql_type(sql_type: str, precision: str = "",
                     scale: str = "") -> str:
    """
    Turn DataStage's numeric SqlType code plus precision/scale into a
    human-readable SQL type like VARCHAR(255) or NUMBER(18,2). If the
    code is unknown or absent, return whatever string we were given
    (already a type name in some CSV inputs) or empty.
    """
    if not sql_type:
        return ""
    code = str(sql_type).strip()
    base = _SQL_TYPE_CODES.get(code)
    if base is None:
        # Already a type name? (csv metadata often is)
        if not code.isdigit() and not code.startswith("-"):
            return code.upper()
        return f"TYPE_{code}"

    prec = str(precision or "").strip()
    sc   = str(scale or "").strip()

    if base in ("CHAR", "VARCHAR", "NCHAR", "NVARCHAR",
                "LONGVARCHAR", "LONGNVARCHAR",
                "BINARY", "VARBINARY", "LONGVARBINARY"):
        return f"{base}({prec})" if prec else base
    if base in ("DECIMAL", "NUMERIC"):
        if prec and sc and sc != "0":
            return f"{base}({prec},{sc})"
        if prec:
            return f"{base}({prec})"
        return base
    return base

