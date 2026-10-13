"""Single source of truth for the conversation shape.

Every case, and every window of a case, builds its messages here. The system prompt, the output rules and
the schema block are rendered once and shared; exactly ONE paragraph and the
shape of the user turn vary by case. If the three prompts drift apart, "markdown
helps" stops being a falsifiable claim, so the shared span is hashed and checked
by report.prompt_integrity().

Messages are built in an ABSTRACT form - image blocks carry a path, not base64 -
so the fingerprint, the token estimate and the redacted request.json can all be
computed without holding megabytes of base64 in memory. materialize() converts to
the OpenAI wire format at the last moment.

build_user_content's pdf_md branch is ported from
D:/POC2/qwen3vl-poc/src/prompting.py, as is conversation_fingerprint.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from common import cfg_path, read_json, read_text, resolve
from render import Page, data_url, encode_for_wire, image_token_estimate

CASES = ("pdf_only", "pdf_md", "md_only")

# Which assets each case consumes. run_cell asserts the image count matches, so a
# pdf_only run that silently sent zero images is caught rather than reported as a
# model failure.
CASE_NEEDS = {
    "pdf_only": {"png": True, "md": False},
    "pdf_md":   {"png": True, "md": True},
    "md_only":  {"png": False, "md": True},
}

# Chars per token, for the text side of the budget estimate. Deliberately
# conservative: over-estimating the prompt costs a preflight warning, while
# under-estimating costs a 400 from vLLM after the images were already uploaded.
CHARS_PER_TOKEN = 3.5


# --------------------------------------------------------------------------- schema


def load_schema(cfg: dict, doc=None) -> dict:
    """The canonical schema for one document.

    Every document carries its own schema path, resolved at discovery time (per
    PDF, else the doc type's single schema, else paths.schema). Nothing about any
    document is in code - the prompt is generated from whichever schema applies.
    """
    p = Path(doc.schema) if (doc is not None and getattr(doc, "schema", None)) else None
    if p is None:
        cfgp = (cfg.get("paths") or {}).get("schema")
        p = cfg_path(cfg, "schema") if cfgp else None
    if p is None or not Path(p).exists():
        where = f" (LOB/doc type {doc.doc_type!r})" if doc is not None and doc.doc_type else ""
        raise FileNotFoundError(
            f"no canonical schema for {getattr(doc, 'doc_id', '?')!r}{where}. Its folder "
            f"needs one schema (or one named after the PDF), and in LOB mode the common "
            f"+ LOB schemas must resolve - `python src/preflight.py --local` says why not."
        )
    return read_json(p)


def doctype_block(cfg: dict, doc_type: str) -> str:
    """The document-type paragraph, plus the LOB's own paragraph if it has one.

    Empty string when the type has no prompt file.
    """
    if not doc_type:
        return ""
    try:
        spec = cfg_path_doctype(cfg, doc_type)
        lob = cfg_path_doctype(cfg, doc_type, "lob_prompt")
    except KeyError:
        return ""
    parts = [read_text(p).strip() for p in (spec, lob) if p and p.exists()]
    return "\n\n".join(p for p in parts if p)


def cfg_path_doctype(cfg: dict, doc_type: str, key: str = "prompt"):
    from common import doc_type_spec
    rel = doc_type_spec(cfg, doc_type).get(key)
    return resolve(rel) if rel else None


def doctype_name(cfg: dict, doc_type: str) -> str:
    """Human name for the user turn ('policy' -> 'Policy Declarations')."""
    from common import doc_type_spec
    if not doc_type:
        return "Insurance Document"
    try:
        return str(doc_type_spec(cfg, doc_type).get("name") or doc_type)
    except KeyError:
        return str(doc_type)


def deref(node: Any, root: dict, _seen: tuple = ()) -> dict:
    """Follow local $ref chains into $defs.

    Real canonical schemas factor shared shapes (Money, Address, Source) into
    $defs and reference them. Without this, every section built from a $ref
    renders as an untyped blank in the outline - which is exactly the part of the
    prompt the model needs most.
    """
    guard = 0
    while isinstance(node, dict) and "$ref" in node and guard < 20:
        guard += 1
        ref = str(node["$ref"])
        if not ref.startswith("#/"):
            break
        target: Any = root
        for part in ref[2:].split("/"):
            if not isinstance(target, dict):
                return {}
            target = target.get(part.replace("~1", "/").replace("~0", "~"), {})
        sibling = {k: v for k, v in node.items() if k != "$ref"}
        node = {**(target if isinstance(target, dict) else {}), **sibling}
    return node if isinstance(node, dict) else {}


def _type_of(node: dict) -> str:
    t = node.get("type")
    if isinstance(t, list):
        t = "/".join(str(x) for x in t if x != "null") or "any"
    if not t:
        if "properties" in node:
            t = "object"
        elif "items" in node:
            t = "array"
        elif "enum" in node:
            t = "enum"
        else:
            t = "any"
    return str(t)


def _prop_order(node: dict) -> list[str]:
    """Property names in the schema's own order, required ones first."""
    props = node.get("properties") or {}
    required = [k for k in (node.get("required") or []) if k in props]
    rest = [k for k in props if k not in set(required)]
    return required + rest


def _describe(name: str, node: dict) -> str:
    bits = [_type_of(node)]
    if node.get("enum"):
        vals = ", ".join(json.dumps(v, ensure_ascii=False) for v in node["enum"][:8])
        bits.append(f"one of: {vals}")
    desc = node.get("description")
    if desc:
        bits.append(str(desc).strip().rstrip("."))
    return f'"{name}" - ' + "; ".join(bits)


def schema_outline(schema: dict, max_depth: int = 3) -> str:
    """A readable numbered outline of the canonical structure.

    POC2's prompt hand-wrote this list for one specific schema. Here it is
    derived through $ref, so swapping in a different canonical structure needs no
    code change.
    """
    root = schema
    lines: list[str] = []
    top = _prop_order(schema)
    if not top:
        return ""

    def walk(node: dict, depth: int, indent: str) -> None:
        if depth > max_depth:
            return
        node = deref(node, root)
        target = node
        if _type_of(node) == "array":
            items = deref(node.get("items") or {}, root)
            if items.get("properties"):
                target = items
        names = _prop_order(target)
        if not names:
            return
        for name in names:
            child = deref((target.get("properties") or {}).get(name) or {}, root)
            lines.append(f"{indent}- {_describe(name, child)}")
            walk(child, depth + 1, indent + "  ")

    for i, name in enumerate(top, start=1):
        node = deref((schema.get("properties") or {}).get(name) or {}, root)
        lines.append(f"{i:2d}. {_describe(name, node)}")
        walk(node, 2, "    ")

    header = ("The canonical structure, in this order. Indented entries are the "
              "fields inside each section; for an array, the fields of each item.")
    return header + "\n" + "\n".join(lines)


def _walk_nodes(schema: dict):
    """Every object node in the schema, including $defs."""
    stack = [schema]
    seen = set()
    while stack:
        node = stack.pop()
        if not isinstance(node, dict) or id(node) in seen:
            continue
        seen.add(id(node))
        yield node
        for key in ("properties", "$defs", "patternProperties"):
            for v in (node.get(key) or {}).values():
                stack.append(v)
        for key in ("items", "additionalProperties", "contains"):
            v = node.get(key)
            if isinstance(v, dict):
                stack.append(v)
        for key in ("oneOf", "anyOf", "allOf"):
            for v in (node.get(key) or []):
                stack.append(v)


def schema_conventions(schema: dict) -> str:
    """Value-format rules READ OFF the schema, not assumed.

    POC2's prompt hard-coded the conventions of one commercial-auto schema
    ("keep $47,149.00 exactly as printed", "keep dates in the document's own
    form"). Applied to a schema whose money is {amount: number} and whose dates
    are format: date, those instructions would actively drive the model away from
    the target. So the rules are derived here and any schema gets the right ones.
    """
    defs = schema.get("$defs") or {}
    rules: list[str] = []

    if schema.get("additionalProperties") is False:
        rules.append(
            "- The schema is CLOSED: every key you emit must be one the schema names. "
            "Do not invent keys and do not add commentary fields. If something on the "
            "page has no home in the schema, put it in the nearest field the schema "
            "does provide for free text rather than inventing a key for it.")
    else:
        rules.append(
            "- The structure is a MINIMUM, not a limit. If the document contains "
            "something the structure does not anticipate, add it under the nearest "
            "matching section rather than dropping it.")

    # Money, in either of the two shapes a canonical schema uses: an object with a
    # numeric `amount`, or a bare numeric alias named for money. Detecting only
    # the first would leave a scalar-money schema with no amount rule at all.
    def _types(node):
        t = node.get("type")
        return {t} if isinstance(t, str) else set(t or [])

    money_obj, money_num = [], []
    for name, node in defs.items():
        if not isinstance(node, dict):
            continue
        if (node.get("properties") or {}).get("amount", {}).get("type") == "number":
            money_obj.append(name)
        elif ("number" in _types(node)
              and any(w in name.lower() for w in ("money", "amount", "currency", "premium"))):
            money_num.append(name)
    money = sorted(money_obj)
    if money_num and not money_obj:
        rules.append(
            f"- Amounts are plain JSON NUMBERS, not strings: no currency symbol, no "
            f"thousands separators, no quotes. An amount printed $87,654.32 is "
            f"written 87654.32, and a blank money box is null - never 0. "
            f"(schema: {', '.join(sorted(money_num))})")
    if money:
        # NOTE: illustrative values here MUST NOT be values from the document
        # being extracted. An example is shown to every cell, so a real answer
        # used as an example is an answer handed to the model for free -
        # preflight's leak check enforces this.
        rules.append(
            f"- Amounts are OBJECTS, not strings: {{\"amount\": <number>, "
            f"\"currency\": \"USD\"}}. The amount is a plain JSON number with no "
            f"currency symbol, no thousands separators and no quotes: an amount "
            f"printed $87,654.32 is written 87654.32. "
            f"(schema: {', '.join(sorted(money))})")

    # Any node declaring a verbatim passthrough alongside typed fields.
    raw_holders = sorted({
        name for name, node in defs.items()
        if isinstance(node, dict) and "raw_text" in (node.get("properties") or {})
    })
    if raw_holders:
        rules.append(
            f"- Where a shape has \"raw_text\", copy the cell's printed text into it "
            f"VERBATIM, exactly as it appears including symbols and separators, and "
            f"put your interpretation of it in the sibling fields. raw_text is the "
            f"audit trail; the typed fields are the reading. "
            f"(schema: {', '.join(raw_holders)})")

    # Dates: `format: date`, or a pattern pinning YYYY-MM-DD. Schemas in the wild
    # use either, and a schema that uses only the pattern still needs the rule.
    def _is_iso_date(node):
        if node.get("format") == "date":
            return True
        pat = node.get("pattern")
        return isinstance(pat, str) and "{4}" in pat and pat.count("-") >= 2

    if any(_is_iso_date(n) for n in _walk_nodes(schema)):
        rules.append(
            "- Dates use ISO YYYY-MM-DD, converted from whatever the page prints. "
            "A date printed 4/06/1988 is written \"1988-04-06\".")

    src = defs.get("Source") or {}
    if isinstance(src, dict) and "page" in (src.get("properties") or {}):
        has_bbox = "bbox" in (src.get("properties") or {})
        rules.append(
            "- \"source\" records where a value came from. Set source.page to the "
            "page number PRINTED ON THE PAGE (usually in the footer, as \"Page N of "
            "M\"). The \"--- File page N of M ---\" header tells you where a page "
            "sits in the file, which is NOT the same thing: pages can be missing "
            "from a supplied file, so file page 2 may be printed page 3. Always "
            "report the printed number, and use the printed numbering for any other "
            "field that names a page."
            + (" Omit bbox and confidence unless you can give them honestly - an "
               "invented coordinate is worse than an absent one." if has_bbox else ""))

    # Shared scalar aliases. When a schema factors a convention into a named
    # $def with a description ("checkbox: true = marked, false = present and
    # unmarked, null = unreadable"), that sentence IS the rule and is worth more
    # than anything inferred. Surface them verbatim.
    aliases: list[str] = []
    for name, node in sorted(defs.items()):
        if not isinstance(node, dict) or node.get("properties") or node.get("items"):
            continue
        desc = str(node.get("description") or "").strip()
        if desc:
            aliases.append(f"    {name} ({_type_of(node)}): {desc}")
    if aliases:
        rules.append(
            "- Shared value types the schema defines, and what each one means:\n"
            + "\n".join(aliases))

    # Enumerated vocabularies: the model must choose, not paraphrase.
    enum_fields: list[str] = []
    for node in _walk_nodes(schema):
        for pname, pnode in (node.get("properties") or {}).items():
            if isinstance(pnode, dict) and pnode.get("enum"):
                vals = ", ".join(json.dumps(v) for v in pnode["enum"][:14])
                enum_fields.append(f'    "{pname}": one of {vals}')
    if enum_fields:
        uniq = sorted(set(enum_fields))
        rules.append(
            "- Enumerated fields accept ONLY the listed values. Choose the closest "
            "one; never invent a new value and never paraphrase:\n" + "\n".join(uniq))

    return "Value conventions this schema requires:\n" + "\n".join(rules)


def _outline_block(cfg: dict, schema: dict) -> str:
    style = str((cfg.get("prompt") or {}).get("schema_style", "both")).lower()
    if style not in ("outline", "both"):
        return ""
    depth = int((cfg.get("prompt") or {}).get("outline_max_depth", 3))
    return schema_outline(schema, depth)


def _schema_json_block(cfg: dict, schema: dict) -> str:
    style = str((cfg.get("prompt") or {}).get("schema_style", "both")).lower()
    if style not in ("json", "both"):
        return ""
    return ("\nThe full JSON Schema the output must satisfy:\n"
            + json.dumps(schema, indent=1, ensure_ascii=False))


def build_system_prompt(cfg: dict, case: str, schema: dict, doc_type: str = "") -> str:
    """The skeleton with the doc-type block, the modality paragraph and the schema.

    Two axes vary here and they are checked separately:
      - doc_type varies BETWEEN document types and is constant across the six
        cells of one type;
      - modality varies BETWEEN cases and is constant across models.
    Everything else is invariant, which is what keeps both comparisons honest.
    """
    if case not in CASES:
        raise ValueError(f"unknown case {case!r}; expected one of {CASES}")
    skeleton = read_text(cfg_path(cfg, "system_prompt"))
    case_file = (cfg.get("paths", {}).get("case_prompts") or {}).get(case)
    if not case_file:
        raise KeyError(f"paths.case_prompts.{case} missing from config")
    modality = read_text(resolve(case_file)).strip()

    text = (skeleton
            .replace("{{DOCTYPE}}", doctype_block(cfg, doc_type))
            .replace("{{MODALITY}}", modality)
            .replace("{{SCHEMA_CONVENTIONS}}", schema_conventions(schema))
            .replace("{{SCHEMA_OUTLINE}}", _outline_block(cfg, schema))
            .replace("{{SCHEMA_JSON}}", _schema_json_block(cfg, schema)))
    return text.strip() + "\n"


def shared_prompt_span(cfg: dict, schema: dict, doc_type: str = "") -> str:
    """The system prompt with the modality paragraph removed.

    This must hash identically for every case and window OF ONE DOCUMENT. It is what
    makes the comparison a comparison rather than six unrelated experiments.
    Different doc types legitimately hash differently - that is the point of a
    doc-type-specific prompt - so report.prompt_integrity() groups by type.
    """
    skeleton = read_text(cfg_path(cfg, "system_prompt"))
    return (skeleton
            .replace("{{DOCTYPE}}", doctype_block(cfg, doc_type))
            .replace("{{MODALITY}}", "")
            .replace("{{SCHEMA_CONVENTIONS}}", schema_conventions(schema))
            .replace("{{SCHEMA_OUTLINE}}", _outline_block(cfg, schema))
            .replace("{{SCHEMA_JSON}}", _schema_json_block(cfg, schema))).strip()


# --------------------------------------------------------------------------- user turn


def _page_markdown(page: Page, max_chars: int, notice: str) -> str:
    text = page.markdown()
    if max_chars and len(text) > max_chars:
        text = text[:max_chars].rstrip() + "\n" + notice
    return text


def user_preamble(cfg: dict, case: str, doc_type: str) -> str:
    """The user turn's opening lines: what document this is, and what is attached."""
    rel = (cfg.get("paths", {}).get("user_prompts") or {}).get(case)
    if not rel:
        return ""
    return read_text(resolve(rel)).replace(
        "{{DOCTYPE_NAME}}", doctype_name(cfg, doc_type)).strip()


def _template(cfg: dict, key: str) -> str:
    rel = (cfg.get("paths") or {}).get(key)
    return read_text(resolve(rel)).strip() if rel else ""


def window_note(cfg: dict, first: int, last: int, total: int,
                reference: list[int] | None = None) -> str:
    """The user-turn note telling the model it sees only part of the document,
    and what the reference pages in front of its pages are."""
    note = (_template(cfg, "window_prompt")
            .replace("{{FIRST}}", str(first))
            .replace("{{LAST}}", str(last))
            .replace("{{TOTAL}}", str(total)))
    if reference:
        ref = _template(cfg, "window_reference_prompt").replace(
            "{{PAGES}}", ", ".join(str(p) for p in reference))
        note = f"{note}\n\n{ref}" if ref else note
    return note.strip()


def context_note(cfg: dict, digest: str, through: int) -> str:
    """The digest of what earlier windows extracted, framed for the model."""
    if not digest:
        return ""
    return (_template(cfg, "window_context_prompt")
            .replace("{{THROUGH}}", str(through))
            .replace("{{DIGEST}}", digest)).strip()


def _page_blocks(case: str, page: Page, header: str, max_md_chars: int,
                 truncate_notice: str) -> list[dict]:
    needs = CASE_NEEDS[case]
    blocks: list[dict] = [{"type": "text", "text": header}]
    if needs["png"]:
        blocks.append({"type": "image", "path": str(Path(page.png).as_posix()),
                       "name": Path(page.png).name})
    if needs["md"]:
        text = _page_markdown(page, max_md_chars, truncate_notice)
        if text:
            body = f"OCR text of page {page.index}:\n{text}"
        elif case == "pdf_md":
            body = (f"OCR text of file page {page.index}: "
                    f"(none - read this page from the image)")
        else:
            body = (f"OCR text of file page {page.index}: "
                    f"(none - this page produced no text)")
        blocks.append({"type": "text", "text": body})
    return blocks


def build_user_content(case: str, pages: list[Page], max_md_chars: int = 20000,
                       truncate_notice: str = "...[page markdown truncated]",
                       preamble: str = "", total_pages: int | None = None,
                       note: str = "", context: str = "",
                       reference_pages: list[Page] | None = None) -> list[dict]:
    """User turn: one chunk per page, in reading order.

    pdf_md interleaves per page - header, that page's image, that page's OCR -
    rather than all images followed by one slab of text. A page's picture and the
    text read off that same page sit next to each other, so the model never has
    to work out which markdown belongs to which image, which gets harder the more
    pages a document has. Ported from POC2's build_user_content.

    The header line is present in every case so the "account for every page"
    instruction is equally supported across all three. For a window, `pages` is
    the window's pages and `total_pages` the whole document's count, so the
    headers carry the page's real position in the file.

    Window order: preamble, window note, the digest of earlier windows, the
    reference pages (marked as such), then the pages to extract. Reference
    pages are given in the case's own modality, like every other page.
    """
    total = total_pages or len(pages)
    content: list[dict] = []
    for text in (preamble, note, context):
        if text:
            content.append({"type": "text", "text": text})
    for page in reference_pages or []:
        content += _page_blocks(
            case, page,
            f"--- Reference page: file page {page.index} of {total} "
            f"(context only - not to be extracted here) ---",
            max_md_chars, truncate_notice)
    for page in pages:
        content += _page_blocks(case, page, f"--- File page {page.index} of {total} ---",
                                max_md_chars, truncate_notice)
    return content


def build_messages(cfg: dict, case: str, pages: list[Page],
                   schema: dict, doc_type: str = "",
                   total_pages: int | None = None,
                   reference_pages: list[Page] | None = None,
                   digest: str = "", digest_through: int | None = None) -> list[dict]:
    """The abstract [system, user] conversation for one cell, or one window of it.

    `pages` is the pages this request carries. When they are not the whole
    document (total_pages is larger), the window note is added to the user turn,
    plus - when given - the reference pages and the digest of earlier windows.
    The system prompt is identical for every window and every document of a LOB,
    so vLLM's prefix cache can reuse it.
    """
    prompt_cfg = cfg.get("prompt") or {}
    total = total_pages or len(pages)
    note = context = ""
    refs = reference_pages or []
    if pages and len(pages) < total:
        note = window_note(cfg, pages[0].index, pages[-1].index, total,
                           [p.index for p in refs])
        context = context_note(cfg, digest, digest_through or pages[0].index - 1)
    else:
        refs = []
    return [
        {"role": "system", "content": build_system_prompt(cfg, case, schema, doc_type)},
        {"role": "user", "content": build_user_content(
            case, pages,
            int(prompt_cfg.get("max_md_chars_per_page") or 0),
            str(prompt_cfg.get("truncate_md_notice") or "...[truncated]"),
            user_preamble(cfg, case, doc_type),
            total, note, context, refs,
        )},
    ]


# --------------------------------------------------------------------------- fingerprint


def conversation_fingerprint(messages: list[dict]) -> str:
    """Hash of the prompt and modality.

    Image blocks contribute their file NAME only, never the full path, so moving
    the project between machines is not read as prompt drift. Ported from POC2.

    Stored per cell. The report asserts that all six differ (nothing was
    accidentally run twice) and that the shared span is identical across them.
    """
    h = hashlib.sha256()
    for msg in messages:
        if msg.get("role") == "assistant":
            continue
        h.update(str(msg.get("role", "")).encode("utf-8"))
        content = msg.get("content")
        if isinstance(content, str):
            h.update(content.encode("utf-8"))
            continue
        for block in content or []:
            kind = str(block.get("type", ""))
            h.update(kind.encode("utf-8"))
            if kind == "image":
                h.update(str(block.get("name") or Path(str(block.get("path", ""))).name)
                         .encode("utf-8"))
            else:
                h.update(str(block.get("text", "")).encode("utf-8"))
    return h.hexdigest()[:16]


# --------------------------------------------------------------------------- wire


class PayloadTooLarge(RuntimeError):
    pass


def materialize(messages: list[dict], cfg: dict) -> tuple[list[dict], dict]:
    """Abstract messages -> OpenAI chat format, with the payload accounted for.

    Refuses rather than warns above render.max_total_payload_mb. A request that
    is too big does not fail cleanly on the far side of a proxy; it fails slowly
    and ambiguously, after the upload.
    """
    rcfg = cfg.get("render") or {}
    fmt = str(rcfg.get("wire_format") or "jpeg")
    quality = int(rcfg.get("jpeg_quality") or 88)
    cap_mb = float(rcfg.get("max_total_payload_mb") or 0)

    out: list[dict] = []
    n_images = 0
    raw_bytes = 0
    b64_chars = 0
    text_chars = 0

    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_chars += len(content)
            out.append({"role": msg["role"], "content": content})
            continue
        blocks: list[dict] = []
        for block in content or []:
            if block.get("type") == "image":
                b64, nbytes = encode_for_wire(Path(block["path"]), fmt, quality)
                raw_bytes += nbytes
                b64_chars += len(b64)
                n_images += 1
                blocks.append({"type": "image_url",
                               "image_url": {"url": data_url(b64, fmt)}})
            else:
                text_chars += len(block.get("text", ""))
                blocks.append({"type": "text", "text": block.get("text", "")})
        out.append({"role": msg["role"], "content": blocks})

    payload_mb = (b64_chars + text_chars) / (1024 * 1024)
    stats = {
        "n_images": n_images,
        "image_raw_bytes": raw_bytes,
        "image_b64_chars": b64_chars,
        "text_chars": text_chars,
        "payload_mb": round(payload_mb, 3),
        "wire_format": fmt,
        "jpeg_quality": quality,
    }
    if cap_mb and payload_mb > cap_mb:
        raise PayloadTooLarge(
            f"request payload is {payload_mb:.1f} MB, over the "
            f"render.max_total_payload_mb cap of {cap_mb} MB. Lower "
            f"render.jpeg_quality or render.max_long_side_px - both change what "
            f"EVERY cell sees, so change them before the run, not during it."
        )
    return out, stats


def redact(messages: list[dict]) -> list[dict]:
    """The conversation with images reduced to a name, size and hash.

    Written to <doc>.request.json so any cell is reproducible and diffable
    against any other without storing megabytes of base64 six times over.
    """
    import hashlib as _h

    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            out.append({"role": msg["role"], "content": content})
            continue
        blocks = []
        for block in content or []:
            if block.get("type") == "image":
                p = Path(block["path"])
                raw = p.read_bytes() if p.exists() else b""
                blocks.append({
                    "image": p.name,
                    "bytes": len(raw),
                    "sha256": _h.sha256(raw).hexdigest()[:16],
                })
            else:
                blocks.append({"text": block.get("text", "")})
        out.append({"role": msg["role"], "content": blocks})
    return out


# --------------------------------------------------------------------------- budget


def estimate_tokens(messages: list[dict], pages: list[Page], case: str,
                    max_tokens: int, extra_text_chars: int = 0) -> dict:
    """Projected prompt size for one request, against the server's context window.

    Images dominate and are estimated from real pixel dimensions; text is
    estimated by characters. `pages` must include any reference pages, so their
    dimensions are known. extra_text_chars reserves room for text not known in
    advance - the digest of earlier windows, which depends on what they extract.
    """
    text_chars = int(extra_text_chars)
    image_tokens = 0
    by_index = {p.index: p for p in pages}
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, str):
            text_chars += len(content)
            continue
        for block in content or []:
            if block.get("type") == "image":
                name = block.get("name") or Path(str(block.get("path", ""))).name
                idx = int(Path(name).stem.split("_")[-1]) if "_" in name else 0
                page = by_index.get(idx)
                if page and page.width and page.height:
                    image_tokens += image_token_estimate(page.width, page.height)
                else:
                    image_tokens += 1100  # a 1024px page, if dimensions are unknown
            else:
                text_chars += len(block.get("text", ""))

    text_tokens = int(text_chars / CHARS_PER_TOKEN)
    prompt_tokens = text_tokens + image_tokens
    return {
        "case": case,
        "text_chars": text_chars,
        "text_tokens_est": text_tokens,
        "image_tokens_est": image_tokens,
        "prompt_tokens_est": prompt_tokens,
        "max_tokens": max_tokens,
        "total_tokens_est": prompt_tokens + max_tokens,
    }
