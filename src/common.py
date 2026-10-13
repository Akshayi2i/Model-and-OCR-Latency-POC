"""Shared plumbing: config, paths, JSON IO, JSON recovery, document discovery.

Ported from D:/POC2/qwen3vl-poc/src/common.py. Every path in config.yaml is
relative to the project root (the directory holding config.yaml), so scripts run
from anywhere.

The JSON recovery block is the load-bearing part. A model that hits max_tokens
emits a valid *prefix*, which json.loads rejects outright - so a 99%-complete
answer would score zero fields. repair_truncated_json rewinds to the last point
where every open container could be closed cleanly and parses that. The result is
a strict subset of what the model produced: no value is invented.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "config.yaml"

BACKSLASH = chr(92)


def pymupdf():
    """PyMuPDF, imported under whichever name this version provides.

    The package renamed its module from `fitz` to `pymupdf`; importing the old
    name still works but prints a deprecation warning on every run.
    """
    try:
        import pymupdf as _pymupdf
        return _pymupdf
    except ImportError:
        import fitz as _pymupdf
        return _pymupdf


def setup_logging(name: str, level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, str(level).upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )
    return logging.getLogger(name)


# --------------------------------------------------------------------------- config


def load_config(path: str | Path | None = None) -> dict:
    cfg_file = Path(path) if path else DEFAULT_CONFIG
    if not cfg_file.is_absolute():
        cfg_file = ROOT / cfg_file
    with open(cfg_file, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["_config_file"] = str(cfg_file)
    _resolve_env(cfg)
    return cfg


_ENV_REF = re.compile(r"^\$\{([A-Z_][A-Z0-9_]*)(?::-(.*))?\}$")


def _resolve_env(node: Any) -> None:
    """Expand ``${VAR}`` and ``${VAR:-default}`` in string values, in place.

    Secrets do not belong in a file that gets committed. The pod scripts already
    read POC3_API_KEY from the environment; this makes the client read the same
    way, so config.yaml can be published without redacting anything by hand. A
    reference with no default and no value set is an error at load time rather
    than a 401 twenty minutes into a run.
    """
    items = node.items() if isinstance(node, dict) else enumerate(node)
    for key, val in list(items):
        if isinstance(val, (dict, list)):
            _resolve_env(val)
        elif isinstance(val, str):
            m = _ENV_REF.match(val.strip())
            if not m:
                continue
            name, default = m.group(1), m.group(2)
            got = os.environ.get(name, default)
            if got is None:
                raise RuntimeError(
                    f"config refers to ${{{name}}} but that variable is not set. "
                    f"Set it first:  export {name}=<value>"
                )
            node[key] = got


def resolve(rel: str | Path) -> Path:
    """Resolve a config-relative path against the project root."""
    p = Path(rel)
    return p if p.is_absolute() else (ROOT / p)


def cfg_path(cfg: dict, key: str) -> Path:
    """Look up paths.<key> from the config and resolve it."""
    try:
        return resolve(cfg["paths"][key])
    except KeyError as exc:
        raise KeyError(f"paths.{key} missing from {cfg.get('_config_file')}") from exc


def ensure_dir(p: Path) -> Path:
    Path(p).mkdir(parents=True, exist_ok=True)
    return Path(p)


def model_by_tag(cfg: dict, tag: str) -> dict:
    for m in cfg.get("models", []):
        if str(m.get("tag")) == str(tag):
            return m
    known = ", ".join(str(m.get("tag")) for m in cfg.get("models", []))
    raise KeyError(f"no model tagged {tag!r} in config; known tags: {known}")


def base_url_for(cfg: dict, port: int, override: str | None = None) -> str:
    """Endpoint for a port, or an explicit base_url that bypasses the pod entirely.

    An explicit URL lets the same runner drive any OpenAI-compatible endpoint - a
    hosted provider, a directly exposed TCP port, an SSH tunnel - without the
    RunPod proxy. The comparison only requires that BOTH models are reached the
    same way; where they are served is not the experiment.
    """
    if override:
        return str(override).rstrip("/")
    pod = cfg.get("pod", {}) or {}
    pod_id = str(pod.get("id") or "").strip()
    if not pod_id or pod_id.startswith("REPLACE"):
        raise ValueError(
            "pod.id is not set in config.yaml. Put your RunPod pod id there - it is "
            "the first path segment of the proxy hostname."
        )
    template = pod.get("proxy_template") or "https://{pod_id}-{port}.proxy.runpod.net"
    return template.format(pod_id=pod_id, port=int(port))


def case_dir_name(cfg: dict, case: str) -> str:
    """The on-disk folder name for a case id ('pdf_md' -> 'PDF+MD')."""
    return str((cfg.get("case_dir_names") or {}).get(case, case))


def lob_mode(cfg: dict) -> bool:
    """True when document types are lines of business discovered from folders."""
    return cfg.get("lobs") is not None


def manifest_path(cfg: dict) -> Path | None:
    """lobs.manifest, resolved - None in folder layout."""
    rel = (cfg.get("lobs") or {}).get("manifest")
    return resolve(rel) if rel else None


def read_manifest(cfg: dict) -> dict:
    """The manifest: {"documents": [{id, pdf, gold, schema, lob, ...}]}.

    Entry paths are relative to the manifest's own folder. Raises with a clear
    message when it is missing or malformed - a manifest layout without its
    manifest has no way to know which schema a PDF belongs to.
    """
    p = manifest_path(cfg)
    if p is None or not p.exists():
        raise FileNotFoundError(f"lobs.manifest points at {p}, which does not exist")
    try:
        man = read_json(p)
    except (ValueError, OSError) as exc:
        raise ValueError(f"{p.name} is not readable JSON: {exc}") from exc
    docs = man.get("documents") if isinstance(man, dict) else None
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"{p.name} has no 'documents' list")
    for i, d in enumerate(docs):
        missing = [k for k in ("pdf", "schema", "lob") if not d.get(k)]
        if missing:
            raise ValueError(f"{p.name} documents[{i}] lacks {missing}")
    return man


def lob_specs(cfg: dict) -> list[dict]:
    """One spec per line of business, in name order.

    Two layouts:
      folders  - each folder directly under input_dir is one LOB (folders
                 starting with "_" or "." are not);
      manifest - lobs.manifest lists every document with its LOB and schema,
                 for a flat input/<type>/{pdfs,gold,schema} layout.

    The spec has the same shape as a configured doc type, so everything keyed on
    doc_type - output folders, prompt integrity, the report's grouping - works on
    LOBs unchanged.
    """
    lobs = cfg.get("lobs") or {}
    input_dir = cfg_path(cfg, "input_dir")
    # Looked up many times per request (prompt blocks, output folders), so the
    # layout is read once per input_dir and the result kept on the config.
    cache = cfg.setdefault("_lob_specs_cache", {})
    if str(input_dir) in cache:
        return [dict(s) for s in cache[str(input_dir)]]
    names = {str(k): str(v) for k, v in (lobs.get("names") or {}).items()}
    kind = str(lobs.get("doc_type_name") or "Policy")
    prompt_dir = lobs.get("lob_prompt_dir")

    if manifest_path(cfg) is not None:
        try:
            ids = sorted({slugify(str(d["lob"])) for d in read_manifest(cfg)["documents"]})
        except (FileNotFoundError, ValueError):
            return []
        dirs = {i: i for i in ids}
    else:
        if not input_dir.is_dir():
            return []
        folders = sorted(p for p in input_dir.iterdir()
                         if p.is_dir() and not p.name.startswith(("_", ".")))
        dirs = {slugify(d.name): d.name for d in folders}
        ids = list(dirs)

    out: list[dict] = []
    for lid in ids:
        pretty = names.get(lid) or names.get(dirs[lid]) or dirs[lid]
        spec = {"id": lid, "dir": dirs[lid], "name": f"{kind} - {pretty}",
                "prompt": lobs.get("prompt"), "lob": lid}
        if prompt_dir:
            spec["lob_prompt"] = str(Path(prompt_dir) / f"{lid}.txt")
        out.append(spec)
    cache[str(input_dir)] = out
    return [dict(s) for s in out]


def doc_type_specs(cfg: dict) -> list[dict]:
    """The document types, in order: discovered LOB folders, or configured doc_types."""
    if lob_mode(cfg):
        return lob_specs(cfg)
    return [dict(d) for d in (cfg.get("doc_types") or [])]


def doc_type_spec(cfg: dict, doc_type: str) -> dict:
    for d in doc_type_specs(cfg):
        if str(d.get("id")) == str(doc_type):
            return dict(d)
    raise KeyError(
        f"unknown doc_type {doc_type!r}; config.yaml doc_types defines "
        f"{[d.get('id') for d in doc_type_specs(cfg)]}")


def doc_type_dir_name(cfg: dict, doc_type: str) -> str:
    """The on-disk folder name for a doc type ('loss_run' -> 'loss runs')."""
    try:
        return str(doc_type_spec(cfg, doc_type).get("dir") or doc_type)
    except KeyError:
        return str(doc_type)


def out_dir_for(cfg: dict, model_tag: str, case: str, doc_type: str | None = None) -> Path:
    """output/<tag>/<case folder>/<doc type folder>/ - e.g. output/8B/PDF+MD/policy/."""
    base = cfg_path(cfg, "output_dir") / str(model_tag) / case_dir_name(cfg, case)
    return base / doc_type_dir_name(cfg, doc_type) if doc_type else base


def ocr_dir_for(cfg: dict, doc_id: str) -> Path:
    """Per-document asset folder: work/ocr/<doc_id>/page_N.png + page_N.md.

    doc_id is unique across doc types (discover_documents guarantees it), so two
    same-named PDFs filed under different types cannot share a cache folder.
    """
    return cfg_path(cfg, "ocr_dir") / doc_id


# --------------------------------------------------------------------------- IO


def read_text(p: Path) -> str:
    return Path(p).read_text(encoding="utf-8")


def write_text(p: Path, text: str) -> None:
    p = Path(p)
    ensure_dir(p.parent)
    p.write_text(text, encoding="utf-8", newline="\n")


def read_json(p: Path) -> Any:
    return json.loads(read_text(p))


def write_json(p: Path, obj: Any, indent: int = 2) -> None:
    write_text(p, json.dumps(obj, indent=indent, ensure_ascii=False, default=str) + "\n")


def compact_json(obj: Any) -> str:
    """Single-line JSON."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def pretty_json(obj: Any, indent: int = 2) -> str:
    """Indented JSON - the shape a model actually emits, so the shape to budget for."""
    return json.dumps(obj, indent=indent, ensure_ascii=False, default=str)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def is_stale(target: Path, sources) -> str | None:
    """Is `target` missing, or older than any of `sources`?

    Returns a short reason when the target needs rebuilding, else None.
    """
    target = Path(target)
    if not target.exists():
        return f"{target.name} does not exist"
    stamp = target.stat().st_mtime
    for src in sources:
        src = Path(src)
        if src.exists() and src.stat().st_mtime > stamp:
            return f"{src.name} is newer than {target.name}"
    return None


# --------------------------------------------------------------------------- corpus


DEFAULT_GOLD_SUFFIXES = (
    "extraction", "extracted", "gold", "golden", "groundtruth", "gt",
    "expected", "label", "labels", "labelled", "labeled",
    "annotation", "annotations",
)


def name_key(stem: str, gold_suffixes=DEFAULT_GOLD_SUFFIXES) -> str:
    """Normalize a file stem so a PDF and its gold JSON collapse to the same key.

    'New Business 6'            -> 'newbusiness6'
    'New_Business_6_extraction' -> 'newbusiness6'
    """
    key = re.sub(r"[^a-z0-9]+", "", str(stem).lower())
    changed = True
    while changed:
        changed = False
        for suffix in gold_suffixes:
            if key.endswith(suffix) and len(key) > len(suffix):
                key = key[: -len(suffix)]
                changed = True
    return key


def name_tokens(stem: str, gold_suffixes=DEFAULT_GOLD_SUFFIXES) -> set[str]:
    """Distinctive tokens of a file stem, for pairing files whose names differ.

    Real labelling conventions are descriptive, not stem-matching:
    'BINDER- ACORD 75.pdf' is labelled 'acord75_binder_4885144.golden.json'. Exact
    key equality cannot pair those, so pairing falls back to token overlap.

    'BINDER- ACORD 75'      -> {'binder', 'acord', '75'}
    'acord75_binder_4885144'-> {'acord', '75', 'binder', '4885144'}
    """
    text = str(stem).lower()
    # split on non-alphanumerics AND on letter<->digit boundaries, so 'acord75'
    # and 'ACORD 75' produce the same tokens.
    text = re.sub(r"(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])", " ", text)
    toks = {t for t in re.split(r"[^a-z0-9]+", text) if t}
    return {t for t in toks if t not in set(gold_suffixes) and t not in _WEAK_TOKENS}


# Tokens too common in this corpus to identify anything on their own.
_WEAK_TOKENS = {"acord", "form", "forms", "pdf", "doc", "document", "final", "copy",
                "v1", "v2", "schema", "canonical", "the", "of", "and"}


def token_match(pdf_stem: str, candidates: list, gold_suffixes=DEFAULT_GOLD_SUFFIXES):
    """Best token-overlap match for a PDF among candidate JSONs, or None.

    Requires a unique winner that shares at least one distinctive token, so a
    near-miss ('ACORD24' against the ACORD 25 gold, sharing nothing but the weak
    token 'acord') pairs nothing rather than pairing wrongly.
    """
    want = name_tokens(pdf_stem, gold_suffixes)
    if not want:
        return None
    scored = []
    for c in candidates:
        have = name_tokens(Path(c).stem, gold_suffixes)
        shared = want & have
        if shared:
            # A shared digit run is far more discriminative than a shared word.
            score = sum(2 if t.isdigit() else 1 for t in shared)
            scored.append((score, str(c), c))
    if not scored:
        return None
    scored.sort(key=lambda t: (-t[0], t[1]))
    if len(scored) > 1 and scored[0][0] == scored[1][0]:
        return None          # ambiguous: refuse rather than guess
    return scored[0][2]


def slugify(stem: str) -> str:
    """Filesystem-safe id used for this document's work and output folders."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(stem).lower()).strip("_")
    return slug or "document"


def looks_like_json_schema(obj: Any) -> bool:
    """Is this a JSON Schema document rather than an instance of one?

    POC2's discover_documents globbed *.json and would happily pair
    input/schema.json as a document's gold. Several independent tells, because a
    schema may legitimately omit "$schema", and a gold instance may carry a
    "$schema" key pointing at the schema it conforms to - so the key alone is
    not enough, its value must name a JSON Schema dialect.
    """
    if not isinstance(obj, dict):
        return False
    dialect = obj.get("$schema")
    if isinstance(dialect, str) and "json-schema.org" in dialect:
        return True
    props = obj.get("properties")
    if (obj.get("type") == "object" and isinstance(props, dict)
            and all(isinstance(v, (dict, bool)) for v in props.values())):
        return True
    # A LOB schema built purely by composition: allOf/$defs and nothing else.
    keys = set(obj) - {"$id", "$comment", "title", "description", "type",
                       "additionalProperties", "required", "properties"}
    return bool(keys) and keys <= {"allOf", "$defs", "definitions", "$ref"}


class Document:
    """One item: a PDF and (optionally) the golden canonical JSON labelling it.

    gold is optional on purpose - inference and scoring are separable stages
    here, so an unpaired PDF still runs, it just goes unscored.
    """

    __slots__ = ("doc_id", "pdf", "gold", "n_pages", "doc_type", "schema")

    def __init__(self, doc_id: str, pdf: Path, gold: Path | None = None,
                 n_pages: int = 0, doc_type: str = "", schema: Path | None = None):
        self.doc_id = doc_id
        self.pdf = Path(pdf)
        self.gold = Path(gold) if gold else None
        self.n_pages = n_pages
        self.doc_type = doc_type
        self.schema = Path(schema) if schema else None

    def __repr__(self) -> str:
        return (f"Document({self.doc_id}, type={self.doc_type or '?'}, "
                f"gold={'yes' if self.gold else 'NO'}, "
                f"schema={'yes' if self.schema else 'NO'})")


def classify_jsons(paths, gold_suffixes) -> tuple[dict, dict]:
    """Split JSON files into (schemas_by_key, golds_by_key), keyed by normalised stem.

    A file is a schema if it looks like one structurally - a root $schema, or
    properties + type: object - not because of where it sits or what it is
    called. That is what lets a schema and its gold live side by side in one
    folder without either being mistaken for the other.
    """
    schemas: dict = {}
    golds: dict = {}
    for f in paths:
        if f.name.startswith("."):
            continue
        try:
            doc = read_json(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError):
            continue
        key = name_key(f.stem, gold_suffixes)
        (schemas if looks_like_json_schema(doc) else golds).setdefault(key, []).append(f)
    return schemas, golds


def discover_documents(
    cfg: dict,
    gold_suffixes=DEFAULT_GOLD_SUFFIXES,
    resolve_schemas: bool = True,
) -> tuple[list[Document], list[Path], list[Path]]:
    """Pair every PDF with its doc type, its schema and its gold.

    Each configured doc type owns a folder under input/. Within that folder,
    layout is free: PDFs anywhere, JSONs anywhere. Pairing is by normalised file
    stem, so input/policy/pdfs/HOME.pdf takes input/policy/gold/HOME.json and
    input/policy/schema/HOME.schema.json.

    Schema resolution per PDF, first hit wins:
      1. a schema whose stem matches this PDF
      2. the single schema in this doc type's folder, if there is exactly one
      3. paths.schema in config.yaml, if set

    (2) is what makes "one schema per doc type" work without renaming anything,
    while (1) still lets a single document override it.

    In LOB mode the schema found this way is the LOB's own schema, and the
    document is given the RESOLVED schema instead: common + LOB combined into one
    file under paths.resolved_schema_dir (schema_resolver.py). Pass
    resolve_schemas=False when only the documents are needed (the report's
    timing tables): nothing is resolved or written, and doc.schema is the LOB's
    own schema file.

    Returns (documents, unpaired_pdfs, unpaired_golds).
    """
    input_dir = cfg_path(cfg, "input_dir")
    fallback = cfg.get("paths", {}).get("schema")
    fallback_schema = cfg_path(cfg, "schema") if fallback else None
    resolved_cache: dict = {}
    if lob_mode(cfg):
        import schema_resolver
        if resolve_schemas:
            schema_resolver.ERRORS.clear()   # report what THIS discovery found

    documents: list[Document] = []
    unpaired_pdfs: list[Path] = []
    unmatched_golds: list[Path] = []
    seen_ids: set[str] = set()

    if lob_mode(cfg) and manifest_path(cfg) is not None:
        return _discover_manifest(cfg, resolve_schemas)

    specs = doc_type_specs(cfg)
    if not specs:
        if lob_mode(cfg):
            return [], [], []    # no LOB folder yet; callers say where to put them
        raise ValueError("config.yaml defines no doc_types")

    for spec in specs:
        tid = str(spec.get("id"))
        tdir = input_dir / str(spec.get("dir") or tid)
        if not tdir.exists():
            continue
        pdfs = sorted(f for f in tdir.rglob("*.pdf") if not f.name.startswith("."))
        schemas, golds = classify_jsons(sorted(tdir.rglob("*.json")), gold_suffixes)

        all_schemas = [f for fs in schemas.values() for f in fs]
        if lob_mode(cfg) and len(all_schemas) > 1:
            # A LOB schema split across files (the LOB schema + a definitions
            # file it $refs): only the file no sibling references is a
            # candidate, so a parts file can never be picked for a document.
            roots = set(schema_resolver.root_schemas(all_schemas))
            all_schemas = [f for f in all_schemas if f in roots]
            schemas = {k: [f for f in fs if f in roots] for k, fs in schemas.items()}
        type_schema = all_schemas[0] if len(all_schemas) == 1 else None

        matched: set[Path] = set()
        for pdf in pdfs:
            key = name_key(pdf.stem, gold_suffixes)

            # Gold: exact key, then token overlap, then the single-document
            # fallback. Labelling conventions are descriptive in practice, so
            # exact stem equality pairs almost nothing on a real corpus.
            gold = (golds.get(key) or [None])[0]
            if gold is None:
                all_golds = [f for fs in golds.values() for f in fs]
                gold = token_match(pdf.stem, all_golds, gold_suffixes)
            if gold is None and len(pdfs) == 1 and len(golds) == 1:
                gold = next(iter(golds.values()))[0]
            if gold is not None:
                matched.add(gold)

            schema = (schemas.get(key) or [None])[0]
            if schema is None:
                schema = token_match(pdf.stem, all_schemas, gold_suffixes)
            schema = schema or type_schema
            # The global fallback only for a folder with NO candidate schema: with
            # several, falling back would hide the ambiguity reported below.
            if schema is None and not (lob_mode(cfg) and len(all_schemas) > 1):
                schema = fallback_schema
            if lob_mode(cfg) and schema is None and len(all_schemas) > 1:
                # Several candidate LOB schemas and none named after this PDF:
                # guessing would silently extract against the wrong one, and
                # falling back to common-only would silently drop every LOB field.
                msg = (f"{pdf.name}: the folder holds {len(all_schemas)} LOB schemas "
                       f"({', '.join(f.name for f in all_schemas)}) and none is named "
                       f"after this PDF - keep one schema per LOB folder, or name a "
                       f"schema after the PDF")
                if resolve_schemas:
                    schema_resolver.ERRORS.setdefault(tid, []).append(msg)
                    logging.getLogger("discover").error("%s", msg)
                schema = False   # resolved below to None, without resolving common-only
            if lob_mode(cfg) and resolve_schemas and schema is not False:
                ck = (tid, str(schema))
                if ck not in resolved_cache:
                    variant = (None if schema is None or schema == type_schema
                               else Path(schema).name.split(".")[0])
                    try:
                        resolved_cache[ck] = schema_resolver.resolved_schema_path(
                            cfg, tid, schema, variant)
                    except schema_resolver.ResolveError as exc:
                        # The document stays discoverable with no schema, so
                        # preflight can report the reason instead of crashing.
                        logging.getLogger("discover").error("%s", exc)
                        resolved_cache[ck] = None
                schema = resolved_cache[ck]
            elif schema is False:
                schema = None

            doc_id = slugify(pdf.stem)
            if doc_id in seen_ids:
                # Same PDF name under two doc types would otherwise collide in
                # work/ocr and overwrite one another's rendered pages.
                doc_id = f"{tid}-{doc_id}"
            seen_ids.add(doc_id)

            if gold is None:
                unpaired_pdfs.append(pdf)
            documents.append(Document(doc_id, pdf, gold, doc_type=tid, schema=schema))

        unmatched_golds += [f for fs in golds.values() for f in fs if f not in matched]

    documents.sort(key=lambda d: (d.doc_type, d.doc_id))
    return documents, unpaired_pdfs, unmatched_golds


def _discover_manifest(cfg: dict, resolve_schemas: bool = True
                       ) -> tuple[list[Document], list[Path], list[Path]]:
    """Documents exactly as lobs.manifest lists them - no pairing by name.

    Each entry names its PDF, gold and LOB schema, so nothing is guessed. A PDF or
    gold JSON in the input folder that the manifest does not list is returned as
    unpaired, so preflight can point it out.
    """
    import schema_resolver
    if resolve_schemas:
        schema_resolver.ERRORS.clear()
    try:
        man = read_manifest(cfg)
    except (FileNotFoundError, ValueError) as exc:
        logging.getLogger("discover").error("%s", exc)
        return [], [], []
    base = manifest_path(cfg).parent
    documents: list[Document] = []
    resolved_cache: dict = {}
    listed: set[Path] = set()
    seen_ids: set[str] = set()
    for entry in man["documents"]:
        pdf = (base / entry["pdf"]).resolve()
        gold = (base / entry["gold"]).resolve() if entry.get("gold") else None
        schema = (base / entry["schema"]).resolve()
        lob = slugify(str(entry["lob"]))
        listed |= {pdf} | ({gold} if gold else set())
        base_id = slugify(str(entry.get("id") or pdf.stem))
        doc_id, n = base_id, 1
        while doc_id in seen_ids:
            # Ids name the OCR cache and the output files: a collision would
            # make two documents overwrite each other.
            n += 1
            doc_id = f"{lob}-{base_id}" if n == 2 else f"{lob}-{base_id}-{n}"
        seen_ids.add(doc_id)
        problems = [f"{k} {p.name} not found" for k, p in
                    (("pdf", pdf), ("gold", gold), ("schema", schema)) if p and not p.exists()]
        if problems:
            msg = f"manifest entry {doc_id}: " + "; ".join(problems)
            if resolve_schemas:
                schema_resolver.ERRORS.setdefault(lob, []).append(msg)
            logging.getLogger("discover").error("%s", msg)
            if not pdf.exists():
                continue
            gold = gold if gold and gold.exists() else None
        resolved = schema if schema.exists() else None
        if resolve_schemas and resolved is not None:
            if str(schema) not in resolved_cache:
                # Two schema files under one LOB must not share a resolved file:
                # one named for the LOB keeps the plain name, any other gets a
                # suffix.
                variant = None if slugify(schema.stem) == lob else schema.stem
                try:
                    resolved_cache[str(schema)] = schema_resolver.resolved_schema_path(
                        cfg, lob, schema, variant)
                except schema_resolver.ResolveError as exc:
                    logging.getLogger("discover").error("%s", exc)
                    resolved_cache[str(schema)] = None
            resolved = resolved_cache[str(schema)]
        documents.append(Document(doc_id, pdf, gold, doc_type=lob, schema=resolved))

    # Files sitting in the folders the manifest uses but not listed in it -
    # most likely a document someone added and forgot to register.
    pdf_dirs = {p.parent for p in listed if p.suffix.lower() == ".pdf"}
    gold_dirs = {p.parent for p in listed if p.suffix.lower() == ".json"}
    unpaired = sorted(p.resolve() for d in pdf_dirs for p in d.glob("*.pdf")
                      if p.resolve() not in listed and not p.name.startswith("."))
    orphans = sorted(p.resolve() for d in gold_dirs for p in d.glob("*.json")
                     if p.resolve() not in listed and not p.name.startswith("."))
    documents.sort(key=lambda d: (d.doc_type, d.doc_id))
    return documents, unpaired, orphans


def select_documents(cfg: dict, resolve_schemas: bool = True,
                     require_gold: bool | None = None) -> list[Document]:
    """Documents this run covers, honouring corpus.documents and corpus.require_gold.

    require_gold overrides corpus.require_gold (timing needs no gold);
    resolve_schemas=False lists documents without touching resolved schemas.
    """
    corpus = cfg.get("corpus", {}) or {}
    docs, _unpaired, _ = discover_documents(
        cfg, tuple(corpus.get("gold_suffixes") or DEFAULT_GOLD_SUFFIXES), resolve_schemas)
    wanted = corpus.get("documents")
    if wanted:
        wanted = {str(w) for w in wanted}
        docs = [d for d in docs if d.doc_id in wanted or d.doc_type in wanted]
    need_gold = corpus.get("require_gold", True) if require_gold is None else require_gold
    if need_gold:
        docs = [d for d in docs if d.gold is not None]
    return docs


# --------------------------------------------------------------------------- JSON recovery


_THINK_RE = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_DANGLING_THINK_RE = re.compile(r"^.*?</think>", flags=re.DOTALL | re.IGNORECASE)
_CONTROL_RE = re.compile(r"^(?:<\|[^|]*\|>\s*)+")


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Remove <think> blocks and leading chat control tokens.

    The Instruct variants of both models emit neither. But a mis-pointed model id
    (the Thinking variant) would wrap the answer, and extract_json_object's fence
    regex would then happily grab a brace from inside the reasoning. The returned
    flag is recorded in meta.json: a silent difference between the 7B and 8B cells
    is exactly the kind of thing that invalidates a comparison.
    """
    if not text:
        return text or "", False
    original = text
    text = _CONTROL_RE.sub("", text.lstrip())
    if "</think>" in text.lower():
        text = _THINK_RE.sub("", text)
        if "</think>" in text.lower():
            text = _DANGLING_THINK_RE.sub("", text)
    text = text.strip()
    return text, text != original.strip()


def extract_json_object(text: str) -> tuple[Any | None, str]:
    """Pull the first complete JSON object out of raw model output.

    Returns (parsed_or_None, reason). Tolerates code fences and trailing prose,
    which the prompt forbids but models still occasionally emit.
    """
    if text is None:
        return None, "empty output"
    cleaned, _ = strip_reasoning(text)
    if not cleaned:
        return None, "empty output"

    fence = re.search(r"```(?:json)?\s*(.*?)```", cleaned, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        cleaned = fence.group(1).strip()

    try:
        return json.loads(cleaned), "clean parse"
    except json.JSONDecodeError:
        pass

    start = cleaned.find("{")
    if start == -1:
        return None, "no open brace in output"

    depth, in_str, escaped = 0, False, False
    for i in range(start, len(cleaned)):
        ch = cleaned[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == BACKSLASH:
                escaped = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                candidate = cleaned[start : i + 1]
                try:
                    return json.loads(candidate), "recovered balanced object"
                except json.JSONDecodeError as exc:
                    return None, f"balanced object did not parse: {exc}"
    return None, "unterminated JSON object"


def repair_truncated_json(text: str) -> tuple[Any | None, str]:
    """Recover the complete prefix of a JSON object cut off mid-generation.

    A model that hits max_tokens emits a valid *prefix*, which json.loads rejects
    outright - so a 99%-complete answer would otherwise score zero fields. This
    rewinds to the last position where every container could be closed cleanly,
    closes them, and parses that.

    Returns (parsed_or_None, reason). The result is a strict subset of what the
    model produced: no value is invented, the tail is dropped.
    """
    if not text:
        return None, "empty output"
    text, _ = strip_reasoning(text)
    start = text.find("{")
    if start == -1:
        return None, "no open brace in output"

    stack: list[str] = []
    safe: list[tuple[int, tuple[str, ...]]] = []  # (cut index, open containers)
    in_str = escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if escaped:
                escaped = False
            elif ch == BACKSLASH:
                escaped = True
            elif ch == '"':
                in_str = False
        elif ch == '"':
            in_str = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack or stack[-1] != ch:
                return None, "mismatched brackets"
            stack.pop()
            safe.append((i + 1, tuple(stack)))
            if not stack:
                break
        elif ch == ",":
            safe.append((i, tuple(stack)))  # cut before the comma

    # Newest safe points first: keep as much of the answer as parses.
    for cut, open_containers in reversed(safe[-2000:]):
        candidate = text[start:cut] + "".join(reversed(open_containers))
        try:
            return json.loads(candidate), (
                f"repaired truncated JSON: kept {cut - start} of "
                f"{len(text) - start} chars, closed {len(open_containers)} container(s)"
            )
        except json.JSONDecodeError:
            continue
    return None, "could not repair truncated JSON"


_ESCAPE_RE = re.compile(r'\\(["\\/bfnrt]|u[0-9a-fA-F]{4})?')


def drop_invalid_escapes(text: str) -> tuple[str, int]:
    """Drop every backslash that starts no valid JSON escape. Returns (text, count).

    MinerU writes markdown, and markdown escapes characters: a blank on a form
    ("NO. ____") comes out as "NO. \\_\\_\\_\\_". A model that copies the value
    verbatim puts "\\_" in a JSON string, which json.loads rejects outright - a
    whole window lost to one character. Without the backslash the value reads as
    the page does. Valid escapes, an escaped backslash included, are kept.
    """
    n = 0

    def fix(m: re.Match) -> str:
        nonlocal n
        if m.group(1):
            return m.group(0)
        n += 1
        return ""

    return _ESCAPE_RE.sub(fix, text), n


def recover_json_object(text: str) -> tuple[Any | None, str, bool]:
    """Parse model output, falling back to truncation repair.

    Returns (parsed_or_None, reason, repaired). `repaired` True means the output
    was NOT valid JSON and only a prefix was recovered - callers must keep
    reporting it as invalid while still scoring the fields it did contain.
    """
    parsed, reason = extract_json_object(text)
    if parsed is not None:
        return parsed, reason, False
    # Stray markdown escapes are the one syntax error whose fix loses nothing:
    # drop the backslashes and parse again before cutting anything off.
    fixed, n_fixed = drop_invalid_escapes(text or "")
    if n_fixed:
        parsed, fixed_reason = extract_json_object(fixed)
        if parsed is not None:
            return parsed, (f"{fixed_reason} after dropping {n_fixed} invalid "
                            f"backslash escape(s)"), False
        text = fixed
    parsed, repair_reason = repair_truncated_json(text or "")
    if parsed is not None:
        return parsed, f"{reason}; {repair_reason}", True
    return None, f"{reason}; {repair_reason}", False


def validate_against_schema(obj: Any, schema_path: Path) -> tuple[bool, list[str]]:
    """Best-effort jsonschema validation. Absent jsonschema/schema => (True, [])."""
    schema_path = Path(schema_path)
    if not schema_path.exists():
        return True, []
    try:
        import jsonschema
    except ImportError:
        return True, []
    validator = jsonschema.Draft202012Validator(read_json(schema_path))
    errors = [
        f"{'.'.join(str(x) for x in e.path) or '<root>'}: {e.message}"
        for e in sorted(validator.iter_errors(obj), key=lambda e: list(e.path))
    ]
    return (not errors), errors
