"""Common schema + per-LOB schema -> one resolved canonical schema per LOB.

    python src/schema_resolver.py              # resolve every LOB, print what happened

The canonical JSON for a policy is the common part every line of business shares
plus the part specific to its LOB. A LOB schema can build on the common one in
any of three ways, and all three are handled:

  1. by reference    "$ref": "../common.schema.json#/$defs/Insured"
  2. by composition  "allOf": [{"$ref": "../common.schema.json"}, {...LOB part...}]
  3. side by side    two standalone schemas whose top-level fields are combined

The resolved schema has no external $ref: referenced definitions are copied
into its own $defs, an allOf of plain object parts in the LOB schema is merged
into one object, and (unless the LOB already included it, or the common schema
is a definitions library with no fields of its own) the common schema's
top-level fields are merged in. An allOf that carries rules - if/then/else,
anyOf, oneOf, not - is kept as written, as are the copied definitions (the
common model's FieldValue keeps its two if/then rules). That matters for three
consumers at once:
  * the prompt builder reads `properties` and local `$defs` - an allOf or an
    external $ref would render as an empty outline;
  * the validator - "additionalProperties": false inside one allOf branch
    rejects every field the other branch adds, a classic composition trap;
  * the scorer and the merge, which walk one concrete structure.

When both schemas define the same field, the LOB's definition wins on a direct
clash (type, enum, format...) and the two are merged where they can be
(properties, required). Every overlap and clash is written to
<lob>.resolve_notes.json next to the resolved schema, so nothing is decided
silently.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402

DIALECT = "https://json-schema.org/draft/2020-12/schema"
META_KEYS = ("$schema", "$id", "$defs", "definitions", "$comment")
# Keywords where a difference between common and LOB is a real clash worth
# recording. Others (title, description...) differ routinely and harmlessly.
CLASH_KEYS = {"type", "enum", "const", "format", "pattern", "items", "minimum",
              "maximum", "minItems", "maxItems", "$ref"}


class ResolveError(RuntimeError):
    pass


# Problems found while resolving, by LOB id, for THIS process. Preflight reads
# these straight after discovery, so it reports what happened now - never a
# notes file left over from an earlier run.
ERRORS: dict[str, list[str]] = {}


def _read(path: Path) -> dict:
    """A schema file, or a ResolveError naming it - never a bare JSON traceback."""
    try:
        doc = common.read_json(path)
    except (ValueError, OSError) as exc:
        raise ResolveError(f"{Path(path).name} is not readable JSON: {exc}") from exc
    if not isinstance(doc, dict):
        raise ResolveError(f"{Path(path).name} is not a JSON object")
    return doc


def _ref_targets(doc: Any) -> set[str]:
    """File names this schema $refs (the part of each $ref before '#')."""
    out: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str) and not ref.startswith("#"):
                out.add(Path(ref.partition("#")[0]).name)
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(doc)
    return out


def root_schemas(paths: list[Path]) -> list[Path]:
    """The schema files in a LOB folder that no sibling schema $refs.

    A LOB schema may be split across files - the LOB schema plus a definitions
    file it references. Only the file nothing else points at is the LOB's own
    schema; the others are its parts and must never be picked for a document.
    An unreadable file is kept as a candidate so resolving it reports the error.
    """
    referenced: set[str] = set()
    for p in paths:
        try:
            referenced |= _ref_targets(common.read_json(p))
        except (ValueError, OSError):
            continue
    return [p for p in paths if Path(p).name not in referenced]


# --------------------------------------------------------------------------- merge


def merge_schema(a: dict, b: dict, path: str, notes: dict) -> dict:
    """Deep-merge schema b into schema a. b wins on a direct clash."""
    out = copy.deepcopy(a)
    for key, bv in b.items():
        if key not in out:
            out[key] = copy.deepcopy(bv)
            continue
        av = out[key]
        if key in ("properties", "$defs", "patternProperties") \
                and isinstance(av, dict) and isinstance(bv, dict):
            merged = dict(av)
            for name, sub in bv.items():
                where = f"{path}/{key}/{name}"
                if name in merged and isinstance(merged[name], dict) and isinstance(sub, dict):
                    if merged[name] != sub:
                        notes["overlaps"].append(where)
                    merged[name] = merge_schema(merged[name], sub, where, notes)
                else:
                    merged[name] = copy.deepcopy(sub)
            out[key] = merged
        elif key == "required" and isinstance(av, list) and isinstance(bv, list):
            out[key] = av + [x for x in bv if x not in av]
        elif key == "additionalProperties":
            if isinstance(av, dict) and isinstance(bv, dict):
                out[key] = merge_schema(av, bv, f"{path}/{key}", notes)
            else:
                # Closed if either side is closed: the union of both property
                # sets is then the complete vocabulary.
                out[key] = False if (av is False or bv is False) else copy.deepcopy(bv)
        elif key == "items" and isinstance(av, dict) and isinstance(bv, dict):
            out[key] = merge_schema(av, bv, f"{path}/items", notes)
        elif key == "allOf" and isinstance(av, list) and isinstance(bv, list):
            out[key] = av + bv
        else:
            if av != bv and key in CLASH_KEYS:
                notes["clashes"].append({"path": f"{path}/{key}", "common": av, "lob": bv,
                                         "kept": "lob"})
            out[key] = copy.deepcopy(bv)
    return out


# --------------------------------------------------------------------------- bundling


def _pointer(doc: Any, frag: str) -> Any:
    node = doc
    for part in [p for p in frag.split("/") if p]:
        part = part.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and part in node:
            node = node[part]
        elif isinstance(node, list) and part.isdigit() and int(part) < len(node):
            node = node[int(part)]
        else:
            raise ResolveError(f"$ref fragment #{frag} does not resolve")
    return node


def _normalise_defs(doc: dict) -> dict:
    """draft-07 'definitions' -> 2020-12 '$defs', refs rewritten to match."""
    if "definitions" in doc and "$defs" not in doc:
        doc = copy.deepcopy(doc)
        doc["$defs"] = doc.pop("definitions")

        def fix(node: Any) -> None:
            if isinstance(node, dict):
                ref = node.get("$ref")
                if isinstance(ref, str) and "#/definitions/" in ref:
                    node["$ref"] = ref.replace("#/definitions/", "#/$defs/")
                for v in node.values():
                    fix(v)
            elif isinstance(node, list):
                for v in node:
                    fix(v)
        fix(doc)
    return doc


class Bundler:
    """Rewrite every external $ref in a root schema into something local.

    A ref into another file's $defs copies that definition into the root's $defs
    (renamed if the name is taken) and points at the copy. A ref to a whole file,
    or to any other pointer inside it, is replaced by a copy of what it points
    at. Definitions are copied lazily, on first reference, so a recursive
    definition resolves to itself rather than looping.
    """

    def __init__(self, root: dict, root_path: Path, aliases: dict[str, Path], notes: dict):
        self.root_path = root_path.resolve()
        self.aliases = aliases          # $id or bare file name -> path
        self.notes = notes
        self.docs: dict[Path, dict] = {self.root_path: root}
        self.out_defs: dict[str, Any] = {}
        self.taken: set[str] = set((root.get("$defs") or {}).keys())
        self.maps: dict[Path, dict[str, str]] = {}
        self.inlined_roots: set[Path] = set()
        self.referenced: set[Path] = set()

    def doc(self, path: Path) -> dict:
        if path not in self.docs:
            self.docs[path] = _normalise_defs(_read(path))
        return self.docs[path]

    def _target(self, uri: str, here: Path) -> Path:
        if not uri:
            return here
        if uri in self.aliases:
            return self.aliases[uri]
        for base in (here.parent, self.root_path.parent):
            cand = (base / uri).resolve()
            if cand.exists():
                return cand
        name = Path(uri).name
        if name in self.aliases:
            return self.aliases[name]
        raise ResolveError(f"$ref {uri!r} (from {here.name}) names a file that was not found")

    def def_name(self, path: Path, name: str) -> str:
        m = self.maps.setdefault(path, {})
        if name in m:
            return m[name]
        new = name
        if new in self.taken:
            new = f"{common.slugify(path.stem)}__{name}"
            n = 2
            while new in self.taken:
                new = f"{common.slugify(path.stem)}__{name}_{n}"
                n += 1
            self.notes["renamed_defs"].append({"file": path.name, "from": name, "to": new})
        m[name] = new
        self.taken.add(new)
        body = (self.doc(path).get("$defs") or {}).get(name)
        if body is None:
            raise ResolveError(f"{path.name} has no $defs/{name}")
        self.out_defs[new] = {}  # placeholder: a self-reference resolves to this name
        self.out_defs[new] = self.walk(copy.deepcopy(body), path)
        return new

    def walk(self, node: Any, here: Path) -> Any:
        if isinstance(node, list):
            return [self.walk(x, here) for x in node]
        if not isinstance(node, dict):
            return node
        is_root_doc = here == self.root_path
        out: dict = {}
        for k, v in node.items():
            if k == "$ref":
                continue
            if k == "$defs" and not is_root_doc:
                continue   # another file's definitions are copied lazily, on reference
            out[k] = self.walk(v, here)
        ref = node.get("$ref")
        if not isinstance(ref, str):
            if "$ref" in node:
                out["$ref"] = ref
            return out

        uri, _, frag = ref.partition("#")
        target = self._target(uri, here)
        if target == self.root_path:
            # A pointer into the root itself is already local.
            if frag.startswith("/$defs/") or not uri:
                out["$ref"] = "#" + frag
                return out
        self.referenced.add(target)
        if frag.startswith("/$defs/"):
            name = frag[len("/$defs/"):].split("/", 1)[0].replace("~1", "/").replace("~0", "~")
            out["$ref"] = f"#/$defs/{self.def_name(target, name)}"
            return out
        # Whole file, or another pointer inside it: inline a copy.
        src = self.doc(target)
        pointed = _pointer(src, frag) if frag.strip("/") else src
        if not frag.strip("/"):
            self.inlined_roots.add(target)
            pointed = {k: v for k, v in pointed.items() if k not in META_KEYS}
        inlined = self.walk(copy.deepcopy(pointed), target)
        return merge_schema(inlined, out, f"<{target.name}>", self.notes) if out else inlined


def flatten_allof(node: Any, defs: dict, notes: dict, path: str = "") -> Any:
    """Merge every allOf of plain object schemas into its parent node."""
    if isinstance(node, list):
        return [flatten_allof(x, defs, notes, path) for x in node]
    if not isinstance(node, dict):
        return node
    node = {k: flatten_allof(v, defs, notes, f"{path}/{k}") for k, v in node.items()}
    members = node.get("allOf")
    if not isinstance(members, list) or not members:
        return node
    concrete = []
    for m in members:
        if isinstance(m, dict) and isinstance(m.get("$ref"), str) \
                and m["$ref"].startswith("#/$defs/") and len(m) == 1:
            name = m["$ref"][len("#/$defs/"):]
            if name in defs and isinstance(defs[name], dict):
                m = flatten_allof(copy.deepcopy(defs[name]), defs, notes, path)
        if not isinstance(m, dict) or "$ref" in m or any(
                k in m for k in ("anyOf", "oneOf", "not", "if", "then", "else")):
            # Not plain - leave this allOf for the validator. Conditionals in
            # particular must stay separate: merged, a later if/then would
            # silently replace an earlier one.
            return node
        concrete.append(m)
    # Members first, in order (a later member wins a clash, so the LOB part
    # placed after the common one wins), then the node's own keywords on top.
    acc: dict = {}
    for m in concrete:
        acc = merge_schema(acc, m, path or "<root>", notes)
    acc = merge_schema(acc, {k: v for k, v in node.items() if k != "allOf"},
                       path or "<root>", notes)
    notes["flattened_allof"].append(path or "<root>")
    return acc


def _check_refs(schema: dict) -> list[str]:
    bad: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            ref = node.get("$ref")
            if isinstance(ref, str):
                if not ref.startswith("#"):
                    bad.append(f"external $ref left: {ref}")
                else:
                    try:
                        _pointer(schema, ref[1:])
                    except ResolveError:
                        bad.append(f"dangling $ref: {ref}")
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
    walk(schema)
    return bad


# --------------------------------------------------------------------------- resolve


def resolve(common_path: Path | None, lob_path: Path | None,
            merge_root: str = "auto") -> tuple[dict, dict]:
    """(resolved schema, notes). Raises ResolveError on a schema it cannot resolve."""
    notes: dict = {"common": str(common_path) if common_path else None,
                   "lob": str(lob_path) if lob_path else None,
                   "mode": [], "overlaps": [], "clashes": [], "renamed_defs": [],
                   "flattened_allof": [], "common_fields": [], "lob_fields": []}
    if not lob_path and not common_path:
        raise ResolveError("neither a common schema nor a LOB schema was found")

    common_doc = _normalise_defs(_read(common_path)) if common_path else None
    if not lob_path:
        notes["mode"].append("common schema only (this LOB folder has no schema)")
        root_path, root = Path(common_path), common_doc
    else:
        root_path, root = Path(lob_path), _normalise_defs(_read(lob_path))

    aliases: dict[str, Path] = {}
    if common_path:
        cp = Path(common_path).resolve()
        aliases[cp.name] = cp
        if isinstance(common_doc.get("$id"), str):
            aliases[common_doc["$id"]] = cp

    b = Bundler(root, root_path, aliases, notes)
    walked = b.walk(root, root_path.resolve())
    if b.referenced:
        notes["mode"].append("resolved $ref to " + ", ".join(sorted(p.name for p in b.referenced)))

    common_included = bool(common_path) and Path(common_path).resolve() in b.inlined_roots
    common_has_fields = bool(common_doc and (common_doc.get("properties") or common_doc.get("allOf")))
    do_merge = bool(common_path and lob_path) and common_has_fields and (
        merge_root == "always" or (merge_root == "auto" and not common_included))
    if common_included:
        notes["mode"].append("LOB schema includes the whole common schema itself")
    if common_path and lob_path and not common_has_fields:
        # A definitions library: it contributes only what the LOB schema $refs,
        # which the bundler has already copied in. Copying every definition
        # would only inflate the prompt with shapes this LOB never uses.
        notes["mode"].append(
            f"common schema is a definitions library: {len(b.out_defs)} of its "
            f"{len(common_doc.get('$defs') or {})} definitions are used by this LOB")
    if do_merge:
        # The common root joins as a parent: its top-level fields come first, the
        # LOB's are added after them, and the LOB wins a direct clash. Only the
        # definitions its fields reference come with it.
        cp = Path(common_path).resolve()
        common_root = {k: v for k, v in common_doc.items() if k not in META_KEYS}
        common_walked = b.walk(copy.deepcopy(common_root), cp)
        walked = merge_schema(common_walked, walked, "<root>", notes)
        notes["mode"].append("merged common top-level fields into the LOB schema")

    defs = dict(walked.get("$defs") or {})
    defs.update(b.out_defs)
    walked = flatten_allof(walked, defs, notes)
    defs.update(walked.get("$defs") or {})
    if defs:
        walked["$defs"] = defs

    resolved = {"$schema": DIALECT}
    title = walked.get("title")
    if common_doc and lob_path and common_doc.get("title"):
        title = f"{title or root_path.stem} + {common_doc['title']} (resolved)"
    if title:
        resolved["title"] = title
    resolved.update({k: v for k, v in walked.items()
                     if k not in ("$schema", "$id", "title")})

    problems = _check_refs(resolved)
    if problems:
        raise ResolveError("; ".join(problems[:5]))
    try:
        import jsonschema
        jsonschema.Draft202012Validator.check_schema(resolved)
    except ImportError:
        pass
    except Exception as exc:  # noqa: BLE001 - jsonschema.SchemaError
        raise ResolveError(f"resolved schema is not a valid JSON Schema: {exc}") from exc
    top = list((resolved.get("properties") or {}).keys())
    from_common = list((common_doc.get("properties") or {}).keys()) if common_doc else []
    notes["top_level_fields"] = top
    notes["common_fields"] = [f for f in from_common if f in top]
    notes["lob_fields"] = [f for f in top if f not in from_common]   # LOB-only
    return resolved, notes


def out_dir(cfg: dict) -> Path:
    """Where resolved schemas live: paths.resolved_schema_dir, else <work>/schemas."""
    paths = cfg.get("paths") or {}
    return (common.cfg_path(cfg, "resolved_schema_dir") if paths.get("resolved_schema_dir")
            else common.cfg_path(cfg, "work_dir") / "schemas")


def notes_path_for(resolved: Path) -> Path:
    """<x>.resolved.schema.json -> <x>.resolve_notes.json, side by side."""
    name = Path(resolved).name.replace(".resolved.schema.json", "")
    return Path(resolved).with_name(f"{name}.resolve_notes.json")


def resolved_schema_path(cfg: dict, lob_id: str, lob_schema: Path | None,
                         variant: str | None = None) -> Path | None:
    """Resolve one LOB schema, write it under out_dir(cfg), return its path.

    The file is <lob>.resolved.schema.json for the LOB's own schema, and
    <lob>--<variant>.resolved.schema.json for a per-document schema in the same
    folder - so two schemas in one LOB never overwrite each other. It is
    rewritten only when its content changes. Errors are raised AND recorded in
    ERRORS for preflight; the notes file sits next to the resolved schema.
    """
    paths = cfg.get("paths") or {}
    cpath = common.cfg_path(cfg, "common_schema") if paths.get("common_schema") else None
    if cpath is not None and not cpath.exists():
        cpath = None
    if lob_schema is None and cpath is None:
        msg = ("no schema: the LOB folder holds none and paths.common_schema is "
               "unset or missing")
        ERRORS.setdefault(lob_id, []).append(msg)
        raise ResolveError(f"LOB {lob_id!r}: {msg}")
    if lob_schema is not None and cpath is not None \
            and Path(lob_schema).resolve() == cpath.resolve():
        lob_schema = None
    merge_root = str((cfg.get("schema_resolution") or {}).get("merge_common_root", "auto"))
    base = f"{lob_id}--{common.slugify(variant)}" if variant else lob_id
    target = common.ensure_dir(out_dir(cfg)) / f"{base}.resolved.schema.json"
    try:
        resolved, notes = resolve(cpath, Path(lob_schema) if lob_schema else None, merge_root)
    except ResolveError as exc:
        ERRORS.setdefault(lob_id, []).append(str(exc))
        raise ResolveError(f"LOB {lob_id!r}: {exc}") from exc
    if not target.exists() or common.read_json(target) != resolved:
        common.write_json(target, resolved)
    # notes["lob"] is the LOB schema file (None = the common schema alone).
    # Written only when it changes, like the schema: several runner processes
    # (one per case) resolve the same LOB at start-up.
    notes_doc = {"lob_id": lob_id, **notes}
    npath = notes_path_for(target)
    try:
        same = npath.exists() and common.read_json(npath) == notes_doc
    except (ValueError, OSError):
        same = False
    if not same:
        common.write_json(npath, notes_doc)
    return target


def load_notes(resolved: Path | None) -> dict:
    """The notes written next to a resolved schema, or {}."""
    if not resolved:
        return {}
    p = notes_path_for(Path(resolved))
    return common.read_json(p) if p.exists() else {}


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = common.load_config(args.config)
    docs, _u, _o = common.discover_documents(cfg)
    lobs = sorted({d.doc_type for d in docs})
    if not lobs:
        print(f"no LOB folders with PDFs under {common.cfg_path(cfg, 'input_dir')}")
        return 2
    for lob in lobs:
        for err in ERRORS.get(lob, []):
            print(f"\n{lob}\n  ERROR {err}")
        for schema in sorted({str(d.schema) for d in docs if d.doc_type == lob and d.schema}):
            show(lob, Path(schema))
    print(f"\nresolved schemas in {out_dir(cfg)}")
    return 1 if ERRORS else 0


def show(lob: str, resolved: Path) -> None:
    n = load_notes(resolved)
    print(f"\n{lob}  ->  {resolved.name}")
    for m in n.get("mode", []):
        print(f"  - {m}")
    print(f"  top-level fields: {len(n.get('top_level_fields', []))}"
          f"  (common {len(n.get('common_fields', []))}, LOB {len(n.get('lob_fields', []))})")
    if n.get("clashes"):
        print(f"  {len(n['clashes'])} clash(es), LOB kept: "
              + ", ".join(c['path'] for c in n['clashes'][:6]))
    if n.get("renamed_defs"):
        print("  renamed defs: " + ", ".join(f"{r['from']}->{r['to']}" for r in n['renamed_defs']))


if __name__ == "__main__":
    raise SystemExit(main())
