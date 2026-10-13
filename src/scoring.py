"""Order-invariant, normalized, per-field scoring against the golden canonical JSON.

Ported from D:/RunPod/eval_fields.py with the loss-run hard-coding removed, plus
the identifier guard and wider date table from D:/POC2/qwen3vl-poc/src/compare.py.

WHY NOT A SIMPLE FLATTEN-AND-COMPARE
------------------------------------
Flattening to dotted paths that include list indices
(`vehicles[0].vin`) and scoring a leaf correct only on exact path + value means:
  - one missing/extra/reordered vehicle shifts EVERY downstream index, so recall
    collapses for a near-perfect extraction;
  - "$47,149.00" != "47149" and "9/1/2026" != "2026-09-01", so formatting counts
    as a content error.
Both make real quality differences between the models invisible, which is the
one thing this POC exists to see. So:

  * ENTITY MATCHING BY CONTENT, NOT POSITION. Lists named in
    scoring.entity_lists are aligned by identity key (vin, policy_number,
    location_number...), falling back to value similarity, and only then to
    position. Reordering a vehicle affects that vehicle's fields and nothing else.
  * VALUE NORMALISATION. Money ($47,149.00 == 47149), dates -> ISO
    (9/1/2026 == 2026-09-01), strings case-folded.
  * NULL == ABSENT. null / "" / "N/A" / "not provided" all mean the same thing,
    and an absent gold value is not counted as a gold leaf.
  * IDENTIFIERS STAY STRINGS. zip "01234" must not score equal to 1234, and NAIC
    "035882" must not equal 35882 - unless the gold itself is a number there.

strict_score() runs the harsh index-path exact-match metric alongside. The GAP
between the two is the interesting number: it separates real extraction error
from formatting and ordering drift.
"""

from __future__ import annotations

import re
from collections import defaultdict
from datetime import datetime
from itertools import zip_longest
from typing import Any, Callable, Iterable

# --------------------------------------------------------------------------- config


DEFAULT_ABSENT = {"", "n/a", "na", "none", "null", "-", "--", "nil",
                  "not stated", "not provided"}

# Wider than eval_fields' six formats: the declaration prints M/D/YYYY, a gold
# file may be ISO, and a model may volunteer "September 1, 2026".
DATE_FORMATS = (
    "%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%m-%d-%Y", "%d-%m-%Y", "%Y/%m/%d",
    "%m/%d/%y", "%d-%b-%Y", "%d %B %Y", "%d %b %Y", "%B %d, %Y", "%b %d, %Y",
    "%B %d %Y", "%b %d %Y", "%d.%m.%Y", "%Y%m%d",
)

MONEY_RE = re.compile(r"^[^\d\-+(]*[(]?([-+]?\d[\d,\s']*(?:\.\d+)?)[)]?\s*[A-Za-z$\u20ac\u00a3\u00a5%]{0,3}$")

DEFAULT_IDENTIFIER_RE = (
    r"(number|naic|zip|vin|license|code|account|territory|sequence"
    r"|certificate|identifier|_id$|_no$)"
)
DEFAULT_DATE_RE = r"(date|dated|expiry|expiration|effective|inception|birth|printed_on|issued)"


class NormOpts:
    """Everything normalisation needs, resolved once from config."""

    __slots__ = ("absent", "id_re", "date_re", "value_keys")

    def __init__(self, cfg: dict | None = None):
        sc = ((cfg or {}).get("scoring") or {})
        self.absent = {str(s).casefold() for s in (sc.get("absent_strings") or DEFAULT_ABSENT)}
        self.id_re = re.compile(sc.get("identifier_key_regex") or DEFAULT_IDENTIFIER_RE, re.I)
        self.date_re = re.compile(sc.get("date_key_regex") or DEFAULT_DATE_RE, re.I)
        # An object carrying one of these keys is ONE value in a wrapper (a
        # FieldValue {raw, parsed, confidence, page_ref}); where a single value is
        # needed - an identity key, a digest label - the first non-null of them
        # is used.
        self.value_keys = tuple(str(k) for k in (sc.get("value_keys") or ("parsed", "raw")))


DEFAULT_OPTS = NormOpts()


def unwrap(value: Any, opts: NormOpts = DEFAULT_OPTS) -> Any:
    """A wrapped value's own value ({raw: "X", parsed: "x", ...} -> "x"); else as is."""
    if isinstance(value, dict) and any(k in value for k in opts.value_keys):
        for k in opts.value_keys:
            if value.get(k) is not None:
                return value[k]
        return None
    return value


_SEG_SUFFIX = re.compile(r"(\[\d*\])*(#count)?$")


def segments(spec: str) -> list[str]:
    """Field names along a spec path: 'a[].b.page_ref[0]' -> ['a', 'b', 'page_ref']."""
    return [_SEG_SUFFIX.sub("", s) for s in str(spec).split(".") if s]


# --------------------------------------------------------------------------- normalisation


MIN_YEAR, MAX_YEAR = 1900, 2100


def as_date(value: Any) -> str | None:
    """Parse a date-looking string to ISO 'YYYY-MM-DD', else None.

    Tolerates a trailing time or zone ('12/11/2020 10:44 EST' -> '2020-12-11').

    The year is range-checked, and that check is load-bearing rather than
    cosmetic: '%Y%m%d' happily reads the NAIC code '035882' as the year 358, and
    an 8-digit claim number as a date. Anything outside 1900-2100 is not a date
    on an insurance document, so rejecting it stops a bare number being silently
    reinterpreted.
    """
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    candidates = (text, text.split()[0]) if " " in text else (text,)
    for candidate in candidates:
        for fmt in DATE_FORMATS:
            try:
                d = datetime.strptime(candidate, fmt).date()
            except ValueError:
                continue
            if MIN_YEAR <= d.year <= MAX_YEAR:
                return d.isoformat()
    try:
        d = datetime.fromisoformat(text.replace("Z", "")).date()
    except ValueError:
        return None
    return d.isoformat() if MIN_YEAR <= d.year <= MAX_YEAR else None


def as_number(value: str) -> float | None:
    """Money / plain number -> float. '$47,149.00' -> 47149.0, '(500)' -> -500.0."""
    m = MONEY_RE.match(value.strip())
    if not m:
        return None
    digits = re.sub(r"[,\s']", "", m.group(1))
    negate = value.strip().startswith("(") and value.strip().endswith(")")
    try:
        n = round(float(digits), 2)
    except ValueError:
        return None
    return -n if (negate and n > 0) else n


def _clean_str(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip(" .,;:").casefold()


def norm_leaf(path: str, value: Any, keep_identifier: bool = True,
              opts: NormOpts = DEFAULT_OPTS) -> Any:
    """Map one raw leaf to its comparable form. None means 'absent'.

    keep_identifier stops identifier-named fields being read as numbers, so zip
    "01234" is not scored equal to 1234. Callers turn it off when the GOLD value
    is itself a number - a genuinely numeric "sequence" field should compare
    numerically, and only the gold can settle which it is.
    """
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return round(float(value), 2)
    if isinstance(value, (dict, list)):
        return value
    s = str(value).strip()
    if s.casefold() in opts.absent:
        return None
    if opts.date_re.search(path):
        iso = as_date(s)
        if iso is not None:
            return iso
    if keep_identifier and opts.id_re.search(path):
        return _clean_str(s)
    number = as_number(s)
    if number is not None:
        return number
    iso = as_date(s)
    if iso is not None:
        return iso
    return _clean_str(s)


# --------------------------------------------------------------------------- spec paths


def child_spec(spec: str, key: str) -> str:
    """Spec path of a dict child. '' + 'states' -> 'states'."""
    return f"{spec}.{key}" if spec else str(key)


def item_spec(spec: str) -> str:
    """Spec path of a list item. 'states' -> 'states[]'."""
    return f"{spec}[]"


class ScoreSpec:
    """Which lists are entities, how to key them, and what to ignore."""

    __slots__ = ("entity_lists", "auto", "ignore", "ignore_keys", "page_ref_keys",
                 "separate", "keyed_maps", "opts")

    def __init__(self, cfg: dict | None = None):
        sc = ((cfg or {}).get("scoring") or {})
        raw = sc.get("entity_lists") or {}
        self.entity_lists: dict[str, list[list[str]]] = {
            str(k): [list(group) for group in (v or [])] for k, v in raw.items()
        }
        self.auto = bool(sc.get("auto_entity_lists", True))
        self.ignore = set(str(p) for p in (sc.get("ignore_paths") or []))
        # Field NAMES ignored wherever they occur (confidence, flagged...).
        self.ignore_keys = set(str(k) for k in (sc.get("ignore_keys") or []))
        # Where a value is printed: scored separately, never part of identity.
        self.page_ref_keys = set(str(k) for k in (sc.get("page_ref_keys") or []))
        # Top-level sections scored separately from the main F1.
        self.separate = [str(p) for p in (sc.get("separate_sections") or [])]
        # Objects keyed by generated ids, scored as lists matched by content.
        self.keyed_maps = [str(p) for p in (sc.get("keyed_maps") or [])]
        self.opts = NormOpts(cfg)

    def group(self, spec: str) -> str | None:
        """Which separate score a field belongs to; None = the main F1."""
        for p in self.separate:
            if spec == p or spec.startswith(p + ".") or spec.startswith(p + "["):
                return p
        if self.page_ref_keys and any(s in self.page_ref_keys for s in segments(spec)):
            return "page_references"
        return None

    def is_page_ref(self, spec: str) -> bool:
        return bool(self.page_ref_keys) and any(s in self.page_ref_keys for s in segments(spec))

    def is_entity_list(self, spec: str, *values: Any) -> bool:
        """Any list holding dicts is an entity list, always.

        This is deliberately NOT gated on auto_entity_lists. A list of objects
        that is neither configured nor recognised would otherwise be skipped by
        scalar_leaves AND never descended into by tally - it would vanish from
        the metric entirely rather than score badly, which is the worst possible
        failure mode for a benchmark. auto_entity_lists controls only whether
        align() may fall back to value-similarity matching.
        """
        if spec in self.entity_lists:
            return True
        for v in values:
            if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                return True
        return False

    def keyfuncs(self, spec: str) -> list[Callable[[dict], str | None]]:
        return make_keyfuncs(self.entity_lists.get(spec, []), self.opts)

    def ignored(self, spec: str) -> bool:
        if spec in self.ignore:
            return True
        if self.ignore_keys and any(s in self.ignore_keys for s in segments(spec)):
            return True
        return any(spec.startswith(p + ".") or spec.startswith(p + "[]") for p in self.ignore)


def dig(obj: Any, dotted: str) -> Any:
    """Fetch a possibly-dotted field ('address.street') out of a dict."""
    cur = obj
    for part in str(dotted).split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def make_keyfuncs(key_specs: Iterable[Iterable[str]],
                  opts: NormOpts = DEFAULT_OPTS) -> list[Callable[[dict], str | None]]:
    """Turn config key groups into identity functions.

    Each group is an AND-group of field names; the key is their normalized values
    joined. A group yields None when EVERY component is absent, so a partially
    filled entity can still be keyed - matching eval_fields._k_claim_loss, which
    keys on date_of_loss OR claimant_name.
    """
    funcs: list[Callable[[dict], str | None]] = []
    for group in key_specs:
        fields = [str(f) for f in group]

        def kf(entity: dict, _fields=tuple(fields)) -> str | None:
            parts = []
            seen_any = False
            for f in _fields:
                # A wrapped field compares by its value, never by its page_ref
                # or confidence riding along in the wrapper.
                v = norm_leaf(f, unwrap(dig(entity, f), opts), True, opts)
                if v is not None:
                    seen_any = True
                parts.append("" if v is None else str(v))
            return ("|".join(_fields) + "=" + "|".join(parts)) if seen_any else None

        funcs.append(kf)
    return funcs


# --------------------------------------------------------------------------- leaves


def scalar_leaves(obj: Any, spec: str, sspec: ScoreSpec) -> dict[str, Any]:
    """Raw scalar leaves of one entity, keyed by full spec path.

    Entity lists are NOT descended into - they are handled by alignment, which is
    the whole point. Leaves whose value means 'absent' are omitted, so a null
    gold field is not counted against anyone.

    Lists of plain scalars are indexed in SORTED normalized order rather than
    document order, so reordering a list of strings is not scored as two errors.
    """
    out: dict[str, Any] = {}
    if not isinstance(obj, dict):
        return out
    for k, v in obj.items():
        key = child_spec(spec, k)
        if sspec.ignored(key):
            continue
        if isinstance(v, dict):
            out.update(scalar_leaves(v, key, sspec))
        elif isinstance(v, list):
            if sspec.is_entity_list(key, v):
                continue
            scalars = []
            for x in v:
                if isinstance(x, dict):
                    continue  # a dict inside a non-entity list: handled by alignment
                nv = norm_leaf(key, x, True, sspec.opts)
                if nv is not None:
                    scalars.append((str(nv), x))
            scalars.sort(key=lambda t: t[0])
            for i, (_, raw) in enumerate(scalars):
                out[f"{key}[{i}]"] = raw
        else:
            if norm_leaf(key, v, True, sspec.opts) is not None:
                out[key] = v
    return out


def _norm_map(obj: Any, spec: str, sspec: ScoreSpec) -> dict[str, Any]:
    """Normalized leaves, used only for value-similarity matching in align()."""
    return {k: norm_leaf(k, v, True, sspec.opts)
            for k, v in scalar_leaves(obj, spec, sspec).items()}


# --------------------------------------------------------------------------- alignment


def align(gs: list, ps: list, keyfuncs: list, spec: str,
          sspec: ScoreSpec) -> list[tuple[dict | None, dict | None]]:
    """Pair gold vs predicted entities ORDER-INDEPENDENTLY.

    Ported from eval_fields._align. A key is used only when it is UNIQUE on both
    sides, so three vehicles sharing a blank VIN fall through to the next key
    rather than being cross-paired. Whatever is left is matched by value
    similarity (most equal scalar leaves), and only what survives that is paired
    positionally.
    """
    gs = [x for x in gs if isinstance(x, dict)]
    ps = [x for x in ps if isinstance(x, dict)]
    gleft, pleft = list(range(len(gs))), list(range(len(ps)))
    pairs: list[tuple[dict | None, dict | None]] = []

    for kf in keyfuncs:
        gkeys: dict[str, list[int]] = defaultdict(list)
        pkeys: dict[str, list[int]] = defaultdict(list)
        for i in gleft:
            k = kf(gs[i])
            if k is not None:
                gkeys[k].append(i)
        for j in pleft:
            k = kf(ps[j])
            if k is not None:
                pkeys[k].append(j)
        for k, gidxs in gkeys.items():
            pidxs = pkeys.get(k)
            if len(gidxs) == 1 and pidxs and len(pidxs) == 1:  # unambiguous only
                gi, pj = gidxs[0], pidxs[0]
                pairs.append((gs[gi], ps[pj]))
                gleft.remove(gi)
                pleft.remove(pj)

    istub = item_spec(spec)
    g_leaves = {i: _norm_map(gs[i], istub, sspec) for i in gleft}
    p_leaves = {j: _norm_map(ps[j], istub, sspec) for j in pleft}
    while gleft and pleft and sspec.auto:
        best = (0, None, None)
        for i in gleft:
            gl = g_leaves[i]
            for j in pleft:
                pl = p_leaves[j]
                score = sum(1 for f, v in gl.items() if pl.get(f) == v)
                if score > best[0]:
                    best = (score, i, j)
        if best[0] <= 0:
            break
        _, gi, pj = best
        pairs.append((gs[gi], ps[pj]))
        gleft.remove(gi)
        pleft.remove(pj)

    # Last resort: pair the remainder positionally. Handles entities with no
    # distinguishing key OR value; their children re-align by their own keys in
    # the recursion, so this stays order-robust in practice.
    for gi, pj in zip_longest(gleft, pleft):
        pairs.append((gs[gi] if gi is not None else None,
                      ps[pj] if pj is not None else None))
    return pairs


# --------------------------------------------------------------------------- tally


# counts per field spec path: [matched, gold_present, pred_present]
Counts = dict


def tally(gold: Any, pred: Any, counts: Counts, spec: str, sspec: ScoreSpec,
          list_counts: dict) -> None:
    """Recursively score one aligned entity pair, then descend into its entity lists.

    Replaces eval_fields.tally_doc's hard-coded carriers->policies->claims walk.
    """
    gl = scalar_leaves(gold, spec, sspec)
    pl = scalar_leaves(pred, spec, sspec)

    for f, gv in gl.items():
        counts.setdefault(f, [0, 0, 0])[1] += 1
        if f in pl:
            # Only the gold can settle whether a field is an identifier or a
            # number: if the gold is a number there, compare numerically.
            keep_id = isinstance(gv, str)
            if norm_leaf(f, gv, keep_id, sspec.opts) == norm_leaf(f, pl[f], keep_id, sspec.opts):
                counts[f][0] += 1
    for f in pl:
        counts.setdefault(f, [0, 0, 0])[2] += 1

    gd = gold if isinstance(gold, dict) else {}
    pd = pred if isinstance(pred, dict) else {}
    for k in dict.fromkeys(list(gd.keys()) + list(pd.keys())):
        key = child_spec(spec, k)
        if sspec.ignored(key):
            continue
        gv, pv = gd.get(k), pd.get(k)
        if not (isinstance(gv, list) or isinstance(pv, list)):
            if isinstance(gv, dict) or isinstance(pv, dict):
                # Nested plain objects are covered by scalar_leaves, but they can
                # themselves hold entity lists, so keep descending.
                _descend_dict(gv, pv, counts, key, sspec, list_counts)
            continue
        if not sspec.is_entity_list(key, gv, pv):
            continue
        gsl = gv if isinstance(gv, list) else []
        psl = pv if isinstance(pv, list) else []
        entry = list_counts.setdefault(key, {"gold": 0, "pred": 0})
        entry["gold"] += len([x for x in gsl if isinstance(x, dict)])
        entry["pred"] += len([x for x in psl if isinstance(x, dict)])
        for g_item, p_item in align(gsl, psl, sspec.keyfuncs(key), key, sspec):
            tally(g_item, p_item, counts, item_spec(key), sspec, list_counts)


def _descend_dict(gv: Any, pv: Any, counts: Counts, key: str, sspec: ScoreSpec,
                  list_counts: dict) -> None:
    """Walk into a nested plain object looking only for entity lists inside it."""
    gd = gv if isinstance(gv, dict) else {}
    pd = pv if isinstance(pv, dict) else {}
    for k in dict.fromkeys(list(gd.keys()) + list(pd.keys())):
        sub = child_spec(key, k)
        if sspec.ignored(sub):
            continue
        a, b = gd.get(k), pd.get(k)
        if isinstance(a, list) or isinstance(b, list):
            if not sspec.is_entity_list(sub, a, b):
                continue
            gsl = a if isinstance(a, list) else []
            psl = b if isinstance(b, list) else []
            entry = list_counts.setdefault(sub, {"gold": 0, "pred": 0})
            entry["gold"] += len([x for x in gsl if isinstance(x, dict)])
            entry["pred"] += len([x for x in psl if isinstance(x, dict)])
            for g_item, p_item in align(gsl, psl, sspec.keyfuncs(sub), sub, sspec):
                tally(g_item, p_item, counts, item_spec(sub), sspec, list_counts)
        elif isinstance(a, dict) or isinstance(b, dict):
            _descend_dict(a, b, counts, sub, sspec, list_counts)


def prf(c: list[int]) -> tuple[float, float, float]:
    matched, gold, pred = c
    recall = matched / gold if gold else (1.0 if pred == 0 else 0.0)
    precision = matched / pred if pred else (1.0 if gold == 0 else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return precision, recall, f1


def overall(counts: Counts) -> tuple[float, float, float]:
    """Micro-average across every field."""
    tot = [0, 0, 0]
    for c in counts.values():
        tot[0] += c[0]
        tot[1] += c[1]
        tot[2] += c[2]
    return prf(tot)


# --------------------------------------------------------------------------- strict


def flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten to dot/bracket leaf paths. Empty containers become explicit leaves.

    Ported from compare.flatten. Index-sensitive by design - that is what makes
    strict_score strict.
    """
    flat: dict[str, Any] = {}
    if isinstance(obj, dict):
        if not obj:
            flat[prefix or "<root>"] = {}
            return flat
        for key, value in obj.items():
            flat.update(flatten(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, list):
        flat[f"{prefix}[]#count"] = len(obj)
        for i, value in enumerate(obj):
            flat.update(flatten(value, f"{prefix}[{i}]"))
    else:
        flat[prefix or "<root>"] = obj
    return flat


def strict_score(gold: dict, candidate: Any, opts: NormOpts = DEFAULT_OPTS,
                 sspec: "ScoreSpec | None" = None) -> dict:
    """Index-path exact match. The harsh read, reported alongside the field F1.

    The GAP between this and field_f1 is the point: it says how much of the
    difference between two cells is ordering and formatting rather than content.
    Covers the same fields as field_f1: ignored and separately scored fields
    are left out.
    """
    def keep(path: str) -> bool:
        if sspec is None:
            return True
        spec = re.sub(r"\[\d+\]", "[]", path)
        return not sspec.ignored(spec) and sspec.group(spec) is None

    gold_flat = {p: v for p, v in flatten(gold).items() if keep(p)}
    cand_flat = ({p: v for p, v in flatten(candidate).items() if keep(p)}
                 if isinstance(candidate, (dict, list)) else {})

    matches = 0
    for path, gold_value in gold_flat.items():
        if path not in cand_flat:
            continue
        keep_id = isinstance(gold_value, str)
        if norm_leaf(path, gold_value, keep_id, opts) == norm_leaf(path, cand_flat[path], keep_id, opts):
            matches += 1

    total = len(gold_flat)
    return {
        "total_fields": total,
        "matched_fields": matches,
        "match_rate": round(matches / total, 4) if total else 0.0,
        "extra_fields": len(set(cand_flat) - set(gold_flat)),
    }


# --------------------------------------------------------------------------- coverage


STOPWORDS = frozenset("""the a an and or of to in for on at by with from as is are was were
be been this that these those any all not no if then than which who whom whose you your we
our it its will shall may can must do does did have has had""".split())

WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")
NUMBER_RE = re.compile(r"\d[\d,./\-]*")


def _norm_number_token(token: str) -> str:
    return re.sub(r"[,\s$]", "", token).rstrip(".")


def content_tokens(text: str) -> tuple[set[str], set[str]]:
    """(words, numbers) worth checking for. Numbers carry most of the meaning here."""
    words = {w.lower() for w in WORD_RE.findall(text)} - STOPWORDS
    numbers = {n for n in (_norm_number_token(t) for t in NUMBER_RE.findall(text)) if len(n) >= 2}
    return words, numbers


def collect_values(obj: Any) -> str:
    """Every value in the extracted object, flattened to one searchable blob."""
    parts: list[str] = []

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                parts.append(str(key))
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
        elif node is not None:
            parts.append(str(node))

    walk(obj)
    return " ".join(parts)


def coverage_report(ocr_text: str, extracted: Any) -> dict:
    """How much of what OCR read off the pages actually reached the output.

    Ported verbatim from compare.coverage_report. Measured against the DOCUMENT,
    not the gold, so it catches content the gold and the model both dropped. This
    is the key diagnostic for the MD-only case: if number coverage collapses
    there, the markdown lost the table, not the model.

    Caveats that matter: OCR noise produces tokens nothing will ever match, so
    100% is unreachable, and presence is not correctness - a value in the wrong
    field still counts as covered. Read the delta between cells, not the absolute.
    """
    if extracted is None or not (ocr_text or "").strip():
        return {}
    ocr_words, ocr_numbers = content_tokens(ocr_text)
    out_words, out_numbers = content_tokens(collect_values(extracted))

    missing_numbers = sorted(ocr_numbers - out_numbers)
    missing_words = sorted(ocr_words - out_words)
    return {
        "ocr_words": len(ocr_words),
        "ocr_numbers": len(ocr_numbers),
        "word_coverage": round(len(ocr_words & out_words) / len(ocr_words), 4) if ocr_words else None,
        "number_coverage": round(len(ocr_numbers & out_numbers) / len(ocr_numbers), 4)
        if ocr_numbers else None,
        "missing_numbers": missing_numbers[:60],
        "missing_numbers_total": len(missing_numbers),
        "missing_words_sample": missing_words[:60],
        "missing_words_total": len(missing_words),
    }


# --------------------------------------------------------------------------- entry point


def listify_keyed_maps(obj: Any, paths: list[str]) -> Any:
    """Objects keyed by generated ids -> lists of their values, at the given paths.

    A gold section index like {"FORM 01 23|12 Example Road|": {...}} is keyed by
    a string the model would have to reproduce character for character. As a list
    its entries are matched by content, like any other entity list.
    """
    if not paths or not isinstance(obj, dict):
        return obj
    out = dict(obj)
    for p in paths:
        head, _, rest = p.partition(".")
        if head not in out:
            continue
        if rest:
            out[head] = listify_keyed_maps(out[head], [rest])
        elif isinstance(out[head], dict) and all(isinstance(v, dict) for v in out[head].values()):
            out[head] = list(out[head].values())
    return out


def score_document(gold: dict, pred: Any, cfg: dict | None = None,
                   ocr_text: str | None = None) -> dict:
    """Score one prediction against one gold. The primary metric is field_f1.

    field_f1 covers the document's content only. Fields named in
    scoring.ignore_keys (extraction metadata such as confidence) are not scored
    at all; page references (scoring.page_ref_keys) and whole sections named in
    scoring.separate_sections get their own scores under "separate", so they
    are reported without diluting the main number.
    """
    sspec = ScoreSpec(cfg)
    counts: Counts = {}
    list_counts: dict = {}
    gold = listify_keyed_maps(gold, sspec.keyed_maps)
    pred = listify_keyed_maps(pred, sspec.keyed_maps) if isinstance(pred, dict) else pred

    if isinstance(pred, dict):
        tally(gold, pred, counts, "", sspec, list_counts)
    else:
        # Unparseable output still gets a gold-side tally, so recall is 0 rather
        # than the cell silently vanishing from the report.
        tally(gold, {}, counts, "", sspec, list_counts)

    main = {f: c for f, c in counts.items() if sspec.group(f) is None}
    separate: dict[str, dict] = {}
    for f, c in counts.items():
        g = sspec.group(f)
        if g is not None:
            separate.setdefault(g, {})[f] = c

    precision, recall, f1 = overall(main)
    per_field = {
        f: {
            "matched": c[0], "gold": c[1], "pred": c[2],
            "precision": round(prf(c)[0], 4),
            "recall": round(prf(c)[1], 4),
            "f1": round(prf(c)[2], 4),
            **({"group": sspec.group(f)} if sspec.group(f) else {}),
        }
        for f, c in sorted(counts.items())
    }
    missing = sorted(
        (f for f, c in main.items() if c[1] > 0 and c[0] == 0),
        key=lambda f: (-main[f][1], f),
    )
    result = {
        "field_precision": round(precision, 4),
        "field_recall": round(recall, 4),
        "field_f1": round(f1, 4),
        "fields_scored": len(main),
        "gold_leaves": sum(c[1] for c in main.values()),
        "pred_leaves": sum(c[2] for c in main.values()),
        "matched_leaves": sum(c[0] for c in main.values()),
        "separate": {
            g: {"precision": round(overall(cs)[0], 4), "recall": round(overall(cs)[1], 4),
                "f1": round(overall(cs)[2], 4),
                "gold_leaves": sum(c[1] for c in cs.values()),
                "pred_leaves": sum(c[2] for c in cs.values()),
                "matched_leaves": sum(c[0] for c in cs.values())}
            for g, cs in sorted(separate.items())
        },
        "list_counts": {
            k: {**v, "delta": v["pred"] - v["gold"]}
            for k, v in sorted(list_counts.items())
        },
        "fully_missing_fields": missing,
        "per_field": per_field,
    }

    sc = ((cfg or {}).get("scoring") or {})
    if sc.get("strict_score", True):
        result["strict"] = strict_score(gold, pred, sspec.opts, sspec)
    if sc.get("coverage_report", True) and ocr_text:
        result["coverage"] = coverage_report(ocr_text, pred)
    return result
