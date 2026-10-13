"""Page windows, the context each window carries, and merging the window outputs.

A long policy does not go to the model in one request. It is cut into windows of
`chunking.window_pages` pages, each overlapping the previous one by
`chunking.overlap_pages`, and every window is extracted against the SAME schema.
The window outputs are merged into one canonical JSON. The same windows are used
for all three cases, so the case comparison is unaffected by the split.

KEEPING THE CONTEXT OF THE WHOLE DOCUMENT
-----------------------------------------
A window on its own does not know which policy it belongs to, which vehicle a
continuation row describes, or what a schedule's columns are if the heading was
two pages back. Four things carry that context across the split:

  * Overlap. The last page of one window is the first page of the next.
  * Section-aware boundaries (plan_windows). A boundary may move by up to
    section_boundaries.max_shift_pages so the next window starts where a new
    section starts - a heading at the top of a page - rather than mid-table.
    The page signals come from MinerU's markdown; the plan is computed once per
    document and shared by all three cases.
  * Reference pages (context.anchor_pages, applied in runner/prompting). The
    declarations page is shown at the start of every later window as reference
    only, in the case's own modality.
  * A digest of what earlier windows extracted (context_digest). Each window
    is told the policy-level values and every entity found so far by its
    identifying fields, and to reuse those exact names and numbers. Built from
    the model's own merged output, so every case gets the same kind of help and
    PDF only never sees MinerU text.

HOW THE MERGE DECIDES
---------------------
  * A field only one window filled: that value.
  * A field two windows filled with the same value (after the scorer's own
    normalisation - "$1,000" == 1000, "9/1/2026" == "2026-09-01"): that value.
  * A field two windows filled DIFFERENTLY: chunking.scalar_conflict decides
    (default: the earlier window, where the declarations usually are), and the
    clash is recorded - never resolved silently.
  * Lists of objects: concatenated, except that an item from a later window that
    is the SAME entity as one already collected is merged into it instead of
    being added twice. Same entity means: equal on a configured identity key
    (scoring.entity_lists), or, failing that, agreeing on every field both
    filled in - at least two fields, or one identifier-like field (a number,
    VIN, code...). Overlap pages and the declarations-vs-schedule repeat
    produce exactly this case.
  * Items from the SAME window are never merged with each other: the model saw
    them side by side, so two rows there are two rows.
  * Lists of plain values: union, in first-seen order - so a value's page_ref
    collects every page it was seen on.
  * Bookkeeping never decides anything: a value wrapper's confidence/flagged
    (scoring.ignore_keys) and its page_ref (scoring.page_ref_keys) differ
    between windows by design, so they are left out of the same-entity test and
    a difference in them is not recorded as a conflict. A wrapper whose own
    value is null counts as absent, whatever bookkeeping rides along with it.
"""

from __future__ import annotations

import copy
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import scoring

# --------------------------------------------------------------------------- windows


def plan_windows(n_pages: int, size: int, overlap: int = 0,
                 signals: list[int] | None = None,
                 max_shift: int = 0) -> list[tuple[int, int]]:
    """Inclusive 1-based (first, last) windows, boundaries nudged towards sections.

    >>> plan_windows(25, 10, 1)
    [(1, 10), (10, 19), (19, 25)]

    signals[i] describes how file page i+1 OPENS: 0 = with a new section, 1 =
    nothing notable, 2 = in the middle of a table. Each boundary may move by up
    to max_shift pages; the page that will open the next window is chosen by
    (signal, distance from the nominal boundary), so with no useful signal the
    plan is exactly the fixed one.
    """
    if n_pages <= 0:
        return []
    size = max(1, int(size))
    overlap = max(0, min(int(overlap), size - 1))
    shift = max(0, int(max_shift)) if signals and len(signals) >= n_pages else 0
    out: list[tuple[int, int]] = []
    first = 1
    while True:
        nominal = first + size - 1
        if nominal >= n_pages:
            out.append((first, n_pages))
            return out
        last = nominal
        if shift:
            lo = max(first + overlap + 1, nominal - shift)   # always make progress
            hi = min(n_pages - 1, nominal + shift)
            best = None
            for cand in range(lo, hi + 1):
                opens = signals[cand]          # page cand+1 opens the next window's new part
                key = (opens, abs(cand - nominal), cand)
                if best is None or key < best[0]:
                    best = (key, cand)
            if best:
                last = best[1]
        out.append((first, last))
        first = last - overlap + 1


_PAGE_NO = re.compile(r"^(page\s*\d+(\s*(of|/)\s*\d+)?|\d+\s*(of|/)\s*\d+|-?\s*\d+\s*-?)$", re.I)


def _norm_line(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d+", "#", s)).strip().casefold()


def page_start_signals(md_pages: list[str]) -> list[int]:
    """How each page opens: 0 = new section, 1 = neutral, 2 = mid-table.

    Running headers and footers (a line that recurs at the top of many pages,
    page numbers) are skipped first, otherwise every page would look like it
    opens with the same "heading".
    """
    tops = []
    for md in md_pages:
        lines = [ln.strip() for ln in (md or "").splitlines() if ln.strip()]
        tops.append(lines[:6])
    freq = Counter(n for lines in tops for n in {_norm_line(x) for x in lines})
    recurring = {n for n, c in freq.items() if c >= max(3, int(0.3 * len(md_pages)))}

    out = []
    for lines in tops:
        first = next((ln for ln in lines
                      if _norm_line(ln) not in recurring and not _PAGE_NO.match(ln)), "")
        low = first.lower()
        letters = sum(ch.isalpha() for ch in first)
        if not first:
            out.append(1)
        elif low.startswith(("<table", "<tr", "<td", "|")):
            out.append(2)
        elif first.startswith("#") or (first.startswith("**") and first.endswith("**")):
            out.append(0)
        elif 4 <= letters and len(first) <= 80 and first.upper() == first:
            out.append(0)
        else:
            out.append(1)
    return out


def plan_for_document(cfg: dict, ocr_dir: Path, n_pages: int) -> tuple[list[tuple[int, int]], str]:
    """(windows, how they were planned) for one document.

    Section-aware when enabled and every page's markdown is on disk, else fixed.
    Deterministic from the cached markdown, so every case and every re-run of a
    document gets the same plan - and report.prompt_integrity() checks it.
    """
    ch = cfg.get("chunking") or {}
    size, overlap = int(ch.get("window_pages", 10)), int(ch.get("overlap_pages", 0))
    sec = ch.get("section_boundaries") or {}
    if not sec.get("enabled") or n_pages <= size:
        return plan_windows(n_pages, size, overlap), "fixed"
    mds = [Path(ocr_dir) / f"page_{i}.md" for i in range(1, n_pages + 1)]
    if not all(p.exists() for p in mds):
        return plan_windows(n_pages, size, overlap), "fixed (no markdown to find sections in)"
    signals = page_start_signals([p.read_text(encoding="utf-8") for p in mds])
    wins = plan_windows(n_pages, size, overlap, signals, int(sec.get("max_shift_pages", 2)))
    return wins, "section-aware"


def window_label(idx: int, first: int, last: int) -> str:
    return f"w{idx:02d}_p{first:03d}-{last:03d}"


# --------------------------------------------------------------------------- merge


class Merger:
    """Merges window outputs one at a time, in window order.

    Incremental on purpose: the digest handed to window k is built from the
    merge of windows 1..k-1, which is exactly this object's state at that point.
    """

    def __init__(self, cfg: dict):
        self.sspec = scoring.ScoreSpec(cfg)
        self.opts = self.sspec.opts
        self.keep = str((cfg.get("chunking") or {}).get("scalar_conflict", "first")).lower()
        self.conflicts: list[dict] = []
        self.merged_entities = 0
        self.appended_entities = 0
        self.merged: Any = None
        self.used: list[str] = []
        self.skipped: list[str] = []

    # ------------------------------------------------------------ helpers

    def absent(self, v: Any) -> bool:
        if v is None:
            return True
        if isinstance(v, str):
            return v.strip().casefold() in self.opts.absent
        if isinstance(v, list):
            return all(self.absent(x) for x in v)
        if isinstance(v, dict):
            if self.is_wrapper(v):
                # {raw: null, parsed: null, confidence: {...}, page_ref: []} is an
                # absent VALUE, whatever bookkeeping rides along with it.
                return self.absent(scoring.unwrap(v, self.opts))
            return all(self.absent(x) for x in v.values())
        return False

    def is_wrapper(self, v: Any) -> bool:
        return isinstance(v, dict) and any(k in v for k in self.opts.value_keys)

    def bookkeeping(self, spec: str) -> bool:
        """Ignored metadata (confidence...) or a page reference: never identity,
        never a conflict worth reporting."""
        return self.sspec.ignored(spec) or self.sspec.is_page_ref(spec)

    def norm(self, path: str, v: Any) -> Any:
        return scoring.norm_leaf(path, v, True, self.opts)

    # ------------------------------------------------------------ incremental API

    def add(self, label: str, obj: Any) -> None:
        if not isinstance(obj, dict):
            self.skipped.append(label)
            return
        if self.merged is None:
            self.merged = copy.deepcopy(obj)
        else:
            self.merged = self.merge(self.merged, obj, "", label)
        self.used.append(label)

    def report(self) -> dict:
        return {
            "windows_merged": list(self.used),
            "windows_skipped": list(self.skipped),
            "entities_merged_across_windows": self.merged_entities,
            "entities_added_by_later_windows": self.appended_entities,
            "conflict_count": len(self.conflicts),
            "conflicts": self.conflicts[:200],
        }

    # ------------------------------------------------------------ merge

    def merge(self, a: Any, b: Any, spec: str, wb: str) -> Any:
        """Merge b (from window wb) into a (collected from earlier windows)."""
        if self.absent(b):
            return a if a is not None or b is None else b
        if self.absent(a):
            return copy.deepcopy(b)
        if isinstance(a, dict) and isinstance(b, dict):
            out = dict(a)
            for k, bv in b.items():
                key = scoring.child_spec(spec, k)
                out[k] = self.merge(a[k], bv, key, wb) if k in a else copy.deepcopy(bv)
            return out
        if isinstance(a, list) and isinstance(b, list):
            if a + b and all(self.is_wrapper(x) for x in a + b):
                return self.merge_values(a, b, spec, wb)
            if any(isinstance(x, dict) for x in a + b):
                return self.merge_entities(a, b, spec, wb)
            seen = {str(self.norm(spec, x)) for x in a}
            out = list(a)
            for x in b:
                n = str(self.norm(spec, x))
                if n not in seen:
                    seen.add(n)
                    out.append(copy.deepcopy(x))
            return out
        if not isinstance(a, (dict, list)) and not isinstance(b, (dict, list)):
            if self.norm(spec, a) == self.norm(spec, b):
                return a
            if self.bookkeeping(spec):
                return a              # confidence differs between windows: not news
        self.conflicts.append({"path": spec, "earlier_value": a, "later_value": b,
                               "later_window": wb,
                               "kept": "earlier" if self.keep == "first" else "later"})
        return a if self.keep == "first" else copy.deepcopy(b)

    def same_entity(self, x: dict, y: dict, istub: str, keyfuncs: list) -> bool:
        for kf in keyfuncs:
            kx, ky = kf(x), kf(y)
            if kx is not None and ky is not None:
                return kx == ky        # a key both sides carry is decisive
        # Compare the entity's VALUES only: where each window saw it (page_ref)
        # and how sure it was (confidence) differ by design.
        lx = {k: v for k, v in scoring._norm_map(x, istub, self.sspec).items()
              if v is not None and not self.bookkeeping(k)}
        ly = {k: v for k, v in scoring._norm_map(y, istub, self.sspec).items()
              if v is not None and not self.bookkeeping(k)}
        if not lx or not ly:
            return False
        shared = set(lx) & set(ly)
        if not shared or any(lx[k] != ly[k] for k in shared):
            return False
        # A wrapper's raw and parsed are one field, not two.
        fields = {k.rsplit(".", 1)[0] if k.rsplit(".", 1)[-1] in self.opts.value_keys else k
                  for k in shared}
        if len(fields) == 1:
            # A window often sees only one field of an entity (a form's title, an
            # insured's name, an installment's date), so one agreeing field can
            # be identity - if it is an identifier, or its value is distinctive.
            # Two vehicles that agree only on make = "Honda", or two exposures
            # only on state = "NY", stay two.
            field = next(iter(fields))
            if self.opts.id_re.search(scoring.segments(field)[-1]):
                return True
            vals = [str(lx[k]) for k in shared]
            return any(len(v) >= 8 or any(ch.isdigit() for ch in v) for v in vals)
        # Agreeing on incidental fields is not identity either: the overlap must
        # cover at least half of the sparser item.
        return len(shared) >= -(-min(len(lx), len(ly)) // 2)

    def merge_values(self, a: list, b: list, spec: str, wb: str) -> list:
        """A list whose items are each one wrapped value ('states': [{raw, parsed,
        page_ref...}]): a set of values. The same value from two windows is one
        entry, with the pages both saw."""
        out = list(a)
        index = {str(self.norm(spec, scoring.unwrap(x, self.opts))): i for i, x in enumerate(a)}
        for x in b:
            if self.absent(x):
                continue
            key = str(self.norm(spec, scoring.unwrap(x, self.opts)))
            if key in index:
                out[index[key]] = self.merge(out[index[key]], x, scoring.item_spec(spec), wb)
                self.merged_entities += 1
            else:
                index[key] = len(out)
                out.append(copy.deepcopy(x))
                self.appended_entities += 1
        return out

    def merge_entities(self, a: list, b: list, spec: str, wb: str) -> list:
        istub = scoring.item_spec(spec)
        keyfuncs = self.sspec.keyfuncs(spec)
        out = list(a)
        taken: set[int] = set()
        for x in b:
            if not isinstance(x, dict):
                out.append(copy.deepcopy(x))
                continue
            if self.absent(x):
                continue
            hit = None
            for i, y in enumerate(a):           # only items from EARLIER windows
                if i in taken or not isinstance(y, dict):
                    continue
                if self.same_entity(x, y, istub, keyfuncs):
                    hit = i
                    break
            if hit is None:
                out.append(copy.deepcopy(x))
                self.appended_entities += 1
            else:
                taken.add(hit)
                out[hit] = self.merge(out[hit], x, istub, wb)
                self.merged_entities += 1
        return out

    # ------------------------------------------------------------ digest

    def _short(self, v: Any) -> Any:
        """A short plain value to name something by, or None."""
        v = scoring.unwrap(v, self.opts)
        if isinstance(v, (dict, list)) or self.absent(v):
            return None
        if isinstance(v, str) and len(v) > 60:
            return None
        return v

    def _label(self, item: dict, spec: str) -> str:
        """The few fields that identify one entity, for the digest.

        A wrapped field is named by its value (parsed, else raw), never by the
        wrapper's bookkeeping.
        """
        for group in self.sspec.entity_lists.get(spec, []):
            vals = [(f, self._short(scoring.dig(item, f))) for f in group]
            vals = [(f, v) for f, v in vals if v is not None]
            if vals:
                return ", ".join(f"{f}={json.dumps(v, ensure_ascii=False)}" for f, v in vals)
        picked = []
        for k, v in item.items():
            s = scoring.child_spec(scoring.item_spec(spec), k)
            if self.bookkeeping(s) or (isinstance(v, dict) and not self.is_wrapper(v)):
                continue
            v = self._short(v)
            if v is None:
                continue
            picked.append(f"{k}={json.dumps(v, ensure_ascii=False)}")
            if len(picked) == 3:
                break
        return ", ".join(picked) or "(no identifying fields)"

    def digest(self, max_chars: int = 6000, max_items: int = 40) -> str:
        """What the windows so far extracted, compactly, for the next window.

        Values first (a wrapper counts as one value), then every entity list by
        its identifying fields. Bookkeeping (confidence, page references) is left
        out, and a section index (scoring.keyed_maps) is summarised by its size
        and latest entry rather than listed - it is page-local and would crowd
        out the entities.
        """
        if not isinstance(self.merged, dict):
            return ""
        scalars: list[str] = []
        lists: list[str] = []
        keyed = set(self.sspec.keyed_maps)

        def walk(node: dict, spec: str, shown: str, depth: int, in_item: bool) -> None:
            # Inside a list item only NESTED lists are reported: the item's own
            # fields are already named by its label in the entity list.
            for k, v in node.items():
                s = scoring.child_spec(spec, k)
                d = f"{shown}.{k}" if shown else k
                if self.bookkeeping(s):
                    continue
                if s in keyed and isinstance(v, dict):
                    entries = [x for x in v.values() if isinstance(x, dict) and not self.absent(x)]
                    if entries:
                        lists.append(f"- {d}: {len(entries)} so far; the latest is "
                                     "{" + self._label(entries[-1], s) + "}")
                    continue
                if self.is_wrapper(v):
                    val = self._short(v)
                    if not in_item and depth <= 3 and val is not None:
                        scalars.append(f"- {d}: {json.dumps(val, ensure_ascii=False)}")
                elif isinstance(v, dict):
                    walk(v, s, d, depth + 1, in_item)
                elif isinstance(v, list):
                    items = [x for x in v if isinstance(x, dict) and not self.absent(x)]
                    if items:
                        shown_items = items[:max_items]
                        more = len(items) - len(shown_items)
                        lists.append(f"- {d} ({len(items)}): "
                                     + "; ".join("{" + self._label(x, s) + "}" for x in shown_items)
                                     + (f"; ... and {more} more" if more > 0 else ""))
                        for x in shown_items:
                            walk(x, scoring.item_spec(s), f"{d}[{self._label(x, s)}]",
                                 depth + 1, True)
                    elif not in_item and depth <= 2 and v and not self.absent(v):
                        vals = [x for x in v if not isinstance(x, (dict, list))][:10]
                        if vals:
                            scalars.append(f"- {d}: {json.dumps(vals, ensure_ascii=False)}")
                elif not in_item and depth <= 2 and not self.absent(v) and len(str(v)) <= 80:
                    scalars.append(f"- {d}: {json.dumps(v, ensure_ascii=False)}")

        walk(self.merged, "", "", 0, False)
        parts = []
        if scalars:
            parts.append("Values already read:\n" + "\n".join(scalars))
        if lists:
            parts.append("Entities already extracted (count, then each by its identifying "
                         "fields):\n" + "\n".join(lists))
        text = "\n\n".join(parts)
        if len(text) > max_chars:
            text = text[:max_chars].rsplit("\n", 1)[0] + "\n... (digest truncated)"
        return text
