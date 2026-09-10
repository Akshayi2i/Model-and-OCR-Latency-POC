"""Cross-run comparison: the thing the POC actually produces.

    python src/report.py

Reads every output/<tag>/<case>/*.score.json and *.meta.json and writes, into
output/report/:
    comparison.json   everything, machine-readable
    comparison.md     the human report
    matrix.csv        one row per cell
    per_field.csv     field x cell recall grid

THE THREE DELTAS ARE THE POINT:
    8B - 7B, per case          is the bigger model better, and in the same places?
    PDF+MD - PDF only          does MinerU markdown help?
    MD only - PDF+MD           do the page images still earn their cost?

prompt_integrity() runs first and gates the rest. If the six cells were not asked
the same question, the deltas are not measurements and the report says so instead
of printing them.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import prompting  # noqa: E402


def load_cells(cfg: dict) -> list[dict]:
    """Every cell that produced a meta.json, with its score folded in."""
    rows: list[dict] = []
    doc_types = [str(t.get("id")) for t in common.doc_type_specs(cfg)] or [""]
    for m in cfg.get("models", []):
        for case in cfg.get("cases", prompting.CASES):
            for tid in doc_types:
                d = common.out_dir_for(cfg, str(m["tag"]), str(case), tid)
                if not d.exists():
                    continue
                for meta_path in sorted(d.glob("*.meta.json")):
                    doc_id = meta_path.name[: -len(".meta.json")]
                    meta = common.read_json(meta_path)
                    score_path = d / f"{doc_id}.score.json"
                    score = common.read_json(score_path) if score_path.exists() else {}
                    rows.append({
                        "model_tag": str(m["tag"]),
                        "case": str(case),
                        "cell": f"{m['tag']}/{case}",
                        "doc_type": tid,
                        "doc_id": doc_id,
                        "meta": meta,
                        "score": score,
                    })
    return rows


# --------------------------------------------------------------------------- integrity


def prompt_integrity(rows: list[dict], cfg: dict) -> dict:
    """Were all six cells asked the same question?

    Two invariants, and they pull in opposite directions on purpose:
      - the shared prompt span (everything but the modality paragraph) must hash
        IDENTICALLY everywhere;
      - for one case, the conversation fingerprint must be IDENTICAL across
        models - the two models get byte-identical input, only the weights differ;
      - across cases, fingerprints must DIFFER - otherwise a case was silently
        run with another case's input.
    """
    problems: list[str] = []
    shared_by_doc: dict[tuple, set] = {}
    fp_by_key: dict[tuple, set] = {}

    for r in rows:
        key = (r.get("doc_type", ""), r.get("doc_id", ""))
        sha = r["meta"].get("shared_prompt_sha")
        if sha:
            shared_by_doc.setdefault(key, set()).add(sha)
        fp = r["meta"].get("conversation_fingerprint")
        if fp:
            fp_by_key.setdefault((*key, r["case"]), set()).add(fp)

    # The shared span must be constant across the six cells of ONE DOCUMENT. It
    # legitimately differs between documents (each carries its own schema) and
    # between doc types (each carries its own doc-type paragraph), so comparing
    # spans any more widely than this reports a false alarm.
    for (tid, doc_id), shas in shared_by_doc.items():
        if len(shas) > 1:
            problems.append(
                f"{tid}/{doc_id}: the shared prompt span differs between its cells "
                f"({sorted(shas)}). Those cells were not asked the same question, so "
                f"the deltas for this document are not comparable. Did a prompt file "
                f"or its schema change mid-run?")

    # For one document and one case, both models must have seen identical input.
    for (tid, doc_id, case), fps in fp_by_key.items():
        if len(fps) > 1:
            problems.append(
                f"{tid}/{doc_id} case {case!r} sent different input to different "
                f"models ({sorted(fps)}). Both models must see byte-identical input "
                f"for the model delta to mean anything - were pages re-rendered "
                f"between the two runs?")

    # For one document, no two cases may have sent the same input.
    by_doc: dict[tuple, dict] = {}
    for (tid, doc_id, case), fps in fp_by_key.items():
        if len(fps) == 1:
            by_doc.setdefault((tid, doc_id), {})[case] = next(iter(fps))
    for (tid, doc_id), cases in by_doc.items():
        seen: dict[str, str] = {}
        for case, fp in cases.items():
            if fp in seen:
                problems.append(
                    f"{tid}/{doc_id}: cases {seen[fp]!r} and {case!r} sent IDENTICAL "
                    f"input - one of them was run with the other's content.")
            seen[fp] = case

    return {
        "ok": not problems,
        "shared_prompt_sha_by_document": {f"{t}/{d}": sorted(v)
                                          for (t, d), v in sorted(shared_by_doc.items())},
        "n_fingerprint_groups": len(fp_by_key),
        "problems": problems,
    }


# --------------------------------------------------------------------------- tables


def _n(v: Any) -> str:
    """0 is a real token count; only None is missing."""
    return "-" if v is None else f"{v:,}"


def _f(v: Any, nd: int = 3) -> str:
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, (int, float)) else str(v))


def matrix_table(rows: list[dict]) -> list[dict]:
    out = []
    for r in rows:
        m, s = r["meta"], r["score"]
        out.append({
            "cell": r["cell"],
            "model": r["model_tag"],
            "case": r["case"],
            "doc_id": r["doc_id"],
            "doc_type": r.get("doc_type", ""),
            "status": m.get("status"),
            "field_f1": s.get("field_f1"),
            "field_recall": s.get("field_recall"),
            "field_precision": s.get("field_precision"),
            "strict_match_rate": (s.get("strict") or {}).get("match_rate"),
            "number_coverage": (s.get("coverage") or {}).get("number_coverage"),
            "score_is_floor": s.get("score_is_floor", False),
            "json_valid_strict": m.get("json_valid_strict"),
            "repaired": m.get("repaired_from_truncation"),
            "schema_ok": m.get("schema_ok"),
            "n_images": m.get("n_images"),
            "payload_mb": m.get("payload_mb"),
            "prompt_tokens": m.get("prompt_tokens"),
            "completion_tokens": m.get("completion_tokens"),
            "finish_reason": m.get("finish_reason"),
            "first_byte_s": m.get("first_byte_s"),
            "total_s": m.get("total_s"),
            "attempts": m.get("attempts"),
            "error": m.get("error"),
        })
    out.sort(key=lambda r: (r.get("doc_type", ""), r["doc_id"], r["model"], r["case"]))
    return out


def per_field_grid(rows: list[dict], min_gold: int = 1) -> dict:
    """(doc_type, doc_id) -> (cells, {field: {cell: recall}}).

    Grouped BY DOCUMENT on purpose. Every document carries its own canonical
    schema, so a field path exists in one document and not the others; merging
    them into a single grid produces a table of mostly-empty cells whose columns
    repeat once per document. The comparison that means something is between the
    six cells of ONE document.
    """
    per_doc: dict = {}
    for r in rows:
        key = (r.get("doc_type", ""), r["doc_id"])
        d = per_doc.setdefault(key, {"cells": [], "grid": {}, "freq": {}})
        if r["cell"] not in d["cells"]:
            d["cells"].append(r["cell"])
        for field, c in (r["score"].get("per_field") or {}).items():
            if c.get("gold", 0) < min_gold:
                continue
            d["freq"][field] = max(d["freq"].get(field, 0), c.get("gold", 0))
            d["grid"].setdefault(field, {})[r["cell"]] = c.get("recall")
    out: dict = {}
    for key, d in per_doc.items():
        order = sorted(d["grid"], key=lambda f: (-d["freq"].get(f, 0), f))
        out[key] = (sorted(d["cells"]), {f: d["grid"][f] for f in order})
    return out


def cell_summary(rows: list[dict]) -> dict:
    """cell -> aggregate across every document.

    The corpus has many documents; a cell's quality is its behaviour over all of
    them, not over whichever one happened to be processed last.
    """
    by: dict[str, list] = {}
    for r in rows:
        f1 = (r["score"] or {}).get("field_f1")
        if f1 is not None:
            by.setdefault(r["cell"], []).append((r["doc_id"], f1))
    out: dict[str, dict] = {}
    for cell, pairs in by.items():
        vals = sorted(v for _, v in pairs)
        n = len(vals)
        out[cell] = {
            "n_docs": n,
            "mean_f1": round(sum(vals) / n, 4),
            "min_f1": round(vals[0], 4),
            "max_f1": round(vals[-1], 4),
            "median_f1": round(vals[n // 2] if n % 2 else (vals[n // 2 - 1] + vals[n // 2]) / 2, 4),
            "by_doc": dict(pairs),
        }
    return out


def pairwise_deltas(rows: list[dict]) -> dict:
    """Deltas over the WHOLE corpus, with the per-document record behind them.

    A single mean can hide a split decision, so every comparison also reports on
    how many documents the winner actually won. Comparing one arbitrary document
    - which is what keying rows by cell alone would do - is not a corpus result.
    """
    summary = cell_summary(rows)

    def f1(cell: str):
        s = summary.get(cell)
        return s["mean_f1"] if s else None

    def delta(a: str, b: str) -> dict:
        fa, fb = f1(a), f1(b)
        sa = (summary.get(a) or {}).get("by_doc") or {}
        sb = (summary.get(b) or {}).get("by_doc") or {}
        shared = sorted(set(sa) & set(sb))
        wins = sum(1 for d in shared if sa[d] > sb[d])
        ties = sum(1 for d in shared if sa[d] == sb[d])
        return {
            "a": a, "b": b, "f1_a": fa, "f1_b": fb,
            "delta": None if (fa is None or fb is None) else round(fa - fb, 4),
            "n_docs": len(shared), "wins": wins, "ties": ties,
            "losses": len(shared) - wins - ties,
        }

    models = sorted({r["model_tag"] for r in rows})
    cases = [c for c in prompting.CASES if any(r["case"] == c for r in rows)]

    out: dict[str, list] = {"model": [], "markdown_helps": [], "images_earn_their_cost": []}
    if len(models) >= 2:
        big, small = models[-1], models[0]
        for case in cases:
            out["model"].append(delta(f"{big}/{case}", f"{small}/{case}"))
    for mt in models:
        if "pdf_md" in cases and "pdf_only" in cases:
            out["markdown_helps"].append(delta(f"{mt}/pdf_md", f"{mt}/pdf_only"))
        if "pdf_md" in cases and "md_only" in cases:
            out["images_earn_their_cost"].append(delta(f"{mt}/pdf_md", f"{mt}/md_only"))
    return out


# --------------------------------------------------------------------------- render


def render_markdown(cfg: dict, rows: list[dict], integrity: dict, matrix: list[dict],
                    deltas: dict, grids: dict,
                    summary: dict | None = None) -> str:
    summary = summary if summary is not None else cell_summary(rows)
    rep = cfg.get("report") or {}
    L: list[str] = []
    a = L.append

    a("# POC3 - extraction quality by model and input modality")
    a("")
    a(f"Generated {common.utc_now_iso()}")
    a("")

    if not integrity["ok"]:
        a("## THESE RESULTS ARE NOT COMPARABLE")
        a("")
        for p in integrity["problems"]:
            a(f"- {p}")
        a("")
        a("Fix the above and re-run. Everything below is printed for diagnosis only.")
        a("")
    else:
        a("Prompt integrity verified: for every document, all six cells share one "
          "system prompt and one schema block; for every document and case both "
          "models received byte-identical input; and no two cases sent the same "
          "input. Prompts differ between documents and document types by design - "
          "each document carries its own canonical schema.")
        a("")

    mock = any(r["meta"].get("mock") for r in rows)
    if mock:
        a("> **Mock run.** These numbers were synthesized from the gold, not "
          "generated by a model. They prove the pipeline works, nothing about "
          "extraction quality.")
        a("")

    a("## Results by cell, across the whole corpus")
    a("")
    a("Each cell is one model on one input modality, aggregated over every "
      "document it ran on. This is the headline; the per-document detail is below.")
    a("")
    a("| cell | docs | mean F1 | median | min | max |")
    a("|---|---:|---:|---:|---:|---:|")
    for cell in sorted(summary):
        v = summary[cell]
        a(f"| {cell} | {v['n_docs']} | {_f(v['mean_f1'])} | {_f(v['median_f1'])} | "
          f"{_f(v['min_f1'])} | {_f(v['max_f1'])} |")
    a("")

    a("## Results per document")
    a("")
    a("| doc type | document | cell | status | F1 | recall | precision | strict | num.cov | tokens in/out | s |")
    a("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in matrix:
        floor = " *" if r.get("score_is_floor") else ""
        a(f"| {r.get('doc_type','')} | {r['doc_id']} "
          f"| {r['cell']} | {r['status']} | {_f(r['field_f1'])}{floor} | "
          f"{_f(r['field_recall'])} | {_f(r['field_precision'])} | "
          f"{_f(r['strict_match_rate'])} | {_f(r['number_coverage'])} | "
          f"{_n(r['prompt_tokens'])}/{_n(r['completion_tokens'])} | "
          f"{_f(r['total_s'], 1)} |")
    a("")
    if any(r.get("score_is_floor") for r in matrix):
        a("`*` output was truncated and repaired to its valid prefix. That score is "
          "a **floor, not a measurement** - the tail was never generated. Without "
          "this note a more verbose model reads as a worse one.")
        a("")

    a("## The three questions")
    a("")
    titles = {
        "model": "Is the bigger model better? (8B - 7B, per case)",
        "markdown_helps": "Does MinerU markdown help? (PDF+MD - PDF only)",
        "images_earn_their_cost": "Do the page images still earn their cost? (PDF+MD - MD only)",
    }
    for key, title in titles.items():
        entries = deltas.get(key) or []
        if not entries:
            continue
        a(f"### {title}")
        a("")
        a("| comparison | mean F1 | vs | mean F1 | delta | won on |")
        a("|---|---:|---|---:|---:|---:|")
        for d in entries:
            sign = "" if d["delta"] is None else ("+" if d["delta"] >= 0 else "")
            record = f"{d.get('wins', 0)}/{d.get('n_docs', 0)} docs"
            if d.get("ties"):
                record += f" ({d['ties']} tied)"
            a(f"| {d['a']} | {_f(d['f1_a'])} | {d['b']} | {_f(d['f1_b'])} | "
              f"{sign}{_f(d['delta'])} | {record} |")
        a("")
        a("Means are over every document both cells ran on. `won on` counts the "
          "documents where the first cell scored strictly higher - a mean can "
          "hide a split decision.")
        a("")

    top_n = int(rep.get("top_missing_fields", 40))
    sections = []
    for (dtype, doc_id), (cells, grid) in sorted(grids.items()):
        worst = sorted(
            ((f, min((v for v in cm.values() if v is not None), default=None), cm)
             for f, cm in grid.items()),
            key=lambda t: (1 if t[1] is None else t[1]))
        hard = [t for t in worst if t[1] is not None and t[1] < 1.0][:top_n]
        if hard:
            sections.append((dtype, doc_id, cells, hard))
    if sections:
        a("## Fields no cell gets right")
        a("")
        a("Recall per cell, one table per document - each document has its own "
          "canonical schema, so field paths are not comparable across documents. "
          "A row low in every column is a document or schema problem, not a model "
          "problem; a row low in one column is what actually separates the cells.")
        a("")
        for dtype, doc_id, cells, hard in sections:
            a(f"### {dtype}/{doc_id}")
            a("")
            a("| field | " + " | ".join(cells) + " |")
            a("|---" * (len(cells) + 1) + "|")
            for f, _lo, cm in hard:
                a(f"| `{f}` | " + " | ".join(_f(cm.get(c), 2) for c in cells) + " |")
            a("")

    by_doc: dict = {}
    for r in rows:
        lc = r["score"].get("list_counts") or {}
        if lc:
            by_doc.setdefault((r.get("doc_type", ""), r["doc_id"]), []).append(r)
    if by_doc:
        a("## Entity counts")
        a("")
        a("How many objects each cell found, against the gold. A per-field F1 alone "
          "hides a list that was dropped whole. One table per document, for the "
          "same reason as above.")
        a("")
        for (dtype, doc_id), drows in sorted(by_doc.items()):
            cells = sorted({r["cell"] for r in drows})
            names = sorted({k for r in drows for k in (r["score"].get("list_counts") or {})})
            if not names:
                continue
            a(f"### {dtype}/{doc_id}")
            a("")
            a("| list | gold | " + " | ".join(cells) + " |")
            a("|---" * (len(cells) + 2) + "|")
            for name in names:
                gold_n = next((( r["score"]["list_counts"][name] or {}).get("gold")
                               for r in drows if name in (r["score"].get("list_counts") or {})), "-")
                vals = []
                for cell in cells:
                    r = next((x for x in drows if x["cell"] == cell), None)
                    lc = (r["score"].get("list_counts") or {}).get(name) if r else None
                    vals.append(str(lc["pred"]) if lc else "-")
                a(f"| `{name}` | {gold_n} | " + " | ".join(vals) + " |")
            a("")

    a("## Health")
    a("")
    a("| cell | JSON valid | repaired | schema ok | images | payload MB | attempts | error |")
    a("|---|---|---|---|---:|---:|---:|---|")
    for r in matrix:
        a(f"| {r['cell']} | {r['json_valid_strict']} | {r['repaired']} | "
          f"{r['schema_ok']} | {r['n_images']} | {_f(r['payload_mb'], 2)} | "
          f"{r['attempts']} | {(r['error'] or '')[:60]} |")
    a("")
    a("---")
    a("")
    a("Re-scoring never needs the models again: raw.txt holds every generation, so "
      "changing scoring.entity_lists or scoring.ignore_paths and re-running "
      "`python src/report.py` is free.")
    return "\n".join(L) + "\n"


def write_csv(rows: list[dict], path: Path, fields: list[str]) -> None:
    common.ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    log = common.setup_logging("report", (cfg.get("runtime") or {}).get("log_level", "INFO"))

    rows = load_cells(cfg)
    if not rows:
        log.error("no cells found under %s - run src/runner.py first",
                  common.cfg_path(cfg, "output_dir"))
        return 2

    integrity = prompt_integrity(rows, cfg)
    matrix = matrix_table(rows)
    deltas = pairwise_deltas(rows)
    grids = per_field_grid(
        rows, int((cfg.get("report") or {}).get("per_field_min_gold", 1)))

    rep_dir = common.ensure_dir(common.cfg_path(cfg, "report_dir"))
    formats = [str(f).lower() for f in ((cfg.get("report") or {}).get("formats")
                                        or ["json", "md", "csv"])]

    if "json" in formats:
        common.write_json(rep_dir / "comparison.json", {
            "generated_at": common.utc_now_iso(),
            "prompt_integrity": integrity,
            "matrix": matrix,
            "deltas": deltas,
            "cell_summary": cell_summary(rows),
            "per_field": {f"{t}/{d}": g for (t, d), (_c, g) in grids.items()},
            "cells": sorted({r["cell"] for r in rows}),
        })
    if "md" in formats:
        common.write_text(rep_dir / "comparison.md",
                          render_markdown(cfg, rows, integrity, matrix, deltas, grids))
    if "csv" in formats:
        write_csv(matrix, rep_dir / "matrix.csv", list(matrix[0].keys()))
        # One row per (document, field): a field path belongs to one schema, so a
        # single wide grid across documents would be mostly empty.
        all_cells = sorted({r["cell"] for r in rows})
        field_rows = [{"doc_type": t, "document": d, "field": f,
                       **{c: g.get(c) for c in all_cells}}
                      for (t, d), (_c, grid) in sorted(grids.items())
                      for f, g in grid.items()]
        if field_rows:
            write_csv(field_rows, rep_dir / "per_field.csv",
                      ["doc_type", "document", "field"] + all_cells)

    if not integrity["ok"]:
        log.error("PROMPT INTEGRITY FAILED - the cells are not comparable:")
        for p in integrity["problems"]:
            log.error("  %s", p)
    else:
        log.info("prompt integrity OK across %d cells", len(rows))

    log.info("wrote %s", rep_dir / "comparison.md")
    for d in (deltas.get("model") or []):
        log.info("  %s vs %s: %+.3f", d["a"], d["b"], d["delta"] or 0.0)
    return 0 if integrity["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
