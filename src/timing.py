"""How long it takes: MinerU per document and per page, the model per document
and per case, and both together. Called by report.py; writes into the report dir:

    timing.md            the human report
    timing.json          everything, machine-readable
    mineru_timing.csv    one row per document
    model_latency.csv    one row per document x case
    end_to_end.csv       one row per document x case: render + MinerU + model

WHICH CLOCK IS WHICH
--------------------
MinerU  processing_s is measured ON THE POD from job start to job end: MinerU's
        own work, with no upload, proxy or polling in it. client_total_s is what
        this machine waited, all of that included. Every fresh MinerU run is
        kept as a sample (re-run with --force-ocr to add one); the median is used.
Model   model_latency_s is each window's call time added up, retries included:
        what one document costs in model time. model_wall_s is the whole
        document as the caller waited, so it also includes building each request
        on this machine (JPEG-encoding the pages) - slightly above the sum when
        chunking.concurrency is 1, below it when windows run in parallel.
        first_byte_s is the first window's time to first token: upload + queue
        + image encoding on the server + prefill.

THE FIT. Every MinerU run is a fresh process that loads its models before it
reads a page, so its time is a fixed start-up cost plus a cost per page. A
straight-line fit of time against pages over documents of very different
lengths separates the two: the intercept is the fixed cost, the slope the
marginal seconds per page. The same fit is done for the model per case, and the
fitted lines give the projection table ("a 50-page policy takes about...").
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Any

import common
import prompting


def _median(vals: list) -> float | None:
    vals = [v for v in vals if isinstance(v, (int, float))]
    return round(statistics.median(vals), 3) if vals else None


def _div(a, b, nd: int = 3):
    if a is None or not b:
        return None
    return round(a / b, nd)


def linear_fit(xs: list[float], ys: list[float]) -> dict | None:
    """Least-squares y = intercept + slope * x, with R^2. None if under-determined."""
    pts = [(float(x), float(y)) for x, y in zip(xs, ys)
           if isinstance(x, (int, float)) and isinstance(y, (int, float))]
    if len(pts) < 2 or len({x for x, _ in pts}) < 2:
        return None
    n = len(pts)
    mx = sum(x for x, _ in pts) / n
    my = sum(y for _, y in pts) / n
    sxx = sum((x - mx) ** 2 for x, _ in pts)
    sxy = sum((x - mx) * (y - my) for x, y in pts)
    slope = sxy / sxx
    intercept = my - slope * mx
    ss_tot = sum((y - my) ** 2 for _, y in pts)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in pts)
    r2 = 1 - ss_res / ss_tot if ss_tot else 1.0
    return {"intercept": round(intercept, 3), "slope": round(slope, 4),
            "r2": round(r2, 3), "n": n}


# --------------------------------------------------------------------------- collect


def mineru_rows(cfg: dict, docs: list) -> list[dict]:
    rows = []
    for d in docs:
        ocr_dir = common.ocr_dir_for(cfg, d.doc_id)
        meta_p = ocr_dir / "ocr_meta.json"
        if not meta_p.exists():
            continue
        meta = common.read_json(meta_p)
        runs = [r for r in (meta.get("timing_runs") or []) if r.get("processing_s") is not None]
        chars = meta.get("chars_per_page") or []
        n_pages = int(meta.get("n_pages") or len(chars) or 0)
        total_chars = sum(chars)
        proc = _median([r.get("processing_s") for r in runs])
        rows.append({
            "lob": d.doc_type,
            "doc_id": d.doc_id,
            "pages": n_pages,
            "pdf_mb": round(d.pdf.stat().st_size / (1024 * 1024), 2) if d.pdf.exists() else None,
            "source": meta.get("source"),
            "runs": len(runs),
            "processing_s": proc,
            "s_per_page": _div(proc, n_pages),
            "cli_s": _median([r.get("cli_s") for r in runs]),
            "parse_s": _median([r.get("parse_s") for r in runs]),
            "queue_wait_s": _median([r.get("queue_wait_s") for r in runs]),
            "upload_s": _median([r.get("upload_s") for r in runs]),
            "client_total_s": _median([r.get("client_total_s") for r in runs]),
            "total_chars": total_chars,
            "chars_per_page": round(total_chars / n_pages) if n_pages else None,
            "max_chars_page": max(chars) if chars else None,
            "chars_per_s": round(total_chars / proc) if proc else None,
            "s_per_1k_chars": _div(proc, total_chars / 1000.0) if total_chars else None,
            "near_empty_pages": len(meta.get("near_empty_pages") or []),
            "gpu": next((r.get("gpu") for r in reversed(runs) if r.get("gpu")), None),
            "min_s": min((r["processing_s"] for r in runs), default=None),
            "max_s": max((r["processing_s"] for r in runs), default=None),
        })
    rows.sort(key=lambda r: (r["pages"], r["doc_id"]))
    return rows


def render_rows(cfg: dict, docs: list) -> dict[str, dict]:
    out = {}
    for d in docs:
        p = common.ocr_dir_for(cfg, d.doc_id) / "render_meta.json"
        if p.exists():
            m = common.read_json(p)
            out[d.doc_id] = {"render_s": m.get("render_s"), "pages": m.get("n_pages")}
    return out


def _case_order(cfg: dict) -> list[str]:
    """The cases in config order (the order they were asked for), then any others."""
    order = [str(c) for c in (cfg.get("cases") or [])]
    return order + [c for c in prompting.CASES if c not in order]


def model_rows(cells: list[dict], case_order: list[str] = list(prompting.CASES)) -> list[dict]:
    rows = []
    for r in cells:
        m = r["meta"]
        if m.get("status") == "ERROR":
            continue
        rows.append({
            "lob": r.get("doc_type", ""),
            "doc_id": r["doc_id"],
            "model": r.get("model_tag"),
            "case": r["case"],
            "label": r["case"],
            "pages": m.get("n_pages"),
            "windows": m.get("n_windows"),
            "status": m.get("status"),
            "prompt_tokens": m.get("prompt_tokens"),
            "completion_tokens": m.get("completion_tokens"),
            "cached_tokens": m.get("cached_tokens"),
            "first_byte_s": m.get("first_byte_s"),
            "model_latency_s": m.get("model_latency_s", m.get("total_s")),
            "model_wall_s": m.get("model_wall_s"),
            "latency_per_page_s": m.get("latency_per_page_s"),
            "decode_tokens_per_s": m.get("decode_tokens_per_s"),
            "attempts": m.get("attempts"),
            "retried_windows": m.get("retried_windows"),
            "resumed": m.get("wall_is_resumed"),
            "window_plan_method": m.get("window_plan_method"),
            "run_label": m.get("run_label"),
            "mock": m.get("mock"),
            "field_f1": (r.get("score") or {}).get("field_f1"),
        })
    # One model: rows are labelled by case. More than one: by model/case, so two
    # models' latencies are never added together under one label.
    if len({r["model"] for r in rows}) > 1:
        for r in rows:
            r["label"] = f"{r['model']}/{r['case']}"
    order = {c: i for i, c in enumerate(case_order)}
    rows.sort(key=lambda r: (r["pages"] or 0, r["doc_id"], str(r["model"]),
                             order.get(r["case"], 9)))
    return rows


def build(cfg: dict, cells: list[dict], docs: list) -> dict:
    cases = _case_order(cfg)
    mrows = mineru_rows(cfg, docs)
    rrows = render_rows(cfg, docs)
    lrows = model_rows(cells, cases)
    mineru_by = {r["doc_id"]: r for r in mrows}
    model_tags = [str(m.get("tag")) for m in cfg.get("models") or []]
    labels = []
    for tag in model_tags + sorted({str(r["model"]) for r in lrows} - set(model_tags)):
        for case in cases:
            for r in lrows:
                if str(r["model"]) == tag and r["case"] == case and r["label"] not in labels:
                    labels.append(r["label"])

    # MinerU summary + fits
    tot_pages = sum(r["pages"] for r in mrows if r["processing_s"] is not None)
    tot_proc = sum(r["processing_s"] for r in mrows if r["processing_s"] is not None)
    tot_chars = sum(r["total_chars"] for r in mrows if r["processing_s"] is not None)
    mineru_summary = {
        "documents": len([r for r in mrows if r["processing_s"] is not None]),
        "pages": tot_pages,
        "processing_s": round(tot_proc, 3),
        "s_per_page_overall": _div(tot_proc, tot_pages),
        "chars": tot_chars,
        "chars_per_page_overall": round(tot_chars / tot_pages) if tot_pages else None,
        "chars_per_s_overall": round(tot_chars / tot_proc) if tot_proc else None,
        "fit_vs_pages": linear_fit([r["pages"] for r in mrows],
                                   [r["processing_s"] for r in mrows]),
        "fit_vs_kchars": linear_fit([r["total_chars"] / 1000.0 for r in mrows],
                                    [r["processing_s"] for r in mrows]),
        "sources": sorted({str(r["source"]) for r in mrows}),
    }

    # Model summary per case (per model/case with several models) + fits
    case_summary = {}
    for label in labels:
        rs = [r for r in lrows if r["label"] == label and r["model_latency_s"] is not None]
        if not rs:
            continue
        pages = sum(r["pages"] or 0 for r in rs)
        lat = sum(r["model_latency_s"] for r in rs)
        case_summary[label] = {
            "documents": len(rs),
            "pages": pages,
            "model_latency_s": round(lat, 3),
            "s_per_page_overall": _div(lat, pages),
            "median_doc_latency_s": _median([r["model_latency_s"] for r in rs]),
            "median_first_byte_s": _median([r["first_byte_s"] for r in rs]),
            "median_decode_tokens_per_s": _median([r["decode_tokens_per_s"] for r in rs]),
            "prompt_tokens": sum(r["prompt_tokens"] or 0 for r in rs),
            "completion_tokens": sum(r["completion_tokens"] or 0 for r in rs),
            "fit_vs_pages": linear_fit([r["pages"] for r in rs],
                                       [r["model_latency_s"] for r in rs]),
        }

    # End to end, per document and case
    e2e = []
    for r in lrows:
        needs = prompting.CASE_NEEDS.get(r["case"], {})
        rend = (rrows.get(r["doc_id"]) or {}).get("render_s") if needs.get("png") else 0.0
        # MinerU is part of a case's cost if the case sends its text - or if the
        # case's windows were planned from MinerU's headings (PDF only then
        # cannot start until MinerU has run, even though it never sees the text).
        uses_mineru = needs.get("md") or r.get("window_plan_method") == "section-aware"
        ocr = (mineru_by.get(r["doc_id"]) or {}).get("processing_s") if uses_mineru else 0.0
        parts = [rend, ocr, r["model_latency_s"]]
        total = None if any(p is None for p in parts) else round(sum(parts), 3)
        e2e.append({"lob": r["lob"], "doc_id": r["doc_id"], "case": r["label"],
                    "pages": r["pages"], "render_s": rend, "mineru_s": ocr,
                    "model_s": r["model_latency_s"], "total_s": total,
                    "s_per_page": _div(total, r["pages"])})

    # Projection from the fitted lines
    seen = sorted({r["pages"] for r in mrows} | {r["pages"] or 0 for r in lrows})
    sizes = sorted({5, 20, 50, 90} | ({max(seen)} if seen else set()))
    proj = []
    mfit = mineru_summary["fit_vs_pages"]
    for n in sizes:
        row: dict[str, Any] = {"pages": n}
        row["mineru_s"] = round(mfit["intercept"] + mfit["slope"] * n, 1) if mfit else None
        for case, cs in case_summary.items():
            f = cs["fit_vs_pages"]
            row[f"model_{case}_s"] = round(f["intercept"] + f["slope"] * n, 1) if f else None
        proj.append(row)

    return {
        "generated_at": common.utc_now_iso(),
        "models": [str(m.get("model_id") or m.get("tag")) for m in cfg.get("models") or []],
        "mock": any(r.get("mock") for r in lrows),
        "chunking": cfg.get("chunking") or {},
        "mineru": mrows,
        "mineru_summary": mineru_summary,
        "render": rrows,
        "model": lrows,
        "model_summary": case_summary,
        "end_to_end": e2e,
        "projection": proj,
    }


# --------------------------------------------------------------------------- render


def _f(v: Any, nd: int = 2) -> str:
    if v is None:
        return "-"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    return str(v)


def _fit(f: dict | None, unit: str = "page") -> str:
    if not f:
        return "not enough distinct document sizes to fit"
    return (f"time ~= {f['intercept']:.1f} s + {f['slope']:.3f} s per {unit}  "
            f"(R^2 {f['r2']:.2f}, {f['n']} documents)")


def render_md(t: dict) -> str:
    L: list[str] = []
    a = L.append
    models = ", ".join(t.get("models") or []) or "the model"
    a(f"# POC3 - processing time: MinerU and {models}")
    a("")
    a(f"Generated {t['generated_at']}")
    a("")
    if t["mock"]:
        a("> **Mock run.** Model latencies below were synthesized, not measured. "
          "MinerU times are real only if MinerU actually ran.")
        a("")
    ch = t.get("chunking") or {}
    ctx = ch.get("context") or {}
    sec = ch.get("section_boundaries") or {}
    extras = [x for x in (
        f"boundaries moved up to {sec.get('max_shift_pages')} pages to section starts"
        if sec.get("enabled") else "",
        f"reference page(s) {ctx.get('anchor_pages')} shown to every later window"
        if ctx.get("anchor_pages") else "",
        "a digest of earlier windows passed forward (windows run one after another)"
        if ctx.get("carry_forward") else "") if x]
    a(f"Documents are processed in windows of about {ch.get('window_pages')} pages "
      f"(overlap {ch.get('overlap_pages', 0)})"
      + (f"; {'; '.join(extras)}" if extras else "")
      + ". The reference pages and the digest add prompt tokens to every window "
        "after the first, and that cost is included in the model times below.")
    a("")

    ms = t["mineru_summary"]
    a("## MinerU (PDF -> markdown)")
    a("")
    if not t["mineru"]:
        a("No MinerU timings yet - run `python src/runner.py --ocr-only`.")
        a("")
    else:
        a(f"**{ms['documents']} documents, {ms['pages']:,} pages, "
          f"{ms['processing_s']:,.1f} s of MinerU processing: "
          f"{_f(ms['s_per_page_overall'])} s per page and "
          f"{_f(ms['chars_per_page_overall'], 0)} characters per page on average, "
          f"{_f(ms['chars_per_s_overall'], 0)} characters per second.**")
        a("")
        fit = ms["fit_vs_pages"]
        fixed = ("negligible: MinerU ran as a warm service with its models already "
                 "loaded, so a document costs its pages and nothing more"
                 if fit and fit["intercept"] < 1.0 else
                 "MinerU's per-document overhead (start-up, model load, layout set-up), "
                 "paid once per document")
        a(f"- Per page: {_fit(fit)}. The fixed part is {fixed}; the per-page part is "
          f"what each extra page costs.")
        a(f"- Per 1,000 characters: {_fit(ms['fit_vs_kchars'], '1,000 characters')}.")
        not_mineru = [s for s in ms["sources"] if not str(s).startswith("mineru")]
        a(f"- Text source: {', '.join(ms['sources'])}"
          + (f" - {', '.join(not_mineru)} is not a MinerU timing." if not_mineru else "."))
        a("")
        a("Processing time is measured on the pod (no upload, proxy or polling). "
          "`end-to-end` is what this machine waited, all of that included.")
        a("")
        a("| LOB | document | pages | PDF MB | runs | MinerU s | s/page | chars | chars/page | chars/s | s per 1k chars | upload s | end-to-end s |")
        a("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for r in t["mineru"]:
            spread = ""
            if r["runs"] > 1 and r["min_s"] is not None:
                spread = f" ({_f(r['min_s'], 1)}-{_f(r['max_s'], 1)})"
            a(f"| {r['lob']} | {r['doc_id']} | {r['pages']} | {_f(r['pdf_mb'])} | {r['runs']} | "
              f"{_f(r['processing_s'], 1)}{spread} | {_f(r['s_per_page'])} | "
              f"{_f(r['total_chars'])} | {_f(r['chars_per_page'])} | {_f(r['chars_per_s'])} | "
              f"{_f(r['s_per_1k_chars'], 3)} | {_f(r['upload_s'], 1)} | "
              f"{_f(r['client_total_s'], 1)} |")
        a("")
        gpus = sorted({r["gpu"] for r in t["mineru"] if r.get("gpu")})
        if gpus:
            a(f"GPU: {', '.join(gpus)}.")
            a("")

    a(f"## Model latency per document ({models})")
    a("")
    if not t["model"]:
        a("No model runs yet - run `python src/runner.py`.")
        a("")
    else:
        labels = sorted({str(r.get("run_label")) for r in t["model"] if r.get("run_label")})
        unlabelled = sum(1 for r in t["model"] if not r.get("run_label"))
        if labels:
            a("**Run conditions:** " + "; ".join(labels)
              + (f" ({unlabelled} cell(s) without a label)" if unlabelled else "")
              + ". When cases run in parallel they share the GPU, so each latency below "
                "is measured under that load - throughput, not a single document in "
                "isolation.")
            a("")
        a("| case | documents | pages | model time s | s/page | median per doc s | median first byte s | median decode tok/s | tokens in | tokens out |")
        a("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
        for case, cs in t["model_summary"].items():
            a(f"| {case} | {cs['documents']} | {cs['pages']:,} | {_f(cs['model_latency_s'], 1)} | "
              f"{_f(cs['s_per_page_overall'])} | {_f(cs['median_doc_latency_s'], 1)} | "
              f"{_f(cs['median_first_byte_s'])} | {_f(cs['median_decode_tokens_per_s'], 1)} | "
              f"{_f(cs['prompt_tokens'])} | {_f(cs['completion_tokens'])} |")
        a("")
        for case, cs in t["model_summary"].items():
            a(f"- **{case}**: {_fit(cs['fit_vs_pages'])}.")
        a("")
        a("Model time is every window's call added up, retries included. Output "
          "length drives most of it: at a steady decode rate, time grows with tokens "
          "out, which grow with how much the document says, not only how many pages "
          "it has.")
        a("")
        scored = any(r.get("field_f1") is not None for r in t["model"])
        a("| LOB | document | pages | case | windows | status | tokens in | cached | tokens out | first byte s | model s | s/page | decode tok/s | attempts |"
          + (" F1 |" if scored else ""))
        a("|---|---|---:|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|" if scored else ""))
        for r in t["model"]:
            flag = " (resumed)" if r.get("resumed") else ""
            a(f"| {r['lob']} | {r['doc_id']} | {r['pages']} | {r['label']} | {r['windows']} | "
              f"{r['status']} | {_f(r['prompt_tokens'])} | {_f(r['cached_tokens'])} | "
              f"{_f(r['completion_tokens'])} | {_f(r['first_byte_s'])} | "
              f"{_f(r['model_latency_s'], 1)}{flag} | {_f(r['latency_per_page_s'])} | "
              f"{_f(r['decode_tokens_per_s'], 1)} | {_f(r['attempts'])} |"
              + (f" {_f(r['field_f1'], 3)} |" if scored else ""))
        a("")

    if t["end_to_end"]:
        a("## End to end, per document and case")
        a("")
        a("What each case costs in total: page rendering if it sends images, MinerU "
          "if it sends markdown OR its windows were planned from MinerU's headings "
          "(section-aware boundaries - PDF only waits for MinerU then too), and the "
          "model. Added up sequentially - in production rendering and MinerU could "
          "run side by side.")
        a("")
        a("| LOB | document | pages | case | render s | MinerU s | model s | total s | s/page |")
        a("|---|---|---:|---|---:|---:|---:|---:|---:|")
        for r in t["end_to_end"]:
            a(f"| {r['lob']} | {r['doc_id']} | {r['pages']} | {r['case']} | "
              f"{_f(r['render_s'], 1)} | {_f(r['mineru_s'], 1)} | {_f(r['model_s'], 1)} | "
              f"{_f(r['total_s'], 1)} | {_f(r['s_per_page'])} |")
        a("")

    if t["projection"] and (t["mineru_summary"]["fit_vs_pages"] or t["model_summary"]):
        cases = list(t["model_summary"])
        a("## Projection: how long a policy of N pages takes")
        a("")
        a("Read off the fitted lines above. Inside the range of document sizes "
          "measured it is an estimate; outside it, an extrapolation.")
        a("")
        a("| pages | MinerU s | " + " | ".join(f"model {c} s" for c in cases) + " |")
        a("|---:|---:|" + "---:|" * len(cases))
        for p in t["projection"]:
            a(f"| {p['pages']} | {_f(p.get('mineru_s'), 1)} | "
              + " | ".join(_f(p.get(f"model_{c}_s"), 1) for c in cases) + " |")
        a("")
    return "\n".join(L) + "\n"


def _csv(rows: list[dict], path: Path) -> None:
    if not rows:
        return
    common.ensure_dir(path.parent)
    fields = list(rows[0].keys())
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def write(cfg: dict, cells: list[dict], docs: list, rep_dir: Path,
          formats: list[str]) -> dict:
    t = build(cfg, cells, docs)
    if "json" in formats:
        common.write_json(rep_dir / "timing.json", t)
    if "md" in formats:
        common.write_text(rep_dir / "timing.md", render_md(t))
    if "csv" in formats:
        _csv(t["mineru"], rep_dir / "mineru_timing.csv")
        _csv(t["model"], rep_dir / "model_latency.csv")
        _csv(t["end_to_end"], rep_dir / "end_to_end.csv")
    return t
