"""The driver: render, OCR, prompt, generate per window, merge, score, write.

    python src/runner.py                       # every case, every document
    python src/runner.py --only pdf_md         # one case
    python src/runner.py --doc home_policy     # one document
    python src/runner.py --ocr-only            # phase 1 only: render + MinerU (timing run)
    python src/runner.py --ocr-only --force-ocr  # re-run MinerU: adds another timing sample
    python src/runner.py --mock                # everything, no GPU
    python src/runner.py --overwrite           # redo cells that already have output

TWO PHASES, IN THIS ORDER. Phase 1 renders and OCRs EVERY document; phase 2 runs
the model. MinerU and vLLM share one GPU, so interleaving them would let one
slow the other down and blur both sets of timings. Phase 1 also means a MinerU
problem shows up before any GPU time is spent on inference.

ONE CELL = ONE CASE x ONE DOCUMENT, run as one request per page window (see
chunking.py). Each window's raw output is written before it is parsed, so a
crash in parsing, merging or scoring can never cost a GPU call - and a window
that already finished is reused on resume rather than generated again.
Everything downstream of the raw text - repair, merge, schema validation,
scoring, the report - can be re-run for free.
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chunking  # noqa: E402
import common  # noqa: E402
import prompting  # noqa: E402
import render  # noqa: E402
import scoring  # noqa: E402
from vllm_client import MockClient, VLLMClient  # noqa: E402


@dataclass
class Cell:
    model_tag: str
    model_id: str
    served_name: str
    port: int
    case: str
    max_model_len: int
    max_images: int
    base_url: str | None = None

    @property
    def label(self) -> str:
        return f"{self.model_tag}/{self.case}"


def cells(cfg: dict, only: str | None = None) -> list[Cell]:
    out: list[Cell] = []
    for m in cfg.get("models", []):
        for case in cfg.get("cases", prompting.CASES):
            out.append(Cell(
                model_tag=str(m["tag"]),
                model_id=str(m["model_id"]),
                served_name=str(m.get("served_name") or m["model_id"]),
                port=int(m["port"]),
                case=str(case),
                max_model_len=int(m.get("max_model_len", 65536)),
                max_images=int(m.get("max_images", 24)),
                base_url=m.get("base_url"),
            ))
    if only:
        wanted = {w.strip() for w in only.split(",") if w.strip()}
        out = [c for c in out
               if c.label in wanted or c.model_tag in wanted or c.case in wanted]
        if not out:
            raise SystemExit(f"--only {only!r} matched no cell. "
                             f"Try one of: {', '.join(c.label for c in cells(cfg))}")
    return out


def scoring_enabled(cfg: dict) -> bool:
    """scoring.enabled: false = extraction + timing only (no F1, no gold read)."""
    return bool((cfg.get("scoring") or {}).get("enabled", True))


def windows_need_markdown(cfg: dict) -> bool:
    """Section-aware windows are planned from MinerU's headings, so MinerU must
    run even when only PDF only is requested. PDF only never SEES the markdown."""
    sec = (cfg.get("chunking") or {}).get("section_boundaries") or {}
    mode = str((cfg.get("ocr") or {}).get("mode", "pod")).lower()
    return bool(sec.get("enabled")) and mode != "none"


def prepare_document(cfg: dict, doc, needed_cases: list[str], log,
                     force_ocr: bool = False) -> dict[str, list[render.Page]]:
    """Render pages and fetch markdown, but only what the enabled cases need.

    So `--only pdf_only` never touches the pod's MinerU port - unless the window
    plan needs MinerU's headings (chunking.section_boundaries).
    """
    doc.n_pages = doc.n_pages or render.page_count(doc.pdf)
    ocr_dir = common.ensure_dir(common.ocr_dir_for(cfg, doc.doc_id))
    rcfg = cfg.get("render") or {}

    want_png = any(prompting.CASE_NEEDS[c]["png"] for c in needed_cases)
    want_md = (any(prompting.CASE_NEEDS[c]["md"] for c in needed_cases)
               or (windows_need_markdown(cfg) and doc.n_pages > int(
                   (cfg.get("chunking") or {}).get("window_pages", 10))))

    if want_png:
        have = len(list(ocr_dir.glob("page_*.png")))
        if have != doc.n_pages or force_ocr:
            log.info("rendering %d pages of %s at dpi=%s, cap=%spx",
                     doc.n_pages, doc.doc_id, rcfg.get("dpi"), rcfg.get("max_long_side_px"))
            res = render.render_pages(doc.pdf, ocr_dir,
                                      int(rcfg.get("dpi", 200)),
                                      int(rcfg.get("max_long_side_px", 1024)))
            meta = render.write_render_meta(ocr_dir, doc.doc_id, res,
                                            str(rcfg.get("wire_format", "jpeg")),
                                            int(rcfg.get("jpeg_quality", 88)))
            log.info("  %d pages in %.1fs, %.2f MB on the wire, ~%s image tokens",
                     meta["n_pages"], meta["render_s"], meta["total_wire_b64_mb"],
                     f"{meta['total_image_tokens_est']:,}")
        else:
            log.info("page images cached for %s (%d pages)", doc.doc_id, doc.n_pages)

    if want_md:
        import ocr_client
        ocr_client.ensure_page_markdown(cfg, doc, log=log, force=force_ocr)

    per_case: dict[str, list[render.Page]] = {}
    for case in needed_cases:
        needs = prompting.CASE_NEEDS[case]
        per_case[case] = render.require_pages(ocr_dir, case, needs["png"], needs["md"])
    return per_case


def _client_for(cfg: dict, cell: Cell, gold: dict | None, log):
    rt = cfg.get("runtime") or {}
    if rt.get("mock"):
        return MockClient(gold or {}, str(rt.get("mock_profile", "degraded")),
                          cell.served_name)
    pod = cfg.get("pod") or {}
    base = common.base_url_for(cfg, cell.port, cell.base_url)
    return VLLMClient(base, cell.served_name, pod.get("api_key"),
                      cfg.get("inference") or {}, bool(pod.get("verify_tls", True)))


def output_budget(cfg: dict, gold: dict | None, log) -> int | None:
    """How many tokens this document's answer needs, sized from its own gold.

    A budget too small to hold a complete answer does not measure extraction, it
    measures where the ceiling was: the output is cut off mid-JSON, repaired to a
    prefix and scored as a floor. The gold is the only honest estimate of how
    long a complete answer is. It yields ONE number per document, used by every
    window of every case, so the comparison is unaffected. A window emits at most
    the whole document's answer, so the whole-document budget always suffices.

    Returns None to mean "use inference.max_tokens", for a document with no gold.
    """
    inf = cfg.get("inference") or {}
    if not inf.get("budget_from_gold", True) or not gold:
        return None
    pretty = len(common.pretty_json(gold))
    need = int(pretty / float(inf.get("chars_per_token", 3.77))
               * float(inf.get("budget_margin", 1.4)))
    lo, hi = int(inf.get("budget_floor", 8192)), int(inf.get("budget_ceiling", 32768))
    if need > hi:
        log.warning("  gold implies %d output tokens; capped at ceiling %d - a "
                    "complete answer may not fit", need, hi)
    return max(lo, min(hi, need))


def window_is_final(prior: dict, budget: int | None = None,
                    served_name: str | None = None) -> bool:
    """May a stored window be reused rather than generated again?

    Yes if the model answered - even with something unparseable, which at
    temperature 0 it would only answer again - under the same output budget and
    from the same served model. No if the REQUEST failed (timeout, 5xx, a stream
    that died after every retry): that is the network, not the model, and a
    re-run must retry it. No if the budget changed (a window truncated at the
    old ceiling deserves the new one) or the model did.
    """
    return (prior.get("status") in ("OK", "REPAIRED", "INVALID")
            and not prior.get("error")
            and prior.get("budget") == budget
            and prior.get("served_name") == served_name)


def run_window(cfg: dict, cell: Cell, doc, wpages: list[render.Page], n_total: int,
               idx: int, schema: dict, gold: dict | None, budget: int | None,
               win_dir: Path, log, refs: list[render.Page] | None = None,
               digest: str = "", through: int | None = None) -> tuple[dict, Any]:
    """One request: one window of one document in one case. Returns (meta, parsed).

    refs are the reference pages shown before the window's own pages; digest is
    what earlier windows extracted (pages 1..through). Both are part of the
    request, so both are part of its fingerprint: if an earlier window changes,
    this one is regenerated rather than reused.
    """
    first, last = wpages[0].index, wpages[-1].index
    label = chunking.window_label(idx, first, last)
    rt = cfg.get("runtime") or {}
    mock = bool(rt.get("mock"))
    refs = refs or []

    messages = prompting.build_messages(cfg, cell.case, wpages, schema, doc.doc_type,
                                        total_pages=n_total, reference_pages=refs,
                                        digest=digest, digest_through=through)
    fingerprint = prompting.conversation_fingerprint(messages)

    meta_path = win_dir / f"{label}.meta.json"
    json_path = win_dir / f"{label}.json"
    if meta_path.exists() and not rt.get("overwrite"):
        # Resume a window only over a final answer produced the same way from
        # the same input. Anything else is redone rather than silently mixed in.
        try:
            prior = common.read_json(meta_path)
        except (ValueError, OSError):
            prior = {}
        if (prior.get("conversation_fingerprint") == fingerprint
                and prior.get("mock") == mock
                and window_is_final(prior, budget, cell.served_name)):
            parsed = common.read_json(json_path) if json_path.exists() else None
            raw_path = win_dir / f"{label}.raw.txt"
            if prior.get("status") == "INVALID" and raw_path.exists():
                # The answer is kept; only reading it failed. A parser that has
                # learned since may read it now - no model call, same timing.
                parsed, note, repaired = common.recover_json_object(
                    raw_path.read_text(encoding="utf-8"))
                if parsed is not None:
                    common.write_json(json_path, parsed)
                    prior.update(parsed=True, repaired_from_truncation=repaired,
                                 parse_note=note, reparsed_at=common.utc_now_iso(),
                                 status="REPAIRED" if repaired else "OK")
                    common.write_json(meta_path, prior)
                    log.info("  %s re-read from its stored answer: %s", label, note)
            log.info("  %s reused (%s)", label, prior.get("status"))
            prior["reused"] = True
            return prior, parsed
        if prior.get("error"):
            log.info("  %s retrying: last attempt failed with %s", label, prior["error"][:80])

    wire, stats = prompting.materialize(messages, cfg)
    # A pdf_only run that quietly sent zero images would read as a model failure.
    expected_images = 0 if cell.case == "md_only" else len(wpages) + len(refs)
    if stats["n_images"] != expected_images:
        raise RuntimeError(
            f"{cell.label} {label}: built {stats['n_images']} images but case "
            f"{cell.case!r} needs exactly {expected_images}")
    if stats["n_images"] > cell.max_images:
        raise RuntimeError(
            f"{cell.label} {label}: {stats['n_images']} images exceeds the server's "
            f"--limit-mm-per-prompt of {cell.max_images}. Lower chunking.window_pages, "
            f"or raise models[].max_images AND the server flag together.")

    common.write_json(win_dir / f"{label}.request.json", {
        "cell": cell.label, "window": label, "pages": [first, last],
        "reference_pages": [p.index for p in refs], "digest_chars": len(digest),
        "model_id": cell.model_id, "served_name": cell.served_name, "case": cell.case,
        "conversation_fingerprint": fingerprint, **stats,
        "messages": prompting.redact(messages),
    })

    client = _client_for(cfg, cell, gold, log)

    def keep_partial(text: str, attempt: int) -> None:
        p = win_dir / f"{label}.raw.attempt{attempt}.txt"
        common.write_text(p, text)
        log.warning("  %s kept %d chars of partial output at %s", label, len(text), p.name)

    result = client.complete(wire, max_tokens=budget, on_partial=keep_partial)
    # raw first, always, before anything can fail.
    common.write_text(win_dir / f"{label}.raw.txt", result.text or "")

    parsed, parse_note, repaired = common.recover_json_object(result.text or "")
    _stripped, had_reasoning = common.strip_reasoning(result.text or "")
    if parsed is not None:
        common.write_json(json_path, parsed)
    elif json_path.exists():
        json_path.unlink()

    usage = result.usage or {}
    status = ("OK" if parsed is not None and not repaired else
              "REPAIRED" if repaired else "INVALID")
    meta = {
        "window": label,
        "index": idx,
        "pages": [first, last],
        "n_pages": len(wpages),
        "reference_pages": [p.index for p in refs],
        "digest_chars": len(digest),
        "served_name": cell.served_name,
        "budget": budget,
        "mock": mock,
        "run_at": common.utc_now_iso(),
        "conversation_fingerprint": fingerprint,
        **stats,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "cached_tokens": result.cached_tokens,
        "finish_reason": result.finish_reason,
        "hit_max_tokens": result.hit_max_tokens,
        "max_tokens_used": result.max_tokens_used,
        # Latency of the kept attempt, and of the whole call including retries.
        "first_byte_s": result.first_byte_s,
        "decode_s": result.decode_s,
        "total_s": result.total_s,
        "wall_s": result.wall_s,
        "attempt_log": result.attempt_log,
        "n_chunks": result.n_chunks,
        "http_status": result.http_status,
        "attempts": result.attempts,
        "partial": result.partial,
        "error": result.error,
        "raw_chars": len(result.text or ""),
        "parsed": parsed is not None,
        "repaired_from_truncation": repaired,
        "had_reasoning_block": had_reasoning,
        "parse_note": parse_note,
        "status": status,
    }
    common.write_json(meta_path, meta)
    log.info("  %s pages %d-%d  %s  %s in / %s out tok  first byte %ss  %.1fs%s",
             label, first, last, status, meta["prompt_tokens"], meta["completion_tokens"],
             meta["first_byte_s"], result.wall_s,
             f"  ({result.attempts} attempts)" if result.attempts > 1 else "")
    return meta, parsed


def _sum(values) -> int | float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) if vals else None


def run_cell(cfg: dict, cell: Cell, doc, pages: list[render.Page], schema: dict,
             gold: dict | None, log) -> dict:
    """One case, one document: every window, then the merge. Returns the meta record."""
    out_dir = common.ensure_dir(
        common.out_dir_for(cfg, cell.model_tag, cell.case, doc.doc_type))
    stem = doc.doc_id
    win_dir = common.ensure_dir(out_dir / "_windows" / stem)

    # Every page of the document, exactly once, or nothing: a page missing here
    # would silently fall out of every window and the cell would still read OK.
    n_total = int(doc.n_pages or len(pages))
    by_index = {p.index: p for p in pages}
    if sorted(by_index) != list(range(1, n_total + 1)):
        missing = sorted(set(range(1, n_total + 1)) - set(by_index))
        raise RuntimeError(f"{doc.doc_id}: page assets do not cover pages 1-{n_total} "
                           f"(missing {missing[:10]}); re-run with --force-ocr")

    wins, plan_how = chunking.plan_for_document(
        cfg, common.ocr_dir_for(cfg, doc.doc_id), n_total)
    budget = output_budget(cfg, gold, log)
    ch = cfg.get("chunking") or {}
    ctx = ch.get("context") or {}
    anchors = sorted({int(a) for a in (ctx.get("anchor_pages") or []) if 1 <= int(a) <= n_total})
    carry = bool(ctx.get("carry_forward", False)) and len(wins) > 1
    concurrency = max(1, int(ch.get("concurrency", 1)))
    if carry and concurrency > 1:
        log.info("  concurrency %d -> 1: context.carry_forward needs each window's "
                 "predecessors first", concurrency)
        concurrency = 1

    log.info("  %d page(s) -> %d window(s) %s (%s, overlap %s), reference pages %s, "
             "carry-forward %s, budget %s tokens",
             n_total, len(wins), " ".join(f"{a}-{b}" for a, b in wins), plan_how,
             ch.get("overlap_pages", 0), (anchors if len(wins) > 1 else None) or "none",
             "on" if carry else "off",
             budget or (cfg.get("inference") or {}).get("max_tokens", 8192))

    merger = chunking.Merger(cfg)
    prune = bool(ch.get("drop_out_of_window", True)) and len(wins) > 1

    def keep_in_window(m: dict, parsed: Any, first: int, last: int) -> Any:
        """What this window may contribute: only what it says came from its pages."""
        if not prune:
            return parsed
        kept, n = chunking.prune_out_of_window(parsed, first, last, cfg)
        m["dropped_out_of_window"] = n
        if n:
            log.info("  %s: dropped %d value(s) whose page_ref lies outside pages %d-%d",
                     m["window"], n, first, last)
        return kept

    def go(i: int, first: int, last: int, digest: str, through: int | None):
        wpages = [by_index[k] for k in range(first, last + 1)]
        refs = [by_index[a] for a in anchors if not first <= a <= last] if len(wins) > 1 else []
        return run_window(cfg, cell, doc, wpages, n_total, i, schema, gold, budget,
                          win_dir, log, refs, digest, through)

    t0 = time.monotonic()
    results = []
    if concurrency > 1 and len(wins) > 1:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(go, i, a, b, "", None)
                       for i, (a, b) in enumerate(wins, start=1)]
            results = [f.result() for f in futures]
        for (m, parsed), (first, last) in zip(results, wins):
            merger.add(m["window"], keep_in_window(m, parsed, first, last))
    else:
        # Sequential: window k's digest is the merge of windows 1..k-1.
        prev_last = None
        for i, (first, last) in enumerate(wins, start=1):
            digest = ""
            if carry and merger.merged is not None:
                digest = merger.digest(int(ctx.get("digest_max_chars", 6000)),
                                       int(ctx.get("digest_max_items", 40)))
            m, parsed = go(i, first, last, digest, prev_last)
            merger.add(m["window"], keep_in_window(m, parsed, first, last))
            results.append((m, parsed))
            prev_last = last
    cell_wall_s = round(time.monotonic() - t0, 3)

    wmetas = [m for m, _ in results]
    merged, merge_report = merger.merged, merger.report()

    schema_ok, schema_errors = (True, [])
    if merged is not None and doc.schema:
        schema_ok, schema_errors = common.validate_against_schema(merged, doc.schema)

    json_path = out_dir / f"{stem}.json"
    if merged is not None:
        common.write_json(json_path, merged)
    elif json_path.exists():
        json_path.unlink()  # never leave a stale result next to a failed run

    statuses = [m.get("status") for m in wmetas]
    all_ok = all(s == "OK" for s in statuses)
    any_repaired = any(s == "REPAIRED" for s in statuses)
    n_failed = sum(1 for s in statuses if s not in ("OK", "REPAIRED"))
    if merged is None:
        status = "INVALID"
    elif n_failed:
        status = "PARTIAL"       # some windows produced nothing: pages are missing
    elif any_repaired:
        status = "REPAIRED"
    else:
        status = "OK"

    # Latency. Two numbers, because concurrency makes them differ:
    #   model_latency_s - the windows' own call times added up (each includes
    #                     its retries). What the document costs in model calls.
    #   model_wall_s    - start of the first window to end of the last, as the
    #                     user would wait, including building each request here.
    #                     Just above the sum when concurrency = 1.
    # A reused window contributes its recorded latency to the sum; the wall
    # clock of a resumed cell is not a measurement, so it is flagged.
    model_latency_s = round(sum(m.get("wall_s") or 0.0 for m in wmetas), 3)
    reused = sum(1 for m in wmetas if m.get("reused"))
    wall_resumed = bool(reused)
    if wmetas and reused == len(wmetas):
        # Nothing was generated now (a re-merge): keep the wall clock measured
        # when the windows actually ran, not the milliseconds of this pass.
        try:
            before = common.read_json(out_dir / f"{stem}.meta.json")
        except (ValueError, OSError):
            before = {}
        if before.get("model_wall_s") is not None:
            cell_wall_s = before["model_wall_s"]
            wall_resumed = bool(before.get("wall_is_resumed"))
    fps ="|".join(m.get("conversation_fingerprint", "") for m in wmetas)
    decode = _sum(m.get("decode_s") for m in wmetas)
    out_tok = _sum(m.get("completion_tokens") for m in wmetas)

    rt = cfg.get("runtime") or {}
    meta = {
        "cell": cell.label,
        "doc_id": doc.doc_id,
        "lob": doc.doc_type,
        "model_tag": cell.model_tag,
        "model_id": cell.model_id,
        "served_name": cell.served_name,
        "case": cell.case,
        "endpoint": None if rt.get("mock") else common.base_url_for(cfg, cell.port, cell.base_url),
        "mock": bool(rt.get("mock")),
        "mock_profile": rt.get("mock_profile") if rt.get("mock") else None,
        "run_label": rt.get("run_label"),
        "run_at": common.utc_now_iso(),
        "schema_file": str(doc.schema) if doc.schema else None,
        "conversation_fingerprint": hashlib.sha256(fps.encode("utf-8")).hexdigest()[:16],
        "shared_prompt_sha": common.sha256_text(
            prompting.shared_prompt_span(cfg, schema, doc.doc_type))[:16],
        "n_pages": n_total,
        "n_windows": len(wins),
        "window_plan": [list(w) for w in wins],
        "window_plan_method": plan_how,
        "window_pages": int(ch.get("window_pages", 10)),
        "overlap_pages": int(ch.get("overlap_pages", 0)),
        "reference_pages": anchors if len(wins) > 1 else [],
        "carry_forward": carry,
        "concurrency": concurrency,
        "max_tokens_budget": budget,
        "n_images": _sum(m.get("n_images") for m in wmetas),
        "payload_mb": max((m.get("payload_mb") or 0.0) for m in wmetas) if wmetas else None,
        "prompt_tokens": _sum(m.get("prompt_tokens") for m in wmetas),
        "completion_tokens": out_tok,
        "cached_tokens": _sum(m.get("cached_tokens") for m in wmetas),
        "finish_reason": ("length" if any(m.get("hit_max_tokens") for m in wmetas)
                          else (wmetas[-1].get("finish_reason") if wmetas else None)),
        "hit_max_tokens": any(m.get("hit_max_tokens") for m in wmetas),
        "first_byte_s": wmetas[0].get("first_byte_s") if wmetas else None,
        "model_latency_s": model_latency_s,
        "model_wall_s": cell_wall_s,
        "wall_is_resumed": wall_resumed,
        "windows_reused": reused,
        "latency_per_page_s": round(model_latency_s / n_total, 3) if n_total else None,
        "decode_s": decode,
        "decode_tokens_per_s": round(out_tok / decode, 1) if out_tok and decode else None,
        # kept for the report's existing columns
        "total_s": model_latency_s,
        "attempts": _sum(m.get("attempts") for m in wmetas),
        "retried_windows": sum(1 for m in wmetas if (m.get("attempts") or 1) > 1),
        "error": "; ".join(f"{m['window']}: {m['error']}" for m in wmetas if m.get("error")) or None,
        "dropped_out_of_window": _sum(m.get("dropped_out_of_window") for m in wmetas),
        "windows": [{k: m.get(k) for k in (
            "window", "pages", "reference_pages", "digest_chars", "dropped_out_of_window", "status",
            "prompt_tokens", "completion_tokens", "cached_tokens", "first_byte_s",
            "decode_s", "total_s", "wall_s", "attempts", "finish_reason", "n_images",
            "payload_mb", "error", "reused")}
            for m in wmetas],
        "merge": merge_report,
        "parsed": merged is not None,
        "json_valid_strict": all_ok,
        "repaired_from_truncation": any_repaired,
        "failed_windows": n_failed,
        "schema_ok": schema_ok,
        "schema_errors": schema_errors[:20],
        "schema_error_count": len(schema_errors),
        "status": status,
    }
    common.write_json(out_dir / f"{stem}.meta.json", meta)

    if gold is not None and scoring_enabled(cfg):
        ocr_text = ""
        if (cfg.get("scoring") or {}).get("coverage_report", True):
            ocr_text = "\n\n".join(
                f"--- OCR page {p.index} ---\n{p.markdown()}"
                for p in pages if p.md and p.md.exists())
        score = scoring.score_document(gold, merged, cfg, ocr_text or None)
        score["cell"] = cell.label
        score["doc_id"] = doc.doc_id
        score["status"] = status
        score["json_valid_strict"] = meta["json_valid_strict"]
        # A repaired or partial cell is a lower bound: some output was never
        # generated, so its recall understates what the model can do.
        score["score_is_floor"] = status in ("REPAIRED", "PARTIAL")
        common.write_json(out_dir / f"{stem}.score.json", score)
        meta["field_f1"] = score["field_f1"]
        common.write_json(out_dir / f"{stem}.meta.json", meta)
        log.info("  %s  f1=%.3f  recall=%.3f  precision=%.3f  latency %.1fs (%.2fs/page)"
                 "  merge: %d merged, %d conflicts%s",
                 status, score["field_f1"], score["field_recall"], score["field_precision"],
                 model_latency_s, meta["latency_per_page_s"] or 0.0,
                 merge_report["entities_merged_across_windows"],
                 merge_report["conflict_count"],
                 "  (FLOOR)" if score["score_is_floor"] else "")
    else:
        log.info("  %s  latency %.1fs (%.2fs/page), %s tokens out, schema %s%s",
                 status, model_latency_s, meta["latency_per_page_s"] or 0.0,
                 meta["completion_tokens"], "ok" if schema_ok else
                 f"{len(schema_errors)} error(s)",
                 "" if scoring_enabled(cfg) else "  (scoring off)")

    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None,
                    help="cases to run: 'pdf_md', 'md_only,pdf_only', or '8B/pdf_md'")
    ap.add_argument("--doc", default=None, help="restrict to one doc_id")
    ap.add_argument("--input-dir", default=None, help="override paths.input_dir")
    ap.add_argument("--ocr-only", action="store_true",
                    help="phase 1 only: render pages and run MinerU, no model calls")
    ap.add_argument("--mock", action="store_true", help="no network; synthesize from gold")
    ap.add_argument("--mock-profile", default=None,
                    choices=list(MockClient.PROFILES))
    ap.add_argument("--overwrite", action="store_true",
                    help="redo cells and windows that already have output (default: resume)")
    ap.add_argument("--force-ocr", action="store_true",
                    help="re-render and re-run MinerU (adds a MinerU timing sample)")
    ap.add_argument("--fail-fast", action="store_true")
    ap.add_argument("--label", default=None,
                    help="run conditions recorded in every cell, e.g. 'three cases in "
                         "parallel' - the timing report states them next to the latencies")
    args = ap.parse_args()

    cfg = common.load_config(args.config)
    if args.input_dir:
        cfg["paths"]["input_dir"] = args.input_dir
    rt = cfg.setdefault("runtime", {})
    if args.mock:
        rt["mock"] = True
    if args.mock_profile:
        rt["mock_profile"] = args.mock_profile
        rt["mock"] = True
    if args.overwrite:
        rt["overwrite"] = True
    if args.fail_fast:
        rt["fail_fast"] = True
    if args.label:
        rt["run_label"] = args.label

    log = common.setup_logging("runner", rt.get("log_level", "INFO"))

    docs = common.select_documents(cfg)
    if args.doc:
        docs = [d for d in docs if d.doc_id == args.doc]
    if not docs:
        log.error("no documents found under %s", common.cfg_path(cfg, "input_dir"))
        log.error("expected one folder per LOB holding PDFs, their gold JSONs and "
                  "the LOB schema - see README.md")
        return 2

    plan = cells(cfg, args.only)
    needed = sorted({c.case for c in plan})
    log.info("%d document(s) x %d case(s)%s", len(docs), len(plan),
             "  [MOCK]" if rt.get("mock") else "")

    # ---- phase 1: render + MinerU for every document, before any model call
    failures = 0
    prepared: dict[str, tuple] = {}
    for doc in docs:
        try:
            schema = prompting.load_schema(cfg, doc)
        except FileNotFoundError as exc:
            log.error("skipping %s: %s", doc.doc_id, exc)
            failures += len(plan)
            continue
        try:
            per_case = prepare_document(cfg, doc, needed, log, args.force_ocr)
        except Exception as exc:  # noqa: BLE001
            log.error("could not prepare %s: %s", doc.doc_id, exc)
            if rt.get("fail_fast"):
                raise
            failures += len(plan)
            continue
        prepared[doc.doc_id] = (doc, schema, per_case)

    if args.ocr_only:
        log.info("phase 1 done for %d document(s); --ocr-only, so no model calls",
                 len(prepared))
        log.info("now run:  python src/report.py   (writes the MinerU timing table)")
        return 1 if failures else 0

    # ---- phase 2: the model, case by case within each document
    for doc_id, (doc, schema, per_case) in prepared.items():
        # Gold is read only when something uses it: scoring, or sizing the
        # output budget from it. An extraction + timing run touches no gold.
        need_gold = scoring_enabled(cfg) or (cfg.get("inference") or {}).get("budget_from_gold")
        gold = common.read_json(doc.gold) if (doc.gold and need_gold) else None
        for cell in plan:
            out_dir = common.out_dir_for(cfg, cell.model_tag, cell.case, doc.doc_type)
            done = out_dir / f"{doc.doc_id}.meta.json"
            if done.exists() and not rt.get("overwrite"):
                # Resume only over a result produced the same way. Otherwise a real
                # run would quietly keep a mock cell (or vice versa) and the report
                # would mix synthesized and generated numbers in one table.
                try:
                    prior = common.read_json(done)
                except (ValueError, OSError):
                    prior = {}
                # A window whose REQUEST failed is retried; the cell's finished
                # windows are reused, so only the failed ones cost GPU time.
                net_failed = [w.get("window") for w in prior.get("windows") or []
                              if w.get("error")]
                # A window the parser could not read is re-read from its stored
                # answer (no model call), so a cell missing pages is re-merged.
                unread = [w.get("window") for w in prior.get("windows") or []
                          if w.get("status") == "INVALID" and not w.get("error")]
                if bool(prior.get("mock")) == bool(rt.get("mock")) \
                        and prior.get("status") != "ERROR" and not net_failed and not unread:
                    log.info("%s %s - already done, skipping (--overwrite to redo)",
                             cell.label, doc.doc_id)
                    continue
                if net_failed:
                    log.info("%s %s - retrying %d window(s) whose request failed: %s",
                             cell.label, doc.doc_id, len(net_failed), ", ".join(net_failed))
                if unread:
                    log.info("%s %s - re-reading %d window(s) that did not parse: %s",
                             cell.label, doc.doc_id, len(unread), ", ".join(unread))
                if bool(prior.get("mock")) != bool(rt.get("mock")):
                    log.warning("%s %s - existing output is %s but this run is %s; redoing",
                                cell.label, doc.doc_id,
                                "MOCK" if prior.get("mock") else "real",
                                "MOCK" if rt.get("mock") else "real")
            log.info("%s %s (%s)", cell.label, doc.doc_id, doc.doc_type)
            try:
                run_cell(cfg, cell, doc, per_case[cell.case], schema, gold, log)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.error("%s failed: %s", cell.label, exc)
                log.debug("%s", traceback.format_exc())
                common.write_json(common.ensure_dir(out_dir) / f"{doc.doc_id}.meta.json", {
                    "cell": cell.label, "doc_id": doc.doc_id, "lob": doc.doc_type,
                    "case": cell.case, "status": "ERROR",
                    "mock": bool(rt.get("mock")),
                    "error": f"{type(exc).__name__}: {exc}",
                    "run_at": common.utc_now_iso(),
                })
                if rt.get("fail_fast"):
                    raise

    log.info("done%s", f" with {failures} failed cell(s)" if failures else "")
    log.info("now run:  python src/report.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
