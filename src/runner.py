"""The 2x3 driver: render, prompt, generate, parse, score, write.

    python src/runner.py                       # every model x every case
    python src/runner.py --only 8B/pdf_md      # one cell
    python src/runner.py --mock                # whole matrix, no GPU
    python src/runner.py --overwrite           # redo cells that already have output

ORDER INSIDE A CELL MATTERS. raw.txt is written before anything is parsed, so a
crash in parsing or scoring can never cost a GPU call. Everything downstream of
raw.txt - repair, schema validation, scoring, the report - can then be re-run for
free, which is why a scoring config change never requires re-running a model.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

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


def prepare_document(cfg: dict, doc, needed_cases: list[str], log,
                     force_ocr: bool = False) -> dict[str, list[render.Page]]:
    """Render pages and fetch markdown, but only what the enabled cases need.

    So `--only 7B/pdf_only` never touches the pod's MinerU port at all.
    """
    doc.n_pages = doc.n_pages or render.page_count(doc.pdf)
    ocr_dir = common.ensure_dir(common.ocr_dir_for(cfg, doc.doc_id))
    rcfg = cfg.get("render") or {}

    want_png = any(prompting.CASE_NEEDS[c]["png"] for c in needed_cases)
    want_md = any(prompting.CASE_NEEDS[c]["md"] for c in needed_cases)

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
            log.info("  %d pages, %.2f MB on the wire, ~%s image tokens",
                     meta["n_pages"], meta["total_wire_b64_mb"],
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
    log.info("  endpoint %s  model %s", base, cell.served_name)
    return VLLMClient(base, cell.served_name, pod.get("api_key"),
                      cfg.get("inference") or {}, bool(pod.get("verify_tls", True)))


def output_budget(cfg: dict, gold: dict | None, log) -> int | None:
    """How many tokens this document's answer needs, sized from its own gold.

    A budget too small to hold a complete answer does not measure extraction, it
    measures where the ceiling was: the cell is cut off mid-JSON, repaired to a
    prefix and scored as a floor. The gold is the only honest estimate of how
    long a complete answer is. Because it yields ONE number per document, every
    cell of that document gets the same budget and the comparison is unaffected.

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


def run_cell(cfg: dict, cell: Cell, doc, pages: list[render.Page], schema: dict,
             gold: dict | None, log) -> dict:
    """One model, one case, one document. Returns the meta record."""
    out_dir = common.ensure_dir(
        common.out_dir_for(cfg, cell.model_tag, cell.case, doc.doc_type))
    stem = doc.doc_id
    raw_path = out_dir / f"{stem}.raw.txt"

    messages = prompting.build_messages(cfg, cell.case, pages, schema, doc.doc_type)
    fingerprint = prompting.conversation_fingerprint(messages)
    wire, stats = prompting.materialize(messages, cfg)

    # A pdf_only run that quietly sent zero images would read as a model failure.
    expected_images = 0 if cell.case == "md_only" else len(pages)
    if stats["n_images"] != expected_images:
        raise RuntimeError(
            f"{cell.label}: built {stats['n_images']} images but case {cell.case!r} "
            f"needs exactly {expected_images}"
        )
    if stats["n_images"] > cell.max_images:
        raise RuntimeError(
            f"{cell.label}: {stats['n_images']} images exceeds the server's "
            f"--limit-mm-per-prompt of {cell.max_images}; vLLM will reject this. "
            f"Raise models[].max_images AND the server flag together."
        )

    common.write_json(out_dir / f"{stem}.request.json", {
        "cell": cell.label,
        "model_id": cell.model_id,
        "served_name": cell.served_name,
        "case": cell.case,
        "conversation_fingerprint": fingerprint,
        "shared_prompt_sha": common.sha256_text(
            prompting.shared_prompt_span(cfg, schema, doc.doc_type))[:16],
        **stats,
        "messages": prompting.redact(messages),
    })

    client = _client_for(cfg, cell, gold, log)

    def keep_partial(text: str, attempt: int) -> None:
        p = out_dir / f"{stem}.raw.attempt{attempt}.txt"
        common.write_text(p, text)
        log.warning("  kept %d chars of partial output at %s", len(text), p.name)

    budget = output_budget(cfg, gold, log)
    log.info("  %d images, %.2f MB payload, fingerprint %s, budget %s tokens",
             stats["n_images"], stats["payload_mb"], fingerprint,
             budget or (cfg.get("inference") or {}).get("max_tokens", 8192))
    result = client.complete(wire, max_tokens=budget, on_partial=keep_partial)

    # raw.txt first, always, before anything can fail.
    common.write_text(raw_path, result.text or "")

    parsed, parse_note, repaired = common.recover_json_object(result.text or "")
    _stripped, had_reasoning = common.strip_reasoning(result.text or "")
    schema_ok, schema_errors = (True, [])
    if parsed is not None and doc.schema:
        # Validate against THIS document's schema, not a global one - each PDF
        # carries its own canonical structure.
        schema_ok, schema_errors = common.validate_against_schema(parsed, doc.schema)

    json_path = out_dir / f"{stem}.json"
    if parsed is not None:
        common.write_json(json_path, parsed)
    elif json_path.exists():
        json_path.unlink()  # never leave a stale result next to a failed run

    meta = {
        "cell": cell.label,
        "doc_id": doc.doc_id,
        "model_tag": cell.model_tag,
        "model_id": cell.model_id,
        "served_name": cell.served_name,
        "case": cell.case,
        "endpoint": None if (cfg.get("runtime") or {}).get("mock")
                    else common.base_url_for(cfg, cell.port, cell.base_url),
        "mock": bool((cfg.get("runtime") or {}).get("mock")),
        "mock_profile": (cfg.get("runtime") or {}).get("mock_profile")
                        if (cfg.get("runtime") or {}).get("mock") else None,
        "run_at": common.utc_now_iso(),
        "conversation_fingerprint": fingerprint,
        "shared_prompt_sha": common.sha256_text(
            prompting.shared_prompt_span(cfg, schema, doc.doc_type))[:16],
        "n_pages": len(pages),
        **stats,
        "prompt_tokens": (result.usage or {}).get("prompt_tokens"),
        "completion_tokens": (result.usage or {}).get("completion_tokens"),
        "finish_reason": result.finish_reason,
        "hit_max_tokens": result.hit_max_tokens,
        "max_tokens_used": result.max_tokens_used,
        "first_byte_s": result.first_byte_s,
        "total_s": result.total_s,
        "n_chunks": result.n_chunks,
        "http_status": result.http_status,
        "attempts": result.attempts,
        "partial": result.partial,
        "error": result.error,
        "raw_chars": len(result.text or ""),
        "parsed": parsed is not None,
        "json_valid_strict": (parsed is not None) and not repaired,
        "repaired_from_truncation": repaired,
        "had_reasoning_block": had_reasoning,
        "parse_note": parse_note,
        "schema_ok": schema_ok,
        "schema_errors": schema_errors[:20],
        "schema_error_count": len(schema_errors),
    }

    status = ("OK" if meta["json_valid_strict"] else
              "REPAIRED" if repaired else "INVALID")
    meta["status"] = status
    common.write_json(out_dir / f"{stem}.meta.json", meta)

    if gold is not None:
        ocr_text = ""
        if (cfg.get("scoring") or {}).get("coverage_report", True):
            ocr_text = "\n\n".join(
                f"--- OCR page {p.index} ---\n{p.markdown()}"
                for p in pages if p.md and p.md.exists())
        score = scoring.score_document(gold, parsed, cfg, ocr_text or None)
        score["cell"] = cell.label
        score["doc_id"] = doc.doc_id
        score["status"] = status
        score["json_valid_strict"] = meta["json_valid_strict"]
        # A repaired cell is a lower bound: the tail was never generated, so its
        # recall understates what the model can do. Saying so here stops the
        # report reading a verbose model as a worse one.
        score["score_is_floor"] = bool(repaired)
        common.write_json(out_dir / f"{stem}.score.json", score)
        meta["field_f1"] = score["field_f1"]
        log.info("  %s  f1=%.3f  recall=%.3f  precision=%.3f%s",
                 status, score["field_f1"], score["field_recall"],
                 score["field_precision"], "  (FLOOR)" if repaired else "")
    else:
        log.info("  %s  (no gold - not scored)", status)

    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", default=None,
                    help="cells to run: '8B/pdf_md', '7B', 'md_only', comma separated")
    ap.add_argument("--doc", default=None, help="restrict to one doc_id")
    ap.add_argument("--input-dir", default=None, help="override paths.input_dir")
    ap.add_argument("--mock", action="store_true", help="no network; synthesize from gold")
    ap.add_argument("--mock-profile", default=None,
                    choices=list(MockClient.PROFILES))
    ap.add_argument("--overwrite", action="store_true",
                    help="redo cells that already have output (default: resume)")
    ap.add_argument("--force-ocr", action="store_true", help="re-render and re-OCR")
    ap.add_argument("--fail-fast", action="store_true")
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

    log = common.setup_logging("runner", rt.get("log_level", "INFO"))

    docs = common.select_documents(cfg)
    if args.doc:
        docs = [d for d in docs if d.doc_id == args.doc]
    if not docs:
        log.error("no documents found under %s", common.cfg_path(cfg, "input_dir"))
        log.error("expected a PDF plus its golden canonical JSON (and schema.json)")
        return 2

    plan = cells(cfg, args.only)
    log.info("%d document(s) x %d cell(s)%s",
             len(docs), len(plan), "  [MOCK]" if rt.get("mock") else "")

    failures = 0
    for doc in docs:
        try:
            schema = prompting.load_schema(cfg, doc)
        except FileNotFoundError as exc:
            log.error("skipping %s: %s", doc.doc_id, exc)
            failures += len(plan)
            continue
        gold = common.read_json(doc.gold) if doc.gold else None
        needed = sorted({c.case for c in plan})
        try:
            per_case = prepare_document(cfg, doc, needed, log, args.force_ocr)
        except Exception as exc:  # noqa: BLE001
            log.error("could not prepare %s: %s", doc.doc_id, exc)
            if rt.get("fail_fast"):
                raise
            failures += len(plan)
            continue

        for cell in plan:
            out_dir = common.out_dir_for(cfg, cell.model_tag, cell.case, doc.doc_type)
            done = out_dir / f"{doc.doc_id}.meta.json"
            if done.exists() and not rt.get("overwrite"):
                # Resume only over a result produced the same way. Otherwise a real
                # run would quietly keep a mock cell (or vice versa) and the report
                # would mix synthesized and generated numbers in one table with
                # nothing to distinguish them.
                try:
                    prior_mock = bool(common.read_json(done).get("mock"))
                except (ValueError, OSError):
                    prior_mock = None
                if prior_mock == bool(rt.get("mock")):
                    log.info("%s %s - already done, skipping (--overwrite to redo)",
                             cell.label, doc.doc_id)
                    continue
                log.warning("%s %s - existing output is %s but this run is %s; redoing",
                            cell.label, doc.doc_id,
                            "MOCK" if prior_mock else "real",
                            "MOCK" if rt.get("mock") else "real")
            log.info("%s %s", cell.label, doc.doc_id)
            try:
                run_cell(cfg, cell, doc, per_case[cell.case], schema, gold, log)
            except Exception as exc:  # noqa: BLE001
                failures += 1
                log.error("%s failed: %s", cell.label, exc)
                log.debug("%s", traceback.format_exc())
                common.write_json(out_dir / f"{doc.doc_id}.meta.json", {
                    "cell": cell.label, "doc_id": doc.doc_id, "status": "ERROR",
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
