"""Go/no-go checks. Run these before spending GPU time.

    python src/preflight.py --local     inputs, render, budgets. No network.
    python src/preflight.py --pod       both vLLM ports + MinerU. Seconds of GPU.
    python src/preflight.py --all

--local answers: did it find the right PDF, schema and gold; do the pages render
inside the cap; how big is each request; will each cell fit the context window.
--pod answers: are both models actually serving under the names config expects;
does streaming really stream through the proxy (rather than being buffered into
one lump, which would reinstate the 100s timeout); is MinerU's model download
present.

Nothing here writes to output/.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import prompting  # noqa: E402
import render  # noqa: E402

OK, WARN, BAD = "OK  ", "WARN", "FAIL"


class Checks:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, name: str, detail: str = "") -> None:
        self.rows.append((status, name, detail))
        print(f"  [{status}] {name}" + (f"  {detail}" if detail else ""))

    @property
    def failed(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == BAD)

    @property
    def warned(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == WARN)


def _schema_vocabulary(schema: dict) -> set[str]:
    """Values the prompt is SUPPOSED to contain: enum members, key names, defaults.

    These legitimately appear in both the schema block and the gold - the model is
    meant to be given the vocabulary. Only values outside this set count as leaks.
    """
    vocab: set[str] = set()
    for node in prompting._walk_nodes(schema):
        for v in (node.get("enum") or []):
            vocab.add(str(v))
        for key in ("default", "const"):
            if key in node:
                vocab.add(str(node[key]))
        # `examples` are the schema author showing the shape of a value, and are
        # part of what the model is meant to be given.
        for v in (node.get("examples") or []):
            vocab.add(str(v))
        for name in (node.get("properties") or {}):
            vocab.add(str(name))
        for name in (node.get("$defs") or {}):
            vocab.add(str(name))
    return {v.casefold() for v in vocab}


def leaked_gold_values(cfg: dict, schema: dict, gold: dict,
                       doc_type: str = "") -> dict:
    """Distinctive gold answers that appear verbatim in the prompt text.

    An example baked into the prompt is shown to all six cells, so a real answer
    used as an illustration is an answer handed to every model for free - and it
    inflates exactly the fields it illustrates. Enum members and key names are
    excluded: those are schema vocabulary the model is meant to have.

    Only DISTINCTIVE values are checked, because short or common ones ("USD",
    "en", 1) collide with ordinary prompt English and would cry wolf.

    Leaks are split by origin, because the two need different responses:
      harness  - an example hard-coded in the prompt or the derived rules. That
                 is a bug in this repo and must be fixed.
      schema   - an "e.g." inside a description the schema author wrote. That is
                 the schema owner's call: the descriptions are real guidance a
                 production extractor would also ship with, and the effect falls
                 on all six cells equally, so the model COMPARISON is unaffected
                 even though the absolute scores are slightly flattered.
    """
    vocab = _schema_vocabulary(schema)
    harness_text = (prompting.read_text(common.cfg_path(cfg, "system_prompt"))
                    + "\n" + prompting.schema_conventions(schema))
    full_prompt = prompting.build_system_prompt(cfg, "pdf_only", schema)

    values: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif node is not None and not isinstance(node, bool):
            values.add(str(node))

    walk(gold)

    out: dict = {"harness": [], "schema": []}
    for v in sorted(values):
        s = v.strip()
        if not s or s.casefold() in vocab:
            continue
        distinctive = (
            (s.replace(".", "").replace("-", "").isdigit() and len(s.replace(".", "")) >= 4)
            or re.match(r"^\d{4}-\d{2}-\d{2}$", s)
            or len(s) >= 14
        )
        if not distinctive or s not in full_prompt:
            continue
        out["harness" if s in harness_text else "schema"].append(s)
    return out


def count_leaked_leaves(cfg: dict, gold: dict, leaked: set) -> tuple[int, int]:
    """(leaves answerable straight from the prompt, total scored gold leaves)."""
    import scoring
    sspec = scoring.ScoreSpec(cfg)
    counts: dict = {}
    scoring.tally(gold, gold, counts, "", sspec, {})
    total = sum(c[1] for c in counts.values())

    hits = 0

    def walk(node: Any) -> None:
        nonlocal hits
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
        elif isinstance(node, list):
            for v in node:
                walk(v)
        elif isinstance(node, str) and node.strip() in leaked:
            hits += 1

    walk(gold)
    return hits, total


def check_local(cfg: dict, ck: Checks) -> dict:
    print("\n=== inputs " + "=" * 58)
    input_dir = common.cfg_path(cfg, "input_dir")
    if not input_dir.exists():
        ck.add(BAD, "input dir", f"{input_dir} does not exist")
        return {}

    docs, unpaired, orphan_golds = common.discover_documents(
        cfg, tuple((cfg.get("corpus") or {}).get("gold_suffixes")
                   or common.DEFAULT_GOLD_SUFFIXES))
    if not docs:
        ck.add(BAD, "documents",
               f"no PDF found under {input_dir}. Expected one folder per doc type: "
               + ", ".join(str(d.get("dir")) for d in common.doc_type_specs(cfg)))
        return {}

    by_type: dict = {}
    for d in docs:
        by_type.setdefault(d.doc_type, []).append(d)
    for tid, group in by_type.items():
        ck.add(OK, f"doc type {tid}",
               f"{len(group)} document(s) in input/{common.doc_type_dir_name(cfg, tid)}/, "
               f"prompt={Path(str(common.doc_type_spec(cfg, tid).get('prompt'))).name}")

    schemas: dict = {}
    for d in docs:
        bits = [f"pdf={d.pdf.name}"]
        bits.append(f"schema={d.schema.name}" if d.schema else "NO SCHEMA")
        bits.append(f"gold={d.gold.name}" if d.gold else "NO GOLD")
        status = OK if (d.schema and d.gold) else (WARN if d.schema else BAD)
        ck.add(status, f"{d.doc_type}/{d.doc_id}", "  ".join(bits))
        if d.schema:
            try:
                schemas[d.doc_id] = prompting.load_schema(cfg, d)
            except Exception as exc:  # noqa: BLE001
                ck.add(BAD, f"schema unreadable  {d.doc_id}", str(exc))
    for g in orphan_golds:
        ck.add(WARN, "unpaired gold", g.name)

    scored = [d for d in docs if d.gold and d.schema]
    for d in scored:
        gold = common.read_json(d.gold)
        ok, errors = common.validate_against_schema(gold, d.schema)
        if ok:
            ck.add(OK, f"gold validates  {d.doc_id}", "")
        else:
            ck.add(WARN, f"gold vs schema  {d.doc_id}",
                   f"{len(errors)} error(s), first: {errors[0][:90]}")


    print("\n=== render " + "=" * 58)
    rcfg = cfg.get("render") or {}
    per_doc: dict = {}
    for d in docs:
        d.n_pages = render.page_count(d.pdf)
        ocr_dir = common.ensure_dir(common.ocr_dir_for(cfg, d.doc_id))
        try:
            res = render.render_pages(d.pdf, ocr_dir, int(rcfg.get("dpi", 200)),
                                      int(rcfg.get("max_long_side_px", 1024)))
            meta = render.write_render_meta(ocr_dir, d.doc_id, res,
                                            str(rcfg.get("wire_format", "jpeg")),
                                            int(rcfg.get("jpeg_quality", 88)))
        except Exception as exc:  # noqa: BLE001
            ck.add(BAD, f"render {d.doc_id}", str(exc))
            continue
        p1 = res.pages[0]
        ck.add(OK, f"render {d.doc_id}",
               f"{meta['n_pages']} pages, {p1.width}x{p1.height}px, "
               f"{meta['total_wire_b64_mb']:.2f} MB on the wire, "
               f"~{meta['total_image_tokens_est']:,} image tokens")
        cap = float(rcfg.get("max_total_payload_mb") or 0)
        if cap and meta["total_wire_b64_mb"] > cap:
            ck.add(BAD, f"payload {d.doc_id}",
                   f"{meta['total_wire_b64_mb']:.1f} MB over the {cap} MB cap - "
                   f"lower render.jpeg_quality or render.max_long_side_px")
        per_doc[d.doc_id] = {"doc": d, "meta": meta}

    print("\n=== markdown " + "=" * 56)
    import ocr_client
    for d in docs:
        ocr_dir = common.ocr_dir_for(cfg, d.doc_id)
        if ocr_client.has_markdown(ocr_dir, d.n_pages):
            chars = [len((ocr_dir / f"page_{i}.md").read_text(encoding="utf-8").strip())
                     for i in range(1, d.n_pages + 1)]
            thin = [i for i, c in enumerate(chars, 1) if c < 20]
            ck.add(OK if not thin else WARN, f"markdown {d.doc_id}",
                   f"{d.n_pages} pages cached, {sum(chars):,} chars"
                   + (f"; near-empty pages {thin}" if thin else ""))
        else:
            ck.add(WARN, f"markdown {d.doc_id}",
                   "not fetched yet - PDF+MD and MD-only need it "
                   "(preflight --pod, then runner will fetch it)")

    print("\n=== prompt leakage " + "=" * 50)
    for d in scored:
        gold = common.read_json(d.gold)
        leaked = leaked_gold_values(
            cfg, schemas[d.doc_id], gold, d.doc_type)

        if leaked["harness"]:
            ck.add(BAD, f"prompt hard-codes gold  {d.doc_id}",
                   f"{len(leaked['harness'])} real answer(s) are baked into the "
                   f"prompt skeleton or the derived rules - that is a bug in this "
                   f"repo, replace them with values not in the document: "
                   + ", ".join(repr(v) for v in leaked["harness"][:6]))
        else:
            ck.add(OK, f"prompt hard-codes nothing  {d.doc_id}",
                   "no gold answer is baked into the skeleton or the derived rules")

        if leaked["schema"]:
            hits, total = count_leaked_leaves(cfg, gold, set(leaked["schema"]))
            ck.add(WARN, f"schema 'e.g.' leaks  {d.doc_id}",
                   f"{len(leaked['schema'])} example(s) in the schema's own "
                   f"description text are real answers here, making {hits} of "
                   f"{total} scored leaves ({hits / total:.1%}) answerable from the "
                   f"prompt. All six cells get them equally, so the model COMPARISON "
                   f"holds; the absolute scores are flattered by up to that much. "
                   f"Worst: " + ", ".join(repr(v) for v in leaked["schema"][:4]))

    print("\n=== prompt and budget " + "=" * 47)
    # One span per DOCUMENT: each carries its own schema, so spans differ between
    # documents by design. What must hold is that one document's six cells share
    # one span, which is guaranteed here by construction and re-checked in the
    # report against what was actually sent.
    for d in docs:
        if d.doc_id not in schemas:
            continue
        shared = prompting.shared_prompt_span(cfg, schemas[d.doc_id], d.doc_type)
        ck.add(OK, f"shared prompt span  {d.doc_type}/{d.doc_id}",
               f"sha={common.sha256_text(shared)[:16]}  {len(shared):,} chars "
               f"(one span for this document's six cells)")

    fps: dict[str, str] = {}
    for d in docs:
        if d.doc_id not in schemas:
            continue
        schema = schemas[d.doc_id]
        ocr_dir = common.ocr_dir_for(cfg, d.doc_id)
        print(f"\n  {d.doc_id}")
        print(f"    {'cell':16} {'imgs':>5} {'payload MB':>11} {'prompt tok':>11} "
              f"{'+out':>8} {'ctx':>8} {'use':>6}")
        for m in cfg.get("models", []):
            for case in cfg.get("cases", prompting.CASES):
                needs = prompting.CASE_NEEDS[case]
                try:
                    pages = render.require_pages(ocr_dir, case, needs["png"], needs["md"])
                except FileNotFoundError:
                    print(f"    {m['tag'] + '/' + case:16} "
                          f"{'-- markdown not fetched yet --':>50}")
                    continue
                msgs = prompting.build_messages(cfg, case, pages, schema, d.doc_type)
                fps[f"{m['tag']}/{case}"] = prompting.conversation_fingerprint(msgs)
                try:
                    _wire, stats = prompting.materialize(msgs, cfg)
                except prompting.PayloadTooLarge as exc:
                    ck.add(BAD, f"payload {m['tag']}/{case}", str(exc))
                    continue
                est = prompting.estimate_tokens(
                    msgs, pages, case,
                    int((cfg.get("inference") or {}).get("max_tokens", 8192)))
                ctx = int(m.get("max_model_len", 65536))
                use = est["total_tokens_est"] / ctx
                print(f"    {m['tag'] + '/' + case:16} {stats['n_images']:5d} "
                      f"{stats['payload_mb']:11.2f} {est['prompt_tokens_est']:11,d} "
                      f"{est['max_tokens']:8,d} {ctx:8,d} {use:5.0%}")
                if use > 0.85:
                    ck.add(BAD, f"budget {m['tag']}/{case}",
                           f"~{est['total_tokens_est']:,} of {ctx:,} tokens ({use:.0%}). "
                           f"Lower render.max_long_side_px (1024->896->768) or "
                           f"prompt.max_md_chars_per_page, or raise --max-model-len "
                           f"and drop --gpu-memory-utilization to match.")
                elif use > 0.7:
                    ck.add(WARN, f"budget {m['tag']}/{case}",
                           f"~{use:.0%} of the context window")

    if fps:
        by_case: dict[str, set] = {}
        for label, fp in fps.items():
            by_case.setdefault(label.split("/", 1)[1], set()).add(fp)
        same_within = all(len(v) == 1 for v in by_case.values())
        distinct_across = len({next(iter(v)) for v in by_case.values() if len(v) == 1}) \
            == len([v for v in by_case.values() if len(v) == 1])
        ck.add(OK if same_within else BAD, "same input to both models",
               "each case sends byte-identical content to 7B and 8B"
               if same_within else "a case differs between models")
        ck.add(OK if distinct_across else BAD, "cases differ from each other",
               "no two cases send the same content" if distinct_across
               else "two cases produced the same fingerprint")

    return per_doc


def check_pod(cfg: dict, ck: Checks) -> None:
    from vllm_client import VLLMClient
    import ocr_client

    print("\n=== pod " + "=" * 61)
    pod = cfg.get("pod") or {}
    try:
        common.base_url_for(cfg, 8000)
    except ValueError as exc:
        ck.add(BAD, "pod id", str(exc))
        return
    ck.add(OK, "pod id", str(pod.get("id")))
    if not pod.get("api_key"):
        ck.add(WARN, "api key", "pod.api_key is null - the RunPod proxy is PUBLIC, "
                                "anyone with the pod id can use your GPU")

    inf = dict(cfg.get("inference") or {})
    for m in cfg.get("models", []):
        tag, port = str(m["tag"]), int(m["port"])
        base = common.base_url_for(cfg, port)
        client = VLLMClient(base, str(m.get("served_name") or m["model_id"]),
                            pod.get("api_key"), inf, bool(pod.get("verify_tls", True)))
        ok, detail = client.health()
        ck.add(OK if ok else BAD, f"{tag} :{port}", detail)
        if not ok:
            continue

        # A real streaming call. Two things are being proven: time-to-first-byte
        # is inside the budget, and MULTIPLE chunks actually arrive. If the proxy
        # buffered the stream into one lump, n_chunks would be 1 and the 100s
        # idle timeout would be back in play on a real request.
        probe = [{"role": "user",
                  "content": [{"type": "text", "text": "Reply with the single word: ready"}]}]
        t0 = time.monotonic()
        res = client.complete(probe, max_tokens=16)
        if res.error:
            ck.add(BAD, f"{tag} streaming probe", res.error)
            continue
        multi = res.n_chunks > 1
        ck.add(OK if multi else WARN, f"{tag} streaming probe",
               f"first byte {res.first_byte_s}s, {res.n_chunks} chunks, "
               f"{time.monotonic() - t0:.1f}s total"
               + ("" if multi else "  <- one chunk only: the proxy may be buffering, "
                                   "which would reinstate the 100s idle timeout"))
        if res.usage:
            ck.add(OK, f"{tag} usage reported",
                   f"prompt={res.usage.get('prompt_tokens')} "
                   f"completion={res.usage.get('completion_tokens')}")
        else:
            ck.add(WARN, f"{tag} usage", "no usage block - set stream_options.include_usage")

    ocr_cfg = cfg.get("ocr") or {}
    if str(ocr_cfg.get("mode", "pod")).lower() == "pod":
        port = int(ocr_cfg.get("port", 8002))
        mc = ocr_client.MinerUClient(common.base_url_for(cfg, port),
                                     pod.get("ocr_api_key"), ocr_cfg,
                                     bool(pod.get("verify_tls", True)))
        try:
            h = mc.health()
        except Exception as exc:  # noqa: BLE001
            ck.add(BAD, f"MinerU :{port}", f"{type(exc).__name__}: {exc}")
            return
        ck.add(OK, f"MinerU :{port}",
               f"{h.get('mineru_version')} on {h.get('device')}, "
               f"free VRAM {h.get('free_vram_gb')} GB")
        if h.get("models_ready"):
            ck.add(OK, "MinerU models", "weights present")
        else:
            ck.add(WARN, "MinerU models",
                   "models_ready is false. MinerU will exit 0 and write nothing. "
                   "Run: mineru-models-download -s huggingface -m pipeline")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--pod", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--input-dir", default=None)
    args = ap.parse_args()
    if not (args.local or args.pod or args.all):
        args.local = True

    cfg = common.load_config(args.config)
    if args.input_dir:
        cfg["paths"]["input_dir"] = args.input_dir
    ck = Checks()

    if args.local or args.all:
        check_local(cfg, ck)
    if args.pod or args.all:
        check_pod(cfg, ck)

    print("\n" + "=" * 70)
    if ck.failed:
        print(f"PREFLIGHT: NO-GO  ({ck.failed} failure(s), {ck.warned} warning(s))")
        for status, name, detail in ck.rows:
            if status == BAD:
                print(f"  FAIL {name}: {detail}")
    else:
        print(f"PREFLIGHT: GO  ({ck.warned} warning(s))")
    print("=" * 70)
    return 1 if ck.failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
