"""MinerU invocation and per-page markdown reconstruction. Runs ON THE POD.

Ported from D:/POC2/qwen3vl-poc/src/run_ocr.py, which had already paid for the
operational lessons baked in here:

  * MinerU is driven as a subprocess CLI, never imported. Its Python surface
    changes between releases; its CLI does not.
  * Per-page attribution comes from *_content_list.json, not the whole-document
    .md. Without it there is no way to say which markdown belongs to which page,
    and a page-offset bug would corrupt PDF+MD and MD-only identically and
    invisibly.
  * "discarded" blocks are KEPT. MinerU files headers, footers and page numbers
    there - and it also misfiles real content there; on a declarations page a
    premium total landed in it. Dropping them would quietly penalise both
    markdown cases.
  * Exit code 0 with no output written is detected separately from a non-zero
    exit. It is almost always a missing model download, and MinerU's own log is
    the only evidence, so it is surfaced rather than swallowed.
  * The device is passed explicitly. MinerU's auto-detection has silently fallen
    back to CPU on some torch builds, which is about 3x slower.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

log = logging.getLogger("mineru_ocr")


def page_count(pdf: Path) -> int:
    try:
        import pymupdf as fitz
    except ImportError:
        import fitz
    with fitz.open(str(pdf)) as doc:
        return doc.page_count


def md_from_content_list(items: list[dict], n_pages: int,
                         keep_discarded: bool = True) -> list[str]:
    """Group MinerU content_list blocks by page_idx and render markdown per page."""
    per_page: list[list[str]] = [[] for _ in range(n_pages)]
    for block in items:
        idx = int(block.get("page_idx", 0))
        if not 0 <= idx < n_pages:
            continue
        kind = block.get("type")
        if kind == "text":
            text = (block.get("text") or "").strip()
            if not text:
                continue
            level = block.get("text_level")
            per_page[idx].append(f"{'#' * int(level)} {text}" if level else text)
        elif kind == "table":
            body = (block.get("table_body") or "").strip()
            caption = " ".join(block.get("table_caption") or []).strip()
            if caption:
                per_page[idx].append(f"**{caption}**")
            if body:
                per_page[idx].append(body)
            for note in block.get("table_footnote") or []:
                note = (note or "").strip()
                if note:
                    per_page[idx].append(note)
        elif kind == "equation":
            latex = (block.get("text") or "").strip()
            if latex:
                per_page[idx].append(latex)
        elif kind == "discarded":
            if not keep_discarded:
                continue
            text = (block.get("text") or "").strip()
            if text:
                per_page[idx].append(text)
        elif kind == "image":
            caption = " ".join(block.get("image_caption") or []).strip()
            if caption:
                per_page[idx].append(f"![{caption}]()")
    return ["\n\n".join(chunks).strip() for chunks in per_page]


def resolve_device(setting: str | None) -> str:
    setting = (setting or "auto").strip()
    if setting != "auto":
        return setting
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


def mineru_version() -> str:
    exe = shutil.which("mineru") or shutil.which("magic-pdf")
    if not exe:
        return "not installed"
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=60)
        return (p.stdout or p.stderr).strip().splitlines()[0] if (p.stdout or p.stderr) else "unknown"
    except Exception as exc:  # noqa: BLE001
        return f"unknown ({exc})"


def run_mineru(pdf: Path, n_pages: int, backend: str = "pipeline", lang: str = "en",
               device: str | None = None, keep_discarded: bool = True,
               progress_cb=None, timing: dict | None = None) -> list[str]:
    """Run MinerU on one PDF and return one markdown string per page.

    `timing`, if given, is filled with where the time went:
      cli_s    - the MinerU subprocess, start to exit. Includes MinerU's own
                 start-up and model load, which it pays on EVERY run because each
                 run is a fresh process; the report separates that fixed cost
                 from the per-page cost by fitting time against page count.
      parse_s  - rebuilding per-page markdown from content_list.json.
    """
    exe = shutil.which("mineru") or shutil.which("magic-pdf")
    if not exe:
        raise RuntimeError(
            "MinerU CLI not found on PATH. Install it with "
            "pip install 'mineru[pipeline]==2.7.6'"
        )
    timing = timing if timing is not None else {}

    with tempfile.TemporaryDirectory(prefix="mineru_") as tmp:
        tmp_dir = Path(tmp)
        dev = resolve_device(device)
        cmd = [exe, "-p", str(pdf), "-o", str(tmp_dir), "-b", backend, "-l", lang,
               "-d", dev]
        log.info("running: %s", " ".join(cmd))
        if progress_cb:
            progress_cb(0, n_pages)
        t_cli = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True)
        timing["cli_s"] = round(time.perf_counter() - t_cli, 3)
        timing["device"] = dev

        if proc.returncode != 0:
            log.error("MinerU stdout:\n%s", proc.stdout[-4000:])
            log.error("MinerU stderr:\n%s", proc.stderr[-4000:])
            raise RuntimeError(
                f"MinerU exited {proc.returncode}. stderr tail:\n{proc.stderr[-1500:]}"
            )

        produced = [q for q in tmp_dir.rglob("*") if q.is_file()]
        if not produced:
            # Exit 0 having written nothing is the signature of a missing model
            # download. MinerU's own log is the only evidence there is.
            log.error("MinerU exited 0 but wrote nothing to %s", tmp_dir)
            log.error("MinerU stdout:\n%s", proc.stdout[-4000:])
            log.error("MinerU stderr:\n%s", proc.stderr[-4000:])
            raise RuntimeError(
                "MinerU exited 0 and produced no output files at all. This is almost "
                "always a missing model download - run "
                "'mineru-models-download -s huggingface -m pipeline' on the pod. "
                f"stderr tail:\n{proc.stderr[-1500:]}"
            )

        content_lists = sorted(tmp_dir.rglob("*_content_list.json"))
        if content_lists:
            t_parse = time.perf_counter()
            items = json.loads(content_lists[0].read_text(encoding="utf-8"))
            log.info("parsed %s (%d blocks)", content_lists[0].name, len(items))
            if progress_cb:
                progress_cb(n_pages, n_pages)
            pages = md_from_content_list(items, n_pages, keep_discarded)
            timing["parse_s"] = round(time.perf_counter() - t_parse, 3)
            timing["n_blocks"] = len(items)
            return pages

        markdowns = sorted(tmp_dir.rglob("*.md"))
        if not markdowns:
            raise RuntimeError(f"MinerU produced no markdown under {tmp_dir}")
        if n_pages == 1:
            log.warning("no content_list.json; using whole-document markdown for page 1")
            return [markdowns[0].read_text(encoding="utf-8")]
        raise RuntimeError(
            "MinerU produced no *_content_list.json, so per-page text cannot be "
            f"recovered for a {n_pages}-page PDF. Per-page attribution is required: "
            f"without it, PDF+MD would pair each page image with the wrong page's "
            f"text. Check the MinerU version."
        )
