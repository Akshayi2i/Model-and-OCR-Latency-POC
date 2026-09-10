"""Client for the MinerU OCR service on the pod.

MinerU itself runs on the pod (pod/mineru_server.py); this side uploads the PDF,
polls until the job finishes, and caches the per-page markdown under
work/ocr/<doc_id>/page_N.md.

WHY SUBMIT + POLL RATHER THAN ONE POST. MinerU on a 15-page scan takes 30-150
seconds. A blocking POST would send no bytes for that whole window and the RunPod
proxy would kill it at ~100s - an intermittent failure that only shows up on
longer documents, which is the worst kind to debug. The upload takes seconds and
each poll answers instantly, so both stay far inside the proxy's tolerance.

The markdown is cached the moment it arrives, so MinerU is needed exactly once
per document ever. A later pod outage cannot block re-running pdf_md or md_only,
and pdf_only never depends on it at all.
"""

from __future__ import annotations

import time
from pathlib import Path

import requests

from common import (base_url_for, cfg_path, ensure_dir, ocr_dir_for, read_json,
                    write_json, write_text)
from render import page_count, run_pymupdf_text


class MinerUError(RuntimeError):
    pass


class MinerUClient:
    def __init__(self, base_url: str, api_key: str | None = None,
                 ocr_cfg: dict | None = None, verify_tls: bool = True):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.cfg = ocr_cfg or {}
        self.verify = verify_tls
        self.session = requests.Session()

    def _headers(self) -> dict:
        return {"X-POC3-Key": self.api_key} if self.api_key else {}

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/health", headers=self._headers(),
                             timeout=30, verify=self.verify)
        r.raise_for_status()
        return r.json()

    def submit(self, pdf: Path) -> dict:
        params = {
            "backend": self.cfg.get("backend", "pipeline"),
            "lang": self.cfg.get("lang", "en"),
            "device": self.cfg.get("device", "cuda"),
            "keep_discarded": str(bool(self.cfg.get("keep_discarded_blocks", True))).lower(),
        }
        with open(pdf, "rb") as fh:
            r = self.session.post(
                f"{self.base_url}/ocr",
                headers=self._headers(),
                params=params,
                files={"pdf": (Path(pdf).name, fh, "application/pdf")},
                timeout=(30, float(self.cfg.get("upload_timeout_s", 300))),
                verify=self.verify,
            )
        if r.status_code >= 400:
            raise MinerUError(f"submit failed: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def poll(self, job_id: str) -> dict:
        r = self.session.get(f"{self.base_url}/ocr/{job_id}", headers=self._headers(),
                             timeout=(30, 60), verify=self.verify)
        if r.status_code >= 400:
            raise MinerUError(f"poll failed: HTTP {r.status_code}: {r.text[:500]}")
        return r.json()

    def run(self, pdf: Path, n_pages: int, log=None) -> list[str]:
        """Submit, poll to completion, return one markdown string per page."""
        job = self.submit(pdf)
        job_id = job.get("job_id")
        if not job_id:
            raise MinerUError(f"service returned no job_id: {job}")
        if log:
            log.info("MinerU job %s queued (%s pages)", job_id, job.get("n_pages", "?"))

        interval = float(self.cfg.get("poll_interval_s", 3))
        deadline = time.monotonic() + float(self.cfg.get("job_timeout_s", 900))
        last_progress = None
        while True:
            if time.monotonic() > deadline:
                raise MinerUError(f"job {job_id} did not finish within the timeout")
            state = self.poll(job_id)
            status = state.get("status")
            if status == "done":
                break
            if status == "error":
                raise MinerUError(f"MinerU failed: {state.get('error')}")
            prog = state.get("progress")
            if log and prog and prog != last_progress:
                log.info("  MinerU page %s/%s", prog.get("page"), prog.get("of"))
                last_progress = prog
            time.sleep(interval)

        pages = state.get("pages") or []
        md = [str(p.get("markdown") or "") for p in
              sorted(pages, key=lambda p: int(p.get("page", 0)))]
        if len(md) != n_pages:
            raise MinerUError(
                f"MinerU returned {len(md)} pages for a {n_pages}-page PDF. A page "
                f"offset here would corrupt pdf_md and md_only identically and "
                f"invisibly, so this is fatal rather than a warning."
            )
        return md


def has_markdown(ocr_dir: Path, n_pages: int) -> bool:
    return all((Path(ocr_dir) / f"page_{i}.md").exists() for i in range(1, n_pages + 1))


def ensure_page_markdown(cfg: dict, doc, log=None, force: bool = False) -> list[Path]:
    """Per-page markdown for one document: cached, else fetched, else fallback.

    Returns the page_N.md paths. Raises rather than returning partial results -
    a half-OCR'd document would score as a model failure.
    """
    ocr_cfg = cfg.get("ocr") or {}
    ocr_dir = ensure_dir(ocr_dir_for(cfg, doc.doc_id))
    n_pages = doc.n_pages or page_count(doc.pdf)

    mode = str(ocr_cfg.get("mode", "pod")).lower()
    expected_source = {"pod": "mineru-pod", "pymupdf": "pymupdf"}.get(mode)

    if not force and ocr_cfg.get("cache", True) and has_markdown(ocr_dir, n_pages):
        # Cached markdown is only reusable if it came from the engine currently
        # configured. Silently reusing a dry-run's empty pymupdf text, or a
        # placeholder, as if it were MinerU output would make both markdown cases
        # look like model failures - and nothing in the numbers would say so.
        meta_path = ocr_dir / "ocr_meta.json"
        cached_source = None
        if meta_path.exists():
            try:
                cached_source = read_json(meta_path).get("source")
            except (ValueError, OSError):
                cached_source = None
        if expected_source and cached_source != expected_source:
            if log:
                log.warning(
                    "ignoring cached markdown for %s: it came from %r but ocr.mode=%r "
                    "expects %r. Re-fetching.",
                    doc.doc_id, cached_source or "unknown", mode, expected_source)
        else:
            if log:
                log.info("markdown cache hit for %s (%d pages, source=%s)",
                         doc.doc_id, n_pages, cached_source or "unknown")
            return [ocr_dir / f"page_{i}.md" for i in range(1, n_pages + 1)]

    if mode == "none":
        raise MinerUError(
            "ocr.mode is 'none' but a case needs markdown. Set ocr.mode to 'pod'."
        )

    if mode == "pod":
        client = MinerUClient(
            base_url_for(cfg, int(ocr_cfg.get("port", 8002)), ocr_cfg.get("base_url")),
            (cfg.get("pod") or {}).get("ocr_api_key"),
            ocr_cfg,
            bool((cfg.get("pod") or {}).get("verify_tls", True)),
        )
        try:
            pages_md = client.run(doc.pdf, n_pages, log=log)
            source = "mineru-pod"
        except (MinerUError, requests.RequestException) as exc:
            if not ocr_cfg.get("fallback_to_pymupdf", False):
                raise MinerUError(
                    f"{exc}\n\nMinerU is unavailable and ocr.fallback_to_pymupdf is "
                    f"false, which is correct for a scanned document: the pymupdf "
                    f"fallback reads the embedded text layer, and a scan has none, "
                    f"so it would silently produce empty markdown and make the two "
                    f"markdown cases look like model failures."
                ) from exc
            if log:
                log.warning("MinerU unavailable (%s); falling back to pymupdf text", exc)
            pages_md = run_pymupdf_text(doc.pdf)
            source = "pymupdf-fallback"
    else:
        pages_md = run_pymupdf_text(doc.pdf)
        source = "pymupdf"

    if len(pages_md) != n_pages:
        raise MinerUError(f"got {len(pages_md)} page texts for a {n_pages}-page PDF")

    for stale in ocr_dir.glob("page_*.md"):
        stale.unlink()
    paths: list[Path] = []
    empty: list[int] = []
    for i, text in enumerate(pages_md, start=1):
        p = ocr_dir / f"page_{i}.md"
        write_text(p, (text or "").strip() + "\n")
        paths.append(p)
        if len((text or "").strip()) < 20:
            empty.append(i)

    write_json(ocr_dir / "ocr_meta.json", {
        "doc_id": doc.doc_id,
        "source": source,
        "backend": ocr_cfg.get("backend"),
        "lang": ocr_cfg.get("lang"),
        "n_pages": n_pages,
        "chars_per_page": [len((t or "").strip()) for t in pages_md],
        "near_empty_pages": empty,
    })
    if empty and log:
        log.warning("pages %s produced almost no text - check them against the images",
                    empty)
    return paths
