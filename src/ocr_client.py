"""Clients for MinerU on the pod.

Two servers are supported (ocr.mode):
  mineru_api - MinerU 4's own HTTP API (`mineru-api`, /v1/tasks). Per-page text
               comes from its structured_content, timing from its own
               parse.duration_ms. What the provisioned pod runs.
  pod        - this repo's pod/mineru_server.py job API around the MinerU 2.x CLI.
Either way this side uploads the PDF, polls until the job finishes, and caches
the per-page markdown under <ocr_dir>/<doc_id>/page_N.md.

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

from common import (base_url_for, ensure_dir, ocr_dir_for, read_json,
                    utc_now_iso, write_json, write_text)
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

    def run(self, pdf: Path, n_pages: int, log=None) -> tuple[list[str], dict]:
        """Submit, poll to completion; return (one markdown string per page, timing).

        timing holds both views of the same job: what the client waited
        (upload_s, client_total_s - these include the network, the proxy and the
        polling interval) and what the pod measured (processing_s, queue_wait_s,
        cli_s, parse_s - MinerU's own work, network excluded).
        """
        t0 = time.perf_counter()
        job = self.submit(pdf)
        upload_s = round(time.perf_counter() - t0, 3)
        job_id = job.get("job_id")
        if not job_id:
            raise MinerUError(f"service returned no job_id: {job}")
        if log:
            log.info("MinerU job %s queued (%s pages, upload %.1fs)",
                     job_id, job.get("n_pages", "?"), upload_s)

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

        client_total_s = round(time.perf_counter() - t0, 3)
        pages = state.get("pages") or []
        md = [str(p.get("markdown") or "") for p in
              sorted(pages, key=lambda p: int(p.get("page", 0)))]
        if len(md) != n_pages:
            raise MinerUError(
                f"MinerU returned {len(md)} pages for a {n_pages}-page PDF. A page "
                f"offset here would corrupt pdf_md and md_only identically and "
                f"invisibly, so this is fatal rather than a warning."
            )
        server = state.get("timing") or {}
        timing = {
            "job_id": job_id,
            "n_pages": n_pages,
            "pdf_bytes": Path(pdf).stat().st_size,
            "upload_s": upload_s,
            "client_total_s": client_total_s,
            "poll_interval_s": interval,
            # A pod running the older server reports elapsed_s rounded to 0.1s and
            # no breakdown; the fields are simply absent then.
            "processing_s": state.get("elapsed_s"),
            "queue_wait_s": state.get("queue_wait_s"),
            "cli_s": server.get("cli_s"),
            "parse_s": server.get("parse_s"),
            "n_blocks": server.get("n_blocks"),
            "device": server.get("device"),
            "gpu": state.get("gpu"),
        }
        if log:
            log.info("  MinerU done: %.1fs on the pod (%.2fs/page), %.1fs end to end",
                     timing["processing_s"] or 0.0,
                     (timing["processing_s"] or 0.0) / max(1, n_pages), client_total_s)
        return md, timing


# --------------------------------------------------------------------------- MinerU 4 API


def _block_text(block: dict) -> list[str]:
    """One structured_content block -> its markdown lines (captions, body, notes).

    Titles keep their level as '#' headings - the window planner looks for them.
    Headers, footers and page numbers are KEPT: MinerU files real content there
    too (a policy number in a running header), as the pipeline backend's
    'discarded' blocks taught. The cropped table image (image_source) is dropped.
    """
    def text_of(x) -> str:
        if isinstance(x, str):
            return x.strip()
        if isinstance(x, dict):
            return text_of(x.get("content"))
        if isinstance(x, list):
            return "\n".join(t for t in (text_of(i) for i in x) if t)
        return ""

    out = [t for t in (text_of(c) for c in block.get("captions") or []) if t]
    body = text_of(block.get("content"))
    if body:
        if block.get("type") in ("paragraph_title", "title") and "\n" not in body:
            level = int(block.get("level") or 2)
            body = f"{'#' * max(1, min(level, 6))} {body}"
        out.append(body)
    out += [t for t in (text_of(c) for c in block.get("footnotes") or []) if t]
    return out


def md_from_structured_content(sc: dict, n_pages: int) -> list[str]:
    """MinerU 4 structured_content -> one markdown string per page.

    Pages are attributed by page_idx (0-based), never by order, so a page MinerU
    skipped stays empty in its own slot instead of shifting every later page.
    """
    per_page: list[list[str]] = [[] for _ in range(n_pages)]
    for page in (sc or {}).get("pages") or []:
        idx = int(page.get("page_idx", -1))
        if 0 <= idx < n_pages:
            for block in page.get("blocks") or []:
                per_page[idx] += _block_text(block)
    # MinerU escapes "$" as "\$" (a LaTeX-math guard). On a policy every "$" is
    # currency, and a model asked to copy values verbatim would copy the
    # backslash into raw fields.
    return ["\n\n".join(lines).strip().replace("\\$", "$") for lines in per_page]


def _iso(ts: str | None) -> float | None:
    if not ts:
        return None
    from datetime import datetime
    try:
        return datetime.fromisoformat(str(ts).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


class MinerUApiClient:
    """MinerU 4's own HTTP API (mineru-api): POST /v1/tasks, poll, fetch result.

    Asynchronous by design, so a long scan never holds a silent connection open
    through the proxy. Timing comes from MinerU itself: parse.duration_ms is its
    processing time for the document, and created/started_at give the queue
    wait - neither includes the upload or this side's polling.
    """

    DONE = {"completed", "succeeded", "success", "done"}
    FAILED = {"failed", "error", "cancelled", "canceled", "expired"}

    def __init__(self, base_url: str, api_key: str | None = None,
                 ocr_cfg: dict | None = None, verify_tls: bool = True):
        self.base_url = base_url.rstrip("/")
        self.cfg = ocr_cfg or {}
        self.verify = verify_tls
        self.session = requests.Session()
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.tier = str(self.cfg.get("tier", "advanced"))

    def health(self) -> dict:
        r = self.session.get(f"{self.base_url}/v1/health", headers=self.headers,
                             timeout=30, verify=self.verify)
        r.raise_for_status()
        return r.json()

    def tiers(self) -> list[str]:
        r = self.session.get(f"{self.base_url}/v1/tiers", headers=self.headers,
                             timeout=30, verify=self.verify)
        r.raise_for_status()
        return [t.get("id") for t in r.json().get("data") or []]

    def run(self, pdf: Path, n_pages: int, log=None) -> tuple[list[str], dict]:
        params = [("tier", self.tier),
                  ("ocr_mode", str(self.cfg.get("ocr_mode", "ocr"))),
                  ("filename", Path(pdf).name),
                  ("output_formats", "structured_content")]
        t0 = time.perf_counter()
        r = self.session.post(
            f"{self.base_url}/v1/tasks", params=params, data=Path(pdf).read_bytes(),
            headers={**self.headers, "Content-Type": "application/octet-stream"},
            timeout=(30, float(self.cfg.get("upload_timeout_s", 300))), verify=self.verify)
        upload_s = round(time.perf_counter() - t0, 3)
        if r.status_code >= 400:
            raise MinerUError(f"submit failed: HTTP {r.status_code}: {r.text[:500]}")
        task_id = (r.json() or {}).get("task_id")
        if not task_id:
            raise MinerUError(f"MinerU returned no task_id: {r.text[:300]}")
        if log:
            log.info("MinerU task %s queued (%d pages, tier %s, upload %.1fs)",
                     task_id, n_pages, self.tier, upload_s)

        interval = float(self.cfg.get("poll_interval_s", 2))
        deadline = time.monotonic() + float(self.cfg.get("job_timeout_s", 3600))
        last = None
        while True:
            if time.monotonic() > deadline:
                raise MinerUError(f"task {task_id} did not finish within the timeout")
            s = self.session.get(f"{self.base_url}/v1/tasks/{task_id}", headers=self.headers,
                                 timeout=(30, 60), verify=self.verify)
            if s.status_code >= 400:
                raise MinerUError(f"poll failed: HTTP {s.status_code}: {s.text[:300]}")
            state = s.json()
            status = str(state.get("status") or "").lower()
            if status in self.DONE:
                break
            if status in self.FAILED:
                errs = [f.get("error") for f in state.get("files") or [] if f.get("error")]
                raise MinerUError(f"MinerU task {task_id} {status}: {errs or state}")
            if log and status != last:
                log.info("  MinerU %s", status)
                last = status
            time.sleep(interval)

        res = self.session.get(f"{self.base_url}/v1/tasks/{task_id}/result",
                               params={"response_format": "json"}, headers=self.headers,
                               timeout=(30, 600), verify=self.verify)
        if res.status_code >= 400:
            raise MinerUError(f"result fetch failed: HTTP {res.status_code}: {res.text[:300]}")
        client_total_s = round(time.perf_counter() - t0, 3)
        data = res.json()
        files = data.get("files") or []
        if len(files) != 1:
            raise MinerUError(f"expected one file in the result, got {len(files)}")
        f = files[0]
        if str(f.get("status") or "").lower() not in self.DONE:
            raise MinerUError(f"file status {f.get('status')}: {f.get('error')}")
        sc = (f.get("content") or {}).get("structured_content")
        if not isinstance(sc, dict):
            raise MinerUError("result has no structured_content - request it in output_formats")
        reported = int((sc.get("metadata") or {}).get("document", {}).get("page_count") or n_pages)
        if reported != n_pages:
            raise MinerUError(
                f"MinerU saw {reported} pages in a {n_pages}-page PDF. A page offset here "
                f"would corrupt pdf_md and md_only identically and invisibly, so this is "
                f"fatal rather than a warning.")
        md = md_from_structured_content(sc, n_pages)

        parse = f.get("parse") or {}
        created, started, finished = (_iso(data.get(k)) for k in
                                      ("created_at", "started_at", "finished_at"))
        processing = (parse.get("duration_ms") / 1000.0 if parse.get("duration_ms") is not None
                      else (finished - started if finished and started else None))
        timing = {
            "task_id": task_id,
            "engine": f"mineru-api {parse.get('parser_version') or ''} tier={self.tier}".strip(),
            "n_pages": n_pages,
            "pdf_bytes": Path(pdf).stat().st_size,
            "upload_s": upload_s,
            "client_total_s": client_total_s,
            "poll_interval_s": interval,
            "processing_s": round(processing, 3) if processing is not None else None,
            "queue_wait_s": round(started - created, 3) if started and created else None,
            "gpu": self.cfg.get("gpu_label"),
        }
        if log:
            log.info("  MinerU done: %.1fs processing (%.2fs/page), %.1fs end to end",
                     timing["processing_s"] or 0.0,
                     (timing["processing_s"] or 0.0) / max(1, n_pages), client_total_s)
        return md, timing


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
    # The tier is part of the source: markdown from one MinerU tier must never
    # be reused as if another tier had produced it.
    expected_source = {"pod": "mineru-pod", "pymupdf": "pymupdf",
                       "mineru_api": f"mineru-api:{ocr_cfg.get('tier', 'advanced')}"}.get(mode)

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

    if mode in ("pod", "mineru_api"):
        client_cls = MinerUApiClient if mode == "mineru_api" else MinerUClient
        client = client_cls(
            base_url_for(cfg, int(ocr_cfg.get("port", 8002)), ocr_cfg.get("base_url")),
            (cfg.get("pod") or {}).get("ocr_api_key"),
            ocr_cfg,
            bool((cfg.get("pod") or {}).get("verify_tls", True)),
        )
        try:
            pages_md, timing = client.run(doc.pdf, n_pages, log=log)
            source = expected_source
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
            t0 = time.perf_counter()
            pages_md = run_pymupdf_text(doc.pdf)
            timing = {"n_pages": n_pages, "processing_s": round(time.perf_counter() - t0, 3)}
            source = "pymupdf-fallback"
    else:
        t0 = time.perf_counter()
        pages_md = run_pymupdf_text(doc.pdf)
        timing = {"n_pages": n_pages, "processing_s": round(time.perf_counter() - t0, 3)}
        source = "pymupdf"

    if len(pages_md) != n_pages:
        raise MinerUError(f"got {len(pages_md)} page texts for a {n_pages}-page PDF")

    # Every fresh run's timing is kept, so re-running with --force-ocr adds a
    # sample instead of replacing one; the report takes the median.
    meta_path = ocr_dir / "ocr_meta.json"
    runs: list = []
    if meta_path.exists():
        try:
            prior = read_json(meta_path)
            if prior.get("source") == source:
                runs = list(prior.get("timing_runs") or [])
        except (ValueError, OSError):
            runs = []
    chars = [len((t or "").strip()) for t in pages_md]
    timing.update({"source": source, "run_at": utc_now_iso(),
                   "total_chars": sum(chars)})
    runs.append(timing)

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

    write_json(meta_path, {
        "doc_id": doc.doc_id,
        "source": source,
        "backend": ocr_cfg.get("backend"),
        "lang": ocr_cfg.get("lang"),
        "n_pages": n_pages,
        "chars_per_page": chars,
        "near_empty_pages": empty,
        "timing_runs": runs,
    })
    if empty and log:
        log.warning("pages %s produced almost no text - check them against the images",
                    empty)
    return paths
