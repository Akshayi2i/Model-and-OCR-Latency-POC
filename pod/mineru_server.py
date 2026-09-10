"""MinerU as a job API on the pod. Port 8002.

    uvicorn mineru_server:app --host 0.0.0.0 --port 8002 --workers 1

WHY A JOB API RATHER THAN A BLOCKING POST. MinerU on a 15-page scan takes 30-150
seconds. A blocking POST sends no bytes for that entire window, and the RunPod
proxy terminates requests that go quiet for roughly 100 seconds - so a blocking
design would work on short documents and fail on long ones, which is the hardest
kind of bug to pin down. Uploading takes seconds and each poll answers instantly,
so both stay far inside the proxy's tolerance.

One worker thread, deliberately: MinerU shares this GPU with two vLLM servers, so
OCR jobs are serialised rather than allowed to pile up against the KV cache.

The proxy is PUBLIC - anyone with the pod id can reach this port - so set
POC3_API_KEY and the client must send it as X-POC3-Key.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Header, HTTPException, Query, UploadFile

import mineru_ocr

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)-7s [%(name)s] %(message)s")
log = logging.getLogger("mineru_server")

API_KEY = os.environ.get("POC3_API_KEY") or None
WORK = Path(os.environ.get("POC3_OCR_WORK", "/workspace/poc3_ocr"))
WORK.mkdir(parents=True, exist_ok=True)
# Jobs are kept in memory so a result can be re-polled; they are small (markdown
# text) and the process is restarted between runs anyway.
JOB_TTL_S = float(os.environ.get("POC3_JOB_TTL_S", 6 * 3600))

app = FastAPI(title="POC3 MinerU")
POOL = ThreadPoolExecutor(max_workers=1)
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()


def _auth(key: str | None) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="bad or missing X-POC3-Key")


def _free_vram_gb() -> float | None:
    try:
        import torch
        if not torch.cuda.is_available():
            return None
        free, _total = torch.cuda.mem_get_info()
        return round(free / (1024 ** 3), 2)
    except Exception:  # noqa: BLE001
        return None


def _models_ready() -> bool:
    """Have MinerU's pipeline weights actually been downloaded?

    /health reporting this is what turns 'MinerU exits 0 and writes nothing' from
    a mystery into a one-line answer.
    """
    roots = [
        os.environ.get("MINERU_MODEL_SOURCE_DIR"),
        os.environ.get("HF_HOME"),
        str(Path.home() / ".cache" / "huggingface"),
        "/workspace/hf",
    ]
    for root in roots:
        if not root:
            continue
        base = Path(root)
        if not base.exists():
            continue
        # Bounded: look only for a MinerU/PDF-Extract-Kit repo dir holding weights,
        # rather than rglob-ing an entire multi-gigabyte HF cache on every /health.
        for hub in (base / "hub", base):
            if not hub.is_dir():
                continue
            for repo in hub.glob("*PDF-Extract-Kit*"):
                if any(repo.rglob("*.pth")) or any(repo.rglob("*.safetensors")):
                    return True
            for repo in hub.glob("*[Mm]iner[Uu]*"):
                if any(repo.rglob("*.pth")) or any(repo.rglob("*.safetensors")):
                    return True
    return False


def _reap() -> None:
    now = time.time()
    with LOCK:
        for jid in [j for j, s in JOBS.items() if now - s.get("t0", now) > JOB_TTL_S]:
            state = JOBS.pop(jid, None)
            if state and state.get("pdf"):
                Path(state["pdf"]).unlink(missing_ok=True)


@app.get("/health")
def health() -> dict:
    with LOCK:
        running = sum(1 for s in JOBS.values() if s["status"] == "running")
        queued = sum(1 for s in JOBS.values() if s["status"] == "queued")
    return {
        "status": "ok",
        "mineru_version": mineru_ocr.mineru_version(),
        "device": mineru_ocr.resolve_device("auto"),
        "models_ready": _models_ready(),
        "free_vram_gb": _free_vram_gb(),
        "jobs": {"running": running, "queued": queued, "total": len(JOBS)},
        "auth_required": bool(API_KEY),
    }


def _work(job_id: str, pdf: Path, n_pages: int, backend: str, lang: str,
          device: str, keep_discarded: bool) -> None:
    def progress(page: int, of: int) -> None:
        with LOCK:
            if job_id in JOBS:
                JOBS[job_id]["progress"] = {"page": page, "of": of}

    with LOCK:
        JOBS[job_id]["status"] = "running"
    t0 = time.time()
    try:
        pages = mineru_ocr.run_mineru(pdf, n_pages, backend, lang, device,
                                      keep_discarded, progress)
        with LOCK:
            JOBS[job_id].update(
                status="done",
                pages=[{"page": i, "markdown": md} for i, md in enumerate(pages, 1)],
                chars_per_page=[len((m or "").strip()) for m in pages],
                elapsed_s=round(time.time() - t0, 1),
            )
        log.info("job %s done in %.1fs (%d pages)", job_id, time.time() - t0, len(pages))
    except Exception as exc:  # noqa: BLE001 - the client needs the reason, not a 500
        with LOCK:
            JOBS[job_id].update(status="error", error=f"{type(exc).__name__}: {exc}",
                                elapsed_s=round(time.time() - t0, 1))
        log.exception("job %s failed", job_id)
    finally:
        Path(pdf).unlink(missing_ok=True)


@app.post("/ocr")
async def submit(
    pdf: UploadFile = File(...),
    backend: str = Query("pipeline"),
    lang: str = Query("en"),
    device: str = Query("cuda"),
    keep_discarded: bool = Query(True),
    x_poc3_key: str | None = Header(None),
) -> dict:
    _auth(x_poc3_key)
    _reap()

    job_id = uuid.uuid4().hex[:16]
    dest = WORK / f"{job_id}.pdf"
    with open(dest, "wb") as fh:
        shutil.copyfileobj(pdf.file, fh)

    try:
        n_pages = mineru_ocr.page_count(dest)
    except Exception as exc:  # noqa: BLE001
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"not a readable PDF: {exc}") from exc

    with LOCK:
        JOBS[job_id] = {"status": "queued", "t0": time.time(), "n_pages": n_pages,
                        "filename": pdf.filename, "pdf": str(dest), "progress": None}
    POOL.submit(_work, job_id, dest, n_pages, backend, lang, device, keep_discarded)
    log.info("job %s queued: %s (%d pages)", job_id, pdf.filename, n_pages)
    return {"job_id": job_id, "n_pages": n_pages, "status": "queued"}


@app.get("/ocr/{job_id}")
def status(job_id: str, x_poc3_key: str | None = Header(None)) -> dict:
    _auth(x_poc3_key)
    with LOCK:
        state = JOBS.get(job_id)
        if not state:
            raise HTTPException(status_code=404, detail="unknown job_id")
        out = {k: v for k, v in state.items() if k != "pdf"}
    out["elapsed_s"] = out.get("elapsed_s") or round(time.time() - out["t0"], 1)
    return out


@app.delete("/ocr/{job_id}")
def drop(job_id: str, x_poc3_key: str | None = Header(None)) -> dict:
    _auth(x_poc3_key)
    with LOCK:
        state = JOBS.pop(job_id, None)
    if state and state.get("pdf"):
        Path(state["pdf"]).unlink(missing_ok=True)
    return {"deleted": bool(state)}
