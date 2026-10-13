"""PDF -> page images, and page images -> what actually goes on the wire.

Two formats on purpose:
  * PNG on disk. Lossless, so the archived page is exactly what was rendered and
    its sha256 is stable across runs. This is what you open when a number looks
    wrong.
  * JPEG in the request. 15 pages of 1024px PNG is 8-12 MB of base64 per request,
    sent six times; at JPEG q88 it is 2-3 MB. The RunPod proxy and the prefill
    latency both care.

render_pages is ported from D:/POC2/qwen3vl-poc/src/run_ocr.py. Its hard assert
on the long-side cap is kept: it is the only thing standing between this POC and
a 40k-image-token prompt that silently overruns max_model_len.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from common import ensure_dir, pymupdf, read_text, utc_now_iso, write_json

_PAGE_RE = re.compile(r"^page_(\d+)$")

# Qwen3-VL packs 16px patches and merges 2x2, so a page costs about
# (W/32) * (H/32) visual tokens - measured at ~770-800 per 1024px letter page on
# the earlier run. (Qwen2.5-VL was 14px -> 28.) Good to within a few percent,
# which is all a budget check needs.
PATCH = 32


@dataclass
class Page:
    """One page of a document, with whichever assets exist for it."""

    index: int
    png: Path | None = None
    md: Path | None = None
    width: int = 0
    height: int = 0
    png_bytes: int = 0
    md_chars: int = 0

    def markdown(self) -> str:
        return read_text(self.md).strip() if self.md and self.md.exists() else ""


@dataclass
class RenderResult:
    pages: list[Page] = field(default_factory=list)
    dpi: int = 0
    max_long_side: int = 0
    elapsed_s: float = 0.0

    @property
    def total_png_bytes(self) -> int:
        return sum(p.png_bytes for p in self.pages)


def page_count(pdf: Path) -> int:
    fitz = pymupdf()
    with fitz.open(str(pdf)) as doc:
        return doc.page_count


def render_pages(pdf: Path, out_dir: Path, dpi: int = 200,
                 max_long_side: int = 1024) -> RenderResult:
    """Render every page to out_dir/page_N.png, capped on the long side.

    The dpi is a starting point; if it would produce an image longer than
    max_long_side on its longest edge, the zoom is reduced so the cap holds. Two
    models must see identically sized pages or the comparison measures image
    resolution rather than the models, so the cap is enforced, not suggested.
    """
    fitz = pymupdf()
    out_dir = ensure_dir(Path(out_dir))

    for stale in out_dir.glob("page_*.png"):
        stale.unlink()

    t0 = time.perf_counter()
    pages: list[Page] = []
    with fitz.open(str(pdf)) as doc:
        for i, page in enumerate(doc, start=1):
            rect = page.rect
            long_side_pt = max(rect.width, rect.height)
            zoom = dpi / 72.0
            if long_side_pt * zoom > max_long_side:
                zoom = max_long_side / long_side_pt
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), alpha=False)
            if max(pix.width, pix.height) > max_long_side:
                raise RuntimeError(
                    f"page {i} rendered to {pix.width}x{pix.height}, over the "
                    f"{max_long_side}px cap. Every cell must see the same page size; "
                    f"lower render.dpi or raise render.max_long_side_px deliberately."
                )
            png = out_dir / f"page_{i}.png"
            pix.save(str(png))
            pages.append(Page(index=i, png=png, width=pix.width, height=pix.height,
                              png_bytes=png.stat().st_size))

    return RenderResult(pages=pages, dpi=dpi, max_long_side=max_long_side,
                        elapsed_s=round(time.perf_counter() - t0, 3))


def image_token_estimate(width: int, height: int, patch: int = PATCH) -> int:
    """Approximate visual tokens one page image costs."""
    if width <= 0 or height <= 0:
        return 0
    return max(1, -(-width // patch)) * max(1, -(-height // patch))


def encode_for_wire(png: Path, fmt: str = "jpeg", quality: int = 88) -> tuple[str, int]:
    """Read an archived PNG and return (base64 payload, raw byte count).

    JPEG re-encoding happens here rather than at render time so the archived page
    stays lossless and its hash stable.
    """
    fmt = (fmt or "jpeg").lower()
    if fmt in ("png",):
        raw = Path(png).read_bytes()
    else:
        fitz = pymupdf()
        pix = fitz.Pixmap(str(png))
        try:
            raw = pix.tobytes("jpeg", jpg_quality=int(quality))
        except TypeError:  # older PyMuPDF without the quality kwarg
            raw = pix.tobytes("jpeg")
    return base64.b64encode(raw).decode("ascii"), len(raw)


def mime_for(fmt: str) -> str:
    return "image/png" if (fmt or "").lower() == "png" else "image/jpeg"


def data_url(b64: str, fmt: str = "jpeg") -> str:
    return f"data:{mime_for(fmt)};base64,{b64}"


def page_assets(ocr_dir: Path, want_png: bool = True, want_md: bool = True) -> list[Page]:
    """Pages found in a document's work folder, ordered by page number.

    POC2's version returned only pages having BOTH a png and an md, which is
    fatal here: pdf_only has no markdown and md_only has no images. Each side is
    requested explicitly and its absence is reported against the case that needs
    it, not silently swallowed.
    """
    ocr_dir = Path(ocr_dir)
    found: dict[int, Page] = {}

    if want_png:
        for png in ocr_dir.glob("page_*.png"):
            m = _PAGE_RE.match(png.stem)
            if m:
                idx = int(m.group(1))
                found.setdefault(idx, Page(index=idx)).png = png
    if want_md:
        for md in ocr_dir.glob("page_*.md"):
            m = _PAGE_RE.match(md.stem)
            if m:
                idx = int(m.group(1))
                found.setdefault(idx, Page(index=idx)).md = md

    pages = [found[i] for i in sorted(found)]

    meta = _load_render_meta(ocr_dir)
    for p in pages:
        if p.png and p.png.exists():
            p.png_bytes = p.png.stat().st_size
            dims = meta.get(str(p.index))
            if dims:
                p.width, p.height = int(dims.get("width", 0)), int(dims.get("height", 0))
        if p.md and p.md.exists():
            p.md_chars = len(p.markdown())
    return pages


def require_pages(ocr_dir: Path, case: str, want_png: bool, want_md: bool) -> list[Page]:
    """page_assets, but refuse to proceed on a case whose inputs are missing."""
    pages = page_assets(ocr_dir, want_png, want_md)
    if not pages:
        raise FileNotFoundError(
            f"no page assets in {ocr_dir} for case {case!r}. "
            f"Run src/preflight.py --local to render pages"
            + (", and src/runner.py --ocr-only to fetch markdown." if want_md else ".")
        )
    if want_png:
        missing = [p.index for p in pages if not (p.png and p.png.exists())]
        if missing:
            raise FileNotFoundError(
                f"case {case!r} needs page images but pages {missing} have none in {ocr_dir}"
            )
    if want_md:
        missing = [p.index for p in pages if not (p.md and p.md.exists())]
        if missing:
            raise FileNotFoundError(
                f"case {case!r} needs MinerU markdown but pages {missing} have none in "
                f"{ocr_dir}. Run src/runner.py --ocr-only for this document."
            )
    return pages


def _render_meta_path(ocr_dir: Path) -> Path:
    return Path(ocr_dir) / "render_meta.json"


def _load_render_meta(ocr_dir: Path) -> dict:
    p = _render_meta_path(ocr_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8")).get("pages", {})
    except (json.JSONDecodeError, OSError):
        return {}


def write_render_meta(ocr_dir: Path, doc_id: str, result: RenderResult,
                      wire_format: str, jpeg_quality: int) -> dict:
    """Record page dimensions and wire sizes so preflight need not re-render."""
    pages: dict[str, Any] = {}
    wire_total = 0
    for p in result.pages:
        _b64, nbytes = encode_for_wire(p.png, wire_format, jpeg_quality)
        wire_total += nbytes
        pages[str(p.index)] = {
            "width": p.width,
            "height": p.height,
            "png_bytes": p.png_bytes,
            "wire_bytes": nbytes,
            "image_tokens_est": image_token_estimate(p.width, p.height),
        }
    meta = {
        "doc_id": doc_id,
        "n_pages": len(result.pages),
        "dpi": result.dpi,
        "max_long_side_px": result.max_long_side,
        "wire_format": wire_format,
        "jpeg_quality": jpeg_quality,
        "total_png_bytes": result.total_png_bytes,
        "total_wire_bytes": wire_total,
        # base64 inflates by 4/3, and that is what actually crosses the proxy.
        "total_wire_b64_mb": round(wire_total * 4 / 3 / (1024 * 1024), 3),
        "total_image_tokens_est": sum(v["image_tokens_est"] for v in pages.values()),
        # Wall time to rasterise every page on THIS machine (PyMuPDF, CPU).
        "render_s": result.elapsed_s,
        "rendered_at": utc_now_iso(),
        "pages": pages,
    }
    write_json(_render_meta_path(ocr_dir), meta)
    return meta


def run_pymupdf_text(pdf: Path) -> list[str]:
    """Embedded text layer, one string per page. No OCR.

    Only useful as a dry-run stand-in: a scanned declaration has no text layer
    and every page comes back empty.
    """
    fitz = pymupdf()
    out: list[str] = []
    with fitz.open(str(pdf)) as doc:
        for page in doc:
            out.append(page.get_text("text") or "")
    return out
