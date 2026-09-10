# POC3 — PDF→JSON extraction quality: 2 models × 3 input modalities

Which small VLM extracts a Policy Declaration into your canonical JSON more
faithfully, and does MinerU markdown help? Six cells, scored per field against
your golden canonical JSON.

|          | `Qwen2.5-VL-7B-Instruct` → `output/7B/` | `Qwen3-VL-8B-Instruct` → `output/8B/` |
|----------|------------------------------------------|----------------------------------------|
| PDF only | page images                              | page images                            |
| PDF+MD   | images + MinerU markdown, interleaved    | images + MinerU markdown, interleaved  |
| MD only  | MinerU markdown, no images               | MinerU markdown, no images             |

**The measurement is the deliverable.** Six cells only compare models if nothing
else differs between them. One system prompt, one schema block, one set of output
rules, one decoder config, one parser, one scorer — exactly one paragraph and the
shape of the user turn vary by case. `report.py` verifies this and refuses to
present the deltas as results if it does not hold.

## Architecture

```
  WINDOWS  (D:\POC3)                        RUNPOD POD  (RTX PRO 6000, 96 GB)
  ------------------                        ---------------------------------
  render pages (PyMuPDF)
  build prompts from input/schema.json
  stream requests  ──────────────────────►  :8000  vLLM  Qwen2.5-VL-7B
                   ──────────────────────►  :8001  vLLM  Qwen3-VL-8B
  upload PDF, poll ──────────────────────►  :8002  MinerU job API
  parse / repair JSON
  score vs gold, write output/
```

Two constraints shape the whole design:

1. **The RunPod proxy kills any request that sends no bytes for ~100 s.** A
   15-page multimodal prefill and a MinerU run both exceed that. So every model
   call streams (enforced in code, not config), and MinerU is a submit+poll job
   API rather than one long POST.
2. **The two models are different architectures** (`Qwen2_5_VLForConditionalGeneration`
   vs `Qwen3VLForConditionalGeneration`). One install covers both at
   `vllm>=0.11` + `transformers>=4.57`; `setup_pod.sh` asserts that *before*
   downloading 33 GB of weights. All model-specific behaviour stays server-side,
   so this side has zero per-model branching.

## Install (Windows)

```powershell
pip install -r requirements.txt
```

No torch, no vLLM, no transformers, no MinerU locally — all model execution is on
the pod behind an HTTP boundary.

## What goes in `input/`

One folder per document type. Within a type the layout is free - PDFs anywhere,
JSONs anywhere - because pairing is by normalised filename stem and a JSON is
classified as schema or gold by its **content**, not its location.

```
input/
  policy/          Policy Declarations   (4 PDFs, all scanned - MinerU mandatory)
    pdfs/    AUTOP 2.pdf  BOAT 2.pdf  DFIRE Dec Page.pdf  HOME.pdf
    schema/  <stem>.schema.json
    gold/    <stem>.json
  loss runs/       Loss Run              (3 PDFs, digital text layer)
    pdfs/    14864356 2.pdf  AIG_LossRun_Policy.pdf  LibertyMutual-Lossruns.pdf
    schema/  ...   gold/  ...
  acord forms/     ACORD Form            (1 PDF, digital text layer)
    pdfs/    BINDER- ACORD 75.pdf
    schema/  ...   gold/  ...
```

Schema resolution per PDF, first hit wins: a schema whose stem matches the PDF →
the single schema in that doc type's folder → `paths.schema`. So "one schema per
type" needs no renaming, and a single document can still override it.

Output mirrors the three dimensions:

```
output/<model tag>/<case>/<doc type>/<doc>.json
        7B | 8B      PDF only        policy
                     PDF+MD          loss runs
                     MD only         acord forms
```

## How a prompt is assembled

Four blocks, each varying on exactly one axis. Nothing about any document is
hard-coded; the schema half is generated from whichever schema applies.

| Block | Varies by | Source |
|---|---|---|
| role + extraction rules | nothing | `prompts/system_prompt.txt` |
| document-type paragraph | doc type | `prompts/doctype/<type>.txt` |
| modality paragraph | case | `prompts/case_<case>.txt` |
| value conventions, outline, JSON Schema | schema | derived by `prompting.schema_conventions()` |

The user turn adds a per-case preamble (`prompts/user_<case>.txt`) naming the
document type, then the pages themselves.

**Why this keeps the comparison honest.** Within one document type the six cells
share a byte-identical prompt except the modality paragraph, and for any one
document and case both models receive byte-identical input. Prompts differ
*between* document types by design - that is the point of a doc-type-specific
prompt - so `report.prompt_integrity()` groups its checks by type rather than
demanding one hash across the whole run.

### Value conventions are read off the schema, never assumed

A prompt that hard-codes "keep `$2,451` exactly as printed" is right for one
schema and actively wrong for the next. `schema_conventions()` derives the rules
instead, so each document type gets the conventions its own schema declares:

- closed vs open schema (`additionalProperties: false`)
- money as an object `{amount, currency}` **or** as a bare number - the rule
  branches on which shape the schema uses
- ISO dates, detected from `format: date` **or** a `^\d{4}-\d{2}-\d{2}$` pattern
- every scalar `$def` that carries a description, quoted verbatim - when a schema
  says "checkbox: true = marked, false = present and unmarked, null = not
  confidently readable", that sentence *is* the rule
- every `enum`, listed as a closed vocabulary

### What the three document types actually demand

Each doc-type prompt was written against these PDFs, not from a template. The
recurring finding across all three: **the digital text layer is spatially
scrambled**, so label↔value and row↔column binding is visual, not textual. That
is precisely the hypothesis this POC tests.

| Type | What the text layer does | The characteristic failure |
|---|---|---|
| Policy | 4 of 4 are scans with **zero** embedded text | column ownership; `Included` and `***` are values, not nulls; credits print in parentheses; schedules continue across pages |
| Loss run | headers and values arrive in separate streams; row identifiers detach to the end of the page | emitting TOTAL / GRAND TOTAL / ALL YEARS rows as if they were individual claims, double-counting the loss history |
| ACORD | filled values and the pre-printed template are separate layers, so an amount can follow a caption it does not belong to | populating the many empty boxes because their captions exist, manufacturing coverage the insured does not have |

## Run order

Each step is cheap and proves the next one is worth attempting.

### 0. Prove the scorer — no GPU, no network, no pod

```powershell
python src/selftest.py
```

Six assertions against your gold. Every one perturbs the gold in a way that must
not change the score, or must change exactly one field:

| # | Perturbation | Must give | A failure means |
|---|---|---|---|
| 1 | none (gold vs itself) | F1 = 1.000 | normalisation is destroying real values |
| 2 | — | every list of objects reached | a list vanishes from the metric instead of scoring badly |
| 3 | dates reformatted M/D/YYYY ↔ ISO | F1 = 1.000 | date normalisation is incomplete |
| 4 | every list of objects reversed | F1 = 1.000 | **a list lacks a working identity key — the tool names it** |
| 5 | one repeated field dropped | that field recall 0, others 1 | the metric is not field-localised |
| 6 | gold cut at 60% of its characters | repair recovers a partial prefix | truncation repair is broken |

**Assertion 4 is the one that costs real time.** When it fails it names the list;
go add an identity key for it under `scoring.entity_lists` in `config.yaml` and
run again. Tuning that until reversal scores 1.000 is required, and it is free —
do it before the pod exists. Until `input/` is populated the self-test runs
against a built-in fixture, which proves the code but **not** your config; the
entity keys must be tuned against your real gold.

### 1. Local preflight — no network

```powershell
python src/preflight.py --local
```

Prints which file it chose as PDF, schema and gold; validates the gold against
the schema; renders every page; and for each of the six cells prints the image
count, payload size and projected token use against the context window. Refuses
(`NO-GO`) on a payload or budget overrun rather than letting you discover it
after the upload.

### 2. Bring up the pod

RunPod template → **Expose HTTP Ports: `8000,8001,8002`**. Copy `pod/` to
`/workspace/POC3/pod`, then:

```bash
export POC3_API_KEY=<something long and random>
bash setup_pod.sh      # installs, asserts the version floor, pre-downloads weights
bash serve_vllm.sh     # 7B:8000 → health gate → 8B:8001 → health gate → MinerU:8002
```

The proxy is **public** — anyone with your pod id can use your GPU — so
`POC3_API_KEY` is not optional. Put the same value in `config.yaml` under
`pod.api_key` and `pod.ocr_api_key`, along with your `pod.id`.

Startup is sequential behind a health gate on purpose: both servers size their KV
cache from *free* VRAM at startup, so launched together each counts the other's
not-yet-allocated memory as free and they collectively over-commit.

### 3. Pod preflight — seconds of GPU

```powershell
python src/preflight.py --pod
```

Checks both ports serve the model names `config.yaml` expects, then makes a real
streaming call to each. It asserts **more than one chunk arrives** — if the proxy
buffered the stream into a single lump, the 100 s idle timeout would be back in
play on a real request and you would only find out on the full run. Also confirms
MinerU reports `models_ready`.

### 4. Mock the whole matrix — zero GPU

```powershell
python src/runner.py --mock --mock-profile degraded
python src/report.py
```

Synthesizes each answer from your gold and exercises every path: writing,
parsing, truncation repair, schema validation, scoring, reporting. You get a
fully populated `output/` tree and a real `comparison.md`. Run each profile once
— `perfect`, `degraded`, `truncated`, `garbage`. `garbage` must produce a clean
`INVALID` cell, not a stack trace.

### 5. One real cell, read by hand

```powershell
python src/runner.py --only 7B/pdf_only
```

Open its `raw.txt`. This is the only stage where a human reading the output beats
any assertion.

### 6. The full matrix

```powershell
python src/runner.py
python src/report.py
```

Roughly 10–20 minutes of GPU for all six cells.

## Output

```
output/
  7B/  8B/
    PDF only/  PDF+MD/  MD only/
      <doc>.json           the extraction (written even if only a prefix was recoverable)
      <doc>.raw.txt        raw model text, verbatim — written FIRST, before parsing
      <doc>.meta.json      tokens, latency, finish_reason, validity, fingerprint, errors
      <doc>.score.json     per-field P/R/F1 vs gold, entity counts, strict score, coverage
      <doc>.request.json   redacted request (images → name + size + hash), so cells diff
  report/
    comparison.md    the human report
    comparison.json  everything, machine-readable
    matrix.csv       one row per cell
    per_field.csv    field × cell recall grid
```

`raw.txt` is written before anything is parsed, so a crash in scoring can never
cost a GPU call. **Re-scoring never needs the models again**: change
`scoring.entity_lists` or `scoring.ignore_paths`, re-run `python src/report.py`,
and the numbers update for free.

## Reading the report

Three deltas, three questions:

- **8B − 7B, per case** — is the bigger model better, and in the same places?
- **PDF+MD − PDF only** — does MinerU markdown help?
- **PDF+MD − MD only** — do the page images still earn their cost?

Alongside the primary field F1:

- **strict match rate** — index-path exact match. The *gap* between it and F1 says
  how much of a difference is ordering and formatting rather than content.
- **number coverage** — what fraction of the numbers MinerU read off the pages
  reached the output. This is the key diagnostic for MD-only: if it collapses
  there, the markdown lost the table, not the model.
- **entity counts** — gold has 3 vehicles, the cell found 2. A per-field F1 alone
  hides a list dropped whole.
- **`*` on a score** — the output was truncated and repaired to its valid prefix.
  That number is a **floor, not a measurement**. Without this flag, a more verbose
  model reads as a worse one.

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Proxy 502 that looks like "still loading" | server bound `127.0.0.1` | `--host 0.0.0.0` |
| `NO-GO` on budget | too many image tokens | `render.max_long_side_px` 1024→896→768, or `prompt.max_md_chars_per_page` |
| vLLM 400 on image count | `--limit-mm-per-prompt` below the page count | raise it *and* `models[].max_images` together |
| MinerU exits 0, writes nothing | models never downloaded | `mineru-models-download -s huggingface -m pipeline`; `/health` reports `models_ready` |
| Cells score suspiciously alike | prompt drift, or a case run with another's input | `report.py` says so explicitly; check `conversation_fingerprint` in each `meta.json` |
| A whole list scores 0 | no working identity key | run `selftest.py`; assertion 4 names the list |
| Both servers OOM at startup | launched concurrently | start sequentially behind the health gate (`serve_vllm.sh` does) |
| Markdown looks empty or wrong | cached from a different engine | the cache records its `source`; a mismatch is refused and re-fetched automatically |
| Mock and real numbers in one table | resumed across run kinds | refused automatically — a real run never resumes over a mock cell |

## Ported from

This is not greenfield code. The load-bearing parts come from work that already
paid for the lessons:

- `D:/POC2/qwen3vl-poc/src/common.py` — `repair_truncated_json` (rewinds a cut-off
  generation to the last safe cut point and closes open containers, inventing
  nothing), plus the JSON recovery and schema validation around it.
- `D:/POC2/qwen3vl-poc/src/prompting.py` — the per-page interleave and
  `conversation_fingerprint`.
- `D:/POC2/qwen3vl-poc/src/run_ocr.py` — `render_pages`' hard long-side cap, and
  the MinerU operational lore now in `pod/mineru_ocr.py`: per-page attribution
  from `*_content_list.json`, keeping `discarded` blocks (MinerU misfiles real
  content there), and the exit-0-wrote-nothing guard.
- `D:/RunPod/eval_fields.py` — `_align`, the order-invariant entity matcher, and
  the normalisation rules, generalised here from loss-runs to any schema.
