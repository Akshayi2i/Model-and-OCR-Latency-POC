# POC3 — policy extraction with MinerU + Qwen3-VL-8B: quality and time

How well, and how fast, does `Qwen3-VL-8B-Instruct` turn insurance **policy
documents** into your **canonical JSON**, given three kinds of input?

| case | the model receives |
|---|---|
| MD only  | MinerU markdown of every page, no images |
| PDF only | the page images, no OCR text |
| PDF+MD   | each page's image and its MinerU markdown, side by side |

For every document and case it produces the canonical JSON, scores it per field
against your gold JSON, and measures:

- **MinerU time** per document, per page and per 1,000 characters, split into a
  fixed start-up cost and a per-page cost;
- **model latency** per document and case — time to first token, decode speed,
  total, seconds per page;
- **end to end** per case (render + MinerU + model), and a projection of how long
  a policy of N pages takes.

The original two-model run (Qwen2.5-VL-7B vs Qwen3-VL-8B over ACORD forms, loss
runs and policies) is commit `a57bc38` on `main`.

## Architecture

```
  WINDOWS  (D:\POC3)                              RUNPOD POD  (RTX PRO 6000, 96 GB)
  ------------------                              ---------------------------------
  resolve common + LOB schema -> one per LOB
  phase 1  render pages (PyMuPDF)
           upload PDF, poll  ───────────────────►  :8002  MinerU job API   (timed on the pod)
  phase 2  per document, per case, per window:
           stream request    ───────────────────►  :8003  vLLM  Qwen3-VL-8B
           parse / repair JSON
           merge windows -> one canonical JSON
           validate, score vs gold, write output_policies/
  report   comparison.md (accuracy) + timing.md (time)
```

Phase 1 runs MinerU on **every** document before phase 2 makes any model call,
so the two never share the GPU at the same moment and neither timing is
polluted by the other.

Two constraints shape the design:

1. **The RunPod proxy kills any request that sends no bytes for ~100 s.** Every
   model call streams (enforced in code), and MinerU is a submit + poll job API.
2. **Long documents.** A 90-page policy in PDF+MD would be a ~200k-token prompt.
   Instead every document is split into **page windows** (default 10 pages,
   overlapping by 1), each window is one request against the full schema, and
   the window outputs are merged into one canonical JSON. The same windows are
   used for all three cases, so the case comparison is unaffected.

## The test set: `input/policy/`

```
input/policy/
  manifest.json      one entry per document: its PDF, gold JSON, LOB and LOB schema
  pdfs/              the 10 scanned (image-only) policy PDFs
  gold/              one gold JSON per PDF
  schema/            common_model.json + one canonical schema per LOB
  raw data/          the source folders, untouched
```

| id | LOB | pages | PDF |
|---|---|---:|---|
| prop_policy_1 | homeowners | 4 | PROP-Policy-1_scanned.pdf |
| boppr | boppr | 6 | BOPPR_scanned.pdf |
| endorsement_capi056240 | auto | 8 | ENDORSEMENT_CAPI056240_scanned.pdf |
| policy | professional_eo | 28 | Policy_scanned.pdf |
| inmrc | inland_marine | 30 | INMRC_scanned.pdf |
| dec_page | wc | 31 | dec_page_scanned.pdf |
| health_main | health | 50 | HEALTH Main_scanned.pdf |
| shelterpoint_health | health | 50 | shelterpoint_health_scanned.pdf |
| policy_01 | gl | 74 | policy_01_scanned.pdf |
| bop | bopgl | 111 | BOP_scanned.pdf |

**Why a manifest.** Neither a gold file nor a PDF name says which LOB schema it
belongs to (`Policy`, `dec_page`). The raw-data folder it came from was the only
record of that, so `manifest.json` keeps it, and nothing is paired by guessing
names.

**Adding a document.** Put its files in `pdfs/`, `gold/` and `schema/`, and add an
entry to `manifest.json`. A PDF or gold file that sits in those folders but is
missing from the manifest is reported by preflight.

**The common schema.** `common_model.json` is a definitions library: 77 shared
shapes and no fields of its own. Each LOB schema `$ref`s the definitions it uses,
and only those are copied into its resolved schema (48–61 per LOB).

### Alternative layout: one folder per LOB

Leave `lobs.manifest` unset, and each folder directly under `paths.input_dir`
becomes one LOB. Everything in it is discovered by content, so sub-folders and
file names are free.

```
input_policies/
  common.schema.json          the common canonical schema (paths.common_schema)
  Homeowners/                 one folder per LOB
    homeowners.schema.json    that LOB's canonical schema
    HOME 1234.pdf             its policy documents...
    HOME 1234.json            ...and a gold JSON per PDF, paired by file name
  Commercial Auto/
    ...
```

- **Pairing.** A PDF takes the gold JSON whose name matches it after
  normalisation (`HOME 1234.pdf` ↔ `home_1234_golden.json`), else the best
  distinctive-token match.
- **Schema vs gold** is decided by content: a file declaring a JSON Schema
  dialect, or shaped like a schema, is a schema; anything else is a gold.
- **One LOB schema per folder.** A LOB schema may be split across files (the LOB
  schema plus a definitions file it `$ref`s). Only the file that no other schema
  in the folder references counts as the LOB schema. If a folder holds two
  unrelated LOB schemas, preflight fails and names them, rather than guessing.
  The exception is a schema named after a specific PDF, which overrides the LOB
  schema for that document.
- **Resolved schema.** `src/schema_resolver.py` combines `common.schema.json` with
  each LOB schema into one schema per LOB, written to
  `work/schemas/<lob>.resolved.schema.json`. A per-PDF schema is written to
  `<lob>--<pdf>.resolved.schema.json`. It handles the three usual ways a LOB
  schema builds on a common one:
  - `$ref` into the common file (`"../common.schema.json#/$defs/Money"`) — the
    definition is copied in;
  - composition (`"allOf": [{"$ref": "../common.schema.json"}, {...}]`) —
    flattened into one object, so a closed common schema no longer rejects the
    LOB's fields;
  - two standalone schemas — the common top-level fields are merged in.

  Where both define the same field the LOB wins a direct clash. Every overlap and
  clash is listed in `work/schemas/<lob>.resolve_notes.json`, and preflight
  prints a summary. `python src/schema_resolver.py` shows what it did per LOB.
- **Per-LOB prompt (optional).** `prompts/lob/<lob id>.txt` is appended to the
  shared policy paragraph for that LOB. The id is the folder name lower-cased,
  with non-alphanumerics as `_` (`Commercial Auto` → `commercial_auto`).

## How a prompt is assembled

| Block | Varies by | Source |
|---|---|---|
| role + extraction rules | nothing | `prompts/system_prompt.txt` |
| policy paragraph (+ LOB paragraph) | LOB | `prompts/doctype/policy.txt`, `prompts/lob/<lob>.txt` |
| modality paragraph | case | `prompts/case_<case>.txt` |
| value conventions, outline, JSON Schema | resolved schema | derived by `prompting.schema_conventions()` |
| window note (user turn) | window | `prompts/window.txt` |

Within a document every case and window shares a byte-identical system prompt
except the modality paragraph. `report.prompt_integrity()` verifies this from
what was actually sent. Because the system prompt is identical across windows,
vLLM's prefix cache serves most of it after the first request; the timing report
shows the cached tokens.

## Keeping the context of the whole document

A window on its own would not know which policy it belongs to, which vehicle a
continuation row describes, or what a schedule's columns are if the heading was
two pages back. Four mechanisms carry that context across the split (all in
`chunking` in `config.yaml`):

| Mechanism | What it does |
|---|---|
| Overlap (`overlap_pages: 1`) | The last page of one window is the first page of the next. |
| Section-aware boundaries (`section_boundaries`) | Each boundary may move up to 2 pages so the next window starts where a section starts, rather than mid-table. The section starts are found from MinerU's headings, ignoring running headers and page numbers. The plan is made once per document and shared by all three cases. |
| Reference page (`context.anchor_pages: [1]`) | The declarations page is shown at the start of every later window, marked "reference only", in that case's own modality (image, text, or both). |
| Context carried forward (`context.carry_forward`) | Each window is told what earlier windows extracted: the policy-level values, and every vehicle, location, coverage or form found so far by its identifying fields. It is asked to reuse those exact names and numbers, and to attach continuing rows to the same parent. |

Some consequences:
- The carried-forward context is built from the model's own earlier output, never from MinerU text, so PDF only stays image-only.
- Because each window needs the windows before it, windows run one after another.
- The extra prompt tokens are counted in the timing report.
- MinerU runs even for a PDF-only run, because the boundaries are planned from its headings.
- `report.py` refuses to compare a document's cases if they were split differently.

## How windows are merged

`src/chunking.py`:
- A field only one window filled keeps that value.
- The same value from two windows (after the scorer's normalisation —
  `"$1,000"` = 1000, `9/1/2026` = `2026-09-01`) is not a conflict.
- Different values from two windows: the earlier window wins by default
  (`chunking.scalar_conflict`), and the conflict is listed in the cell's
  `meta.json`.
- Lists of objects are concatenated. An item from a later window that is the
  **same entity** as one already collected is merged into it instead of being
  added twice — typically a row on an overlap page, or a vehicle listed on the
  declarations and again on a schedule. Same entity means:
  - equal on an identity key in `scoring.entity_lists`; or
  - agreeing on every field both filled in, which must be at least two fields,
    or one identifier such as a VIN or form number.

  Two vehicles that agree only on make = "Honda" stay two vehicles.
- Items from the same window are never merged with each other.

## Run order

Each step is cheap and proves the next one is worth attempting.

```powershell
pip install -r requirements.txt                 # once; no torch/vLLM locally
python src/selftest.py --gold "input/policy/gold/<doc>.gold.json"   # 0. scorer vs YOUR gold, no GPU
python src/preflight.py --local                  # 1. LOBs, schemas, gold, render, windows, budgets
```

`selftest` assertion 4 names any list of objects without a working identity
key. Add one under `scoring.entity_lists` in `config.yaml`; the merge uses the
same keys. Do this once per LOB schema, before the pod exists.

**2. The pod.** The provisioned pod `peuwpamzad828w` already runs everything
from `/workspace/boot.sh`. Nothing from `pod/` is needed there.

| service | port | started by |
|---|---|---|
| Qwen3-VL-8B-Instruct (vLLM 0.26) | 8000 | `svc-vl.sh` |
| mineru-api (MinerU 4.0.12), used through its `/v1/tasks` API | 8001 | `svc-mineru.sh` |
| MinerU2.5-Pro VLM, internal | 127.0.0.1:18003 | `svc-mineru-vlm.sh` |

All three read their settings from `/workspace/serve.conf`. One change is
needed: the model's context must hold the largest window (about 62k tokens of
prompt plus a 32k answer).

```bash
cd /workspace && cp serve.conf serve.conf.bak
sed -i 's/^VL_LEN=.*/VL_LEN=131072/; s/^VL_UTIL=.*/VL_UTIL=0.62/' serve.conf
pkill -f "vllm serve /workspace/models/vl/"   # svc-vl.sh restarts it with the new values after 15 s
tail -f logs/vllm-vl.log                       # wait for "Application startup complete", then Ctrl+C
```

On Windows, set the key the pod uses, `API_KEY` in `serve.conf`, with
`$env:POC3_API_KEY = "..."`. `pod.id` is set in `config.yaml`.

### Running the whole test on the pod (survives a closed laptop)

The job runs entirely on the pod and talks to vLLM and mineru-api on
`127.0.0.1`, so once it is started it needs no laptop, proxy or internet.

```bash
cd /workspace/POC3/Model-and-OCR-Latency-POC && git pull
# the data is git-ignored: bring input/ and work/ (MinerU cache) from the laptop
runpodctl receive <code>                 # code from: runpodctl send poc3_run.tar.gz
tar xzf poc3_run.tar.gz input work       # only these two folders
python3 -m venv /workspace/POC3/venv && /workspace/POC3/venv/bin/pip install -q -r requirements.txt
nohup setsid bash pod/run_all.sh > /dev/null 2>&1 &
bash pod/status.sh                       # any time, from any terminal
```

`pod/run_all.sh` takes the API key from `/workspace/serve.conf`. It then:
1. runs both preflights, retrying for up to 10 minutes while services start;
2. runs the three cases in parallel;
3. runs a second pass to retry any window that failed on the network;
4. writes the reports.

Logs go to `output_policies/logs/`, and a `DONE` (or `FAILED`) file marks the
end. Starting it again is safe: finished work is skipped.

*On a fresh pod without these services*, use this repo's scripts instead:
expose HTTP ports 8002 and 8003, copy `pod/`, run `bash setup_pod.sh` and
`bash serve_vllm.sh`, and set `ocr.mode: pod`, `ocr.port: 8002`, and the model's
port 8003 and name `qwen3-vl-8b` in `config.yaml`.

```powershell
python src/preflight.py --pod                    # 3. model name, real streaming, MinerU /health
python src/runner.py --mock --mock-profile degraded; python src/report.py   # 4. whole pipeline, no GPU
python src/runner.py --ocr-only                  # 5. MinerU on every document, timed
python src/runner.py --ocr-only --force-ocr      #    (optional) another MinerU timing sample
python src/runner.py --doc <one doc_id> --only md_only   # 6. one real cell - read its raw text
python src/runner.py                             # 7. everything
python src/report.py                             # 8. comparison.md + timing.md
```

A real run never resumes over mock output: any cell or window left by step 4 is
regenerated for real automatically.

Useful flags for `runner.py`:

| flag | effect |
|---|---|
| `--only md_only,pdf_md` | run only these cases |
| `--doc <doc_id>` | one document |
| `--ocr-only` | render + MinerU only, no model calls |
| `--force-ocr` | re-render and re-run MinerU; each run adds a timing sample, and the report uses the median |
| `--overwrite` | redo cells and windows that already have output (by default finished work is resumed) |

**Resuming.** A window whose request failed on the network (timeout, server
error, a dropped stream after every retry) is retried on the next run. All of
that document's finished windows are reused, so only the failed pages cost GPU
time again.

## Output

```
output_policies/
  8B/<MD only | PDF only | PDF+MD>/<LOB folder>/
    <doc>.json            the merged canonical JSON
    <doc>.meta.json       status, tokens, latency, per-window summary, merge report, schema check
    <doc>.score.json      per-field P/R/F1 vs gold, entity counts, strict score, coverage
    _windows/<doc>/
      wNN_pAAA-BBB.raw.txt       raw model text for one window, written FIRST
      wNN_pAAA-BBB.json          that window's parsed JSON
      wNN_pAAA-BBB.meta.json     that window's tokens and timing, per attempt
      wNN_pAAA-BBB.request.json  the redacted request (images -> name + size + hash)
  report/
    comparison.md / .json  accuracy, per case and per document
    matrix.csv, per_field.csv
    timing.md / .json      MinerU and model timing, end to end, projection
    mineru_timing.csv, model_latency.csv, end_to_end.csv
    ocr/<LOB>/<doc>.md     MinerU's text for the document, page by page, with its
                           engine, timing and character counts on top
work/ocr_policies/<doc>/   page_N.png, page_N.md, render_meta.json, ocr_meta.json (MinerU timing samples)
work/schemas/              resolved schemas + resolve notes
```

Raw text is written before anything is parsed, so **re-scoring and re-merging
never need the model again**. Change `scoring.entity_lists` or
`chunking.scalar_conflict`, delete the cell's `<doc>.meta.json`, re-run
`runner.py`, and the finished windows are reused from disk.

## What the accuracy score counts

The canonical model wraps every value as
`{raw, parsed, confidence: {score, source}, page_ref, flagged}`, and the gold
adds `fideon:provenance` and a `text_sections` index of the policy text. Scored
as-is, the actual values would be only 7–27% of each document's score. The
score is therefore split:

| part | how it is scored | config |
|---|---|---|
| **F1** (`field_f1`) | the document's values: `raw`/`parsed` and plain fields such as codes and labels | the default |
| not scored | `confidence`, `flagged`, `fideon:provenance`, which are the gold pipeline's own bookkeeping | `scoring.ignore_keys` |
| **page refs** | `page_ref`/`page_range`: where each value is printed | `scoring.page_ref_keys` |
| **text sec.** | the `text_sections` index, scored as a list matched by content, because its keys are generated strings that no model reproduces | `scoring.separate_sections`, `scoring.keyed_maps` |

The same rules govern the window merge. A wrapper's `confidence` and `page_ref`
never decide whether two windows saw the same entity, and differences in them
are not reported as conflicts. When a value is seen in two windows, its
`page_ref` collects both pages.

## Reading the timing report

| measure | what it covers |
|---|---|
| MinerU s | Measured **on the pod**, from job start to job end: MinerU's own work, excluding upload, proxy and polling. Each run is a fresh MinerU process, so it includes a fixed start-up and model-load cost. The fit `time ≈ a + b × pages` across documents of different lengths separates the fixed cost `a` from the per-page cost `b`. |
| end-to-end s (MinerU) | What this machine waited: upload + queue + processing + polling. |
| first byte s | The first window's time to first token: upload, queue, image encoding and prefill. |
| model s | Every window's call time added up, retries included: what one document costs. |
| decode tok/s | Output tokens per second of generation. Output length, not page count, drives most of the model time. |
| cached | Prompt tokens vLLM served from its prefix cache (the shared schema prompt). |
| projection | Read off the fitted lines: an estimate inside the measured range of document sizes, an extrapolation outside it. |

`*` on an accuracy score means part of the output was never generated: a window
was truncated and repaired (REPAIRED), or a window returned no JSON (PARTIAL).
That score is a **floor, not a measurement**.

## When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| Preflight `schema <lob>: could not resolve` | the LOB schema `$ref`s a file or definition that is not there | see the error, and `work/schemas/<lob>.resolve_notes.json` |
| Gold fails validation against the resolved schema | the gold and the schema disagree, or the common/LOB combination is not what was intended | check `resolve_notes.json` (mode, clashes); set `schema_resolution.merge_common_root` |
| Proxy 502 that looks like "still loading" | server bound `127.0.0.1` | `--host 0.0.0.0` |
| vLLM 400 on image count | a window has more pages than `--limit-mm-per-prompt` | lower `chunking.window_pages`, or raise the flag **and** `models[].max_images` |
| `NO-GO` on budget | the largest window is too big for the context window | lower `chunking.window_pages`, `render.max_long_side_px` or `prompt.max_md_chars_per_page` |
| MinerU exits 0, writes nothing | models never downloaded | `mineru-models-download -s huggingface -m pipeline`; `/health` reports `models_ready` |
| MinerU timing has no breakdown | the pod runs the older `mineru_server.py` | copy the current `pod/` and restart it; preflight warns about this |
| Many merge conflicts on one field | two windows read a value differently, or the field genuinely changes across the document | read `merge.conflicts` in the cell's `meta.json` |
| A whole list scores 0 | no working identity key | run `selftest.py --gold ...`; assertion 4 names the list |
| Markdown looks empty or wrong | cached from a different engine | the cache records its `source`; a mismatch is refused and re-fetched |

## Ported from

- `D:/POC2/qwen3vl-poc/src/common.py` — `repair_truncated_json`, JSON recovery,
  schema validation.
- `D:/POC2/qwen3vl-poc/src/prompting.py` — the per-page interleave and
  `conversation_fingerprint`.
- `D:/POC2/qwen3vl-poc/src/run_ocr.py` — the render cap, and the MinerU lessons
  in `pod/mineru_ocr.py` (per-page attribution from `*_content_list.json`, keeping
  `discarded` blocks, the exit-0-wrote-nothing guard).
- `D:/RunPod/eval_fields.py` — the order-invariant entity matcher and
  normalisation, now also used to merge windows.
