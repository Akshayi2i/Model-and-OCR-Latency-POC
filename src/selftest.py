"""Stage 0: prove the scorer before any GPU or network is touched.

No model output is involved. Every case perturbs the GOLD in a way that must not
change the score (or must change exactly one field), and checks that it doesn't.
If these do not pass, every number this POC later prints is noise.

    python src/selftest.py                # uses input/ gold if present, else the fixture
    python src/selftest.py --fixture      # force the built-in fixture
    python src/selftest.py --gold path.json

Assertion 4 (list reversal) is the one that costs real time. When it fails it
names the list that lacks a working identity key - go add one to
scoring.entity_lists in config.yaml and run again. Tuning that config until
reversal scores 1.000 is a required step, and it is free.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent))

import common  # noqa: E402
import scoring  # noqa: E402

# --------------------------------------------------------------------------- fixture

# A miniature policy declaration exercising every structural feature the real one
# has: top-level scalars, a nested address object, several entity lists, a list
# nested two deep inside another entity list, a list of plain strings, money,
# dates, and identifiers that must not be read as numbers.
FIXTURE: dict[str, Any] = {
    "document_type": "Policy Declaration",
    "carrier_name": "Meridian Casualty Company",
    "transaction": {
        "transaction_type": "NEW BUSINESS",
        "transaction_date": "9/1/2026",
        "policy_number": "CA 0071824 03",
        "naic": "035882",
        "effective_date": "9/1/2026",
        "expiration_date": "9/1/2027",
        "term_amount": "$47,149.00",
    },
    "named_insured": {
        "entity_type": "Corporation",
        "name": "Kestrel Freight Lines, Inc.",
        "address": {"street": "1420 Harbor Way", "city": "Toledo",
                    "state": "OH", "zip": "43604"},
        "additional_named_insureds": ["Kestrel Logistics LLC", "Kestrel Leasing Co"],
    },
    "contact_info": [
        {"type": "Agency", "information": "(419) 555-0182", "contact_name": "D. Alvarez"},
        {"type": "Billing", "information": "billing@meridian.example", "contact_name": None},
    ],
    "line_of_business": {
        "name": "Commercial Auto",
        "premium": "$47,149.00",
        "coverages": [
            {"coverage": "Liability", "limit": "1,000,000", "premium": "$28,400.00"},
            {"coverage": "Uninsured Motorist", "limit": "25,000/50,000", "premium": "$3,120.00"},
            {"coverage": "Comprehensive", "limit": "1,000 Ded", "premium": "$9,415.00"},
        ],
    },
    "additional_interests": [
        {"nature_of_interest": "Loss Payee", "name": "Fifth Third Bank",
         "address": {"street": "38 Fountain Sq", "city": "Cincinnati",
                     "state": "OH", "zip": "45263"},
         "rank": "1", "payor": "No", "account_number": "0098-4471"},
    ],
    "states": [
        {
            "state": "OH",
            "locations": [
                {"location_number": 1, "location_address": "1420 Harbor Way, Toledo, OH",
                 "vehicle_coverages": {"vehicle_columns": ["2025 FREIGHTLINER M2"]}},
                {"location_number": 2, "location_address": "88 Rail St, Akron, OH",
                 "vehicle_coverages": {"vehicle_columns": ["2024 HINO 268"]}},
            ],
        },
        {
            "state": "MI",
            "locations": [
                {"location_number": 1, "location_address": "9 Industrial Dr, Monroe, MI",
                 "vehicle_coverages": {"vehicle_columns": ["2023 ISUZU NPR"]}},
            ],
        },
    ],
    "vehicles": [
        {"sequence": 1, "year": "2025", "make": "FREIGHTLINER", "model": "M2",
         "vin": "3AKJHHDR9NSMZ1234", "body_type": "Tractor", "territory": "007",
         "gvw": "33,000", "registration_state": "OH"},
        {"sequence": 2, "year": "2024", "make": "HINO", "model": "268",
         "vin": "5PVNJ8JT1P4S51987", "body_type": "Truck", "territory": "012",
         "gvw": "25,950", "registration_state": "OH"},
        {"sequence": 3, "year": "2023", "make": "ISUZU", "model": "NPR",
         "vin": "JALC4W169P7000456", "body_type": "Truck", "territory": "104",
         "gvw": "14,500", "registration_state": "MI"},
    ],
    "drivers": [
        {"sequence": 1, "name": "Marcus Ortega", "license_state": "OH",
         "license_number": "RL419882", "date_of_birth": "3/14/1985"},
        {"sequence": 2, "name": "Priya Raman", "license_state": "OH",
         "license_number": "TK802331", "date_of_birth": "11/2/1990"},
    ],
    "locations": [
        {"location_number": 1,
         "address": {"street": "1420 Harbor Way", "city": "Toledo",
                     "state": "OH", "zip": "43604"},
         "legal_description": "Lot 14, Harbor Industrial Park"},
        {"location_number": 2,
         "address": {"street": "88 Rail St", "city": "Akron", "state": "OH", "zip": "44301"},
         "legal_description": None},
    ],
    "pay_plan": {"payment_plan": "Quarterly", "number_of_payments": "4",
                 "bill_method": "Direct", "payor": "Insured"},
    "disclaimer": "This declaration page replaces any previously issued for this policy term.",
    "document_metadata": {"printed_on": "9/1/2026", "printed_at": "10:42 AM", "total_pages": 15},
    "additional_content": [
        {"page": 15, "label": "Form code footer", "content": "CA 00 01 10 13"},
    ],
}


# --------------------------------------------------------------------------- perturbations


def reformat_dates(obj: Any, opts: scoring.NormOpts, path: str = "") -> Any:
    """Rewrite every date-looking value into a different but equivalent format.

    Identifier-named fields are left alone. A policy number or a NAIC code is not
    a date, so rewriting it would not be an equivalence-preserving perturbation -
    the scorer would be right to call the result a mismatch.
    """
    if isinstance(obj, dict):
        return {k: reformat_dates(v, opts, scoring.child_spec(path, k)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [reformat_dates(v, opts, path) for v in obj]
    if isinstance(obj, str) and not opts.id_re.search(path):
        iso = scoring.as_date(obj)
        if iso:
            y, m, d = iso.split("-")
            # M/D/YYYY <-> ISO, whichever the gold is not.
            return iso if "/" in obj else f"{int(m)}/{int(d)}/{y}"
    return obj


def reverse_lists(obj: Any) -> Any:
    """Reverse every list of dicts, at every depth."""
    if isinstance(obj, dict):
        return {k: reverse_lists(v) for k, v in obj.items()}
    if isinstance(obj, list):
        items = [reverse_lists(v) for v in obj]
        if any(isinstance(v, dict) for v in obj):
            items.reverse()
        return items
    return obj


def drop_field(obj: Any, target: str, sspec: scoring.ScoreSpec, path: str = "") -> Any:
    """Remove one field spec path wherever it occurs ('vehicles[].vin')."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            key = scoring.child_spec(path, k)
            if key == target:
                continue
            out[k] = drop_field(v, target, sspec, key)
        return out
    if isinstance(obj, list):
        return [drop_field(v, target, sspec, scoring.item_spec(path)) for v in obj]
    return obj


def all_object_lists(obj: Any, path: str = "") -> set[str]:
    """Every spec path in the gold that holds a list of objects."""
    found: set[str] = set()
    if isinstance(obj, dict):
        for k, v in obj.items():
            key = scoring.child_spec(path, k)
            if isinstance(v, list) and any(isinstance(x, dict) for x in v):
                found.add(key)
                for x in v:
                    found |= all_object_lists(x, scoring.item_spec(key))
            elif isinstance(v, (dict, list)):
                found |= all_object_lists(v, key)
    elif isinstance(obj, list):
        for x in obj:
            found |= all_object_lists(x, path)
    return found


def first_repeated_leaf(gold: dict, sspec: scoring.ScoreSpec) -> str | None:
    """A field spec that occurs in several entities - the useful thing to drop."""
    counts: dict = {}
    lc: dict = {}
    scoring.tally(gold, gold, counts, "", sspec, lc)
    # A named field only: drop_field removes keys, so an element of a plain-value
    # list ('applies_to[0]') cannot be dropped and would read as a scorer failure.
    inside_list = [f for f, c in counts.items()
                   if c[1] >= 2 and "[]" in f and not f.endswith("]")]
    return sorted(inside_list, key=lambda f: (-counts[f][1], f))[0] if inside_list else None


def owning_lists(fields: list[str]) -> list[str]:
    """Map failing field specs back to the entity lists that contain them."""
    lists = set()
    for f in fields:
        if "[]" in f:
            lists.add(f.rsplit("[]", 1)[0])
    return sorted(lists)


# --------------------------------------------------------------------------- cases


def run(gold: dict, cfg: dict) -> int:
    sspec = scoring.ScoreSpec(cfg)
    ok = True

    def head(n: int, title: str) -> None:
        print(f"\n--- {n}. {title} " + "-" * max(0, 58 - len(title)))

    def verdict(passed: bool, detail: str = "") -> None:
        nonlocal ok
        ok = ok and passed
        print(f"    {'PASS' if passed else 'FAIL'}{('  ' + detail) if detail else ''}")

    # 1 -- identity
    head(1, "gold vs itself")
    r = scoring.score_document(gold, copy.deepcopy(gold), cfg)
    print(f"    f1={r['field_f1']:.3f}  fields={r['fields_scored']}  "
          f"gold_leaves={r['gold_leaves']}")
    if r["field_f1"] < 1.0:
        bad = [f for f, c in r["per_field"].items() if c["f1"] < 1.0]
        print("    normalisation is destroying real values in:")
        for f in bad[:15]:
            print(f"      {f}")
    verdict(r["field_f1"] >= 1.0)

    # 1b -- nothing silently escapes the metric
    head(2, "every list of objects is actually reached")
    expected = all_object_lists(gold)
    reached = set(r["list_counts"].keys())
    escaped = sorted(expected - reached)
    print(f"    object lists in gold: {len(expected)}   reached by the scorer: {len(reached)}")
    if escaped:
        print("    NOT SCORED AT ALL -- these vanish from the metric rather than")
        print("    scoring badly, which would silently flatter every model:")
        for lst in escaped:
            print(f"      {lst}")
    verdict(not escaped)

    # 2 -- date reformatting
    head(3, "every date rewritten to the other format")
    r2 = scoring.score_document(gold, reformat_dates(copy.deepcopy(gold), sspec.opts), cfg)
    print(f"    f1={r2['field_f1']:.3f}")
    if r2["field_f1"] < 1.0:
        bad = [f for f, c in r2["per_field"].items() if c["f1"] < 1.0]
        print("    date normalisation is incomplete for:")
        for f in bad[:15]:
            print(f"      {f}")
    verdict(r2["field_f1"] >= 1.0)

    # 3 -- list reversal (the config-tuning loop)
    head(4, "every list of dicts reversed")
    r3 = scoring.score_document(gold, reverse_lists(copy.deepcopy(gold)), cfg)
    print(f"    f1={r3['field_f1']:.3f}")
    if r3["field_f1"] < 1.0:
        bad = [f for f, c in r3["per_field"].items() if c["f1"] < 1.0]
        print("    these lists lack a working identity key -- add one to")
        print("    scoring.entity_lists in config.yaml:")
        for lst in owning_lists(bad):
            print(f"      {lst}")
        print("    affected fields:")
        for f in bad[:15]:
            print(f"      {f}  (f1={r3['per_field'][f]['f1']:.2f})")
    verdict(r3["field_f1"] >= 1.0)

    # 4 -- field localisation
    head(5, "one repeated field dropped everywhere")
    target = first_repeated_leaf(gold, sspec)
    if not target:
        print("    SKIP  no field repeats across entities in this gold")
    else:
        r4 = scoring.score_document(gold, drop_field(copy.deepcopy(gold), target, sspec), cfg)
        tgt = r4["per_field"].get(target, {"recall": None})
        others = [c["recall"] for f, c in r4["per_field"].items()
                  if f != target and c["gold"] > 0]
        worst = min(others) if others else 1.0
        print(f"    dropped {target}")
        print(f"    its recall={tgt['recall']}   worst other field recall={worst:.3f}")
        verdict(tgt["recall"] == 0.0 and worst >= 1.0,
                "the metric must localise to the dropped field only")

    # 5 -- truncation repair
    head(6, "gold serialised then cut at 60%")
    blob = common.compact_json(gold)
    cut = blob[: int(len(blob) * 0.6)]
    parsed, reason, repaired = common.recover_json_object(cut)
    if parsed is None:
        print(f"    could not repair: {reason}")
        verdict(False)
    else:
        r5 = scoring.score_document(gold, parsed, cfg)
        # Judged over EVERYTHING scored - field_f1 plus the separately scored
        # parts - because the cut may fall entirely inside a separate section
        # (a long text_sections index at the end), leaving field_f1 untouched.
        parts = [(r5["matched_leaves"], r5["gold_leaves"], r5["pred_leaves"])] + [
            (s["matched_leaves"], s["gold_leaves"], s.get("pred_leaves", s["matched_leaves"]))
            for s in (r5.get("separate") or {}).values()]
        m, g, p = (sum(x[i] for x in parts) for i in range(3))
        recall, precision = (m / g if g else 1.0), (m / p if p else 1.0)
        print(f"    {reason}")
        print(f"    recovered field_f1={r5['field_f1']:.3f}; over everything scored "
              f"recall={recall:.3f} precision={precision:.3f}")
        print("    (a floor, not a measurement -- json_valid_strict is False here)")
        verdict(repaired and 0.0 < recall < 1.0 and precision >= 0.99,
                "repair must recover a partial PREFIX, inventing nothing")

    print("\n" + "=" * 70)
    print("SELFTEST: " + ("PASS - the scorer is trustworthy" if ok else
                          "FAIL - fix the above before scoring any model output"))
    print("=" * 70)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--gold", default=None, help="score against this gold JSON")
    ap.add_argument("--fixture", action="store_true",
                    help="force the built-in fixture even if input/ has a gold")
    args = ap.parse_args()

    cfg = common.load_config(args.config)

    gold = None
    source = "built-in fixture"
    if args.gold:
        gold = common.read_json(Path(args.gold))
        source = str(args.gold)
    elif not args.fixture:
        docs = common.select_documents(cfg)
        golds = [d for d in docs if d.gold]
        if golds:
            gold = common.read_json(golds[0].gold)
            source = str(golds[0].gold)

    if gold is None:
        gold = copy.deepcopy(FIXTURE)
        if not args.fixture:
            print("No gold found under input/ - using the built-in fixture.")
            print("Re-run once your golden canonical JSON is in place; the entity_lists")
            print("config must be tuned against the REAL gold, not this stand-in.\n")

    print(f"gold: {source}")
    print(f"top-level keys: {len(gold) if isinstance(gold, dict) else 'not an object'}")
    return run(gold, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
