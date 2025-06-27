#!/usr/bin/env python
"""
compare_names.py  –  compare (first_name, last_name) similarity between
                     model output and gold annotations.

Usage:
    python compare_names.py model.jsonl gold.jsonl
"""

import json, sys, statistics as stats
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, Tuple, List
from rapidfuzz import fuzz, utils

# ---------------------------------------------------------------------------
# configuration -------------------------------------------------------------
# ---------------------------------------------------------------------------

TITLE_WORDS = {
    "herr", "frau", "fru", "fruis", "fruher", "señor", "señorita", "mr.", "mrs.",
    "prof.", "prof", "doctor", "dr.", "dr", "sir", "madam", "monsieur",
    "herrn", "ing.", "ing", "pr."  # extend as you like
}

def strip_titles(name: str) -> str:
    """remove common honorifics / titles from a name string"""
    if not name:
        return ""
    tokens = [t for t in utils.default_process(name).split() if t not in TITLE_WORDS]
    return " ".join(tokens)

# ---------------------------------------------------------------------------
# helpers -------------------------------------------------------------------
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> Dict[Tuple[str, str], dict]:
    """Return dict keyed by (file, report_id) with the record as value"""
    with path.open(encoding="utf-8") as fh:
        return {(d["file"], d["report_id"]): d for d in map(json.loads, fh)}

def similarity(a: str, b: str) -> float:
    """token-sort ratio ∈ [0,100]"""
    return fuzz.token_sort_ratio(strip_titles(a), strip_titles(b))

def safe_get(d: dict, field: str) -> str:
    return d.get(field) or ""

# ---------------------------------------------------------------------------
# main ----------------------------------------------------------------------
# ---------------------------------------------------------------------------

def main(generated_path: str, gold_path: str) -> None:
    gen = read_jsonl(Path(generated_path))
    gold = read_jsonl(Path(gold_path))

    first_scores: List[float] = []
    last_scores:  List[float] = []
    per_file: Dict[str, List[Tuple[float,float]]] = defaultdict(list)

    missing = Counter()

    for key, gold_rec in gold.items():
        if key not in gen:
            missing["generated"] += 1
            continue
        gen_rec = gen[key]

        f_score = similarity(safe_get(gen_rec, "first_name"),
                             safe_get(gold_rec, "first_name"))
        l_score = similarity(safe_get(gen_rec, "last_name"),
                             safe_get(gold_rec, "last_name"))

        first_scores.append(f_score)
        last_scores.append(l_score)
        per_file[key[0]].append((f_score, l_score))

    for key in gen.keys() - gold.keys():
        missing["gold"] += 1

    # ------------------ reporting -----------------------------------------
    def _stats(series: List[float]) -> str:
        return f"mean={stats.mean(series):5.1f} std={stats.stdev(series):5.1f} " \
               f"median={stats.median(series):4.1f} perfect={sum(s==100 for s in series)}"

    print("=== PER-FILE STATISTICS ===")
    for fname, pairs in per_file.items():
        f_vals, l_vals = zip(*pairs)
        print(f"{fname:40s}  "
              f"first[{_stats(list(f_vals))}]  "
              f"last[{_stats(list(l_vals))}]")

    print("\n=== OVERALL ===")
    print("First names :", _stats(first_scores))
    print("Last names  :", _stats(last_scores))
    print(f"\nMissing records ->  generated-only: {missing['generated']},  gold-only: {missing['gold']}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit("Usage: asessment.py generated.jsonl gold.jsonl")
    main(*sys.argv[1:])
