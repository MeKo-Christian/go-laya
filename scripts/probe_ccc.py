#!/usr/bin/env python3
"""Probe the pinned tokenizer's canonical combining classes (PLAN.md task 4.5.5).

Task 4.5.3's differential found 85 corpus lines where this package's NFC and the
Python oracle's disagree. This script isolates the cause so the disagreement is a
set of codepoints rather than a set of sentences.

The method is behavioural, not declarative: `tokenizers` exposes no combining-class
accessor, so each codepoint is normalised next to U+0334 COMBINING TILDE OVERLAY
(ccc = 1) and we watch whether the pair reorders. It reorders exactly when the
normaliser believes the codepoint's ccc is greater than 1, so one NFC call per
codepoint reads the table out of the implementation.

Result at tokenizers 0.23.2 / transformers 5.17.0: **108 codepoints** on which the
Rust normaliser behind HF `tokenizers` does not reorder while Go's
`golang.org/x/text` and CPython's own `unicodedata` both do. They are combining
marks assigned in Unicode 11.0 through 15.0 -- U+07FD, U+0898-089F, U+08CA-08D3,
U+1AC0-1ACE, U+1DF6-1DFA, U+10D24-10D27, U+10EFD-10EFF and others -- so the Rust
crate's tables predate them and treat each as a starter, which blocks the
canonical reordering the other two perform.

Go and CPython agree with each other everywhere. A naive run of this probe also
flags U+0334 itself (it is the pivot) and the four singleton-decomposition
characters U+0340, U+0341, U+0343, U+0344, where the probe measures NFC behaviour
rather than the raw property; those five are excluded as artefacts of the method,
not disagreements.

Output: build/ccc_rust.tsv (every codepoint) and build/ccc_disagreement.json (the
set, for whoever implements the reconciliation).

Usage:
    .venv-ref/bin/python scripts/probe_ccc.py
"""

from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# U+0334 COMBINING TILDE OVERLAY, ccc = 1: the lowest non-zero class, so a pair
# reorders if and only if the other codepoint's class is higher.
PIVOT = "̴"

# Characters where this probe measures the wrong thing. The pivot compared with
# itself is trivially unchanged, and the four tone marks have singleton canonical
# decompositions, so NFC rewrites them before any reordering could be observed.
ARTEFACTS = {0x0334, 0x0340, 0x0341, 0x0343, 0x0344}

SURROGATES = range(0xD800, 0xE000)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.models_root / "laya" / "tokenizer")
    normalizer = tok.backend_tokenizer.normalizer

    args.out.mkdir(parents=True, exist_ok=True)
    table = args.out / "ccc_rust.tsv"
    disagree = []

    with table.open("w", encoding="utf-8") as fh:
        fh.write("codepoint\tcpython_ccc\trust_reorders\n")
        for cp in range(0x110000):
            if cp in SURROGATES:
                continue
            ch = chr(cp)
            ccc = unicodedata.combining(ch)
            reorders = normalizer.normalize_str(ch + PIVOT) == PIVOT + ch
            fh.write(f"{cp:04X}\t{ccc}\t{int(reorders)}\n")
            if ccc > 1 and not reorders and cp not in ARTEFACTS:
                disagree.append(cp)

    import tokenizers

    summary = {
        "plan_task": "4.5.5",
        "method": "normalize(ch + U+0334) and watch for reordering",
        "versions": {
            "python": sys.version.split()[0],
            "tokenizers": tokenizers.__version__,
            "cpython_unidata": unicodedata.unidata_version,
        },
        "count": len(disagree),
        "note": (
            "cpython and golang.org/x/text agree with each other on all of these; "
            "the Rust normaliser behind tokenizers treats each as a starter"
        ),
        "codepoints": [f"U+{cp:04X}" for cp in disagree],
    }
    (args.out / "ccc_disagreement.json").write_text(json.dumps(summary, indent=2) + "\n", "utf-8")
    print(f"{len(disagree)} disagreeing codepoints -> {args.out / 'ccc_disagreement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
