#!/usr/bin/env python3
"""Per-stage tokenizer vectors for the differential run (PLAN.md task 4.5.3).

4.5.3 wants mismatches recorded **per stage** -- added-token / normalizer /
pre-tokenizer / BPE / byte fallback -- and not just counted. That needs an oracle
that exposes the intermediate values, which `dump_python_parity.py`'s
`dump_tokenizer` (end-to-end ids only) does not, and which `dump_pretok` only
half does.

`dump_pretok` cannot simply be pointed at a bigger corpus. It composes
``normalize_str(whole_text)`` and then ``pre_tokenize_str(normalized)``, which is
the right order for text containing no added token and the wrong pipeline for text
that does: ``Tokenizer.encode`` splits on the added vocabulary **first** and
normalizes each unclaimed segment separately. That is what `tokenizer/added.go`
implements, and on a 100k-line corpus the added-token path is reached often enough
that recording the wrong composition would manufacture mismatches.

So this dumper reconstructs the segmentation. `tokenizers` does not expose
``AddedVocabulary``, so the split is **derived**: encode once, take the offsets of
every token whose id is an added token, and treat the gaps as unclaimed segments.
Each record says how it was derived, and where the derivation is ambiguous the
record is flagged rather than guessed at -- the Go side then attributes such a
line to "pre-BPE" instead of to a stage the oracle cannot actually prove.

Per corpus line and per checkpoint:

    {"i": 0, "text": ..., "derived": true,
     "segments": [{"added": null | "<tok>", "raw": ..., "norm": ..., "pre": [...]}],
     "ids": [...], "tokens": [...]}

`tokens` comes from ``convert_ids_to_tokens``, never ``Encoding.tokens`` -- the
latter returns the lstripped slice for an lstrip added token, which is the trap
`dump_python_parity.py:275` already documents.

Usage:
    .venv-ref/bin/python scripts/dump_stages.py --corpus build/corpus/corpus.jsonl
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent

# english and typed-decisions ship a byte-identical tokenizer.json (both
# 3583228 bytes, same sha256), so there are two distinct tokenizers across the
# three checkpoints -- which is why testdata/ carries tokenizer_en and
# tokenizer_ml and no third file.
CHECKPOINTS = {
    "en": "",
    "ml": "multilingual",
}


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_tokenizer(models_root: Path, subfolder: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(models_root / "laya" / subfolder / "tokenizer")


def added_token_ids(tok) -> set[int]:
    """Ids the added vocabulary claims, specials included.

    `get_added_vocab` is the documented accessor and returns the full added set,
    which is what decides whether an offset span is a segment boundary.
    """
    return set(tok.get_added_vocab().values())


def segment(text: str, enc, added: set[int]) -> tuple[list[tuple[str | None, str]], bool]:
    """Derive (added_token | None, raw_text) segments from an encoding's offsets.

    Returns the segments and whether the derivation is trustworthy. It is not,
    and says so, when two claimed spans overlap or run backwards -- neither should
    happen, but a silent reordering would move a mismatch onto the wrong stage.
    """
    spans = [
        (start, end, tok)
        for tid, (start, end), tok in zip(enc.ids, enc.offsets, enc.tokens, strict=True)
        if tid in added
    ]
    spans.sort(key=lambda s: s[0])

    ok = True
    out: list[tuple[str | None, str]] = []
    cursor = 0
    for start, end, tok in spans:
        if start < cursor or end < start:
            ok = False
            continue
        if start > cursor:
            out.append((None, text[cursor:start]))
        out.append((tok, text[start:end]))
        cursor = end
    if cursor < len(text):
        out.append((None, text[cursor:]))
    return out, ok


def dump(name: str, subfolder: str, corpus: list[str], models_root: Path, out_dir: Path) -> dict:
    tok = load_tokenizer(models_root, subfolder)
    backend = tok.backend_tokenizer
    added = added_token_ids(tok)

    path = out_dir / f"stages_{name}.jsonl.gz"
    derived_bad = 0

    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as fh:
        head = {
            "kind": "header",
            "checkpoint": name,
            "subfolder": subfolder,
            "generator": "scripts/dump_stages.py",
            "plan_task": "4.5.3",
            "versions": versions(),
            "added_tokens": len(added),
            "normalizer": stage_type(backend.normalizer),
            "pre_tokenizer": stage_type(backend.pre_tokenizer),
        }
        fh.write(json.dumps(head, ensure_ascii=False) + "\n")

        for i, text in enumerate(corpus):
            enc = backend.encode(text, add_special_tokens=False)
            segs, ok = segment(text, enc, added)
            if not ok:
                derived_bad += 1

            records = []
            for tokstr, raw in segs:
                if tokstr is not None:
                    records.append({"added": tokstr, "raw": raw})
                    continue
                norm = backend.normalizer.normalize_str(raw) if backend.normalizer else raw
                pre = [p for p, _ in backend.pre_tokenizer.pre_tokenize_str(norm)]
                records.append({"added": None, "raw": raw, "norm": norm, "pre": pre})

            fh.write(
                json.dumps(
                    {
                        "i": i,
                        "text": text,
                        "derived": ok,
                        "segments": records,
                        "ids": enc.ids,
                        # convert_ids_to_tokens, not enc.tokens: the latter is the
                        # lstripped slice for an lstrip added token.
                        "tokens": tok.convert_ids_to_tokens(enc.ids),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            if i and i % 20000 == 0:
                log(f"{name}: {i}/{len(corpus)}")

    log(f"{name}: wrote {path} ({path.stat().st_size} bytes), {derived_bad} ambiguous splits")
    return {
        "path": str(path),
        "lines": len(corpus),
        "ambiguous_splits": derived_bad,
        "bytes": path.stat().st_size,
    }


def stage_type(stage) -> str:
    if stage is None:
        return "none"
    try:
        return json.loads(stage.__getstate__())["type"]
    except (TypeError, ValueError, KeyError):
        return type(stage).__name__


def versions() -> dict[str, str]:
    import tokenizers
    import transformers

    return {
        "python": sys.version.split()[0],
        "tokenizers": tokenizers.__version__,
        "transformers": transformers.__version__,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", type=Path, default=REPO_ROOT / "build" / "corpus" / "corpus.jsonl")
    ap.add_argument("--out", type=Path, default=None, help="default: the corpus's own directory")
    ap.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--checkpoint", action="append", choices=sorted(CHECKPOINTS))
    ap.add_argument(
        "--limit", type=int, default=0, help="first N corpus lines only (for a smoke run)"
    )
    args = ap.parse_args()

    corpus_path: Path = args.corpus.resolve()
    out_dir: Path = (args.out or corpus_path.parent).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    with corpus_path.open(encoding="utf-8") as fh:
        for line in fh:
            texts.append(json.loads(line)["text"])
            if args.limit and len(texts) >= args.limit:
                break
    log(f"corpus: {len(texts)} lines from {corpus_path}")

    wanted = args.checkpoint or sorted(CHECKPOINTS)
    report: dict[str, Any] = {}
    for name in wanted:
        report[name] = dump(name, CHECKPOINTS[name], texts, args.models_root, out_dir)

    (out_dir / "STAGES.json").write_text(
        json.dumps({"corpus": str(corpus_path), "checkpoints": report}, indent=2) + "\n", "utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
