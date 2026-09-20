#!/usr/bin/env python3
"""Freeze the upstream Python's behaviour as golden vectors under ``testdata/``.

PLAN.md task 1.3. Every later milestone is tested against these files, and
``AGENTS.md`` is categorical about why: *"CI must never need Python or model
weights"*, and *"most of these behaviours are ones where a wrong implementation
returns a plausible answer rather than an error."* Without a corpus, the Go port
would be written by reading ``original/`` and asserting whatever the Go happens to
do -- which is how a port enshrines its own bugs.

Output is **JSONL**, one JSON object per line, first line a header record. Two
reasons rather than the pretty-JSON shape ``export_onnx.py --fixture`` uses:
``treefmt``'s prettier owns ``*.json`` and would reflow a pretty fixture on every
``just fmt``, and a line-per-case diff is reviewable where a reflowed blob is not
(task 1.3.7, R6).

Determinism (task 1.3.7): a fresh ``np.random.default_rng(seed)`` per case, never
torch's global stream -- ``DecisionModel.__init__`` randomly initialises the head
before ``load_state_dict`` overwrites it, so the global stream advances by an amount
that depends on which checkpoints were built. Cases are emitted in sorted name
order, ``torch.set_num_threads(1)``, CPU only.

Usage::

    .venv-ref/bin/python scripts/dump_python_parity.py --all
    just fmt && just ci

Regenerating is a reviewed diff, never a drive-by (``AGENTS.md``).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import math
import sys
import unicodedata
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "original"))

SEED = 0
GENERATOR = "scripts/dump_python_parity.py"
COMMENT = (
    "Generated against the pinned reference environment (scripts/requirements-ref.txt). "
    "Do not hand-edit: regenerating this file is a reviewed diff (PLAN.md R6)."
)

CHECKPOINT_DIRS = {
    "english": "",
    "multilingual": "multilingual",
    "typed-decisions": "typed-decisions",
}


# --------------------------------------------------------------------------- header


def versions() -> dict[str, str]:
    """The pins that make the corpus meaningful.

    ``export_onnx.py``'s ``versions()`` omits ``tokenizers`` and ``safetensors``;
    task 1.3.6 needs both, and ``tokenizers`` is the one R6 actually names.
    """
    import safetensors
    import tokenizers
    import transformers

    return {
        "python": sys.version.split()[0],
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors.__version__,
    }


def torch_version() -> str:
    import torch

    return str(torch.__version__)


def stable_seed(name: str) -> int:
    """A per-case seed that does not move between processes or Python versions."""
    return int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)


def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def tokenizer_config_shas(models_root: Path) -> dict[str, str]:
    """Hash every ``tokenizer_config.json`` into the header.

    ``laya.Agent.__init__`` calls ``_fix_tokenizer_config`` (``agent.py:21-46``),
    which rewrites this file in place, so the local tree can stop matching the
    recorded Hub revision the first time the Python runs. At the pinned revision the
    function is a verified no-op for all three checkpoints -- hashing turns that
    from an assumption into something a regeneration would catch.
    """
    out = {}
    for name, sub in sorted(CHECKPOINT_DIRS.items()):
        p = models_root / "laya" / sub / "tokenizer" / "tokenizer_config.json"
        if p.is_file():
            out[name] = sha256_file(p)
    return out


def checkpoint_sha(models_root: Path) -> str | None:
    p = models_root / "PROVENANCE.json"
    if not p.is_file():
        return None
    return str(json.loads(p.read_text()).get("sha"))


def header(fixture: str, models_root: Path, **extra: Any) -> dict[str, Any]:
    h: dict[str, Any] = {
        "kind": "header",
        "fixture": fixture,
        "generator": GENERATOR,
        "versions": versions(),
        "checkpoint_sha": checkpoint_sha(models_root),
        "tokenizer_config_sha256": tokenizer_config_shas(models_root),
        "seed": SEED,
        "comment": COMMENT,
    }
    h.update(extra)
    return h


def write_jsonl(path: Path, head: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """One header line, then cases in sorted name order (task 1.3.7).

    ``sort_keys`` stays off on purpose: ``criteria`` dict order is load-bearing --
    it is the order ``render_options`` renders in and the order ``system_one`` zips
    probabilities against -- so sorting keys would silently change the fixture's
    meaning. ``ensure_ascii=False`` keeps non-ASCII literal, which is what makes the
    diff reviewable; it does not change what a JSON decoder sees.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(head, ensure_ascii=False)]
    for c in sorted(cases, key=lambda c: c["name"]):
        lines.append(json.dumps({"kind": "case", **c}, ensure_ascii=False))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  {path}  ({len(cases)} cases)")


# ------------------------------------------------------------------ 1.3.1 tokenizer


# PLAN.md task 4.4's required case list. Each entry is (name, text); the record then
# carries both `text` and `" " + text`, because `common.py:68` tokenises every option
# with a leading space and the two checkpoints disagree about whether it survives.
#
# Task 4.4.9 (lone surrogates / invalid UTF-8) is deliberately **absent**. A lone
# surrogate cannot round-trip through JSON: Python emits "\ud800" and Go's decoder
# replaces it with U+FFFD, so the fixture would assert against a corrupted input. The
# task says "decide and document the boundary; sanitize at the API edge" -- this is
# that decision. Go tests it at the edge, without a golden vector.
def tokenizer_cases(mask_token: str) -> list[tuple[str, str]]:
    cases: list[tuple[str, str]] = []

    # 4.4.1 -- consecutive whitespace, the most common drift and crash site.
    for n, t in [
        ("empty", ""),
        ("space", " "),
        ("two-spaces", "  "),
        ("newline", "\n"),
        ("tab", "\t"),
        ("crlf", "\r\n"),
        ("mixed", "   \n   "),
    ]:
        cases.append((f"ws/{n}", t))

    # 4.4.5 -- runs of 2 to 24 spaces are added tokens on English (ids 50254-50275),
    # and `serialize_state` feeds arbitrary user JSON through the tokenizer, so
    # indentation runs are realistic input rather than a synthetic edge case.
    for n in (2, 3, 8, 16, 23, 24, 25):
        cases.append((f"ws/run-{n:02d}", "a" + " " * n + "b"))

    # 4.4.3 -- compact JSON exactly as `serialize_state` emits it.
    for n, obj in [
        ("flat", {"subject": "invoice 4411", "amount": 42}),
        ("nested", {"user": {"id": 7, "tags": ["a", "b"]}, "ok": True, "none": None}),
        ("escapes", {"q": 'he said "hi"\nand left\\', "tab": "a\tb"}),
        ("non-ascii", {"city": "München", "note": "café — 日本"}),
        ("long-array", {"xs": list(range(40))}),
    ]:
        cases.append((f"json/{n}", json.dumps(obj, ensure_ascii=False)))

    # 4.4.4 -- the mask literal inside user text. build_sequence scrubs it to " "
    # before tokenising, but the tokenizer must still be pinned on the raw form.
    cases.append(("mask/bare", mask_token))
    cases.append(("mask/mid-sentence", f"is this {mask_token} spam?"))

    # 4.4.5 -- added tokens, bare and mid-sentence.
    for n, t in [
        ("unused0", "<unused0>"),
        ("start-of-turn", "<start_of_turn>"),
        ("2mass", "<2mass>"),
        ("at-bos", "[@BOS@]"),
        ("ip-address", "|||IP_ADDRESS|||"),
    ]:
        cases.append((f"added/{n}-bare", t))
        cases.append((f"added/{n}-mid", f"before {t} after"))

    # 4.4.6 -- normalisation. English normalises NFC; multilingual does not normalise
    # beyond space -> U+2581, so these two legitimately produce different ids.
    for n, t in [
        ("nfc-e-acute", unicodedata.normalize("NFC", "café")),
        ("nfd-e-acute", unicodedata.normalize("NFD", "café")),
        ("fullwidth-a", "ＡＢＣ"),
        ("ligature-fi", "ﬁnal"),
        ("nbsp", "a b"),
        ("zwsp", "a​b"),
        ("zwj", "a‍b"),
        ("bom", "﻿hello"),
        ("combining", "áêĩ"),
    ]:
        cases.append((f"norm/{n}", t))

    # 4.4.7 -- emoji: ZWJ sequences, skin tones, flags. Byte fallback on multilingual.
    for n, t in [
        ("family-zwj", "\U0001f468‍\U0001f469‍\U0001f467"),
        ("skin-tone", "\U0001f44d\U0001f3fd"),
        ("flag", "\U0001f1e9\U0001f1ea"),
        ("mixed-text", "ship it \U0001f680 now"),
    ]:
        cases.append((f"emoji/{n}", t))

    # 4.4.8 -- scripts.
    for n, t in [
        ("devanagari", "मुंबई में बारिश"),
        ("arabic", "القاهرة مدينة"),
        ("cjk", "東京へようこそ"),
        ("thai", "สวัสดีครับ"),
        ("cyrillic", "Здравствуйте"),
        ("hangul-precomposed", "한국어"),
        ("hangul-jamo", "한국어"),
    ]:
        cases.append((f"script/{n}", t))

    # Ordinary prose, so the corpus is not entirely pathological.
    cases.append(("prose/en", "I was charged twice for invoice 4411, please refund it today."))
    cases.append(("prose/de", "Ich wurde zweimal für Rechnung 4411 belastet."))

    return cases


def dump_tokenizer(name: str, ckpt: str, models_root: Path, out_dir: Path) -> None:
    from transformers import AutoTokenizer

    tok_dir = models_root / "laya" / CHECKPOINT_DIRS[ckpt] / "tokenizer"
    tok = AutoTokenizer.from_pretrained(tok_dir)

    cases = []
    for case_name, text in tokenizer_cases(tok.mask_token):
        ids = tok(text, add_special_tokens=False)["input_ids"]
        ids_ls = tok(" " + text, add_special_tokens=False)["input_ids"]
        cases.append(
            {
                "name": case_name,
                "text": text,
                "ids": ids,
                # Token strings as well as ids: an id diff tells you nothing, while
                # ["a", "b"] vs ["ab"] names the stage that broke (AGENTS.md).
                "tokens": tok.convert_ids_to_tokens(ids),
                "ids_with_leading_space": ids_ls,
                "tokens_with_leading_space": tok.convert_ids_to_tokens(ids_ls),
            }
        )

    head = header(
        name,
        models_root,
        checkpoint=ckpt,
        specials={
            "mask_token": tok.mask_token,
            "mask_token_id": tok.mask_token_id,
            "cls_token_id": tok.cls_token_id,
            "sep_token_id": tok.sep_token_id,
            "pad_token_id": tok.pad_token_id,
            "unk_token_id": tok.unk_token_id,
        },
    )
    write_jsonl(out_dir / f"{name}.jsonl", head, cases)


# --------------------------------------------------------------------- 1.3.3 render


def render_cases() -> list[dict[str, Any]]:
    """`render_criterion` / `render_options` / `serialize_state`, invariants 14-18.

    Ported from ``original/tests/test_criteria.py:33-100`` plus the golden cases
    ``docs/INVARIANTS.md`` #17 names. ``test_criteria.py:103-116`` is deliberately
    left out: it asserts five string literals are present in ``Agent.__init__`` via
    ``inspect.getsource``, which is not a behaviour a port can or should reproduce
    (PLAN.md task 2.5.4).

    Output is always a **string** (or a list of strings). Never a nested object: #17's
    ``{"d": "münchen"}`` -> literal-``ü`` case is only verifiable if the rendered text
    is carried as text.
    """
    from laya.common import render_criterion, render_options, serialize_state

    cases: list[dict[str, Any]] = []

    def crit(name: str, value: Any) -> None:
        cases.append(
            {
                "name": f"criterion/{name}",
                "fn": "render_criterion",
                "input": value,
                "output": render_criterion(value),
            }
        )

    crit("str-passthrough", "phishing or scam")
    crit("dict", {"desc": "phishing"})
    crit("list", ["a", "b"])
    crit("int", 3)
    crit("float", 3.5)
    crit("bool-false", False)
    crit("bool-true", True)
    crit("none", None)
    crit("non-ascii", {"d": "münchen"})
    crit("nested", {"a": {"b": [1, {"c": None}]}})
    crit("empty-string", "")
    crit("empty-dict", {})
    # `default=str` -- the json.dumps fallback for anything unserialisable (Go's
    # equivalent is the fmt.Sprintf("%v") fallback in Task 2.1.4). NOT `object()`:
    # its str() embeds a heap address, so the fixture would differ on every run.
    #
    # The *input* is recorded post-substitution, because a Python date or Decimal
    # cannot be written into JSON and Go could not construct one anyway. Feeding Go
    # the substituted form must produce the same output, which is the property worth
    # asserting; `original_types` says what was really passed in.
    for name, value, subst in [
        ("date", {"when": _dt.date(2026, 9, 20)}, {"when": "2026-09-20"}),
        ("decimal", {"amount": Decimal("1.50")}, {"amount": "1.50"}),
        ("complex", {"z": complex(1, 2)}, {"z": "(1+2j)"}),
    ]:
        cases.append(
            {
                "name": f"criterion/default-str-{name}",
                "fn": "render_criterion",
                "input": subst,
                "output": render_criterion(value),
                "original_types": {k: type(v).__name__ for k, v in value.items()},
                "note": "input recorded after json.dumps(default=str) substitution",
            }
        )

    def opts(name: str, q: dict[str, Any]) -> None:
        cases.append(
            {
                "name": f"options/{name}",
                "fn": "render_options",
                "input": q,
                "output": render_options(q),
            }
        )

    opts("choice/dict-desc", {"t": "choice", "ins": "x", "crit": {"billing": {"desc": "payments"}}})
    # 0 and False are legitimate criterion values; only None and "" mean "bare key".
    opts("choice/bare-forms", {"t": "choice", "ins": "x", "crit": {"tech": None, "sales": ""}})
    opts("choice/zero-and-false", {"t": "choice", "ins": "x", "crit": {"zero": 0, "no": False}})
    opts("choice/strings", {"t": "choice", "ins": "x", "crit": {"a": "first", "b": None}})
    opts(
        "choice/mixed", {"t": "choice", "ins": "x", "crit": {"a": {"n": 1}, "b": [1, 2], "c": 3.5}}
    )
    opts("choice/empty", {"t": "choice", "ins": "x", "crit": {}})
    opts("score/mixed-levels", {"t": "score", "ins": "x", "crit": [{"d": "low"}, "high", 2]})
    opts("score/strings", {"t": "score", "ins": "x", "crit": ["low", "high"]})
    opts("score/with-none", {"t": "score", "ins": "x", "crit": [{"a": 1}, [2], None]})
    opts("score/empty", {"t": "score", "ins": "x", "crit": []})
    opts("noul/defaults", {"t": "noul", "ins": "x", "crit": None})
    opts("noul/empty-dict", {"t": "noul", "ins": "x", "crit": {}})
    opts("noul/strings", {"t": "noul", "ins": "x", "crit": {"true": "yes it is", "false": "no"}})
    # Written true-first on purpose: the rendered order is fixed false=0, true=1
    # regardless of how the dict was written.
    opts(
        "noul/reversed-dict-order",
        {"t": "noul", "ins": "x", "crit": {"true": {"desc": "phishing"}, "false": {"desc": "ok"}}},
    )
    opts("noul/dict-criteria", {"t": "noul", "ins": "x", "crit": {"true": [1], "false": {"z": 0}}})
    # `score` with a dict criteria: enumerate() walks the KEYS, so the level text is
    # the key, not the value. Not an error upstream, just surprising -- pin it.
    opts("score/dict-criteria", {"t": "score", "ins": "x", "crit": {"low": 1, "high": 2}})

    def state(name: str, value: Any) -> None:
        cases.append(
            {
                "name": f"state/{name}",
                "fn": "serialize_state",
                "input": value,
                "output": serialize_state(value),
            }
        )

    state("str-passthrough", "just a string")
    state("str-with-braces", '{"not": "parsed"}')
    state("dict-flat", {"subject": "invoice 4411", "amount": 42})
    state("dict-nested", {"user": {"id": 7, "tags": ["a", "b"]}, "ok": True, "none": None})
    # Insertion order, not sorted: Go's encoding/json sorts map keys, which changes
    # the bytes the model sees (invariant #18).
    state("dict-unsorted-keys", {"z": 1, "a": 2, "m": 3})
    # No HTML escaping: encoding/json escapes <, > and & by default.
    state("dict-html-chars", {"html": "<b>a & b</b>"})
    state("dict-non-ascii", {"city": "München", "emoji": "\U0001f680"})
    state("list", [1, "two", {"three": 3}])
    state("empty-dict", {})
    state("empty-list", [])
    return cases


# ------------------------------------------------------------------- 1.3.6 mailtext


def mailtext_cases() -> list[dict[str, Any]]:
    """`clean_email_body` / `email_state` / `email_questions`, invariants 66-74.

    This block is the only one in ``docs/INVARIANTS.md`` with neither an upstream test
    nor a golden vector: ``original/tests/`` covers criteria, routing and e2e, and
    nothing covers ``email.py``. PLAN.md task 2.3.3 says to "write the tests this port
    deserves" -- writing them against vectors beats writing them against a reading of
    the regexes, which is how a Go bug becomes the expected answer.

    The signature cut is the part worth pinning: the scan window is
    ``range(max(1, min(int(len(lines) * 0.6), len(lines) - 8)), len(lines))``, so
    whether a sign-off is stripped depends on how long the message is.
    """
    from laya.email import clean_email_body, email_questions, email_state

    long_body = "\n".join(f"line {i} of the actual message body" for i in range(20))
    cases: list[dict[str, Any]] = []

    def body(name: str, text: str, max_chars: int | None = None) -> None:
        out = clean_email_body(text) if max_chars is None else clean_email_body(text, max_chars)
        rec: dict[str, Any] = {
            "name": f"clean/{name}",
            "fn": "clean_email_body",
            "input": text,
            "output": out,
        }
        if max_chars is not None:
            rec["max_chars"] = max_chars
        cases.append(rec)

    body("empty", "")
    body("plain", "Hello,\n\nplease refund invoice 4411.\n")
    body("crlf", "Hello,\r\n\r\nplease refund invoice 4411.\r\n")
    body("literal-backslash-n", "Hello,\\n\\nplease refund invoice 4411.")
    body("quoted-gt", "My reply.\n> their text\n> more of it\nstill mine.")
    body("quote-header-on-wrote", "My reply.\nOn Mon, someone wrote:\nold stuff here.")
    body("quote-header-original", "My reply.\n-- Original Message --\nold stuff.")
    body("quote-header-underscores", "My reply.\n________\nold stuff.")
    body("quote-header-from", "My reply.\nFrom: someone@example.com\nold stuff.")
    # A quote header on the FIRST line does not cut: `and lines` is falsy there.
    body("quote-header-first-line", "From: a@b.c\nthe whole message is here.")
    body("signature-dashes", long_body + "\n--\nChristian")
    body("signature-regards", long_body + "\nBest regards,\nChristian")
    body("signature-sent-from", long_body + "\nSent from my iPhone")
    # Counter-intuitive and therefore worth a vector: for a SHORT message the scan
    # window is at its widest, not its narrowest. `min(int(n*0.6), n - 8)` goes
    # negative below nine lines and `max(1, ...)` then pins the start at 1, so the
    # sign-off is stripped here while a longer message would have to reach 60% first.
    body("signature-short-body", "Thanks!\nBest regards,\nChristian")
    body("disclaimer-confidential", "Real content.\n\nThis email is confidential.")
    body(
        "disclaimer-received-in-error",
        "Real content.\n\nIf you received this email in error, delete it.",
    )
    body("collapse-spaces", "a    b\t\tc")
    body("blank-paragraphs", "one\n\n\n\ntwo")
    body("truncation", "x" * 50, 10)

    for name, kwargs in [
        ("minimal", {"subject": " Invoice ", "body": "Please refund.\n"}),
        ("with-sender", {"subject": "S", "body": "B", "sender": "a@b.c"}),
        ("clean-off", {"subject": "S", "body": "B\n> quoted", "clean": False}),
        ("extra-fields", {"subject": "S", "body": "B", "priority": "high", "dropped": None}),
        ("empty-subject", {"subject": None, "body": "B"}),
    ]:
        cases.append(
            {
                "name": f"state/{name}",
                "fn": "email_state",
                "input": kwargs,
                "output": email_state(**kwargs),
            }
        )

    cases.append(
        {
            "name": "questions/defaults",
            "fn": "email_questions",
            "input": None,
            "output": email_questions(),
        }
    )
    cases.append(
        {
            "name": "questions/custom-categories",
            "fn": "email_questions",
            "input": {"a": "first", "b": "second"},
            "output": email_questions({"a": "first", "b": "second"}),
        }
    )
    return cases


# ------------------------------------------------------------------- 1.3.2 sequence


def to_internal(qdef: dict[str, Any]) -> dict[str, Any]:
    """Replicate ``laya.Agent._to_internal`` (``agent.py:229-238``).

    Kept here rather than imported because ``Agent`` cannot be constructed without a
    checkpoint, and 1.3.2 needs no weights. A ``choice`` given a list becomes
    ``{label: None}``, which silently de-duplicates repeated labels.
    """
    t = qdef.get("type", "choice")
    crit = qdef.get("criteria")
    if t == "choice" and isinstance(crit, list):
        crit = {c: None for c in crit}
    ins = qdef.get("instructions", "")
    # The one place laya escapes non-ASCII: ensure_ascii is left at its default True
    # here, unlike every other json.dumps in the package (agent.py:236-237).
    if not isinstance(ins, str):
        ins = json.dumps(ins)
    return {"t": t, "ins": ins, "crit": crit}


def sequence_cases(ckpt: str, tok: Any, max_len: int, head_max_len: int) -> list[dict[str, Any]]:
    """The `(qtype, criteria, state, budget, order, direction)` matrix, invariants 1-12."""
    from laya.common import build_sequence, render_options

    long_state = {"body": " ".join(f"word{i}" for i in range(400)), "subject": "invoice 4411"}
    specs: list[tuple[str, dict[str, Any], Any, dict[str, Any]]] = [
        (
            "choice/basic",
            {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "invoices", "tech": "bugs", "sales": None},
            },
            {"subject": "refund", "body": "charged twice"},
            {},
        ),
        (
            "choice/list-criteria",
            {
                "type": "choice",
                "instructions": "Pick one",
                "criteria": ["billing", "tech", "billing"],
            },
            "plain text state",
            {},
        ),
        (
            "choice/empty-criteria",
            {"type": "choice", "instructions": "Nothing", "criteria": {}},
            "state",
            {},
        ),
        (
            "score/basic",
            {
                "type": "score",
                "instructions": "How urgent?",
                "criteria": ["none", "soon", "blocking"],
            },
            {"body": "asap"},
            {},
        ),
        ("noul/defaults", {"type": "noul", "instructions": "Is this spam?"}, "buy now", {}),
        (
            "noul/criteria",
            {
                "type": "noul",
                "instructions": "Phishing?",
                "criteria": {"true": "scam", "false": "legit"},
            },
            "click here",
            {},
        ),
        # Invariant #6: 77 options at head_max_len=192 gives per=4 and a negative
        # opt_budget, so head_ids falls back to its floor of 8.
        (
            "choice/77-options",
            {
                "type": "choice",
                "instructions": "Overflow the head",
                "criteria": {f"k{i:02d}": f"desc {i}" for i in range(77)},
            },
            "s",
            {},
        ),
        (
            "choice/long-state",
            {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None}},
            long_state,
            {},
        ),
        (
            "choice/truncate-left",
            {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None}},
            long_state,
            {"truncate_left": True},
        ),
        # option_order is never passed in production (agent.py:261 uses the 5-arg
        # form) but it is in the signature, and it permutes marker semantics.
        (
            "choice/option-order",
            {
                "type": "choice",
                "instructions": "Reordered",
                "criteria": {"a": "one", "b": "two", "c": "three"},
            },
            "s",
            {"option_order": [2, 0, 1]},
        ),
        (
            "choice/tiny-max-len",
            {"type": "choice", "instructions": "Cramped", "criteria": {"a": "one", "b": "two"}},
            long_state,
            {"max_len": 32},
        ),
        (
            "choice/tiny-head",
            {
                "type": "choice",
                "instructions": "Cramped head",
                "criteria": {f"k{i}": f"a fairly long description {i}" for i in range(8)},
            },
            "s",
            {"head_max_len": 24},
        ),
        (
            "choice/state-list",
            {"type": "choice", "instructions": "Which team?", "criteria": {"a": None, "b": None}},
            [1, "two", {"three": 3}],
            {},
        ),
        (
            "choice/non-ascii",
            {
                "type": "choice",
                "instructions": "Welche Abteilung?",
                "criteria": {"buchhaltung": "Rechnungen", "technik": "Fehler"},
            },
            {"stadt": "München"},
            {},
        ),
        (
            "choice/mask-in-text",
            {
                "type": "choice",
                "instructions": f"Is {tok.mask_token} relevant?",
                "criteria": {"yes": None, "no": None},
            },
            {"body": f"contains {tok.mask_token} literal"},
            {},
        ),
    ]

    cases = []
    for name, qdef, state, kw in specs:
        q = to_internal(qdef)
        ml = kw.get("max_len", max_len)
        hml = kw.get("head_max_len", head_max_len)
        order = kw.get("option_order")
        tl = kw.get("truncate_left", False)
        ids, markers = build_sequence(tok, state, q, ml, hml, order, tl)
        cases.append(
            {
                "name": name,
                "checkpoint": ckpt,
                "state": state,
                "qdef": qdef,
                "q": q,
                "max_len": ml,
                "head_max_len": hml,
                "option_order": order,
                "truncate_left": tl,
                "ids": ids,
                "markers": markers,
                "n_options": len(render_options(q)),
                # agent.py:262-263 raises when these disagree; the Go port owes the
                # same error, so the fixture has to say when it fires.
                "markers_lost": len(render_options(q)) - len(markers),
            }
        )
    return cases


def dump_sequence(models_root: Path, out_dir: Path) -> None:
    from transformers import AutoTokenizer

    budgets = {"english": (512, 192), "multilingual": (1024, 256), "typed-decisions": (1024, 256)}
    cases: list[dict[str, Any]] = []
    for ckpt in sorted(CHECKPOINT_DIRS):
        tok_dir = models_root / "laya" / CHECKPOINT_DIRS[ckpt] / "tokenizer"
        tok = AutoTokenizer.from_pretrained(tok_dir)
        ml, hml = budgets[ckpt]
        for c in sequence_cases(ckpt, tok, ml, hml):
            c["name"] = f"{ckpt}/{c['name']}"
            cases.append(c)
    write_jsonl(out_dir / "sequence.jsonl", header("sequence", models_root), cases)


# -------------------------------------------------------------------- 1.3.4 answers


def answer_block(
    logit_row: list[float],
    act_row: list[float],
    temperature: list[float],
    temperature_by_options: dict[str, float],
    k: int,
    qtype: str,
    crit: Any,
) -> dict[str, Any]:
    """Replicate ``agent.py:294-343`` for one question.

    Inlined rather than driven through ``Agent.system_one`` because there is no seam:
    the arithmetic lives inside the batch loop. ``dump_logits`` verifies this
    replication against a real ``Agent`` run rather than trusting it.

    ``act_row`` is **pre**-softmax, matching what the model emits; ``agent.py:295``
    softmaxes the whole tensor and then takes column 0.
    """
    from laya.common import QTYPES, confidence_from_probs, temp_bucket

    qt = QTYPES[qtype]
    t_scale = temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
    z = np.array(logit_row, dtype=np.float32)[:k] / max(1e-3, float(t_scale))
    p = np.exp(z - z.max())
    p = p / p.sum()

    a = np.array(act_row, dtype=np.float32)
    a = np.exp(a - a.max())
    a = a / a.sum()

    conf_score = round(confidence_from_probs(p, k), 4)
    ext = {"act_probability": round(float(a[0]), 4)}

    if qtype == "choice":
        keys = list(crit.keys())
        # zip() without strict=: if the criteria and the marker count disagree the
        # upstream silently truncates to the shorter. Reproduced, not fixed.
        return {
            "type": "choice",
            "choice": keys[int(p.argmax())],
            "probabilities": {
                # noqa is the point: strict=True would raise where upstream truncates.
                kk: round(float(v), 4)
                for kk, v in zip(keys, p)  # noqa: B905
            },
            "confidence": conf_score,
            "action": ext,
        }
    if qtype == "score":
        return {
            "type": "score",
            "score": round(float((np.arange(k) * p).sum()), 4),
            "legend": {str(i): c for i, c in enumerate(crit)},
            "probabilities": {str(i): round(float(v), 4) for i, v in enumerate(p)},
            "confidence": conf_score,
            "action": ext,
        }
    return {
        "type": "noul",
        "noul": round(float(p[1]), 4),
        # NOT confidence_from_probs: for k=2 the two genuinely disagree (#26).
        "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
        "action": ext,
    }


def answers_cases(models_root: Path) -> list[dict[str, Any]]:
    """Synthetic logits through the answer path, invariants 22-32."""
    cfg_en = json.loads((models_root / "laya" / "rl_agent_config.json").read_text())
    temp = cfg_en["temperature"]
    # english carries every bucket; multilingual's is {} and exercises the fallback
    # to temperature[qtype], so both are represented.
    tbo_full = cfg_en["temperature_by_options"]
    tbo_empty: dict[str, float] = {}

    specs: list[tuple[str, str, Any, int, dict[str, float]]] = []
    # k sweeps the temp_bucket thresholds: 2, 3-5, 6-10, 11+.
    for k in (2, 3, 5, 6, 10, 11, 13):
        specs.append(
            (
                f"choice/k{k:02d}-buckets",
                "choice",
                {f"key{i}": f"desc {i}" for i in range(k)},
                k,
                tbo_full,
            )
        )
        specs.append(
            (
                f"choice/k{k:02d}-fallback",
                "choice",
                {f"key{i}": f"desc {i}" for i in range(k)},
                k,
                tbo_empty,
            )
        )
    for k in (2, 3, 6, 11):
        specs.append((f"score/k{k:02d}", "score", [f"level {i}" for i in range(k)], k, tbo_full))
        specs.append((f"noul/k{k:02d}", "noul", {"true": "yes", "false": "no"}, k, tbo_full))
    # #31: the legend carries the RAW criterion, so a dict criterion stays a dict.
    specs.append(("score/dict-legend", "score", [{"d": "low"}, "mid", 2], 3, tbo_full))
    # A choice whose criteria are shorter than k is deliberately NOT here. zip()
    # would truncate the probabilities and `keys[p.argmax()]` would then raise
    # IndexError -- but the state is unreachable through the public API, because
    # agent.py:262-263 raises ValueError when len(markers) != len(render_options(q)).
    # sequence.jsonl's `markers_lost` field is where that guard is pinned instead.

    cases: list[dict[str, Any]] = []
    for name, qtype, crit, k, tbo in specs:
        # NOT hash(name): Python randomises str hashing per process unless
        # PYTHONHASHSEED is set, so the corpus would differ on every run (1.3.7).
        rng = np.random.default_rng(SEED + stable_seed(name))
        logits = (rng.standard_normal(max(k, 4)) * 3.0).round(6).tolist()
        # Asymmetric on purpose: act_probability is column 0, and a 0/1 swap has to
        # be detectable (#32).
        act = [float(rng.standard_normal() * 2.0 + 1.5), float(rng.standard_normal() * 2.0 - 1.5)]
        ans = answer_block(logits, act, temp, tbo, k, qtype, crit)
        cases.append(
            {
                "name": name,
                "qtype": qtype,
                "k": k,
                "criteria": crit,
                "logits": logits,
                "act_logits": [round(v, 6) for v in act],
                "temperature": temp,
                "temperature_by_options": tbo,
                "answer": ans,
                # 7.3.2 compares the serialized bytes, not the parsed struct.
                "answer_json": json.dumps(ans, ensure_ascii=False, separators=(", ", ": ")),
            }
        )

    # Hand-injected, because no random sample produces them.
    def inject(
        name: str, qtype: str, crit: Any, k: int, logits: list[float], act: list[float]
    ) -> None:
        ans = answer_block(logits, act, temp, tbo_empty, k, qtype, crit)
        cases.append(
            {
                "name": name,
                "qtype": qtype,
                "k": k,
                "criteria": crit,
                "logits": logits,
                "act_logits": act,
                "temperature": temp,
                "temperature_by_options": tbo_empty,
                "answer": ans,
                "answer_json": json.dumps(ans, ensure_ascii=False, separators=(", ", ": ")),
            }
        )

    # A score of exactly 0.0. `Score` is a pointer field in Go precisely so that this
    # emits rather than being dropped as a zero value (docs/API.md:203-205).
    inject("score/exactly-zero", "score", ["a", "b", "c"], 3, [200.0, 0.0, 0.0], [1.0, -1.0])
    # k=2 under both confidence formulas: for a uniform p the entropy form gives 0.0
    # and noul's max(p1, 1-p1) gives 0.5. They must not be unified.
    inject("choice/k02-uniform", "choice", {"a": None, "b": None}, 2, [0.0, 0.0], [0.0, 0.0])
    inject("noul/k02-uniform", "noul", {"true": "y", "false": "n"}, 2, [0.0, 0.0], [0.0, 0.0])
    # A near-certain outcome: confidence saturates at 1.0 and the clip matters.
    inject("choice/saturated", "choice", {"a": None, "b": None}, 2, [50.0, -50.0], [9.0, -9.0])
    return cases


# ---------------------------------------------------------------------- Round4 ties


def round4_cases() -> list[dict[str, Any]]:
    """`round(x, 4)` is half-to-even on the binary double; Go's math.Round is not.

    Invariant #29, and one of the four ``PLAN.md`` §5 singles out as causing silent
    wrong answers. It gets its own file because no sampled softmax output lands on a
    tie -- the values have to be chosen -- and because Task 2.1.5/2.1.6 implements
    ``Round4`` in ``jsonx`` long before any answer exists to apply it to.
    """
    values = [
        0.00125,
        0.00135,
        0.00115,
        -0.00125,
        -0.00135,
        2.5e-5,
        3.5e-5,
        1.5e-5,
        -2.5e-5,
        0.12345,
        0.123450001,
        0.98765,
        0.5,
        -0.5,
        0.0,
        -0.0,
        1.0,
        0.99995,
        0.99994999,
        1e-9,
        -1e-9,
        1 / 3,
        2 / 3,
        math.pi,
        math.e,
        1e10 + 0.00005,
    ]
    return [
        {
            "name": f"round4/{i:02d}",
            "input": v,
            # repr() pins the exact double the fixture means, independent of how a
            # JSON reader chooses to print it.
            "input_repr": repr(v),
            "output": round(v, 4),
            "output_repr": repr(round(v, 4)),
        }
        for i, v in enumerate(values)
    ]


# --------------------------------------------------------------------- 1.3.5 logits


LOGITS_QUESTIONS: dict[str, dict[str, Any]] = {
    "category": {
        "type": "choice",
        "instructions": "Which team should handle the email in `body`?",
        "criteria": {
            "billing": "invoices, payments, refunds",
            "technical": "bugs, outages, integrations",
            "sales": "pricing, demos, new purchases",
            "other": "none of the above",
        },
    },
    "urgency": {
        "type": "score",
        "instructions": "How urgent is the request in `body`?",
        "criteria": ["no time pressure", "needs attention soon", "blocking issue"],
    },
    "is_spam": {"type": "noul", "instructions": "Is this email unsolicited spam?"},
}

LOGITS_STATES: list[tuple[str, Any]] = [
    (
        "en/billing",
        {"subject": "Duplicate charge", "body": "I was charged twice for invoice 4411."},
    ),
    ("en/technical", {"subject": "API 500s", "body": "Your API returns 500 on every POST."}),
    ("en/sales", {"subject": "Pricing", "body": "Can we see a demo and enterprise pricing?"}),
    ("en/spam", {"subject": "WIN NOW", "body": "Claim your free prize, click here!!!"}),
    (
        "de/billing",
        {"subject": "Doppelte Abbuchung", "body": "Rechnung 4411 wurde zweimal belastet."},
    ),
    ("ja/support", {"subject": "問い合わせ", "body": "ログインできません。"}),
    ("ar/billing", {"subject": "فاتورة", "body": "تم خصم المبلغ مرتين."}),
    ("plain-string", "The customer wants a refund for invoice 4411."),
    ("list-state", [{"role": "user", "text": "refund please"}, {"role": "agent", "text": "ok"}]),
    ("empty-body", {"subject": "", "body": ""}),
]


def dump_logits(models_root: Path, out_dir: Path, checkpoints: list[str]) -> None:
    """End-to-end ``(state, questions) -> (logits, act_logits)``, batch-level.

    Batch-level is not a convenience: ``collate_items`` derives ``L`` and ``kmax``
    from the whole batch, and the action features soft-max over the padded ``kmax``
    (``common.py:119-125``), so one question in isolation is a *different*
    computation. Every record therefore carries the whole ``questions`` dict.

    Only final outputs are stored. Invariants #35-38 (the manual head loop, the
    ``-1e4`` masking, the four act features) need intermediates, but they are only
    asserted if the head is reimplemented, which is M8 -- deferred post-1.0. Task 8.8
    regenerates with intermediates when it starts.
    """
    import torch
    from transformers import AutoTokenizer

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    from export_onnx import build_decision_model

    from laya.common import build_sequence, collate_items, render_options

    torch.set_num_threads(1)

    cases: list[dict[str, Any]] = []
    for ckpt in checkpoints:
        ckpt_dir = models_root / "laya" / CHECKPOINT_DIRS[ckpt]
        cfg = json.loads((ckpt_dir / "rl_agent_config.json").read_text())
        tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
        print(f"  building {ckpt} ...", flush=True)
        model = build_decision_model(ckpt_dir, attn="sdpa")
        max_len, head_max_len = cfg.get("max_len", 512), cfg.get("head_max_len", 192)

        for state_name, state in LOGITS_STATES:
            items, qids = [], []
            for qid, qdef in LOGITS_QUESTIONS.items():
                q = to_internal(qdef)
                ids, markers = build_sequence(tok, state, q, max_len, head_max_len)
                # The same guard the Agent applies (agent.py:262-263). A fixture case
                # that silently dropped a marker would be a wrong golden vector.
                if len(markers) != len(render_options(q)):
                    raise SystemExit(
                        f"{ckpt}/{state_name}/{qid}: options exceed head_max_len="
                        f"{head_max_len}; upstream raises ValueError here"
                    )
                items.append({"ids": ids, "markers": markers, "qtype": QTYPES_MAP[q["t"]]})
                qids.append(qid)
            b = collate_items([items], tok.pad_token_id)
            with torch.no_grad():
                logits, act_logits = model(
                    b["input_ids"],
                    b["attention_mask"],
                    b["marker_pos"],
                    b["marker_mask"],
                    b["qtype"],
                    False,
                )
            cases.append(
                {
                    "name": f"{ckpt}/{state_name}",
                    "checkpoint": ckpt,
                    "state": state,
                    "questions": LOGITS_QUESTIONS,
                    "qids": qids,
                    "qtypes": [QTYPES_MAP[to_internal(LOGITS_QUESTIONS[q])["t"]] for q in qids],
                    # #33: summed over the WHOLE batch, not per question.
                    "input_tokens": int(b["attention_mask"].sum()),
                    "logits": tensor_rec(logits.float().numpy()),
                    "act_logits": tensor_rec(act_logits.float().numpy()),
                }
            )
        del model

    head = header(
        "logits",
        models_root,
        torch=torch_version(),
        note=(
            "Final outputs only. Invariants 35-38 need head intermediates and are "
            "only asserted if the head is reimplemented (M8, deferred); PLAN.md task "
            "8.8 regenerates this file with intermediates when it starts."
        ),
    )
    write_jsonl(out_dir / "logits.jsonl", head, cases)


def tensor_rec(a: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.reshape(-1).tolist()}


QTYPES_MAP = {"choice": 0, "score": 1, "noul": 2}


# ---------------------------------------------------------------------------- main


def verify_answer_block(models_root: Path) -> None:
    """Check `answer_block` against a real `Agent.system_one` before trusting it.

    ``answers.jsonl`` is generated by a hand-inlined copy of ``agent.py:294-343``,
    because the arithmetic has no seam -- it lives inside the batch loop. A copy that
    drifts from the original would produce a corpus that is internally consistent and
    wrong, which is the worst possible outcome for a parity fixture. So: run the real
    ``Agent`` once, run the copy on the same forward pass, and compare every field.
    """
    import torch
    from transformers import AutoTokenizer

    import laya
    from laya.common import build_sequence, collate_items

    torch.set_num_threads(1)
    ckpt_dir = models_root / "laya"
    cfg = json.loads((ckpt_dir / "rl_agent_config.json").read_text())
    state = {"subject": "Duplicate charge", "body": "I was charged twice for invoice 4411."}

    agent = laya.load(str(ckpt_dir), device="cpu")
    want = agent.system_one(state, LOGITS_QUESTIONS)["answers"]

    # The same forward pass, driven by hand, then through the inlined copy.
    tok = AutoTokenizer.from_pretrained(ckpt_dir / "tokenizer")
    items, qids = [], []
    for qid, qdef in LOGITS_QUESTIONS.items():
        q = to_internal(qdef)
        ids, markers = build_sequence(
            tok, state, q, cfg.get("max_len", 512), cfg.get("head_max_len", 192)
        )
        items.append({"ids": ids, "markers": markers, "qtype": QTYPES_MAP[q["t"]]})
        qids.append(qid)
    b = collate_items([items], tok.pad_token_id)
    with torch.no_grad():
        logits, act = agent.model(
            b["input_ids"],
            b["attention_mask"],
            b["marker_pos"],
            b["marker_mask"],
            b["qtype"],
            False,
        )
    logits, act = logits.float().numpy(), act.float().numpy()

    bad = []
    for r, qid in enumerate(qids):
        q = to_internal(LOGITS_QUESTIONS[qid])
        got = answer_block(
            logits[r].tolist(),
            act[r].tolist(),
            cfg["temperature"],
            cfg["temperature_by_options"],
            len(items[r]["markers"]),
            q["t"],
            q["crit"],
        )
        if got != want[qid]:
            bad.append(f"{qid}:\n      copy {got}\n      real {want[qid]}")
    if bad:
        raise SystemExit("answer_block verification FAILED:\n    " + "\n    ".join(bad))
    print(f"  answer_block matches laya.Agent.system_one on all {len(qids)} questions")


FIXTURES = (
    "tokenizer_en",
    "tokenizer_ml",
    "sequence",
    "render",
    "answers",
    "round4",
    "mailtext",
    "logits",
)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fixture", choices=FIXTURES, action="append", default=[])
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "testdata")
    ap.add_argument(
        "--checkpoint",
        choices=sorted(CHECKPOINT_DIRS),
        action="append",
        default=[],
        help="restrict the logits fixture; the others always cover every checkpoint",
    )
    ap.add_argument(
        "--verify-agent",
        action="store_true",
        help="cross-check answer_block against laya.Agent (needs weights)",
    )
    args = ap.parse_args()

    wanted = list(FIXTURES) if args.all or not args.fixture else args.fixture
    print(json.dumps(versions(), indent=2))

    if "tokenizer_en" in wanted:
        dump_tokenizer("tokenizer_en", "english", args.models_root, args.out)
    if "tokenizer_ml" in wanted:
        dump_tokenizer("tokenizer_ml", "multilingual", args.models_root, args.out)
    if "sequence" in wanted:
        dump_sequence(args.models_root, args.out)
    if "render" in wanted:
        write_jsonl(args.out / "render.jsonl", header("render", args.models_root), render_cases())
    if "answers" in wanted:
        write_jsonl(
            args.out / "answers.jsonl",
            header("answers", args.models_root),
            answers_cases(args.models_root),
        )
    if "round4" in wanted:
        write_jsonl(args.out / "round4.jsonl", header("round4", args.models_root), round4_cases())
    if "mailtext" in wanted:
        write_jsonl(
            args.out / "mailtext.jsonl", header("mailtext", args.models_root), mailtext_cases()
        )
    if args.verify_agent:
        verify_answer_block(args.models_root)
    if "logits" in wanted:
        cks = args.checkpoint or sorted(CHECKPOINT_DIRS)
        dump_logits(args.models_root, args.out, cks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
