#!/usr/bin/env python3
"""Diff the reference environment's numerics against a transformers 4.x build.

PLAN.md task 1.6. The reference environment resolved to **transformers 5.17.0**, but
laya 0.3.4 declares ``transformers>=4.45.0`` with no upper bound and predates 5.x --
indeed its declared floor predates ModernBERT itself, which landed in 4.48. That the
checkpoint loads with ``load_state_dict(strict=True)`` proves the *parameter set*
matches; it says nothing about numerics.

The surface under test is ``AutoModel.from_config(ecfg, attn_implementation="sdpa")``
(``original/laya/common.py:134``). ModernBERT's sliding-window and global attention
masks are built by ``masking_utils.py`` in 5.x and inside the model in 4.x, and a
different mask builder is exactly how padded-position numerics diverge without any
parameter changing. ``reference_compile`` is load-bearing under 4.x and swallowed
under 5.x (``original/laya/agent.py:189-192``), so it is forced off either way.

Tokenizer ids are diffed too. It is the same question -- R6 is *"upstream
tokenizer/transformers version drift silently changes golden vectors"* -- and the two
environments happen to resolve different ``tokenizers`` builds, so the check is free.

This must run **before** task 1.3 freezes any golden vector: if the two disagree
beyond float noise, 1.6.2 says re-pin the reference environment, which would
invalidate every vector written before it.

**Result (2026-09-20): keep 5.17.0. Do not apply 1.6.2.** The two ModernBERT-large
checkpoints are bit-identical across the major. ``multilingual`` diverges hugely --
logits by 6.96, act_logits by 1879 -- and the cause is a parsing bug in 4.x, not a
numerics difference. mmBERT's config sets ``rope_parameters.sliding_attention.
rope_theta = 160000``; transformers 4.57.6 loads it as ``local_rope_theta = 10000.0``,
its default, so every sliding-attention layer runs on the wrong RoPE frequencies.
Setting that one value by hand makes 4.57.6 bit-identical to 5.17.0, which is the
whole difference. English is unaffected only because its sliding ``rope_theta``
genuinely is 10000.0 and so survives being ignored. Pinning to 4.x -- what 1.6.2
presumes is the safe direction -- would silently corrupt the multilingual checkpoint.

Usage::

    .venv-ref/bin/python  scripts/crosscheck_transformers.py --all --out build/x-5.json
    .venv-ref4/bin/python scripts/crosscheck_transformers.py --all --out build/x-4.json
    .venv-ref/bin/python  scripts/crosscheck_transformers.py \
        --compare build/x-4.json build/x-5.json

Neither this script nor its output is needed to build or test the Go module.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from export_onnx import CHECKPOINTS, build_decision_model, example_inputs  # noqa: E402

# Probe strings for the tokenizer half. Deliberately the drift-prone shapes from
# PLAN.md task 4.4 rather than prose: whitespace runs, added tokens, normalisation
# and byte fallback are where a tokenizers bump would actually show.
PROBES = (
    "",
    " ",
    "   \n   ",
    "billing: refund the duplicate charge",
    '{"amount": 42, "currency": "EUR"}',
    "café",  # precomposed
    "café",  # decomposed
    "Ａﬁ",  # fullwidth A, ligature fi
    "\U0001f1e9\U0001f1ea \U0001f469‍\U0001f4bb",  # flag, ZWJ sequence
    "मुंबई القاهرة 東京",
    "a" + " " * 24 + "b",
)


def versions() -> dict[str, str]:
    """The pins that make a diff meaningful. No onnx here: the 4.x venv has none."""
    import safetensors
    import tokenizers
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "tokenizers": tokenizers.__version__,
        "numpy": np.__version__,
        "safetensors": safetensors.__version__,
    }


def tensor(a: np.ndarray) -> dict[str, Any]:
    return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.reshape(-1).tolist()}


def probe_tokenizer(ckpt_dir: Path) -> dict[str, Any]:
    """Encode the probe strings exactly as ``build_sequence`` would.

    ``add_special_tokens=False`` and the ``" " + text`` variant are the two things
    ``original/laya/common.py:63-68`` actually does; nothing else is parity surface.
    """
    from transformers import AutoTokenizer

    tok_dir = ckpt_dir / "tokenizer"
    if not tok_dir.is_dir():
        return {"error": f"{tok_dir} is missing"}
    tok = AutoTokenizer.from_pretrained(tok_dir)
    return {
        "specials": {
            "mask_token": tok.mask_token,
            "mask_token_id": tok.mask_token_id,
            "cls_token_id": tok.cls_token_id,
            "sep_token_id": tok.sep_token_id,
            "pad_token_id": tok.pad_token_id,
        },
        "probes": [
            {
                "text": t,
                "ids": tok(t, add_special_tokens=False)["input_ids"],
                "ids_with_leading_space": tok(" " + t, add_special_tokens=False)["input_ids"],
            }
            for t in PROBES
        ],
    }


def run_one(name: str, models_root: Path) -> dict[str, Any]:
    """Build one checkpoint and run a single fixed batch through it, fp32 on CPU."""
    ckpt = next(c for c in CHECKPOINTS if c.name == name)
    ckpt_dir = ckpt.dir(models_root)

    # sdpa, not the exporter's eager default: sdpa is what laya.common.build_model
    # asks for, and attention-implementation dispatch is half of what 4.x/5.x changed.
    model = build_decision_model(ckpt_dir, attn="sdpa")
    inputs = example_inputs(model)
    with torch.no_grad():
        logits, act_logits = model(*inputs, False)

    return {
        "checkpoint": name,
        "encoder": ckpt.encoder,
        "inputs": {"seq": int(inputs[0].shape[1]), "batch": int(inputs[0].shape[0])},
        "logits": tensor(logits.float().numpy()),
        "act_logits": tensor(act_logits.float().numpy()),
        "tokenizer": probe_tokenizer(ckpt_dir),
    }


def diff_tensor(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    x, y = np.array(a["data"], dtype=np.float64), np.array(b["data"], dtype=np.float64)
    if a["shape"] != b["shape"]:
        return {"shape_mismatch": [a["shape"], b["shape"]]}
    abs_d = np.abs(x - y)
    # Scaled, not raw relative: logits carry a -1e4 masked_fill sentinel, so a plain
    # x/y ratio is dominated by the padding rather than by anything the model computed.
    scale = max(float(np.abs(x).max()), 1e-9)
    return {"max_abs": float(abs_d.max()), "max_scaled": float(abs_d.max() / scale)}


def compare(a_path: Path, b_path: Path) -> int:
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    print(f"A {a_path}: {a['versions']}")
    print(f"B {b_path}: {b['versions']}\n")

    worst = 0.0
    tok_mismatch = 0
    for name in sorted(set(a["results"]) & set(b["results"])):
        ra, rb = a["results"][name], b["results"][name]
        if "error" in ra or "error" in rb:
            print(f"{name}: SKIPPED  A={ra.get('error', 'ok')}  B={rb.get('error', 'ok')}")
            continue
        for out in ("logits", "act_logits"):
            d = diff_tensor(ra[out], rb[out])
            worst = max(worst, d.get("max_abs", float("inf")))
            print(
                f"{name:16s} {out:11s} max_abs={d.get('max_abs'):.3e} "
                f"max_scaled={d.get('max_scaled'):.3e}"
            )

        pa, pb = ra["tokenizer"], rb["tokenizer"]
        if pa.get("specials") != pb.get("specials"):
            tok_mismatch += 1
            print(f"{name:16s} tokenizer   SPECIALS DIFFER: {pa['specials']} vs {pb['specials']}")
        for ca, cb in zip(pa.get("probes", []), pb.get("probes", []), strict=True):
            for field in ("ids", "ids_with_leading_space"):
                if ca[field] != cb[field]:
                    tok_mismatch += 1
                    print(
                        f"{name:16s} tokenizer   {field} DIFFER for {ca['text']!r}: "
                        f"{ca[field]} vs {cb[field]}"
                    )
    for name in sorted(set(a["results"]) ^ set(b["results"])):
        print(f"{name}: present in only one run")

    print(f"\nworst max_abs across every output: {worst:.3e}")
    print(f"tokenizer mismatches: {tok_mismatch}")
    # 1e-5 is the bar the plan set: identical weights in fp32 on CPU should agree to
    # roughly float32 epsilon on these magnitudes. Anything larger is a real
    # divergence and task 1.6.2 applies.
    verdict = worst <= 1e-5 and tok_mismatch == 0
    print("VERDICT:", "MATCH" if verdict else "DIVERGENCE")
    return 0 if verdict else 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--checkpoint", choices=[c.name for c in CHECKPOINTS], action="append", default=[]
    )
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--models-root", type=Path, default=REPO_ROOT / "models")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        return compare(*args.compare)
    if args.out is None:
        ap.error("--out is required unless --compare is given")

    # Single-threaded: thread count changes reduction order and therefore the last
    # bits, which would show up as a version difference that is not one.
    torch.set_num_threads(1)
    torch.manual_seed(0)

    names = [c.name for c in CHECKPOINTS] if args.all or not args.checkpoint else args.checkpoint
    print(json.dumps(versions(), indent=2), flush=True)

    results: dict[str, Any] = {}
    for name in names:
        print(f"--- {name}", flush=True)
        try:
            results[name] = run_one(name, args.models_root)
        except Exception as exc:  # a cross-check records how it failed; it does not hide it
            print(f"    FAILED: {type(exc).__name__}: {exc}", flush=True)
            results[name] = {"checkpoint": name, "error": f"{type(exc).__name__}: {exc}"}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"versions": versions(), "results": results}) + "\n")
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
