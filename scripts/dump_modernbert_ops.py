#!/usr/bin/env python3
"""Dump ModernBERT building blocks from the pinned torch, for the native backend's tests.

``PLAN.md`` M8 reimplements ModernBERT and mmBERT in pure Go (D8). The golden vectors under
``testdata/`` only see the whole forward pass, which is too coarse to tell which block broke,
so each block gets its own oracle here: the *real* transformers modules of a tiny
``ModernBertModel``, run on seeded inputs and written to
``internal/modernbert/testdata/ops.json``. CI reads that file; it never needs Python.

Task 8.2 writes the norms. They come from the modules, not a hand-written ``nn.LayerNorm``:
``embeddings.norm``, an encoder layer's ``attn_norm`` and ``mlp_norm``, layer 0's
``attn_norm`` (an ``nn.Identity()`` upstream, ``modeling_modernbert.py:309-310``) and
``final_norm``. Every weight is drawn at random, so a weight of ones cannot pass by accident,
and one input sits at +1000, where a float32 single-pass ``E[x^2] - E[x]^2`` variance
cancels to nothing.

The file is meant to grow: Task 8.3 appends MLP cases. Each case draws from its own generator,
seeded from ``SEED`` and its name, so adding, removing or reordering cases leaves every other
record byte-identical. The header holds nothing that depends on which cases exist.

Format, one JSON object::

    {"header": {"versions": {...}, "seed": 82, "config": {...}, ...},
     "cases": [{"op": "layernorm", "name": ..., "module": ..., "module_class": ...,
                "seed": ..., "input": T, "weight": T | null, "output": T}, ...]}

where ``T`` is ``{"dtype": "float32", "shape": [...], "data": [...]}``, row-major, each value
the shortest decimal that round-trips to its float32. ``"weight": null`` means the module has
no norm (``nn.Identity``), which is not the same as a weight of ones. Each case sits on one
line, so a diff shows which case moved; ``treefmt.toml`` keeps prettier off the file.

Usage::

    .venv-ref/bin/python -I scripts/dump_modernbert_ops.py
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch import nn
from transformers import ModernBertConfig, ModernBertModel

REPO = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO / "scripts" / "requirements-ref.txt"
DEFAULT_OUT = REPO / "internal" / "modernbert" / "testdata" / "ops.json"

# Any fixed value; each case derives its own seed from it (case_seed).
SEED = 82

# Both checkpoints' encoder config.json say norm_eps 1e-05 and norm_bias false
# (models/laya/encoder and models/laya/multilingual/encoder). Set explicitly rather than
# trusting the ModernBertConfig defaults, which are not the contract.
NORM_EPS = 1e-5
NORM_BIAS = False

# Tiny on purpose: the fixture must stay a few KB. Two layers, because layer 0 is the only
# one whose attn_norm is the identity. intermediate_size is for Task 8.3's MLP cases.
CONFIG = {
    "hidden_size": 8,
    "intermediate_size": 6,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "vocab_size": 16,
    "pad_token_id": 0,
    "bos_token_id": 1,
    "eos_token_id": 2,
    "sep_token_id": 3,
    "cls_token_id": 4,
    "norm_eps": NORM_EPS,
    "norm_bias": NORM_BIAS,
}

# The versions that decide the numbers. The Go test asserts the same pins.
PINNED = ("torch", "transformers", "numpy")


def pins() -> dict[str, str]:
    """The ``name==version`` lines of requirements-ref.txt, the parity contract (R6)."""
    out = {}
    for line in REQUIREMENTS.read_text().splitlines():
        name, sep, version = line.strip().partition("==")
        if sep:
            out[name] = version
    return out


def versions() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "numpy": np.__version__,
    }


def check_environment() -> None:
    """Refuse to write a fixture from any environment but the pinned one."""
    want, got = pins(), versions()
    bad = [f"{n} {got[n]} (pinned {want.get(n)})" for n in PINNED if got[n] != want.get(n)]
    if bad:
        sys.exit(
            f"refusing to dump: {', '.join(bad)}. Run under .venv-ref, installed from "
            f"{REQUIREMENTS.relative_to(REPO)} (scripts/README.md)."
        )


def case_seed(name: str) -> int:
    """A per-case seed that does not move when other cases are added or reordered."""
    return int(hashlib.sha256(f"{SEED}:{name}".encode()).hexdigest()[:8], 16)


def f32(v: float) -> float:
    """The shortest decimal that parses back to the same float32, as a JSON number."""
    s = str(np.float32(v))
    out = float(s)
    if np.float32(out) != np.float32(v):
        raise AssertionError(f"{s} does not round-trip to float32 {v!r}")
    return out


def tensor_rec(t: torch.Tensor) -> dict[str, Any]:
    a = t.detach().numpy()
    if a.dtype != np.float32:
        raise AssertionError(f"want float32, got {a.dtype}")
    return {"dtype": "float32", "shape": list(a.shape), "data": [f32(v) for v in a.reshape(-1)]}


def float64_layernorm(x: np.ndarray, w: np.ndarray | None) -> np.ndarray:
    """The exact result the float32 outputs approximate, for the summary only."""
    if w is None:
        return x.astype(np.float64)
    x = x.astype(np.float64)
    mu = x.mean(-1, keepdims=True)
    var = ((x - mu) ** 2).mean(-1, keepdims=True)
    return (x - mu) / np.sqrt(var + NORM_EPS) * w.astype(np.float64)


Input = Callable[[torch.Generator, int], torch.Tensor]


def randn(g: torch.Generator, *shape: int) -> torch.Tensor:
    return torch.randn(*shape, generator=g, dtype=torch.float32)


# (name, module, input). Inputs carry a non-zero mean throughout: a norm that skips the
# centring must not be able to pass. Shapes stay small; the last dimension is hidden_size.
NORM_CASES: list[tuple[str, str, Input]] = [
    # Per-row offsets, so each row has its own mean.
    ("embeddings_norm", "embeddings.norm", lambda g, h: randn(g, 2, h) + 2 * randn(g, 2, 1)),
    # +1000: float32 E[x^2] is ~1e6 with a ulp of 0.06, and the variance is ~1. A
    # float32 single-pass variance returns noise here; a two-pass one does not.
    ("attn_norm_offset", "layers.1.attn_norm", lambda g, h: randn(g, 2, h) + 1000),
    # Variance ~5e-6, half of eps itself: eps 1e-6 instead of 1e-5 moves these outputs by
    # half or more. The mean stays small, since a mean near 0.5 costs ~1e-5 of float32
    # precision at this scale. The last row is constant, which tests the variance-0 path
    # (output exactly 0, whatever eps is) rather than eps.
    (
        "mlp_norm_tiny_variance",
        "layers.1.mlp_norm",
        lambda g, h: torch.cat([3e-3 * randn(g, 2, h) + 1e-3, torch.full((1, h), 0.25)]),
    ),
    # Rank 3, as hidden states are [batch, seq, hidden].
    ("mlp_norm_rank3", "layers.0.mlp_norm", lambda g, h: 3 * randn(g, 2, 2, h) - 2),
    ("final_norm_wide", "final_norm", lambda g, h: 50 * randn(g, 2, h) + 7),
    # Layer 0 has no attn_norm: its output is its input.
    ("attn_norm_layer0_identity", "layers.0.attn_norm", lambda g, h: randn(g, 2, h) + 3),
]


def norm_case(model: ModernBertModel, name: str, path: str, make_input: Input) -> dict[str, Any]:
    hidden = model.config.hidden_size
    # A copy, so that the weights written below do not leak into later cases: a case that
    # runs a whole layer (Tasks 8.5-8.6) must not depend on which cases ran before it.
    mod = copy.deepcopy(model.get_submodule(path))
    seed = case_seed(name)
    g = torch.Generator().manual_seed(seed)

    weight = None
    if isinstance(mod, nn.LayerNorm):
        if mod.bias is not None or mod.eps != NORM_EPS or mod.normalized_shape != (hidden,):
            raise AssertionError(f"{path} is {mod}, want bias-free eps={NORM_EPS} over {hidden}")
        # Uniform in [0.5, 1.5]: far enough from ones that a dropped weight shows.
        weight = 0.5 + torch.rand(hidden, generator=g, dtype=torch.float32)
        with torch.no_grad():
            mod.weight.copy_(weight)
    elif not isinstance(mod, nn.Identity):
        raise AssertionError(f"{path} is {type(mod).__name__}, want LayerNorm or Identity")

    x = make_input(g, hidden)
    with torch.no_grad():
        y = mod(x)
    if weight is None and not torch.equal(x, y):
        raise AssertionError(f"{path}: Identity changed its input")

    return {
        "op": "layernorm",
        "name": name,
        "module": path,
        "module_class": type(mod).__name__,
        "seed": seed,
        "input": tensor_rec(x),
        "weight": None if weight is None else tensor_rec(weight),
        "output": tensor_rec(y),
    }


def build_model() -> ModernBertModel:
    torch.manual_seed(SEED)
    model = ModernBertModel(ModernBertConfig(**CONFIG)).eval()
    if not isinstance(model.layers[0].attn_norm, nn.Identity):
        raise AssertionError("layer 0's attn_norm is no longer nn.Identity in this transformers")
    return model


def header(model: ModernBertModel) -> dict[str, Any]:
    cfg = model.config.to_dict()
    return {
        "_comment": (
            "Generated by scripts/dump_modernbert_ops.py against the pinned reference "
            "environment. Do not hand-edit: regenerating it is a reviewed diff (PLAN.md R6)."
        ),
        "fixture": "ops",
        "versions": versions(),
        "seed": SEED,
        "torch_threads": torch.get_num_threads(),
        "config": {k: cfg[k] for k in CONFIG} | {"layer_types": cfg["layer_types"]},
    }


def write(path: Path, head: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """One case per line, so appending a case is a diff of added lines only."""
    body = ",\n".join("    " + json.dumps(c) for c in cases)
    text = f'{{\n  "header": {json.dumps(head)},\n  "cases": [\n{body}\n  ]\n}}\n'
    json.loads(text)  # the hand-built layout must still be one valid document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="fixture path")
    args = ap.parse_args()

    check_environment()
    torch.set_num_threads(1)
    model = build_model()
    cases = [norm_case(model, *c) for c in NORM_CASES]
    write(args.out, header(model), cases)

    print(json.dumps(versions()))
    for c in cases:
        x = np.array(c["input"]["data"], dtype=np.float32).reshape(c["input"]["shape"])
        w = None if c["weight"] is None else np.array(c["weight"]["data"], dtype=np.float32)
        y = np.array(c["output"]["data"], dtype=np.float32).reshape(c["output"]["shape"])
        diff = np.abs(y - float64_layernorm(x, w)).max()
        print(f"{c['name']:28} {c['module']:20} {str(c['input']['shape']):12} torch-f64 {diff:.2e}")
    print(f"wrote {len(cases)} cases, {args.out.stat().st_size} bytes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
