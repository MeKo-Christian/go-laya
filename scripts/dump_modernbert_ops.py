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

Task 8.3 appends the MLPs: ``layers.N.mlp``, the real ``ModernBertMLP`` (GeGLU over a fused
``Wi``, no bias), with seeded random ``Wi`` and ``Wo``. Most inputs put the gate
preactivations in the band where the exact erf GELU and its tanh approximation differ most
(|u| ~ 2.7, up to 4.7e-4); one is large and mostly negative, for GELU's tails. The summary
prints how far each wrong reading of the module -- tanh GELU, ``erf(x)`` for
``erf(x/sqrt 2)``, the halves swapped, ``Wo`` read as ``[I, H]`` -- moves the output, so a
case that cannot tell them apart shows.

Tasks 8.4 and 8.5 append the attention: ``layers.N.attn``, the real ``ModernBertAttention``
(fused ``Wqkv``, no bias) with seeded random ``Wqkv`` and ``Wo``, fed the cos/sin of the real
``ModernBertRotaryEmbedding`` and the mask that ``create_bidirectional_mask`` or
``create_bidirectional_sliding_window_mask`` builds, exactly as ``ModernBertModel.forward``
does for sdpa. They run on their own model (ATTN_CONFIG), because the window and the per-type
theta have to be small enough to bite inside 12 tokens; the header describes the base model
and stays as it was. Each record carries what decides it -- layer type, theta, window, the
padding mask and the mask transformers built -- and the cos/sin it ran with. Two "rope"
records hold the real ``ModernBertRotaryEmbedding`` at the checkpoints' head_dim 64 and
thetas, out to position 8191, where a one-ulp slip in ``inv_freq`` would have grown into the
angle.

The file is meant to grow. Each case draws from its own generator,
seeded from ``SEED`` and its name, so adding, removing or reordering cases leaves every other
record byte-identical. The header holds nothing that depends on which cases exist.

Format, one JSON object::

    {"header": {"versions": {...}, "seed": 82, "config": {...}, ...},
     "cases": [{"op": "layernorm", "name": ..., "module": ..., "module_class": ...,
                "seed": ..., "input": T, "weight": T | null, "output": T},
               {"op": "mlp", "name": ..., "module": ..., "module_class": "ModernBertMLP",
                "seed": ..., "hidden_activation": "gelu", "mlp_bias": false,
                "input": T, "wi": T, "wo": T, "output": T},
               {"op": "attention", "name": ..., "module": ..., "module_class":
                "ModernBertAttention", "seed": ..., "layer_type": ..., "config": {...},
                "padding_mask": [[0|1]], "mask": [[[0|1]]], "input": T, "wqkv": T, "wo": T,
                "cos": T, "sin": T, "output": T},
               {"op": "rope", "name": ..., "module_class": "ModernBertRotaryEmbedding",
                "layer_type": ..., "rope_type": "default", "rope_theta": ..., "head_dim": 64,
                "attention_scaling": 1.0, "positions": [...], "inv_freq": T, "cos": T,
                "sin": T}, ...]}

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
import math
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import transformers
from torch import nn
from transformers import ModernBertConfig, ModernBertModel
from transformers.activations import GELUActivation
from transformers.masking_utils import (
    create_bidirectional_mask,
    create_bidirectional_sliding_window_mask,
)
from transformers.models.modernbert.modeling_modernbert import (
    ModernBertAttention,
    ModernBertMLP,
    ModernBertRotaryEmbedding,
)

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

# Both checkpoints' encoder config.json say hidden_activation "gelu" and mlp_bias false. They
# are the ModernBertConfig defaults, so CONFIG leaves them out (and the header unchanged);
# mlp_case asserts them instead.
HIDDEN_ACTIVATION = "gelu"
MLP_BIAS = False

# The attention cases' model: CONFIG with a window and per-type thetas that bite inside 12
# tokens. local_attention 6 is config.sliding_window 3, the stand-in for the checkpoints'
# 128 and 64. The thetas differ by four orders of magnitude so that a swapped theta shows:
# with head_dim 4 only the second frequency, theta^-1/2, depends on it (0.0025 against
# 0.32). 160000 is the checkpoints' full-attention theta. "sdpa" is what the reference runs
# (docs/ARCHITECTURE.md 1.2), and with it ModernBertAttention.sliding_window (window + 1)
# is never read: the mask alone decides the window.
ATTN_CONFIG = CONFIG | {
    "local_attention": 6,
    "rope_parameters": {
        "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10.0},
    },
    "attn_implementation": "sdpa",
}

# The rope cases' config: ModernBERT-large's head_dim (1024 / 16 = 64, as mmBERT-base's 768 /
# 12) and its rope_parameters (models/laya/encoder/config.json). mmBERT-base has 160000 on
# both types, which the full-attention case covers.
ROPE_CONFIG = CONFIG | {
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "rope_parameters": {
        "full_attention": {"rope_type": "default", "rope_theta": 160000.0},
        "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
    },
}
# Short positions and the longest max_position_embeddings allows: an error in inv_freq grows
# with the position, so 8191 shows what 12 tokens cannot.
ROPE_POSITIONS = [0, 1, 511, 1023, 8191]
ROPE_CASES = [("rope_full_d64", "full_attention"), ("rope_sliding_d64", "sliding_attention")]

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


# The one norm upstream leaves out (modeling_modernbert.py:309-310).
IDENTITY_PATH = "layers.0.attn_norm"


# (name, module, input), the last dimension hidden_size. With Wi drawn at 1/sqrt(hidden), a
# preactivation has the spread of one input element.
MLP_CASES: list[tuple[str, str, Input]] = [
    # Preactivations ~N(0, 2.5^2): half fall in 1 < |u| < 4, where tanh GELU is off by 1e-4
    # and more.
    ("mlp_layer0", "layers.0.mlp", lambda g, h: 2.5 * randn(g, 3, h)),
    # Large inputs, mostly negative: preactivations ~N(0, 11^2), deep in GELU's tails (this
    # draw puts 4 of the 12 activated ones below -5), where an activation that is not ~0
    # there shows. Wi has zero mean, so the input's offset does not shift them.
    ("mlp_large_negative", "layers.1.mlp", lambda g, h: 10 * randn(g, 2, h) - 5),
    # Rank 3, as hidden states are [batch, seq, hidden].
    ("mlp_rank3", "layers.1.mlp", lambda g, h: 2 * randn(g, 2, 2, h) + 1),
]


# (name, module, real tokens per row). Every input is [2, 12, hidden]: row 0 is unpadded, row
# 1 has 4 real tokens and 8 of right padding, more than the window of 3. On the sliding layer
# its queries 7-11 have no key left (each real key is 4 or more away); sdpa returns zeros
# there. Row 0 holds query/key pairs at distance 3, the last the mask allows, and at 4, the
# first it does not. Inputs at 1.5 sigma put the scaled scores at a spread of ~2, where the
# softmax is neither flat nor one-hot, so the scale and the mask both move the output.
ATTN_CASES: list[tuple[str, str, list[int]]] = [
    ("attn_full_padded", "layers.0.attn", [12, 4]),
    ("attn_sliding_padded", "layers.1.attn", [12, 4]),
]
ATTN_SEQ = 12


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
    elif isinstance(mod, nn.Identity):
        # Only layer 0 lacks its attn_norm upstream. Anywhere else an Identity is a drift
        # that removed a norm, and recording it would make a null weight the oracle.
        if path != IDENTITY_PATH:
            raise AssertionError(f"{path} is Identity; only {IDENTITY_PATH} may be")
    else:
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


def mlp_case(model: ModernBertModel, name: str, path: str, make_input: Input) -> dict[str, Any]:
    cfg = model.config
    hidden, inter = cfg.hidden_size, cfg.intermediate_size
    if cfg.hidden_activation != HIDDEN_ACTIVATION or cfg.mlp_bias != MLP_BIAS:
        raise AssertionError(
            f"config has hidden_activation {cfg.hidden_activation!r}, mlp_bias {cfg.mlp_bias}; "
            f"the checkpoints have {HIDDEN_ACTIVATION!r}, {MLP_BIAS}"
        )
    mod = copy.deepcopy(model.get_submodule(path))  # as in norm_case
    if not isinstance(mod, ModernBertMLP):
        raise AssertionError(f"{path} is {type(mod).__name__}, want ModernBertMLP")
    if mod.Wi.bias is not None or mod.Wo.bias is not None:
        raise AssertionError(f"{path} has an MLP bias")
    # "gelu" is GELUActivation calling nn.functional.gelu, whose default is the exact erf
    # form; the Go side reimplements exactly that.
    if not isinstance(mod.act, GELUActivation) or mod.act.act is not nn.functional.gelu:
        raise AssertionError(f"{path}.act is {mod.act}, want GELUActivation(nn.functional.gelu)")
    if mod.Wi.weight.shape != (2 * inter, hidden) or mod.Wo.weight.shape != (hidden, inter):
        raise AssertionError(f"{path}: Wi {mod.Wi.weight.shape}, Wo {mod.Wo.weight.shape}")

    seed = case_seed(name)
    g = torch.Generator().manual_seed(seed)
    wi = randn(g, 2 * inter, hidden) / math.sqrt(hidden)
    wo = randn(g, hidden, inter) / math.sqrt(inter)
    with torch.no_grad():
        mod.Wi.weight.copy_(wi)
        mod.Wo.weight.copy_(wo)

    x = make_input(g, hidden)
    with torch.no_grad():
        y = mod(x)

    return {
        "op": "mlp",
        "name": name,
        "module": path,
        "module_class": type(mod).__name__,
        "seed": seed,
        "hidden_activation": cfg.hidden_activation,
        "mlp_bias": cfg.mlp_bias,
        "input": tensor_rec(x),
        "wi": tensor_rec(wi),
        "wo": tensor_rec(wo),
        "output": tensor_rec(y),
    }


def mask_bits(mask: torch.Tensor | None, batch: int, seq: int) -> list[list[list[int]]]:
    """A transformers sdpa mask as [batch][query][key] 0/1, None meaning all allowed."""
    if mask is None:
        return [[[1] * seq for _ in range(seq)] for _ in range(batch)]
    if mask.dtype != torch.bool or mask.shape != (batch, 1, seq, seq):
        want = f"bool [{batch}, 1, {seq}, {seq}]"
        raise AssertionError(f"want a {want} mask, got {mask.dtype} {list(mask.shape)}")
    return mask[:, 0].int().tolist()


def attention_case(
    model: ModernBertModel, name: str, path: str, lengths: list[int]
) -> dict[str, Any]:
    cfg = model.config
    hidden, heads = cfg.hidden_size, cfg.num_attention_heads
    impl = cfg._attn_implementation
    if impl != "sdpa" or cfg.attention_bias:
        raise AssertionError(f"attention is {impl!r} with bias {cfg.attention_bias}, want sdpa")
    layer = model.get_submodule(path.removesuffix(".attn"))
    layer_type = layer.attention_type
    mod = copy.deepcopy(layer.attn)  # as in norm_case
    if not isinstance(mod, ModernBertAttention):
        raise AssertionError(f"{path} is {type(mod).__name__}, want ModernBertAttention")
    if mod.Wqkv.bias is not None or mod.Wo.bias is not None:
        raise AssertionError(f"{path} has an attention bias")
    if mod.head_dim != hidden // heads:
        raise AssertionError(f"{path}: head_dim {mod.head_dim} for hidden {hidden}, {heads} heads")
    if mod.Wqkv.weight.shape != (3 * hidden, hidden) or mod.Wo.weight.shape != (hidden, hidden):
        raise AssertionError(f"{path}: Wqkv {mod.Wqkv.weight.shape}, Wo {mod.Wo.weight.shape}")
    rope = cfg.rope_parameters[layer_type]
    if rope["rope_type"] != "default":
        raise AssertionError(f"{layer_type} rope_type {rope['rope_type']!r}, want 'default'")
    if getattr(model.rotary_emb, f"{layer_type}_attention_scaling") != 1.0:
        raise AssertionError(f"{layer_type}: default RoPE with an attention scaling")

    seed = case_seed(name)
    g = torch.Generator().manual_seed(seed)
    wqkv = randn(g, 3 * hidden, hidden) / math.sqrt(hidden)
    wo = randn(g, hidden, hidden) / math.sqrt(hidden)
    with torch.no_grad():
        mod.Wqkv.weight.copy_(wqkv)
        mod.Wo.weight.copy_(wo)

    batch, seq = len(lengths), ATTN_SEQ
    x = 1.5 * randn(g, batch, seq, hidden)
    padding = torch.tensor([[1] * n + [0] * (seq - n) for n in lengths])

    # ModernBertModel.forward, for one layer: both masks from the 2-D padding mask, the one
    # for this layer's type, and the cos/sin of its theta over positions 0..seq-1.
    mask_kwargs = {"config": cfg, "inputs_embeds": x, "attention_mask": padding}
    masks = {
        "full_attention": create_bidirectional_mask(**mask_kwargs),
        "sliding_attention": create_bidirectional_sliding_window_mask(**mask_kwargs),
    }
    position_ids = torch.arange(seq).unsqueeze(0)
    with torch.no_grad():
        cos, sin = model.rotary_emb(x, position_ids, layer_type)
        y, _ = mod(x, position_embeddings=(cos, sin), attention_mask=masks[layer_type])

    bits = mask_bits(masks[layer_type], batch, seq)
    # A query with no key left is what this case is for on the sliding layer. If torch ever
    # returns NaN or an average there instead of zeros, the Go side's contract moved: stop.
    if torch.isnan(y).any():
        raise AssertionError(f"{path}: NaN in the output")
    for b, rows in enumerate(bits):
        for q, row in enumerate(rows):
            if not any(row) and torch.count_nonzero(y[b, q]):
                raise AssertionError(f"{path}: query {b},{q} has no key but output {y[b, q]}")

    return {
        "op": "attention",
        "name": name,
        "module": path,
        "module_class": type(mod).__name__,
        "seed": seed,
        "layer_type": layer_type,
        "config": {
            "hidden_size": hidden,
            "num_attention_heads": heads,
            "head_dim": mod.head_dim,
            "attn_implementation": impl,
            "attention_bias": cfg.attention_bias,
            "rope_type": rope["rope_type"],
            "rope_theta": rope["rope_theta"],
            "local_attention": cfg.local_attention,
            "sliding_window": cfg.sliding_window,
            # What ModernBertAttention stores and passes on (window + 1, or None). sdpa
            # ignores it; it is recorded so that the difference stays visible.
            "module_sliding_window": mod.sliding_window,
            "scaling": mod.head_dim**-0.5,
        },
        "padding_mask": padding.tolist(),
        "mask": bits,
        "input": tensor_rec(x),
        "wqkv": tensor_rec(wqkv),
        "wo": tensor_rec(wo),
        "cos": tensor_rec(cos[0]),
        "sin": tensor_rec(sin[0]),
        "output": tensor_rec(y),
    }


def rope_case(name: str, layer_type: str) -> dict[str, Any]:
    cfg = ModernBertConfig(**ROPE_CONFIG)
    rot = ModernBertRotaryEmbedding(cfg)
    rope = cfg.rope_parameters[layer_type]
    scaling = getattr(rot, f"{layer_type}_attention_scaling")
    if rope["rope_type"] != "default" or scaling != 1.0:
        raise AssertionError(f"{layer_type}: rope_type {rope['rope_type']!r}, scaling {scaling}")
    inv_freq = getattr(rot, f"{layer_type}_inv_freq")
    position_ids = torch.tensor([ROPE_POSITIONS])
    cos, sin = rot(torch.zeros(1), position_ids, layer_type)
    return {
        "op": "rope",
        "name": name,
        "module_class": type(rot).__name__,
        "layer_type": layer_type,
        "rope_type": rope["rope_type"],
        "rope_theta": rope["rope_theta"],
        "head_dim": cfg.hidden_size // cfg.num_attention_heads,
        "attention_scaling": scaling,
        "positions": ROPE_POSITIONS,
        "inv_freq": tensor_rec(inv_freq),
        "cos": tensor_rec(cos[0]),
        "sin": tensor_rec(sin[0]),
    }


def summarize_rope(c: dict[str, Any]) -> None:
    d = c["head_dim"]
    inv = c["rope_theta"] ** -(np.arange(0, d, 2) / d)
    ang = np.array(c["positions"])[:, None] * np.concatenate([inv, inv])[None, :]
    diff = max(
        np.abs(as_array(c["cos"]) - np.cos(ang)).max(),
        np.abs(as_array(c["sin"]) - np.sin(ang)).max(),
    )
    where = f"theta {c['rope_theta']:g}, positions {c['positions']}"
    print(f"{c['name']:28} {where:41}  torch-f64 {diff:.2e}")


def float64_mlp(
    x: np.ndarray,
    wi: np.ndarray,
    wo: np.ndarray,
    act: Callable[[np.ndarray], np.ndarray],
    *,
    swap: bool = False,
    wo_transposed: bool = False,
) -> np.ndarray:
    """ModernBertMLP in float64, or one of the wrong readings of it, for the summary only."""
    x, wi, wo = x.astype(np.float64), wi.astype(np.float64), wo.astype(np.float64)
    inp, gate = np.split(x @ wi.T, 2, axis=-1)
    if swap:
        inp, gate = gate, inp
    h = act(inp) * gate
    return h @ wo.reshape(wo.shape[1], wo.shape[0]) if wo_transposed else h @ wo.T


_erf = np.vectorize(math.erf)


def gelu_erf(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * (1 + _erf(u / math.sqrt(2)))


def gelu_tanh(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * (1 + np.tanh(math.sqrt(2 / math.pi) * (u + 0.044715 * u**3)))


def gelu_erf_unscaled(u: np.ndarray) -> np.ndarray:
    return 0.5 * u * (1 + _erf(u))


def as_array(rec: dict[str, Any]) -> np.ndarray:
    return np.array(rec["data"], dtype=np.float32).reshape(rec["shape"])


def summarize_norm(c: dict[str, Any]) -> None:
    x = as_array(c["input"])
    w = None if c["weight"] is None else as_array(c["weight"])
    diff = np.abs(as_array(c["output"]) - float64_layernorm(x, w)).max()
    print(f"{c['name']:28} {c['module']:20} {str(c['input']['shape']):12} torch-f64 {diff:.2e}")


def summarize_mlp(c: dict[str, Any]) -> None:
    x, wi, wo, y = (as_array(c[k]) for k in ("input", "wi", "wo", "output"))
    exact = float64_mlp(x, wi, wo, gelu_erf)
    wrong = {
        "tanh": float64_mlp(x, wi, wo, gelu_tanh),
        "erf(x)": float64_mlp(x, wi, wo, gelu_erf_unscaled),
        "swap": float64_mlp(x, wi, wo, gelu_erf, swap=True),
        "woT": float64_mlp(x, wi, wo, gelu_erf, wo_transposed=True),
    }
    moved = " ".join(f"{k} {np.abs(v - exact).max():.1e}" for k, v in wrong.items())
    diff = np.abs(y - exact).max()
    print(
        f"{c['name']:28} {c['module']:20} {str(c['input']['shape']):12} torch-f64 {diff:.2e}"
        f"  moved by: {moved}"
    )


def float64_attention(c: dict[str, Any], variant: str = "") -> np.ndarray:
    """ModernBertAttention in float64, or one wrong reading of it, for the summary only."""
    x, wqkv, wo = (as_array(c[k]).astype(np.float64) for k in ("input", "wqkv", "wo"))
    cfg = c["config"]
    batch, seq, hidden = x.shape
    heads, d = cfg["num_attention_heads"], cfg["head_dim"]
    theta = cfg["rope_theta"]
    if variant == "theta":  # the other layer type's theta
        theta = ATTN_CONFIG["rope_parameters"]["full_attention"]["rope_theta"]
        if c["layer_type"] == "full_attention":
            theta = ATTN_CONFIG["rope_parameters"]["sliding_attention"]["rope_theta"]
    window = cfg["sliding_window"] if c["layer_type"] == "sliding_attention" else None
    if window is not None and variant in ("window+1", "window-1"):
        window += 1 if variant == "window+1" else -1

    qkv = x @ wqkv.T
    order = (heads, 3, d) if variant == "split" else (3, heads, d)
    qkv = qkv.reshape(batch, seq, *order)
    if variant == "split":
        qkv = qkv.transpose(0, 1, 3, 2, 4)
    q, k, v = (qkv[:, :, i].transpose(0, 2, 1, 3) for i in range(3))  # [B, heads, S, d]
    if variant == "swap":
        q, k = k, q

    inv = 1.0 / theta ** (np.arange(0, d, 2) / d)
    ang = np.arange(seq)[:, None] * inv[None, :]
    if variant == "interleaved":
        cos, sin = np.repeat(np.cos(ang), 2, -1), np.repeat(np.sin(ang), 2, -1)

        def rot(t: np.ndarray) -> np.ndarray:
            return np.stack([-t[..., 1::2], t[..., ::2]], -1).reshape(t.shape)
    else:
        cos, sin = np.cos(np.concatenate([ang, ang], -1)), np.sin(np.concatenate([ang, ang], -1))

        def rot(t: np.ndarray) -> np.ndarray:
            return np.concatenate([-t[..., d // 2 :], t[..., : d // 2]], -1)

    q, k = q * cos + rot(q) * sin, k * cos + rot(k) * sin
    scale = 1 / d if variant == "scale" else d**-0.5
    scores = q @ k.transpose(0, 1, 3, 2) * scale
    pad = np.array(c["padding_mask"], dtype=bool)
    if variant == "nopad":
        pad[:] = True
    allowed = np.broadcast_to(pad[:, None, None, :], scores.shape).copy()
    if window is not None:
        dist = np.abs(np.arange(seq)[:, None] - np.arange(seq)[None, :])
        allowed &= dist <= window
    scores = np.where(allowed, scores, -np.inf)
    top = scores.max(-1, keepdims=True)
    e = np.where(allowed, np.exp(scores - np.where(np.isfinite(top), top, 0)), 0)
    total = e.sum(-1, keepdims=True)
    p = np.divide(e, total, out=np.zeros_like(e), where=total > 0)  # sdpa: no key, zeros
    out = (p @ v).transpose(0, 2, 1, 3).reshape(batch, seq, hidden)
    return out @ wo.T


ATTN_VARIANTS = ("swap", "split", "interleaved", "theta", "window+1", "window-1", "nopad", "scale")


def summarize_attention(c: dict[str, Any]) -> None:
    exact = float64_attention(c)
    want = np.array(c["mask"], dtype=bool)
    if c["layer_type"] == "sliding_attention":
        seq = want.shape[-1]
        dist = np.abs(np.arange(seq)[:, None] - np.arange(seq)[None, :])
        w = c["config"]["sliding_window"]
        # The window the sdpa mask applies, read off it: distance w allowed, w + 1 not.
        last = int(dist[want[0]].max())
        print(
            f"  mask's last allowed distance {last} (config.sliding_window {w}, "
            f"module sliding_window {c['config']['module_sliding_window']})"
        )
    moved = " ".join(
        f"{v} {np.abs(float64_attention(c, v) - exact).max():.1e}" for v in ATTN_VARIANTS
    )
    diff = np.abs(as_array(c["output"]) - exact).max()
    print(
        f"{c['name']:28} {c['module']:20} {str(c['input']['shape']):12} torch-f64 {diff:.2e}"
        f"  moved by: {moved}"
    )


def build_model(config: dict[str, Any] = CONFIG) -> ModernBertModel:
    torch.manual_seed(SEED)
    model = ModernBertModel(ModernBertConfig(**config)).eval()
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
    """One case per line, so appending a case adds lines and a comma to the last old one."""
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
    cases += [mlp_case(model, *c) for c in MLP_CASES]
    attn_model = build_model(ATTN_CONFIG)
    cases += [attention_case(attn_model, *c) for c in ATTN_CASES]
    cases += [rope_case(*c) for c in ROPE_CASES]
    write(args.out, header(model), cases)

    print(json.dumps(versions()))
    summarize = {
        "layernorm": summarize_norm,
        "mlp": summarize_mlp,
        "attention": summarize_attention,
        "rope": summarize_rope,
    }
    for c in cases:
        summarize[c["op"]](c)
    print(f"wrote {len(cases)} cases, {args.out.stat().st_size} bytes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
