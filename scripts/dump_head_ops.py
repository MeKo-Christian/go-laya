#!/usr/bin/env python3
"""Dump the laya decision head from the pinned torch, for the native backend's tests.

``PLAN.md`` Task 8.6 reimplements in pure Go everything ``DecisionModel.forward`` does after
the encoder (``original/laya/common.py:108-126``): the type embedding, the manual loop over
the two ``nn.TransformerEncoderLayer`` (norm_first, ReLU), the marker gather, the scorer with
its ``-1e4`` fill, the act features and the act head. This script is its oracle. It imports
``DecisionModel`` from ``original/`` unchanged and runs it as ``agent.py`` does on the CPU:
``eval()``, ``torch.no_grad()``, float32, no autocast (``agent.py:196-215, 268-277``). It
writes ``internal/head/testdata/head.json``. CI reads that file; it never needs Python.

Random weights at the checkpoints' d=1024 would be megabytes, so the model is tiny:

* ``decision_model``: the whole head at d=8, so nhead = max(1, 8 // 64) = 1, with every
  parameter drawn at random (torch initialises the biases and norms to 0 and 1, which would
  hide a dropped bias). The encoder is a stub that only satisfies the constructor and returns
  the recorded ``h`` as ``last_hidden_state``. The batch is right-padded and has three rows:
  a ``choice`` with four markers, a ``noul`` with two and a ``score`` with one, so the last
  two carry marker fill, and the last has fewer than two markers, where the entropy's
  ``clamp(min=2)`` decides the denominator. Besides ``logits`` and ``act_logits`` the record
  holds the intermediates of invariants #35-38: h after the type embedding, after each layer,
  the gathered marker rows, the probabilities and the features. The layer outputs come from
  calling the model's own layers in the loop ``forward`` runs; the script checks that they
  reproduce what ``forward`` gathered, bit for bit.
* ``encoder_layer``: one ``nn.TransformerEncoderLayer(8, nhead=2, 32, batch_first=True,
  norm_first=True)`` with a key-padding mask, for the head split with nhead > 1.
* ``topk_error``: the forward pass with fewer than two marker columns. ``topk(2)`` raises
  there (``common.py:122``); the record holds the message.

Which ``TransformerEncoderLayer`` path ran is recorded, by counting calls to
``torch._transformer_encoder_layer_fwd``, the fused kernel the fast path calls
(``torch/nn/modules/transformer.py``, ``TransformerEncoderLayer.forward``). A forward hook on
the layers would itself disable the fast path, so the layers carry none. The fast path needs
an even head count: the checkpoints' 16 and 12 heads take it, the d=8 head's one head does
not ("num_head is odd"). The ``encoder_layer`` record therefore holds both: the output of the
fast path, which is what the checkpoints run, and of the slow path with the fast path
disabled.

Each case draws from its own generator, seeded from ``SEED`` and its name, so a rerun
reproduces the file byte for byte and adding a case leaves the others unchanged.

Format, one JSON object::

    {"header": {"versions": {...}, "seed": 86, "torch_threads": 1, ...},
     "cases": [{"op": "decision_model", "name": ..., "seed": ..., "d": 8, "nhead": 1,
                "head_layers": 2, "n_act": 2, "path": "slow", "fast_path_calls": 0,
                "batch": {"input_ids": [[...]], "attention_mask": [[...]],
                          "marker_pos": [[...]], "marker_mask": [[...]], "qtype": [...]},
                "marker_fill": 0, "h": T, "weights": {"type_emb.weight": T, ...},
                "h_typed": T, "h_layers": [T, T], "gathered": T, "logits": T,
                "probs": T, "feats": T, "act_logits": T,
                "fill_minus_one": {"logits_equal": true, "act_logits_equal": true}},
               {"op": "encoder_layer", "name": ..., "seed": ..., "d": 8, "nhead": 2,
                "dim_feedforward": 32, "path": "fast", "fast_path_calls": 1,
                "padding_mask": [[0|1]], "input": T, "weights": {...}, "output": T,
                "output_slow": T},
               {"op": "topk_error", "name": ..., "kmax": 1, "error": "RuntimeError: ..."}]}

where ``T`` is ``{"dtype": "float32", "shape": [...], "data": [...]}``, row-major, each value
the shortest decimal that round-trips to its float32. ``weights`` uses the state_dict names
of the module, ``DecisionModel``'s without the stub encoder and the ``temperature`` buffer
inference never reads: ``head.layers.N.*``, ``type_emb.weight``, ``scorer.*``,
``act_head.*``. ``padding_mask`` and ``attention_mask`` are 1 over real tokens, as the
collator writes them; torch gets ``~attention_mask``.

Usage::

    .venv-ref/bin/python -I scripts/dump_head_ops.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import transformers
from torch import nn

REPO = Path(__file__).resolve().parent.parent
REQUIREMENTS = REPO / "scripts" / "requirements-ref.txt"
DEFAULT_OUT = REPO / "internal" / "head" / "testdata" / "head.json"

# original/ is the frozen upstream: import it, never copy it, and leave no __pycache__ in it.
sys.dont_write_bytecode = True
sys.path.insert(0, str(REPO / "original"))
from laya.common import DecisionModel  # noqa: E402

# Any fixed value; each case derives its own seed from it (case_seed).
SEED = 86

# The versions that decide the numbers. The Go test asserts the same pins.
PINNED = ("torch", "transformers", "numpy")

# The whole head's size. d=8 gives nhead 1 (common.py:96); head_layers and n_act are
# build_model's defaults and the shipped checkpoints' values (common.py:136-137).
D = 8
HEAD_LAYERS = 2
N_ACT = 2

# The batch, in the collator's layout (common.collate_items, internal/prompt/collate.go):
# right-padded to the longest row, marker_pos zero-filled to the most markers any row has.
# Row 0 is a choice with four options, row 1 a noul (two), row 2 a score with one level,
# which build_sequence and the Go validation both accept. Markers never sit at 0, where
# [CLS] is, so a pooled vector read from the first marker cannot pass for h[:, 0].
ROWS = [
    {"qtype": 0, "length": 9, "markers": [2, 3, 5, 7]},
    {"qtype": 2, "length": 6, "markers": [2, 4]},
    {"qtype": 1, "length": 4, "markers": [2]},
]
SEQ = max(r["length"] for r in ROWS)
KMAX = max(len(r["markers"]) for r in ROWS)

# The single layer for the head split with more than one head: two heads keep the fast path
# eligible, as the checkpoints' 16 and 12 do.
LAYER_D, LAYER_NHEAD, LAYER_FF = 8, 2, 32
LAYER_LENGTHS = [7, 4]
LAYER_SEQ = 7


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
    a = t.detach().contiguous().numpy()
    if a.dtype != np.float32:
        raise AssertionError(f"want float32, got {a.dtype}")
    return {"dtype": "float32", "shape": list(a.shape), "data": [f32(v) for v in a.reshape(-1)]}


class FastPathCounter:
    """Counts calls to the fused kernel TransformerEncoderLayer's fast path runs.

    The layer looks ``torch._transformer_encoder_layer_fwd`` up at call time, so wrapping the
    attribute sees every fast-path call without touching the layer: no hook, which would
    itself make the layer take the slow path.
    """

    def __init__(self) -> None:
        self.calls = 0
        self._orig = torch._transformer_encoder_layer_fwd

    def __enter__(self) -> FastPathCounter:
        def counted(*args: Any, **kwargs: Any) -> Any:
            self.calls += 1
            return self._orig(*args, **kwargs)

        torch._transformer_encoder_layer_fwd = counted
        return self

    def __exit__(self, *exc: object) -> None:
        torch._transformer_encoder_layer_fwd = self._orig


class StubEncoder(nn.Module):
    """Satisfies DecisionModel's constructor and hands back the recorded hidden states."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size)
        self.h: torch.Tensor | None = None

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> SimpleNamespace:
        if self.h is None or self.h.shape[:2] != input_ids.shape:
            raise AssertionError("set StubEncoder.h to a [batch, seq, d] tensor first")
        return SimpleNamespace(last_hidden_state=self.h)


def randomize(module: nn.Module, g: torch.Generator) -> None:
    """Draw every parameter at random, in state_dict order.

    torch initialises biases and LayerNorm to 0 and 1, which would let a dropped bias or
    weight pass. Matrices get 1/sqrt(fan_in), so a layer keeps its input's spread; norm
    weights sit around 1, biases around 0, both with a spread that a dropped one shows.
    """
    with torch.no_grad():
        for name, p in module.named_parameters():
            if p.dim() == 2 and not name.startswith("type_emb"):
                p.copy_(torch.randn(p.shape, generator=g) / math.sqrt(p.shape[1]))
            elif "norm" in name or name.startswith("scorer.0"):
                if name.endswith("weight"):
                    p.copy_(1 + 0.5 * torch.randn(p.shape, generator=g))
                else:
                    p.copy_(0.5 * torch.randn(p.shape, generator=g))
            else:
                p.copy_(0.5 * torch.randn(p.shape, generator=g))


def weights_rec(module: nn.Module, skip: tuple[str, ...] = ()) -> dict[str, Any]:
    return {
        name: tensor_rec(t) for name, t in module.state_dict().items() if not name.startswith(skip)
    }


def path_of(calls: int, layers: int) -> str:
    if calls == 0:
        return "slow"
    if calls == layers:
        return "fast"
    raise AssertionError(f"{calls} fast-path calls for {layers} layers")


def build_batch(rows: list[dict[str, Any]], g: torch.Generator) -> dict[str, torch.Tensor]:
    ids = torch.zeros(len(rows), SEQ, dtype=torch.long)
    att = torch.zeros(len(rows), SEQ, dtype=torch.long)
    pos = torch.zeros(len(rows), KMAX, dtype=torch.long)
    mask = torch.zeros(len(rows), KMAX, dtype=torch.bool)
    for i, r in enumerate(rows):
        n = r["length"]
        ids[i, :n] = torch.randint(5, 100, (n,), generator=g)
        att[i, :n] = 1
        pos[i, : len(r["markers"])] = torch.tensor(r["markers"])
        mask[i, : len(r["markers"])] = True
    qtype = torch.tensor([r["qtype"] for r in rows], dtype=torch.long)
    return {
        "input_ids": ids,
        "attention_mask": att,
        "marker_pos": pos,
        "marker_mask": mask,
        "qtype": qtype,
    }


def run(model: DecisionModel, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    return model(b["input_ids"], b["attention_mask"], b["marker_pos"], b["marker_mask"], b["qtype"])


def decision_model_case(name: str) -> dict[str, Any]:
    seed = case_seed(name)
    g = torch.Generator().manual_seed(seed)
    enc = StubEncoder(D)
    model = DecisionModel(enc, HEAD_LAYERS, N_ACT).eval()
    randomize(model, g)
    first = model.head.layers[0]
    nhead = first.self_attn.num_heads
    if nhead != max(1, D // 64) or not first.norm_first:
        raise AssertionError(f"head has {nhead} heads, norm_first {first.norm_first}")

    b = build_batch(ROWS, g)
    # Padded positions get large values, so a key the mask fails to drop moves the output.
    h = torch.randn(len(ROWS), SEQ, D, generator=g)
    h = torch.where(b["attention_mask"][:, :, None].bool(), h, 5 * h + 3)
    enc.h = h

    # What forward gathers and what act_head reads, from hooks on modules outside the head
    # layers (a hook on a layer would push it off the fast path).
    seen: dict[str, torch.Tensor] = {}
    model.scorer[0].register_forward_pre_hook(lambda _m, a: seen.__setitem__("gathered", a[0]))
    model.act_head[0].register_forward_pre_hook(lambda _m, a: seen.__setitem__("act_in", a[0]))

    with torch.no_grad(), FastPathCounter() as fp:
        logits, act = run(model, b)
    path = path_of(fp.calls, HEAD_LAYERS)
    if path != "slow" or nhead % 2 == 0:
        raise AssertionError(f"d={D} with {nhead} head(s) took the {path} path")

    # The intermediates, through the model's own modules in forward's order (common.py:109-
    # 116). Calling the layers again outside forward takes the same path: the conditions
    # do not depend on the caller.
    with torch.no_grad(), FastPathCounter() as fp2:
        h_typed = h + model.type_emb(b["qtype"])[:, None, :]
        pad = ~b["attention_mask"].bool()
        h_layers = []
        x = h_typed
        for layer in model.head.layers:
            x = layer(x, src_key_padding_mask=pad)
            h_layers.append(x)
        idx = b["marker_pos"].clamp(min=0)[:, :, None].expand(-1, -1, D)
        gathered = torch.gather(x, 1, idx)
        probs = torch.softmax(logits, -1)
        feats = seen["act_in"][:, D:]
    if fp2.calls != fp.calls:
        raise AssertionError("the intermediates took another path than forward")
    if not torch.equal(gathered, seen["gathered"]):
        raise AssertionError("the manual layer loop does not reproduce forward's gather")
    if not torch.equal(seen["act_in"][:, :D], x[:, 0]):
        raise AssertionError("act_head's pooled input is not h[:, 0] after the head layers")

    # A fill of -1 instead of 0: clamp(min=0) reads row 0 either way, and the fill's logit is
    # overwritten by -1e4, so the outputs must not move at all.
    b_neg = dict(b)
    b_neg["marker_pos"] = torch.where(b["marker_mask"], b["marker_pos"], -1)
    with torch.no_grad():
        logits_neg, act_neg = run(model, b_neg)
    fill_neg = {
        "logits_equal": bool(torch.equal(logits, logits_neg)),
        "act_logits_equal": bool(torch.equal(act, act_neg)),
    }
    if not all(fill_neg.values()):
        raise AssertionError(f"a fill of -1 moved the outputs: {fill_neg}")

    return {
        "op": "decision_model",
        "name": name,
        "seed": seed,
        "d": D,
        "nhead": nhead,
        "head_layers": HEAD_LAYERS,
        "n_act": N_ACT,
        "path": path,
        "fast_path_calls": fp.calls,
        "batch": {
            "input_ids": b["input_ids"].tolist(),
            "attention_mask": b["attention_mask"].tolist(),
            "marker_pos": b["marker_pos"].tolist(),
            "marker_mask": b["marker_mask"].tolist(),
            "qtype": b["qtype"].tolist(),
        },
        "marker_fill": 0,
        "h": tensor_rec(h),
        "weights": weights_rec(model, skip=("encoder.", "temperature")),
        "h_typed": tensor_rec(h_typed),
        "h_layers": [tensor_rec(t) for t in h_layers],
        "gathered": tensor_rec(gathered),
        "logits": tensor_rec(logits),
        "probs": tensor_rec(probs),
        "feats": tensor_rec(feats),
        "act_logits": tensor_rec(act),
        "fill_minus_one": fill_neg,
    }


def encoder_layer_case(name: str) -> dict[str, Any]:
    seed = case_seed(name)
    g = torch.Generator().manual_seed(seed)
    layer = nn.TransformerEncoderLayer(
        LAYER_D, LAYER_NHEAD, LAYER_FF, batch_first=True, norm_first=True
    ).eval()
    randomize(layer, g)

    att = torch.tensor([[1] * n + [0] * (LAYER_SEQ - n) for n in LAYER_LENGTHS])
    x = 1.5 * torch.randn(len(LAYER_LENGTHS), LAYER_SEQ, LAYER_D, generator=g)
    x = torch.where(att[:, :, None].bool(), x, 5 * x + 3)
    pad = ~att.bool()

    with torch.no_grad(), FastPathCounter() as fp:
        y = layer(x, src_key_padding_mask=pad)
    path = path_of(fp.calls, 1)
    if path != "fast":
        raise AssertionError(f"a {LAYER_NHEAD}-head layer took the {path} path")

    torch.backends.mha.set_fastpath_enabled(False)
    try:
        with torch.no_grad(), FastPathCounter() as fp_off:
            y_slow = layer(x, src_key_padding_mask=pad)
    finally:
        torch.backends.mha.set_fastpath_enabled(True)
    if fp_off.calls != 0:
        raise AssertionError("the fast path ran while disabled")

    return {
        "op": "encoder_layer",
        "name": name,
        "seed": seed,
        "d": LAYER_D,
        "nhead": LAYER_NHEAD,
        "dim_feedforward": LAYER_FF,
        "path": path,
        "fast_path_calls": fp.calls,
        "padding_mask": att.tolist(),
        "input": tensor_rec(x),
        "weights": weights_rec(layer),
        "output": tensor_rec(y),
        "output_slow": tensor_rec(y_slow),
    }


def topk_error_case(name: str, kmax: int) -> dict[str, Any]:
    """What forward does with fewer than two marker columns: topk(2) has nothing to take."""
    g = torch.Generator().manual_seed(case_seed(name))
    enc = StubEncoder(D)
    model = DecisionModel(enc, HEAD_LAYERS, N_ACT).eval()
    rows = [{"qtype": 0, "length": 4, "markers": [2][:kmax]}]
    b = build_batch(rows, g)
    b["marker_pos"] = b["marker_pos"][:, :kmax]
    b["marker_mask"] = b["marker_mask"][:, :kmax]
    enc.h = torch.randn(1, SEQ, D, generator=g)
    try:
        with torch.no_grad():
            run(model, b)
    except RuntimeError as e:
        return {"op": "topk_error", "name": name, "kmax": kmax, "error": f"RuntimeError: {e}"}
    raise AssertionError(f"forward with kmax {kmax} did not raise")


def header() -> dict[str, Any]:
    return {
        "_comment": (
            "Generated by scripts/dump_head_ops.py against the pinned reference "
            "environment. Do not hand-edit: regenerating it is a reviewed diff (PLAN.md R6)."
        ),
        "fixture": "head",
        "versions": versions(),
        "seed": SEED,
        "torch_threads": torch.get_num_threads(),
        "upstream": "original/laya/common.py:88-126 (DecisionModel)",
        "mode": {"device": "cpu", "dtype": "float32", "eval": True, "no_grad": True},
    }


def write(path: Path, head: dict[str, Any], cases: list[dict[str, Any]]) -> None:
    """One case per line, so a diff shows which case moved."""
    # allow_nan=False: Go's encoding/json rejects NaN and Infinity, so refuse them here.
    body = ",\n".join("    " + json.dumps(c, allow_nan=False) for c in cases)
    text = f'{{\n  "header": {json.dumps(head, allow_nan=False)},\n  "cases": [\n{body}\n  ]\n}}\n'
    json.loads(text)  # the hand-built layout must still be one valid document
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def summarize(c: dict[str, Any]) -> None:
    if c["op"] == "decision_model":
        logits = np.array(c["logits"]["data"]).reshape(c["logits"]["shape"])
        feats = np.array(c["feats"]["data"]).reshape(c["feats"]["shape"])
        print(
            f"{c['name']:22} d={c['d']} nhead={c['nhead']} path={c['path']} "
            f"fill -1 equal={all(c['fill_minus_one'].values())}"
        )
        print(f"  logits {logits.round(4).tolist()}")
        print(f"  feats  {feats.round(4).tolist()}")
    elif c["op"] == "encoder_layer":
        y = np.array(c["output"]["data"])
        ys = np.array(c["output_slow"]["data"])
        print(
            f"{c['name']:22} d={c['d']} nhead={c['nhead']} path={c['path']} "
            f"|fast - slow| max {np.abs(y - ys).max():.2e}"
        )
    else:
        print(f"{c['name']:22} kmax={c['kmax']} {c['error']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="fixture path")
    args = ap.parse_args()

    check_environment()
    torch.set_num_threads(1)
    if not torch.backends.mha.get_fastpath_enabled():
        raise AssertionError("the MHA fast path is disabled in this process")
    cases = [decision_model_case("head_d8_choice_noul_score")]
    cases.append(encoder_layer_case("layer_d8_nhead2_padded"))
    cases += [topk_error_case(f"topk_kmax{k}", k) for k in (1, 0)]
    write(args.out, header(), cases)

    print(json.dumps(versions()))
    for c in cases:
        summarize(c)
    print(f"wrote {len(cases)} cases, {args.out.stat().st_size} bytes to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
