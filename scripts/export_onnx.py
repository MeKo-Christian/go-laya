#!/usr/bin/env python3
"""Export the whole laya ``DecisionModel`` graph (encoder + decision head) to ONNX.

This is Spike S1 from ``PLAN.md``. Decision D1 rests on the answer: the entire
``DecisionModel.forward`` is one static graph -- no KV cache, no loop over time, no
control flow on tensor values -- so it should export to a *single* ONNX file, which
leaves exactly one numerics-parity surface for the Go port to test.

Working recipe (S1.6) -- these versions are the contract, see scripts/requirements-ref.txt:

    python 3.12.4 - torch 2.14.0+cpu - transformers 5.17.0 - onnxscript 0.7.2
    onnx 1.23.0 - onnxruntime 1.30.0 - numpy 2.5.3 - safetensors 0.8.0
    dynamo=True, opset 18, torch.backends.mha.set_fastpath_enabled(False)
    checkpoints: convaiinnovations/laya @ 1c5edc17a7acd8701df6fc341c0d179f1c62c982

    .venv-ref/bin/python scripts/export_onnx.py --all --dynamo

``--dynamo`` is load-bearing. The legacy TorchScript exporter also produces a file, and
that file runs correctly -- but only at the sequence length it was traced with, because
transformers' mask builder resolves a Python ``bool`` on ``attention_mask.shape[-1]``
(``masking_utils.py:213``) and bakes the length in. It warns and carries on; ONNX Runtime
then rejects every other length in a ``Reshape``. Hence four validation shapes, not one.
Asking the dynamo exporter for opset 17 also yields a graph ORT refuses to load: the
18->17 downconversion leaves ``Split`` holding its opset-18 ``num_outputs`` attribute.

Two notes on how this differs from ``laya.common.build_model``:

* ``build_model`` hardcodes ``attn_implementation="sdpa"``. Exporting wants "eager",
  so the encoder is constructed here rather than through ``build_model``. ``--check-attn``
  then proves eager and sdpa agree, otherwise we would be exporting a different model
  than the one parity is later measured against.
* ``reference_compile`` was ModernBERT's ``torch.compile`` switch and is what
  huggingface/transformers#35545 (PLAN.md R2) tripped over. transformers 5.x dropped it:
  there is no ``torch.compile`` left in ``modeling_modernbert.py`` and the config key is
  only popped for backwards compatibility. It is still set here when the attribute exists,
  so the script keeps working against a 4.x reference environment.

The blocker that actually bites is neither of those, and it is not in the encoder at all.
``nn.TransformerEncoderLayer.forward`` dispatches to the fused kernel
``aten::_transformer_encoder_layer_fwd`` in eval mode, and that op has no ONNX symbolic:

    UnsupportedOperatorError: Exporting the operator
    'aten::_transformer_encoder_layer_fwd' to ONNX opset version 17 is not supported

It is the *decision head* that trips, not ModernBERT -- the encoder traces all the way
through. ``norm_first=True`` does not disable the fused path (the fast path takes
``norm_first`` as an argument); the supported switch is
``torch.backends.mha.set_fastpath_enabled(False)``, which is the first condition
``why_not_sparsity_fast_path`` tests. That routes the layer through PyTorch's own
reference implementation -- the ``_sa_block``/``_ff_block`` composition -- rather than a
reimplementation of ours, and ``--check-attn`` measures the two against each other.

Usage::

    .venv-ref/bin/python scripts/export_onnx.py --all
    .venv-ref/bin/python scripts/export_onnx.py --checkpoint english --out build/onnx
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
# The frozen upstream Python is the parity reference; import DecisionModel from it
# rather than restating the architecture here, so the two cannot drift.
sys.path.insert(0, str(REPO_ROOT / "original"))

# PLAN.md S1.2 asks for opset >= 17. The TorchScript exporter emits 17; the dynamo
# exporter has implementations from 18 and its 18->17 downconversion produces an
# invalid graph (Split keeps the opset-18 `num_outputs` attribute), so it gets 18.
OPSET_TORCHSCRIPT = 17
OPSET_DYNAMO = 18


@dataclass(frozen=True)
class Checkpoint:
    """One of the three checkpoints bundled in ``convaiinnovations/laya``."""

    name: str
    subfolder: str  # "" for the English checkpoint, which lives at the repo root
    encoder: str

    def dir(self, models_root: Path) -> Path:
        base = models_root / "laya"
        return base / self.subfolder if self.subfolder else base


CHECKPOINTS = (
    Checkpoint("english", "", "answerdotai/ModernBERT-large"),
    Checkpoint("multilingual", "multilingual", "jhu-clsp/mmBERT-base"),
    Checkpoint("typed-decisions", "typed-decisions", "answerdotai/ModernBERT-large"),
)


class DecisionModelWrapper(nn.Module):
    """Pin ``detach_encoder=False`` so the exported graph has exactly five inputs.

    ``DecisionModel.forward``'s last parameter is a Python bool, not a tensor
    (``original/laya/common.py:105``). Tracing it as an input would be wrong; baking it
    as a constant is what the trace does anyway.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        return self.model(input_ids, attention_mask, marker_pos, marker_mask, qtype, False)


def build_decision_model(ckpt_dir: Path, attn: str = "eager") -> nn.Module:
    """Replicate ``laya.common.build_model`` with an export-friendly attention impl.

    Everything is resolved from ``ckpt_dir``: no ``snapshot_download``, no hub fallback
    through ``cfg["encoder"]``. The state dict is loaded with ``strict=True``, which is
    the structural proof that this reference environment builds the architecture the
    weights were trained with.
    """
    from safetensors.torch import load_file
    from transformers import AutoConfig, AutoModel

    from laya.common import DecisionModel

    with (ckpt_dir / "rl_agent_config.json").open() as f:
        cfg = json.load(f)

    encoder_dir = ckpt_dir / "encoder"
    if not encoder_dir.is_dir():
        raise FileNotFoundError(
            f"{encoder_dir} is missing; the exporter refuses to fall back to the hub "
            f"because that would silently export a different encoder config."
        )

    ecfg = AutoConfig.from_pretrained(encoder_dir)
    # Pre-5.x transformers torch.compile ModernBERT unless this is off (PLAN.md R2).
    if hasattr(ecfg, "reference_compile"):
        ecfg.reference_compile = False
    enc = AutoModel.from_config(ecfg, attn_implementation=attn)

    model = DecisionModel(enc, cfg.get("head_layers", 2), len(cfg.get("act_costs", {})) + 1)
    model.load_state_dict(load_file(ckpt_dir / "model.safetensors"), strict=True)
    # Stored weights are fp16; CPU kernels largely are not, and the export must be fp32.
    model.float().eval()
    return model


def example_inputs(model: nn.Module, seq: int = 48, k: int = 4, batch: int = 2) -> tuple:
    """A batch that exercises every dynamic axis and every masked path.

    ``k >= 2`` is not optional: ``forward`` calls ``p.topk(2, -1)``
    (``original/laya/common.py:122``), which fails to trace with fewer than two options.
    The last row leaves a padded tail and a masked-out marker so the ``masked_fill`` and
    the ``attention_mask`` branches are actually covered rather than traced as no-ops.
    """
    vocab = model.encoder.config.vocab_size
    g = torch.Generator().manual_seed(0)
    input_ids = torch.randint(0, vocab, (batch, seq), generator=g, dtype=torch.long)
    attention_mask = torch.ones((batch, seq), dtype=torch.long)
    attention_mask[-1, seq // 2 :] = 0
    marker_pos = torch.zeros((batch, k), dtype=torch.long)
    for b in range(batch):
        for i in range(k):
            marker_pos[b, i] = 1 + i * 3
    marker_mask = torch.ones((batch, k), dtype=torch.bool)
    marker_mask[-1, -1] = False
    qtype = torch.tensor([i % 3 for i in range(batch)], dtype=torch.long)
    return input_ids, attention_mask, marker_pos, marker_mask, qtype


class _no_mha_fastpath:
    """Route ``nn.TransformerEncoderLayer`` through its reference implementation.

    The fused ``aten::_transformer_encoder_layer_fwd`` has no ONNX symbolic. This is the
    supported way off it: ``torch.backends.mha.get_fastpath_enabled()`` is the first thing
    ``why_not_sparsity_fast_path`` checks (``torch/nn/modules/transformer.py:842``).
    """

    def __enter__(self) -> None:
        self._prev = torch.backends.mha.get_fastpath_enabled()
        torch.backends.mha.set_fastpath_enabled(False)

    def __exit__(self, *exc: object) -> None:
        torch.backends.mha.set_fastpath_enabled(self._prev)


def check_fastpath_equivalence(model: nn.Module, inputs: tuple) -> dict[str, float]:
    """Prove the un-fused head matches the fused one we would otherwise be running."""
    with torch.no_grad():
        fused = model(*inputs, False)
        with _no_mha_fastpath():
            plain = model(*inputs, False)
    return {
        "logits": float((fused[0] - plain[0]).abs().max()),
        "act_logits": float((fused[1] - plain[1]).abs().max()),
    }


def check_attention_equivalence(model: nn.Module, inputs: tuple) -> dict[str, float]:
    """Prove that exporting under "eager" does not change the model's numbers.

    transformers dispatches on ``config._attn_implementation`` at forward time
    (``modeling_modernbert.py:282``), so this flips the attribute rather than rebuilding
    the model -- a second 421M-parameter fp32 copy does not fit comfortably in RAM.
    """
    cfg = model.encoder.config
    original = cfg._attn_implementation
    with torch.no_grad():
        eager = model(*inputs, False)
        try:
            cfg._attn_implementation = "sdpa"
            sdpa = model(*inputs, False)
        finally:
            cfg._attn_implementation = original
    return {
        "logits": float((eager[0] - sdpa[0]).abs().max()),
        "act_logits": float((eager[1] - sdpa[1]).abs().max()),
    }


DYNAMIC_AXES = {
    "input_ids": {0: "batch", 1: "seq"},
    "attention_mask": {0: "batch", 1: "seq"},
    "marker_pos": {0: "batch", 1: "k"},
    "marker_mask": {0: "batch", 1: "k"},
    "qtype": {0: "batch"},
    "logits": {0: "batch", 1: "k"},
    "act_logits": {0: "batch"},
}

INPUT_NAMES = ["input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype"]
OUTPUT_NAMES = ["logits", "act_logits"]

# The shape frozen into the Go spike fixture (PLAN.md S2.1). Deliberately small: the
# file is checked in, and one row is enough to prove the binding feeds tensors
# correctly. k >= 2 still applies -- see example_inputs.
FIXTURE_SEQ, FIXTURE_K, FIXTURE_BATCH = 16, 3, 1


def export(
    model: nn.Module, inputs: tuple, out_path: Path, dynamo: bool = False, opset: int | None = None
) -> None:
    """S1.2: opset >= 17, dynamic {batch, seq} and {batch, k}.

    ``dynamo=False`` is the legacy TorchScript exporter. It produces a file, but
    transformers' mask builder bakes the traced sequence length into the graph, so the
    result only runs at the shape it was traced with -- see ``--dynamo`` and PLAN.md S1.3.
    """
    opset = opset or (OPSET_DYNAMO if dynamo else OPSET_TORCHSCRIPT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = DecisionModelWrapper(model).eval()
    with torch.no_grad(), _no_mha_fastpath():
        torch.onnx.export(
            wrapper,
            inputs,
            out_path.as_posix(),
            input_names=INPUT_NAMES,
            output_names=OUTPUT_NAMES,
            dynamic_axes=DYNAMIC_AXES,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=dynamo,
        )


def head_activation(onnx_path: Path) -> dict[str, int]:
    """S1.4: the head FFN is ReLU while the encoder body is GeGLU.

    ``nn.TransformerEncoderLayer``'s default activation is relu and laya never overrides
    it (``original/laya/common.py:97``). Counting the op types in the exported graph is
    the only way to confirm the export preserved that rather than folding in a GELU.
    """
    import onnx

    model = onnx.load(onnx_path.as_posix(), load_external_data=False)
    counts: dict[str, int] = {}
    for node in model.graph.node:
        counts[node.op_type] = counts.get(node.op_type, 0) + 1
    return {op: counts.get(op, 0) for op in ("Relu", "Gelu", "Erf", "Tanh")}


def compare_with_ort(onnx_path: Path, model: nn.Module, inputs: tuple) -> dict[str, Any]:
    """S1 exit criterion: the file loads in ORT and its logits match PyTorch.

    Also runs a *different* shape from the traced one, because an export whose dynamic
    axes silently baked to constants still passes a same-shape check.
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])
    names = [i.name for i in sess.get_inputs()]

    def run(batch: tuple) -> dict[str, float]:
        feed = {n: t.numpy() for n, t in zip(names, batch, strict=True)}
        got = sess.run(["logits", "act_logits"], feed)
        with torch.no_grad(), _no_mha_fastpath():
            want = model(*batch, False)
        return {
            "logits": float(np.abs(got[0] - want[0].numpy()).max()),
            "act_logits": float(np.abs(got[1] - want[1].numpy()).max()),
        }

    def attempt(label: str, batch: tuple) -> dict[str, Any]:
        try:
            return run(batch)
        except Exception as exc:  # a shape that will not run IS the result
            return {"error": f"{type(exc).__name__}: {str(exc).strip().splitlines()[0]}"}

    return {
        "outputs": [o.name for o in sess.get_outputs()],
        "traced_shape": attempt("traced", inputs),
        # Different on every dynamic axis at once. transformers emits a TracerWarning
        # about baking the sequence length into the attention mask, so these are the
        # checks that say whether the warning mattered.
        "other_seq": attempt("other_seq", example_inputs(model, seq=61, k=4, batch=2)),
        "other_k": attempt("other_k", example_inputs(model, seq=48, k=7, batch=2)),
        "other_batch": attempt("other_batch", example_inputs(model, seq=48, k=4, batch=3)),
    }


def dump_fixture(onnx_path: Path, model: nn.Module, ckpt: Checkpoint, out_path: Path) -> dict[str, Any]:
    """Freeze one forward pass so the Go spike can assert against it (PLAN.md S2.1).

    Stores *both* runtimes' outputs on purpose. Go loads the graph through whatever
    libonnxruntime is installed -- 1.23.x, the only C API version
    ``onnxruntime-purego`` implements -- while these numbers come from the pinned
    reference environment, which is on a newer onnxruntime. Keeping the PyTorch
    outputs next to the ORT ones separates "the Go binding fed the tensors wrong"
    from "the two ORT builds disagree slightly".
    """
    import onnxruntime as ort

    inputs = example_inputs(model, seq=FIXTURE_SEQ, k=FIXTURE_K, batch=FIXTURE_BATCH)
    sess = ort.InferenceSession(onnx_path.as_posix(), providers=["CPUExecutionProvider"])

    got_names = [i.name for i in sess.get_inputs()]
    if got_names != INPUT_NAMES:
        raise RuntimeError(f"graph inputs are {got_names}, expected {INPUT_NAMES}")

    feed = {n: t.numpy() for n, t in zip(INPUT_NAMES, inputs, strict=True)}
    got = sess.run(OUTPUT_NAMES, feed)
    with torch.no_grad(), _no_mha_fastpath():
        want = model(*inputs, False)

    def tensor(a: np.ndarray) -> dict[str, Any]:
        return {"dtype": str(a.dtype), "shape": list(a.shape), "data": a.reshape(-1).tolist()}

    fixture = {
        "_comment": (
            "Generated by scripts/export_onnx.py --fixture against the pinned reference "
            "environment. Do not hand-edit: regenerating it is a reviewed diff (PLAN.md R6)."
        ),
        "checkpoint": ckpt.name,
        "onnx": onnx_path.name,
        "versions": versions(),
        "input_names": INPUT_NAMES,
        "output_names": OUTPUT_NAMES,
        "inputs": {n: tensor(feed[n]) for n in INPUT_NAMES},
        "ort": {n: tensor(a) for n, a in zip(OUTPUT_NAMES, got, strict=True)},
        "torch": {n: tensor(t.numpy()) for n, t in zip(OUTPUT_NAMES, want, strict=True)},
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(fixture, indent=2) + "\n")
    return {
        "path": out_path.as_posix(),
        "shape": {"batch": FIXTURE_BATCH, "seq": FIXTURE_SEQ, "k": FIXTURE_K},
        "ort_vs_torch": {
            n: float(np.abs(a - t.numpy()).max())
            for n, a, t in zip(OUTPUT_NAMES, got, want, strict=True)
        },
    }


def _onnxscript_version() -> str:
    try:
        import onnxscript
    except ImportError:
        return "<not installed; required for --dynamo>"
    return onnxscript.__version__


def versions() -> dict[str, str]:
    import onnx
    import onnxruntime
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime.__version__,
        "numpy": np.__version__,
        "onnxscript": _onnxscript_version(),
    }


def run_one(
    ckpt: Checkpoint,
    models_root: Path,
    out_dir: Path,
    check_attn: bool,
    reuse: bool = False,
    dynamo: bool = False,
    suffix: str = "",
    opset: int | None = None,
    fixture: Path | None = None,
) -> dict[str, Any]:
    ckpt_dir = ckpt.dir(models_root)
    print(f"\n=== {ckpt.name} ({ckpt_dir}) ===", flush=True)
    report: dict[str, Any] = {"checkpoint": ckpt.name, "dir": ckpt_dir.as_posix()}

    model = build_decision_model(ckpt_dir, attn="eager")
    report["hidden_size"] = model.encoder.config.hidden_size
    report["params"] = sum(p.numel() for p in model.parameters())
    print(f"loaded strict=True: {report['params']:,} params, d={report['hidden_size']}", flush=True)

    inputs = example_inputs(model)

    if check_attn:
        report["eager_vs_sdpa"] = check_attention_equivalence(model, inputs)
        print(f"eager vs sdpa max abs diff: {report['eager_vs_sdpa']}", flush=True)
        report["fused_vs_reference_head"] = check_fastpath_equivalence(model, inputs)
        print(f"fused vs reference head max abs diff: {report['fused_vs_reference_head']}", flush=True)

    out_path = out_dir / f"laya-{ckpt.name}{suffix}.onnx"
    report["exporter"] = "dynamo" if dynamo else "torchscript"
    report["opset"] = opset or (OPSET_DYNAMO if dynamo else OPSET_TORCHSCRIPT)
    if reuse and out_path.exists():
        print(f"reusing existing {out_path}", flush=True)
    else:
        export(model, inputs, out_path, dynamo=dynamo, opset=opset)
    report["onnx"] = out_path.as_posix()
    report["bytes"] = sum(p.stat().st_size for p in out_dir.glob(f"{out_path.stem}.onnx*"))
    print(f"exported -> {out_path} ({report['bytes'] / 1e6:.0f} MB)", flush=True)

    report["head_ops"] = head_activation(out_path)
    print(f"activation op counts: {report['head_ops']}", flush=True)

    report["ort"] = compare_with_ort(out_path, model, inputs)
    print(f"ORT vs PyTorch: {report['ort']}", flush=True)

    if fixture is not None:
        report["fixture"] = dump_fixture(out_path, model, ckpt, fixture)
        print(f"fixture -> {report['fixture']}", flush=True)

    del model
    gc.collect()
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", choices=[c.name for c in CHECKPOINTS], action="append", default=[])
    ap.add_argument("--all", action="store_true", help="export all three checkpoints (S1.5)")
    ap.add_argument(
        "--models-root",
        type=Path,
        default=Path(os.environ.get("LAYA_MODELS", REPO_ROOT / "models")),
        help="directory holding the downloaded laya/ snapshot (default: $LAYA_MODELS)",
    )
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "build" / "onnx")
    ap.add_argument("--no-check-attn", dest="check_attn", action="store_false")
    ap.add_argument("--reuse", action="store_true", help="skip the export if the .onnx is already there")
    ap.add_argument(
        "--dynamo",
        action="store_true",
        help="use the torch.export-based exporter instead of TorchScript (PLAN.md S1.3)",
    )
    ap.add_argument("--suffix", default="", help="appended to the output filename")
    ap.add_argument("--opset", type=int, default=None, help="override the default opset")
    ap.add_argument("--report", type=Path, default=None, help="write the JSON report here")
    ap.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="write a one-forward-pass JSON fixture for the Go spike here (PLAN.md S2.1)",
    )
    args = ap.parse_args()

    selected = CHECKPOINTS if args.all or not args.checkpoint else tuple(
        c for c in CHECKPOINTS if c.name in args.checkpoint
    )

    if args.fixture is not None and len(selected) != 1:
        ap.error("--fixture writes a single file; select exactly one --checkpoint")

    print(json.dumps(versions(), indent=2), flush=True)
    torch.manual_seed(0)

    reports, failures = [], []
    for ckpt in selected:
        try:
            reports.append(
                run_one(
                    ckpt, args.models_root, args.out, args.check_attn,
                    args.reuse, args.dynamo, args.suffix, args.opset, args.fixture,
                )
            )
        except Exception as exc:  # a spike records how it failed; it does not hide it
            import traceback

            traceback.print_exc()
            failures.append({"checkpoint": ckpt.name, "error": f"{type(exc).__name__}: {exc}"})

    summary = {"versions": versions(), "exported": reports, "failed": failures}
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(summary, indent=2) + "\n")
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
