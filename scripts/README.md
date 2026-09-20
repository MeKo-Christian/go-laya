# `scripts/` — the Python reference harness

Nothing in here runs in CI, and nothing in here is needed to build or test the Go module.
These scripts exist to produce artefacts that _are_ checked in (golden vectors under
`testdata/`) or that are too large to check in (ONNX exports), from the frozen upstream
Python in `original/`.

| Script                       | Purpose                                                                                | `PLAN.md` |
| ---------------------------- | -------------------------------------------------------------------------------------- | --------- |
| `export_onnx.py`             | Export the whole `DecisionModel` graph (encoder + head) to ONNX                        | Spike S1  |
| `export_onnx.py --fixture`   | One forward pass with inputs, Python-ORT and PyTorch outputs, for the Go binding tests | Spike S2  |
| `dump_python_parity.py`      | Generate the golden vectors under `testdata/`                                          | Task 1.3  |
| `crosscheck_transformers.py` | Diff a transformers 4.x build against the pinned 5.17.0                                | Task 1.6  |
| `requirements-ref.txt`       | The pinned reference environment — **this file is the contract**                       | Task 1.1  |

## Regenerating the golden vectors

```bash
.venv-ref/bin/python scripts/dump_python_parity.py --all
just fmt && just ci
git diff --stat testdata/     # review it; this is never a drive-by
```

Eight JSONL files, one header record per file carrying the `transformers` /
`tokenizers` / `numpy` / `safetensors` versions, the Hub revision and a sha256 of
each `tokenizer_config.json`. `TestGoldenProvenance` fails the whole suite if a
header drifts from what the Go port targets (R6).

Seven of the eight need no model weights — only `logits.jsonl` loads a checkpoint.
`--verify-agent` additionally checks the generator's inlined copy of
`agent.py:294-343` against a real `laya.Agent.system_one` run, because a copy that
drifts would produce a corpus that is internally consistent and wrong.

`testdata/*.jsonl` is excluded from `treefmt` (see `treefmt.toml`): the vectors are
compared byte-for-byte, so nothing may reflow them.

## Checking the transformers major version

```bash
uv venv --python 3.12 .venv-ref4
uv pip install --python .venv-ref4/bin/python --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    'torch==2.14.0+cpu' 'transformers==4.57.6' 'numpy==2.5.3' 'safetensors==0.8.0'

.venv-ref/bin/python  scripts/crosscheck_transformers.py --all --out build/x-5.json
.venv-ref4/bin/python scripts/crosscheck_transformers.py --all --out build/x-4.json
.venv-ref/bin/python  scripts/crosscheck_transformers.py --compare build/x-4.json build/x-5.json
```

**Answered 2026-09-20: stay on 5.17.0.** 4.57.6 mis-parses mmBERT's
`rope_parameters.sliding_attention.rope_theta`, silently substituting its own
default. Details in the script's docstring and `PLAN.md` task 1.6.

## The reference environment

The system Python's torch is broken (`libtorch_global_deps.so: cannot open shared object
file`). Do not repair it; the reference environment is deliberately isolated:

```bash
uv venv --python 3.12 .venv-ref
uv pip install --python .venv-ref/bin/python torch --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-ref/bin/python transformers safetensors huggingface_hub numpy onnx onnxruntime
uv pip freeze --python .venv-ref/bin/python >> scripts/requirements-ref.txt
```

`requirements-ref.txt` carries a hand-written header with the `--extra-index-url` that the
CPU torch wheel needs, so a bare `uv pip freeze >` would throw it away. Truncate the file
back to that header before appending, or re-add it afterwards.

To reproduce an existing environment exactly, install from the lock instead:

```bash
uv venv --python 3.12 .venv-ref
uv pip install --python .venv-ref/bin/python --index-strategy unsafe-best-match \
    -r scripts/requirements-ref.txt
```

`--index-strategy unsafe-best-match` is not optional and cannot live in the requirements
file. uv defaults to first-index-wins, so it finds plain `torch` on PyPI, sees that the
pinned `2.14.0+cpu` is not there, and stops rather than looking at the extra index.

`.venv-ref/` is gitignored. `requirements-ref.txt` is not: **the versions are part of the
parity contract** (`PLAN.md` R6). A tokenizer or transformers upgrade can silently change
the golden vectors, which is the realistic way parity regresses, so regenerating
`testdata/` after changing these pins is a reviewed diff and never a drive-by.

### transformers 5.x and `reference_compile`

`PLAN.md` R2 expects `torch.onnx.export` to trip over ModernBERT's `torch.compile`
decorators (huggingface/transformers#35545). **That blocker is gone in transformers 5.x**:
there is no `torch.compile` left in `modeling_modernbert.py`, and `reference_compile`
survives only as a key that `ModernBertConfig.to_dict` pops for backwards compatibility.
`export_onnx.py` still sets it when the attribute exists, so it keeps working against a
4.x environment.

Upstream declares `transformers>=4.45.0` with no upper bound, and the checkpoint loads
under 5.17.0 with `load_state_dict(strict=True)` — which proves the parameter set matches.
It does not prove the _numerics_ match 4.x. See the open task under `PLAN.md` M1.

## The checkpoints

~2.4 GB, Apache-2.0, ungated, no HF token:

```bash
export LAYA_MODELS=$PWD/models
.venv-ref/bin/python -c "from huggingface_hub import snapshot_download as d; d('convaiinnovations/laya', local_dir='$LAYA_MODELS/laya')"
```

`models/` is gitignored. The resolved Hub revision is written to `models/PROVENANCE.json`
at download time, because **the local tree stops matching that revision as soon as the
Python is used**: `laya.Agent.__init__` calls `_fix_tokenizer_config`
(`original/laya/agent.py:21-46`), which rewrites `tokenizer/tokenizer_config.json` in
place to satisfy newer transformers. Golden vectors are only meaningful against a known
revision, so record it before anything loads the model, not after.

The three checkpoints are the repo root (english), `multilingual/` and `typed-decisions/`.

## Exporting to ONNX

```bash
.venv-ref/bin/python scripts/export_onnx.py --all --dynamo --report build/onnx/report.json
```

`--dynamo` is not optional if the export has to run at more than one sequence length.
Both exporters produce a file; only one of them produces a _usable_ one:

| Exporter              | Opset | `batch` | `k`     | `seq`                           |
| --------------------- | ----- | ------- | ------- | ------------------------------- |
| TorchScript (default) | 17    | dynamic | dynamic | **frozen at the traced length** |
| `--dynamo`            | 18    | dynamic | dynamic | dynamic                         |

The TorchScript exporter warns about it and carries on:

```text
TracerWarning: Converting a tensor to a Python boolean might cause the trace to be
incorrect  ...  transformers/masking_utils.py:213
```

The resulting graph then fails in ONNX Runtime at any other length, with a `Reshape`
complaining about the shape it was traced with. It loads and runs perfectly at the traced
shape, which is why `export_onnx.py` validates against **four** shapes and not one.

`--dynamo` needs `onnxscript` (in `requirements-ref.txt`) and opset 18. Asking it for 17
produces a file that ONNX Runtime rejects outright — the 18→17 downconversion leaves
`Split` carrying its opset-18 `num_outputs` attribute.

Output goes to `build/onnx/` (gitignored). Each export is ~1.7 GB: the stored weights are
fp16, CPU kernels largely have no fp16 path, so the export is fp32. Above the 2 GB
protobuf limit the dynamo exporter splits the file into `laya-<name>.onnx` plus
`laya-<name>.onnx.data`.

For each checkpoint the script reports the parameter count, eager-vs-sdpa and
fused-vs-reference-head differences, the activation op counts in the exported graph, and
the max absolute difference between ONNX Runtime and PyTorch at each of the four shapes.
Measured numbers live in `PLAN.md` under Spike S1.
