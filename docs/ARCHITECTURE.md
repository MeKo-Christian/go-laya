# Architecture

> What go-laya is and the facts it rests on. There is no history here, only the current state.
> The decisions behind it, with their evidence, are in [`DECISIONS.md`](DECISIONS.md). The public
> types and JSON shapes are in [`API.md`](API.md), the 74 behaviours a test must assert are in
> [`INVARIANTS.md`](INVARIANTS.md), and measured latency is in [`../BENCHMARKS.md`](../BENCHMARKS.md).

## 1. What is being ported

### 1.1 The three checkpoints

`convaiinnovations/laya` bundles all three. It is Apache-2.0, **not gated, and needs no token**.
Golden vectors describe revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982`. The default loader pins the bundle repo
to it (D17).

| Name            | Subfolder          | Encoder                        | Params      | `max_len` / `head_max_len` | Weights                                 |
| --------------- | ------------------ | ------------------------------ | ----------- | -------------------------- | --------------------------------------- |
| english         | _(root)_           | `answerdotai/ModernBERT-large` | 421,293,830 | 512 / 192                  | 842 MB, **fp16** (+ fp32 `temperature`) |
| multilingual    | `multilingual/`    | `jhu-clsp/mmBERT-base`         | 321,908,998 | 1024 / 256                 | 644 MB, fp16                            |
| typed-decisions | `typed-decisions/` | `answerdotai/ModernBERT-large` | 421,293,830 | 1024 / 256                 | 842 MB, all fp16                        |

`amp_dtype: "bf16"` refers to autocast; the stored weights are fp16. The `temperature` tensor
(F32/F32/F16) is **dead at inference**: `common.py:102` registers it and `forward` never reads it.
Calibration comes from `cfg["temperature"]` / `cfg["temperature_by_options"]` in
`rl_agent_config.json` (`agent.py:194-195,304`). Its dtype matters only to a strict safetensors
loader.

`laya.Agent.__init__` rewrites `tokenizer/tokenizer_config.json` in place (`_fix_tokenizer_config`,
`agent.py:21-46`). Every fixture header therefore records a sha256 of each config.

### 1.2 Encoder

**ModernBERT-large:** `model_type: modernbert`, hidden 1024, 28 layers, 16 heads (head_dim 64),
intermediate 2624, vocab 50368.

- **RoPE only**, with no positional-embedding tensor. Every 3rd layer (0, 3, …, 27) is
  `full_attention` at `rope_theta=160000`. The rest are `sliding_attention`, window 128 (±64),
  at `rope_theta=10000`.
- **Bias-free LayerNorm** (eps 1e-5). Layer 0 has no `attn_norm`.
- **GeGLU**, no bias: `mlp.Wi [5248,1024]` is fused gate+up, `mlp.Wo [1024,2624]`.
- **Fused QKV** `attn.Wqkv [3072,1024]`, `attn.Wo [1024,1024]`, no attention bias.
- The reference runs `attn_implementation="sdpa"`: padded, no unpadding, no flash-attn.

**mmBERT-base deltas:** hidden 768, 22 layers, 12 heads, intermediate 1152, vocab 256000. Both
attention types use `rope_theta=160000`.

> ⚠️ `multilingual/encoder/config.json` says `cls_token_id: 1`, but the tokenizer says `<bos>` =
> **2**. The reference reads `tok.cls_token_id`, so 2 is what the weights were trained with, and
> the Go port resolves special tokens from the tokenizer config.

### 1.3 The decision head

```
type_emb.weight                             [3, d]
head.layers.{0,1}.self_attn.in_proj_weight  [3d, d]  .in_proj_bias  [3d]
head.layers.{0,1}.self_attn.out_proj.{weight,bias}
head.layers.{0,1}.linear1.{weight,bias}     [4d, d]
head.layers.{0,1}.linear2.{weight,bias}     [d, 4d]
head.layers.{0,1}.norm1.{weight,bias}  norm2.{weight,bias}
scorer.0.{weight,bias}   # LayerNorm
scorer.1.{weight,bias}   [d, d]
scorer.3.{weight,bias}   [1, d]
act_head.0.{weight,bias} [256, d+4]
act_head.2.{weight,bias} [n_act, 256]   # n_act = len(cfg["act_costs"]) + 1 (common.py:137); 2 on all shipped checkpoints
temperature              [3]            # never read at inference
```

> ⚠️ **The head FFN is ReLU, not GELU** (`nn.TransformerEncoderLayer`'s default). The encoder
> body is GeGLU.
>
> ⚠️ The head layers run in a manual loop (`for layer in self.head.layers`), bypassing
> `nn.TransformerEncoder`'s optional final norm. A reimplementation must reproduce the loop.

`act_logits` sits downstream of a softmax, an entropy and a `topk(2)` difference
(`common.py:119-125`), so small logit differences are amplified before `act_head` sees them. It
is always looser than `logits` in every parity measurement.

## 2. Repository layout

```
go-laya/
  laya.go        Agent interface (D13), Version; planned: a public Open (PLAN 7.7.5)
  answer.go      Result, Answer, Probs, AnswerSet and their Python-byte JSON; formatAnswer
  systemone.go   SystemOne: validate, BuildSequence, Collate, one Forward, formatAnswer
  question.go    type aliases re-exporting question/ (D11)
  router.go      Router, RouteDecision, model registry; lru.go the agent cache
  loader.go      the default agent loader: snapshot, config, tokenizer, local ONNX export (D24)
  options.go  errors.go
  lang/          script + language detection            (public, zero deps)
  mailtext/      email cleaning                         (public, zero deps)
  presets/       the five question presets              (public, zero deps)
  jsonx/         ordered JSON with Python byte parity; Round4, Repr, ReprString (D12)
  question/      Question interface + Choice/Score/Noul (leaf, D11)
  tokenizer/     Tokenizer interface + pure-Go tokenizer.json loader (D10)
  backend/       Backend interface + Batch — public leaf, stdlib only (D9)
  internal/prompt/        render_options, render_criterion, serialize_state, BuildSequence, Collate
  internal/calib/         temperatures, float32 softmax, entropy confidence, ECE, Brier, numpy exp/log, ATen act softmax
  internal/golden/        the testdata/ loader shared by every package
  internal/hub/           HF resolve + verified cache + snapshot + offline
  internal/checkpoint/    rl_agent_config.json loading and validation
  internal/onnxheader/    ONNX graph header validation (protowire, no generated types)
  internal/safetensors/   safetensors header validation
  internal/ortlib/        pinned, hash-verified ONNX Runtime download
  internal/backend/fake/  replays testdata/logits.jsonl per checkpoint
  internal/backend/onnx/  ONNX Runtime backend
  cmd/laya-ort/           downloads the pinned ORT library
  scripts/       dump_python_parity.py, export_onnx.py, crosscheck_transformers.py, … (scripts/README.md)
  testdata/      golden vectors, checked in; CI never needs Python
  original/      frozen upstream Python (D4)
```

**Dependency rule.** `lang`, `mailtext`, `presets`, `backend`, `jsonx` and `question` never
depend on the ONNX binding. `TestNoMLDependency` (`deps_test.go`) runs `go list -deps` over them
with `GOOS=linux` pinned. `TestRuntimeImportedOnlyByBackend` requires `internal/backend/onnx` to
be the binding's only direct importer. The root package, and so the Router, may depend on the
binding. Under purego nothing is `dlopen`ed until `onnx.Open`, so a routing-only caller gains a
build-graph edge but never loads ONNX Runtime.

## 3. Tokenizer

Both checkpoints ship only `tokenizer/tokenizer.json` + `tokenizer_config.json`.
`typed-decisions` is byte-identical to English, so there are exactly **two pipelines**.

**English (ModernBERT):** NFC → `ByteLevel{add_prefix_space:false, use_regex:true}` → BPE (vocab
50280, 50009 merges, `byte_fallback:false`, `unk:null`). There are 116 added tokens:

- 23 space runs of 2–24 at ids 50254–50276 (24 spaces = 50254)
- `|||IP_ADDRESS|||` (0), `|||EMAIL_ADDRESS|||` (50277), `|||PHONE_NUMBER|||` (50278)
- `[unused0..82]` (50285–50367)
- the specials **UNK 50280, CLS 50281, SEP 50282, PAD 50283, MASK 50284** (`lstrip:true`)

**Multilingual (Gemma):** `Replace{" " → "▁"}` → `Metaspace{prepend_scheme:always, split:true}` →
BPE (vocab 256000, 580604 merges, **`byte_fallback:true`, `fuse_unk:true`**). There are 249 added
tokens, all `normalized:false`, including runs of `\n`, `\t` and `▁`. Specials: **PAD 0, SEP 1,
CLS 2, UNK 3, MASK 4**.

Semantics the implementation gets right from config, not by convention:

1. **Added tokens are matched first, in HF's two phases.** Phase 1 matches `normalized:false`
   tokens on the raw text, and phase 2 matches `normalized:true` tokens on each normalised
   segment. Matching is leftmost-longest, and `lstrip` extends left over Unicode `White_Space` but
   never past the previous match. On English, `[MASK]` swallows spaces before the space-run tokens
   see them. On multilingual, `▁` runs match before `Replace`.
2. **`tok(" " + text)` behaves oppositely per checkpoint.** On English the space becomes `Ġ`. On
   multilingual, `Replace` makes the text start with `▁`, and Metaspace's `!starts_with` guard
   suppresses the prepend, so `tok(" x") == tok("x")`. The exception: when the text begins with an
   added token, the leading `" "` is its own segment and becomes `▁` (id 235248).
3. **ByteLevel is a hand-written rune scanner**, not a regex. HF's pattern has a `\s+(?!\S)`
   lookahead RE2 cannot express, and its `\s` is Unicode `White_Space`. A whitespace run of
   length ≥ 2 gives back its last char unless it reaches the segment end. The lookahead is
   evaluated inside the added-token segment, so `'a\t\t[unused0]'` keeps its tab run whole.
4. **Byte coverage is a load-time assertion.** Multilingual has 255 of 256 `<0xXX>` tokens (no
   `<0x09>`, harmless). English lacks the 13 byte-chars that never occur in valid UTF-8, which is
   why `Encode` sanitises first (D15).
5. **BPE** does per-char vocab lookup, applies byte fallback per char before merging (only if
   every byte is in the vocab), then `fuse_unk`, then a rank/position heap. Every merge resolves to
   a `newID` at load, and a merge whose concatenation is not in the vocab fails the load.

**Differential result (157 281 lines × 2 tokenizers).** There is one divergence class: 85 lines,
all in NFC on English/typed-decisions. `tokenizers` 0.23.2's Rust tables lack the canonical
combining class of 108 codepoints assigned in Unicode 11–15. Go and CPython agree with each other,
so the oracle is the outlier. No natural text, vocabulary entry or UI string in the corpus reaches
it, and multilingual (no NFC) cannot. `scripts/probe_ccc.py` enumerates the set. Whether to
reproduce the Rust tables is PLAN Task B.2.

**Cost.** `Open`: english 68 ms / 19.6 MB, multilingual 0.6–1.2 s / 207 MB (the 34 MB JSON
decode, PLAN Task B.1). `Encode`: 191–456 µs.

## 4. Prompt assembly

`internal/prompt.BuildSequence` ports `common.py:49-86`: the token budget (`opt_budget`, the
`< 16` fallback, 48-id option cap), truncation direction, the final `ids[:max_len]` clamp, and the
marker filter. Python slice semantics are kept exactly, including `st[-0:]` being the **whole**
state. `option_order` and `truncate_left` exist only on this internal signature.
`Collate` ports `collate_items`: right-pad with `pad_id`, `kmax` over the batch, `marker_mask`
false beyond each row. The graph sees the padded batch, so a question batched with others is a
different computation from the same question alone.

## 5. ONNX export and backend

**Export recipe** (`scripts/export_onnx.py`, recorded in its docstring): `DecisionModel.forward`
itself (so the manual head loop comes for free), `attn_implementation="eager"` (D19),
`torch.backends.mha.set_fastpath_enabled(False)` (the fused
`aten::_transformer_encoder_layer_fwd` has no ONNX symbolic), and `dynamo=True` at **opset 18**.
The TorchScript exporter silently freezes the sequence axis, and opset 17 under dynamo yields a
graph ORT refuses. The export needs `onnxscript`. Files over 2 GB split into `.onnx` +
`.onnx.data`.

**Parity, measured.** These are scaled diffs `|got−want| / max(1, |want|)`, because `logits`
carries the −1e4 mask sentinel.

| Checkpoint      | ONNX vs PyTorch (S1, abs, logits / act) | Go ORT at real prompts (logits / act) | Go ORT, S1 shapes, worst (vs Python ORT / vs PyTorch) |
| --------------- | --------------------------------------- | ------------------------------------- | ----------------------------------------------------- |
| english         | 6.4e-05 / 2.0e-03                       | 9.4e-06 / 3.9e-06                     | 7.0e-05 / 3.1e-05                                     |
| multilingual    | 4.9e-04 / 8.4e-02                       | 9.8e-06 / 1.4e-06                     | 6.4e-05 / 4.7e-04                                     |
| typed-decisions | 6.1e-06 / 4.2e-03                       | 5.8e-06 / 1.2e-06                     | 2.6e-05 / 3.2e-05                                     |

`matrixTol` is about 5× the matrix worst cases and `goldenTol` a 5e-05 ceiling. Tolerances are **per checkpoint**, never global: on random inputs multilingual is ~15× looser
against PyTorch. PyTorch's own eager-vs-sdpa gap (absolute, logits / act) is english
1.0e-06 / 4.9e-04, multilingual 5.1e-05 / 1.2e-02, typed-decisions 1.5e-06 / 7.3e-04.

**Backend constraints** (`internal/backend/onnx`):

- Every `*Value` is closed explicitly (D5). `LAYA_ORT_FINALIZER=1` keeps the race reproduction.
- Sessions are built **from a path**, because ORT resolves `.onnx.data` relative to it.
- `Session.Run` ignores its `ctx` (NULL `RunOptions`), so `Forward` checks `ctx` before and after
  and cannot cancel a pass in flight.
- `Forward` validates shapes (`ErrBadBatch`), checks `logits` width against the batch's `kmax` and
  `act_logits` against `Options.ActWidth` (= `len(act_costs)+1`), and folds outputs by their
  returned shape.
- `Open` validates the graph header (`internal/onnxheader`) before loading the library: IR 3–11,
  `ai.onnx` opset ≤ 23, external data local, regular, non-symlinked and in range, and the IO names
  and `act_logits` width. Any failure wraps `ErrIncompatibleCheckpoint`.
- `Close` releases session, env and runtime, waits for in-flight `Forward`s, and is idempotent.
- `IntraOpThreads` defaults to physical cores, not `runtime.NumCPU()` (D6).
- Device: `""`/`auto` walks cuda → coreml → cpu. An explicit device the library lacks falls back
  to CPU with one `slog` warning, as does `cuda` for now (D22).
- Builds only on D21's platform list; elsewhere `ErrUnsupportedPlatform`.

**Runtime library.** The binding speaks C API 23. `Open` resolves `LAYA_ORT_LIB`, then
`ORT_LIBRARY_PATH` (set but missing is an error, never a fallback), then the verified download in
`$LAYA_CACHE/onnxruntime/` (a hash failure is an error), then platform paths. It then requires
version `1.M.P` with M ≥ 23 (D20).

## 6. Downloads and trust

The default loader downloads a checkpoint's config and tokenizer, and the planned
`laya.Open("someone/their-model")` (PLAN Task 7.3) will do the same for any repo, so the download
layer verifies everything before use (R7). The graph itself is never downloaded: it is a local
export, `laya-<name>.onnx` in `WithONNXDir`, `$LAYA_ONNX_DIR` or `<cache>/onnx` (D24).

- **`internal/hub`**: HEAD then GET on `/{repo}/resolve/{rev}/{path}`. Redirects are followed by
  hand, so the token goes only to the Hub's host. The first hop's `X-Linked-Etag` is the git blob
  sha1 (regular file) or the sha256 (LFS file). A weak, missing or malformed tag is
  `ErrUnverifiable` (fail closed), and a mismatch is `ErrHashMismatch`. Files land at
  `Dir/owner/name/<commit>/path` via `.partial-*` then rename, after the hash matched.
  `Snapshot` lists the revision, filters with fnmatch semantics (`*` crosses `/`), and fetches at
  the listing's commit (`ErrCommitMismatch` otherwise). Remote names are validated before use.
  `Offline` resolves from recorded refs and listings with no network call. Everything honours
  `context.Context`, and a cancelled download leaves nothing behind.
- **`internal/ortlib`**: archive sha256 and size checked before parsing, only the library member
  extracted, and that member checked by size and sha256, `Sync`ed and renamed. `Cached`
  re-hashes on every call and `Lstat`s every directory, so a symlink is rejected rather than
  followed.
- **`internal/checkpoint`**: `rl_agent_config.json` is capped at 1 MiB and needs `encoder` and
  `head_layers`. `ActWidth` uses Python's `len` semantics.
- **`internal/safetensors`**: the safetensors library's own header rules (≤ 100 MB JSON header,
  known dtypes, spans tiling the buffer exactly, overflow-checked).
- CI forbids `encoding/gob` and `os/exec` outside tests, and filesystem `replace` directives.

## 7. Numerics parity rules

Each of these gives a plausible wrong answer rather than an error if done the obvious Go way.

- **JSON bytes.** `encoding/json` emits `{"a":1}`, sorts map keys and HTML-escapes. `jsonx` writes
  Python's `", "`/`": "`, preserves order, emits non-ASCII literally, writes U+2028/U+2029
  literally and `\b` rather than `\u0008`, and refuses Go maps. It must be the outer encoder
  (D12).
- **Float repr.** Python writes `1.0`, `1e+16`, `1e-05`, `-0.0`; `jsonx.Repr` reimplements
  CPython's shortest-round-trip rule. `jsonx.ReprString` is `str.__repr__`, which `strconv.Quote`
  disagrees with three ways.
- **Rounding.** `round()` is half-to-even on the exact double. `Round4` formats and reparses, and
  keeps `-0.0`.
- **Dict order.** `detect_script`'s tie-break is encounter order with `latin` assigned last. The
  emitted `script_profile` seeds `latin` first. `_STOP`'s language order is the tie-break. Ordered
  slices everywhere.
- **Unicode classes.** Python's `\w`/`\s` are Unicode, RE2's are ASCII, and `str.strip()` strips
  U+001C–U+001F, which `unicode.IsSpace` does not.
- **float32 where numpy is float32.** Softmax and entropy confidence run in float32. The softmax
  sum is numpy's `pairwise_sum` (left to right below 8 elements). `score` is float64 over a
  float32 `p`, and `noul` confidence is `max(p1, 1−p1)`, never the entropy. The temperature floor
  `max(1e-3, t)` sits in the softmax. `ln k` in the confidence is a Python float (`math.log`),
  narrowed.
- **exp and log are numpy's kernel**, bit for bit (`internal/calib/npyf32.go`). This is one lane
  of numpy 2.5.3's AVX2+FMA3 `simd_exp_FLOAT`/`simd_log_FLOAT` (dispatch target `X86_V3`), with a
  single-rounding float32 FMA built from an exact float64 product plus round-to-odd. It has been
  verified on all 2^32 inputs. The fixtures record `"simd": "X86_V3"`, and the generator refuses
  other targets, because a host without FMA gives different reference numbers.
- **The act head is torch, not numpy** (D26). `act_probability` is torch's float32 softmax:
  ATen's `_vec_softmax_lastdim` on AVX2, i.e. a lane-wise max, SLEEF's `xexpf`
  (`Sleef_expf8_u10`), a left-to-right sum and a multiply by `1/sum`. `calib.ActSoftmax` ports
  it, including the vectorized reduction ATen switches to at 8 values, and is bit-exact on all
  8757 rows (widths 1–20, 32, 33) of `act_softmax.jsonl`. numpy's `exp32` with a divide
  misses 2234 of them in the last bits, and 96 after `Round4`. Only the AVX2 kernel is verified.
- **`p.argmax()` is the first max.**

## 8. Golden corpus

`scripts/dump_python_parity.py` in `.venv-ref` (`scripts/requirements-ref.txt`, transformers
5.17.0, D7) writes `testdata/*.jsonl`. Line 1 is a header carrying library versions, the Hub sha,
tokenizer-config sha256s and a `compute` block `{device: cpu, dtype: float32, attn: sdpa,
torch_threads: 1}`. `TestGoldenProvenance` globs every file and checks the header.
Regeneration is deterministic (per-case sha256 seeds, sorted cases, one thread) and is a reviewed
diff (R6).

| File                      | Holds                                                                                                                                                                                                                                       |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tokenizer_{en,ml}.jsonl` | 103 cases each, ids **and** token strings, bare and with a leading space. Needs `LAYA_MODELS`, so this is a local gate.                                                                                                                     |
| `pretok_{en,ml}.jsonl`    | Normalizer output and pre-tokenizer pieces, so CI tests those stages without checkpoints.                                                                                                                                                   |
| `render.jsonl`            | `render_options`, `render_criterion`, `serialize_state`, byte for byte; inputs are key-ordered objects.                                                                                                                                     |
| `sequence.jsonl`          | 45 `build_sequence` cases over all three checkpoints.                                                                                                                                                                                       |
| `logits.jsonl`            | 10 states × 3 questions × 3 checkpoints, batch-level: the collated tensors and the PyTorch outputs.                                                                                                                                         |
| `answers.jsonl`           | 33 `system_one` results, including `temperature_by_options` buckets, six `precision/*` cases that a float64 port or a sequential sum rounds differently, and exact `answer_json` bytes. Checked against a real `Agent` by `--verify-agent`. |
| `mailtext.jsonl`          | 26 `email.py` cases (upstream has no tests for it).                                                                                                                                                                                         |
| `round4.jsonl`            | 26 `round()` ties, including `-2.5e-05 → -0.0`.                                                                                                                                                                                             |
| `ece.jsonl`               | 16 upstream `ece_score` cases, plus 6 Brier cases (our definition).                                                                                                                                                                         |
| `f32math.jsonl`           | numpy float32 exp/log on 15 423 inputs, plus 64 softmax rows, all as bit patterns.                                                                                                                                                          |
| `act_softmax.jsonl`       | torch 2.14.0+cpu `softmax(-1)` on 8757 float32 rows of widths 1–20, 32 and 33 (AVX2), as bit patterns; 2800 two-wide rows sit within 3 ulp of a `Round4` tie. The generator refuses a torch not on AVX2.                                    |

The ONNX fixtures live beside the backend: `internal/backend/onnx/testdata/forward_pass.json` and
`matrix/forward-<checkpoint>.json` (S1's four shapes, from `export_onnx.py --fixture-matrix`).
`internal/backend/fake` replays `logits.jsonl` keyed on all five tensors **and** the checkpoint:
english and typed-decisions record identical batches with different logits. It fails with
`ErrUnknownBatch`, naming the first differing cell.
