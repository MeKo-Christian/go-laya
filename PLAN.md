# go-laya — Python → Go Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Reimplement the Python `laya` 0.3.4 decision engine as an idiomatic Go library that produces
byte-identical prompts and numerically equivalent decisions from the same Hugging Face checkpoints.

**Architecture:** Three layers, separately importable. (1) A dependency-free pure-Go core — question
rendering, language/script routing, email cleaning, presets, calibration math — which is 51 % of the
package and needs no ML runtime at all. (2) A pure-Go tokenizer that loads the checkpoint's
`tokenizer.json` directly. (3) A `Backend` interface for the neural net, implemented first against an
ONNX export of the _whole_ `DecisionModel` graph (encoder + head fused), with a pure-Go
safetensors backend landing later behind the same interface.

**Tech stack:** Go 1.26 · ONNX Runtime (binding chosen in Spike S2) · a forked pure-Go
`tokenizer.json` loader · `just` + `treefmt` + `golangci-lint` per house convention.

---

## 0. Decisions already made

| #   | Decision                                                                        | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                     |
| --- | ------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | **ONNX first, pure-Go native backend later**, both behind a `Backend` interface | The entire `DecisionModel.forward` is one static graph — no KV cache, no loop, no control flow — so it exports to a single ONNX file and there is exactly _one_ numerics-parity surface to test. Reimplementing ModernBERT (RoPE, alternating local/global attention, GeGLU) up front is weeks of work before anything runs. In-house precedent: `yalue/onnxruntime_go` in `pogo`, `FlashSR`, `go-autoresearch`, `Emanetics`. |
| D2  | **Pure-Go tokenizer, no CGO**                                                   | User decision. Keeps the build CGO-free and cross-compilable. This is the single largest correctness risk in the port (see §6 R1) and is therefore front-loaded: golden corpus before implementation.                                                                                                                                                                                                                         |
| D3  | **Inference only**                                                              | `proper_reward`, `td_lambda_targets` and `collate_items` are RLCD training math a Go port cannot use. Deliberate API narrowing; document it in the README.                                                                                                                                                                                                                                                                    |
| D4  | **Python moves to `original/`**                                                 | Same pattern as `go-pocket-tts`. Upstream Python is frozen as the parity reference and drives `scripts/dump_python_parity.py`. Also keeps the Apache-2.0 derivative-work attribution honest.                                                                                                                                                                                                                                  |

**D2 has a consequence that needs resolving in Spike S2:** `yalue/onnxruntime_go` requires CGO (it
`dlopen`s the shared library _through_ cgo). A CGO-free tokenizer paired with a CGO ONNX binding
gives up the benefit. `go-pocket-tts` already uses `github.com/shota3506/onnxruntime-purego`, which
is CGO-free — but its own PLAN.md records a real defect: _"local ONNX-backed native parity tests can
panic inside `onnxruntime-purego` with `runtime.AddCleanup`"_. S2 decides between the two; default to
purego for consistency with D2, fall back to `yalue/onnxruntime_go` if purego proves unstable.

---

## 1. What we are porting — verified facts

All of the following was read from the live Hub repo, not inferred.

### 1.1 The three checkpoints

`convaiinnovations/laya` bundles all three; only the requested subfolder is downloaded.
Apache-2.0, **not gated, no HF token required**.

| Name            | Subfolder          | Encoder                        | Params      | `max_len` / `head_max_len` | Weights                                 |
| --------------- | ------------------ | ------------------------------ | ----------- | -------------------------- | --------------------------------------- |
| english         | _(root)_           | `answerdotai/ModernBERT-large` | 421,293,830 | 512 / 192                  | 842 MB, **fp16** (+ fp32 `temperature`) |
| multilingual    | `multilingual/`    | `jhu-clsp/mmBERT-base`         | 321,908,998 | 1024 / 256                 | 644 MB, fp16                            |
| typed-decisions | `typed-decisions/` | `answerdotai/ModernBERT-large` | 421,293,830 | 1024 / 256                 | 842 MB, all fp16                        |

Note `amp_dtype: "bf16"` in the config refers to autocast during compute — the **stored** weights are fp16.
The `temperature` tensor is F32 in two checkpoints and F16 in the third; a loader must not hardcode it.

### 1.2 Encoder architecture (ModernBERT-large, English + typed-decisions)

`model_type: modernbert` · hidden 1024 · 28 layers · 16 heads (head_dim 64) · intermediate 2624 ·
vocab 50368 · max_position_embeddings 8192.

- **RoPE only.** No positional-embedding tensor exists in the checkpoint. Every 3rd layer (0, 3, …, 27)
  is `full_attention` with `rope_theta=160000`; the rest are `sliding_attention`, window 128 (±64),
  `rope_theta=10000`. No ALiBi.
- **Bias-free LayerNorm** (`norm_bias: false`, eps 1e-5). Layer 0 has no `attn_norm` tensor.
- **GeGLU**, no MLP bias: `mlp.Wi [5248,1024]` is the fused gate+up (2 × 2624), `mlp.Wo [1024,2624]`.
- **Fused QKV**: `attn.Wqkv [3072,1024]`, `attn.Wo [1024,1024]`, no attention bias.
- Reference path uses `attn_implementation="sdpa"` with `reference_compile=False` — padded SDPA,
  no unpadding, no flash-attn.

**mmBERT-base deltas:** hidden 768, 22 layers, 12 heads, intermediate 1152, vocab 256000,
_both_ attention types use `rope_theta=160000`, `position_embedding_type: "sans_pos"`.

> ⚠️ `multilingual/encoder/config.json` says `cls_token_id: 1` while its `tokenizer_config.json` says
> `cls_token: "<bos>"` (id **2**). The reference code reads `tok.cls_token_id`, so **2** is what the
> weights were trained with. Match the tokenizer, not the encoder config.

### 1.3 The decision head (not part of the HF encoder)

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
act_head.2.{weight,bias} [2, 256]
temperature              [3]
```

> ⚠️ **The head FFN activation is ReLU, not GELU.** `nn.TransformerEncoderLayer`'s default activation
> is ReLU; the encoder body is GeGLU. This is the single easiest thing to get wrong in a reimplementation.
>
> ⚠️ The head layers are run manually (`for layer in self.head.layers`), which bypasses
> `nn.TransformerEncoder`'s optional final norm. With the default `norm=None` this is equivalent —
> but an ONNX export must reproduce the manual loop, not `nn.TransformerEncoder.forward`.

### 1.4 Tokenizers

Both checkpoints ship `tokenizer/tokenizer.json` + `tokenizer_config.json`, and **no** `tokenizer.model`,
`vocab.txt` or `merges.txt`. The `tokenizer.json` is the only source of truth.

**English (ModernBERT):** normalizer NFC · pre-tokenizer `ByteLevel{add_prefix_space:false, use_regex:true}` ·
BPE, vocab 50280, 50009 merges (new array form), `byte_fallback:false`, `unk:null` ·
116 added tokens, 109 of them non-special with `normalized:true` (runs of 1–24 spaces, ids 50254–50275) ·
**UNK 50280, CLS 50281, SEP 50282, PAD 50283, MASK 50284** (mask has `lstrip:true`).

**Multilingual (Gemma/mmBERT):** normalizer `Replace{" " → "▁"}` · pre-tokenizer
`Metaspace{replacement:"▁", prepend_scheme:"always", split:true}` · BPE, vocab 256000, 580604 merges,
**`byte_fallback:true`, `fuse_unk:true`** · 249 added tokens ·
**PAD 0, EOS/SEP 1, BOS/CLS 2, UNK 3, MASK 4**.

**Two semantics the Go tokenizer must get right, from config and not by convention:**

1. `tok(" " + text)` — which `laya/common.py:68` does for every option — behaves _oppositely_ in the two
   checkpoints. On multilingual, the `Replace` normalizer turns `" "` into `"▁"` before Metaspace runs,
   so the text already starts with `▁` and HF's `if !starts_with(replacement)` guard **suppresses the
   prepend**: `tok(" x") == tok("x")`. On English (ByteLevel, `add_prefix_space:false`) the leading
   space is real and becomes `Ġ`: `tok(" x") != tok("x")`.
2. Added tokens are pre-split out of the input **before** normalizer and pre-tokenizer run. Since
   `build_sequence` feeds `serialize_state(state)` — arbitrary user JSON — through the tokenizer,
   indentation runs are realistic input and the 109 whitespace-run added tokens are load-bearing.

### 1.5 Prior art that can be reused in-house

- `../go-pocket-tts/internal/safetensors/` — reader/store/writer.
- `../go-pocket-tts/internal/runtime/tensor/` — `MatMul`, `Linear`, `LayerNorm`, `Softmax`, plus
  hand-written amd64/arm64 assembly for `dot` and `axpy`.
- `../go-pocket-tts/internal/runtime/ops/` — `Attention` (incl. fused 4D, parallel), `RoPE`, `MLP`.
- `../go-pocket-tts/internal/tokenizer/sentencepiece_bytes.go` — byte-fallback handling.
- `../go-pocket-tts/scripts/dump_python_parity.py` + `internal/*/parity*.go` — the parity-harness pattern
  this plan copies wholesale.

These make Milestone M8 (the pure-Go native backend) far more tractable than a from-scratch estimate suggests.

---

## 2. Target repository layout

```
go-laya/
  go.mod                       module github.com/MeKo-Christian/go-laya
  justfile  treefmt.toml  .golangci.yml
  PLAN.md  README.md  LICENSE  NOTICE
  laya.go        Agent, Open, SystemOne, Result/Answer
  question.go    Question interface + Choice/Score/Noul
  router.go      Router, RouteDecision, model registry, LRU
  options.go     functional options
  errors.go      sentinel errors
  lang/          script + language detection      (public, zero deps)
  mailtext/      email cleaning                   (public, zero deps)
  presets/       the five question presets        (public, zero deps)
  jsonx/         ordered JSON with Python byte-parity
  tokenizer/     Tokenizer interface + pure-Go tokenizer.json loader
  internal/prompt/   render_options, render_criterion, build_sequence
  internal/calib/    softmax, entropy confidence, temperature, py-round
  internal/hub/      HF resolve + local cache
  internal/backend/  Backend interface, ONNX impl, (later) native impl
  cmd/laya/          optional CLI
  scripts/dump_python_parity.py
  scripts/export_onnx.py
  testdata/          golden vectors (checked in; CI never needs Python)
  original/          frozen upstream Python, reference only
```

`lang`, `mailtext` and `presets` have **no dependency on the runtime**, so a user who only wants
routing or email cleaning never links ONNX Runtime. That is the biggest structural win over the Python
layout, where `import laya` drags in torch.

---

## 3. Spikes — do these before writing library code

These can invalidate the plan. Budget 1–2 days total.

### Spike S1: Does `torch.onnx.export` survive the full `DecisionModel`?

**Files:** create `scripts/export_onnx.py`, `original/` must already exist (Task 1.3).

ModernBERT decorates functions with `torch.compile`, which `torch.onnx.export(dynamo=False)` rejects
(huggingface/transformers#35545, still open). `optimum` works around it with a
`DisableCompileContextManager` and by forcing `attn_implementation="eager"`.

1. Build the model with `reference_compile=False` **and** `attn_implementation="eager"`.
2. Export at opset ≥17 with dynamic axes `{batch, seq}` for `input_ids`/`attention_mask` and
   `{batch, k}` for `marker_pos`/`marker_mask`.
3. Verify the tail exports: `torch.gather`, `topk(2)`, `masked_fill(-1e4)`. Try `dynamo=True` if not.
4. Repeat for **all three** checkpoints. No ModernBERT-_large_ ONNX exists on `onnx-community`;
   only base variants are published, so large is unproven.

**Exit criteria:** three `.onnx` files that load in ORT and produce `logits`/`act_logits` for a batch.

**Fallback if export fails:** a third-party export already exists — `sevenreasons/laya-onnx-fp16`
(Apache-2.0, 846 MB, inputs `input_ids i64[B,S]`, `attention_mask i64[B,S]`, `marker_pos i64[B,K]`,
`marker_mask bool[B,K]`, `qtype i64[B]`; outputs `logits[B,K]`, `act_logits[B,2]`; dynamic B/S/K;
claims max logits diff 0.00416 vs PyTorch). Also `Mattepiu/laya-onnx` (fp32 + int8). Use one to unblock
M6 while fixing our own exporter — but **never ship a third-party artifact as the default**: verify it
ourselves or export our own.

### Spike S2: Which ONNX binding — and does CGO-free hold?

**Exit criteria:** a Go program that loads the S1 export and runs one forward pass, built with
`CGO_ENABLED=0` if using `shota3506/onnxruntime-purego`. Record whether the
`runtime.AddCleanup` panic that `go-pocket-tts` hit reproduces. If it does, switch to
`yalue/onnxruntime_go` and record in this file that D2's CGO-free promise covers the tokenizer only.

### Spike S3: Latency on real hardware

FLOP-derived estimate: ≈360 GFLOP for ModernBERT-large at 512 tokens; ≈256 GFLOP for mmBERT-base at 1024. Measured on comparable hardware, gonum `Sgemm` reaches ~27–32 GFLOPS and 2-thread OpenBLAS
~54 GFLOPS — which puts a _hand-written_ pure-Go forward pass at roughly **12 s/sequence**, against the
README's 33 ms on a T4. ORT-CPU with MLAS should land in the 0.3–5 s range depending on cores.

**Measure before promising "CPU-first" anywhere in the README.** If too slow, the levers are int8
dynamic quantization (typically 2–4×) or defaulting the Router to mmBERT-base. If int8 is used,
measure **ECE and Brier, not just accuracy** — this model's whole selling point is calibrated
probabilities, and a calibration regression will not show up in an argmax test.

---

## 4. Milestones and tasks

Each task is TDD: write the failing test, watch it fail, implement minimally, watch it pass, commit.
Run `just check` before every commit.

### M0 — Scaffolding

**Task 0.1: Branch.**

```bash
git checkout -b feat/go-port
```

**Task 0.2: Go module + house tooling.**

- Create: `go.mod` (`module github.com/MeKo-Christian/go-laya`, `go 1.26`), `justfile`, `treefmt.toml`, `.golangci.yml`.
- Copy `treefmt.toml` and `.golangci.yml` from `../algo-fft/`; copy the justfile targets
  `build test test-race lint lint-fix fmt fmt-check cover check` from `../algo-fft/justfile`
  and `check-tidy`/`ci` from `../go-pocket-tts/justfile`.
- Verify: `just fmt-check && just lint` on an empty module.

**Task 0.3: Freeze the Python upstream.**

```bash
git mv laya original/laya
git mv tests original/tests
git mv setup.py pyproject.toml notebooks original/
git commit -m "chore: freeze upstream Python under original/ as the port's parity reference"
```

Keep `LICENSE` at the root. Add a `NOTICE` recording that this is a derivative of
`NandhaKishorM/laya` © Convai Innovations, Apache-2.0, and what was modified.

**Task 0.4: Replace CI.**

- Modify: `.github/workflows/ci.yml` → `go test ./... -race`, `go vet`, `golangci-lint`, `just check-tidy`,
  matrix over `ubuntu-latest`/`macos-latest`, Go 1.26.
- Modify: `.github/dependabot.yml` → `gomod` instead of `pip`.
- Modify: `.github/workflows/security.yml` — **keep the policy, translate the mechanism.** The upstream
  greps `laya/` for `pickle`, `torch.load(`, `eval(`, `subprocess.` because laya downloads checkpoints
  from the Hub and a pickle path would be RCE on `laya.load("someone/their-model")`. The Go equivalents:
  forbid `encoding/gob` and `os/exec` on downloaded artifacts, verify safetensors/ONNX headers before
  use, pin the ORT version, and check each download's ETag/sha against Hub metadata. Keep both gitleaks
  scans (PR diff **and** working tree — a PyPI token once survived in `main`; see commit `31280fc`).
- Modify: `.github/workflows/release.yml` → keep the tag-vs-version gate, drop PyPI.

**Task 0.5: Single version constant.** Python keeps `0.3.4` in three files. Go: one
`const Version = "0.1.0"` in `laya.go`, asserted against the tag in `release.yml`.

### M1 — The Python reference harness

Nothing downstream can be trusted without this, and CI must never need it.

**Task 1.1: Reference env.** The system Python's torch is broken
(`libtorch_global_deps.so: cannot open shared object file`). Do not repair it; create an isolated one:

```bash
uv venv --python 3.12 .venv-ref
. .venv-ref/bin/activate
uv pip install torch --index-url https://download.pytorch.org/whl/cpu
uv pip install transformers safetensors huggingface_hub numpy onnx onnxruntime
uv pip freeze > scripts/requirements-ref.txt   # the versions ARE part of the contract
```

Add `.venv-ref/` to `.gitignore`.

**Task 1.2: Fetch the checkpoints.** ~2.4 GB, anonymous, no token:

```bash
export LAYA_MODELS=${LAYA_MODELS:-$HOME/laya_models}
python -c "from huggingface_hub import snapshot_download as d; d('convaiinnovations/laya', local_dir='$LAYA_MODELS/laya')"
```

Disk check: 549 GB free on `/mnt/projekte`.

**Task 1.3: `scripts/dump_python_parity.py`.** Modelled on `../go-pocket-tts/scripts/dump_python_parity.py`.
Emits into `testdata/`, and records `transformers.__version__` / `tokenizers.__version__` in every
file header:

- `testdata/tokenizer_{en,ml}.jsonl` — `{text, ids, ids_with_leading_space, tokens}` per case.
- `testdata/sequence.jsonl` — `(state, question, max_len, head_max_len, option_order, truncate_left) → (ids, markers)`.
- `testdata/render.jsonl` — `render_criterion` / `render_options` / `serialize_state` inputs and outputs.
- `testdata/answers.jsonl` — `(logits, act_logits, temperature, k, qtype, criteria) → answer JSON`.
- `testdata/logits.jsonl` — end-to-end `(state, questions) → (logits, act_logits)` for ~50 fixed inputs.

**Task 1.4: Provenance test.** `TestGoldenProvenance` asserts the recorded tokenizer version matches
what the Go implementation claims to target. A silent upstream tokenizer change is the realistic way
parity regresses.

**Commit:** `feat(testdata): golden parity vectors generated from upstream Python`

### M2 — Tier-1 core (no ML runtime, ~620 lines of Python)

> **Three Python-isms will silently corrupt results if translated naively.** Do M2 in this order.

**Task 2.1: `jsonx` — Python-compatible JSON.** _Do this first; everything downstream depends on it._

`encoding/json` gets all three of these wrong: it emits `{"a":1}` instead of `{"a": 1}`, it **sorts map
keys**, and it **HTML-escapes** `<`, `>`, `&` by default. The Python code feeds
`json.dumps(state, ensure_ascii=False)` straight into the tokenizer, so any of the three changes the
bytes the model sees.

- Create: `jsonx/jsonx.go`, `jsonx/jsonx_test.go`.
- `type Obj []Field` with `Field{Key string; Value any}` — an _ordered_ object.
- `Obj.MarshalJSON` reproduces `json.dumps(x, ensure_ascii=False)`: separators `", "` and `": "`,
  no key sorting, no HTML escaping, non-ASCII emitted literally.
- `Compact(v any) string` reproduces `json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)`,
  falling back to `fmt.Sprintf("%v", v)` for anything `encoding/json` refuses (Python's `default=str`).
- `Round4(x float64) float64` — **Python's `round()` is half-to-even on the exact binary double; Go's
  `math.Round` is half-away-from-zero.** Implement as
  `strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)` and test the tie cases (`0.00125`, `2.5e-5`).
- Test against `testdata/render.jsonl`.

**Task 2.2: `lang`.** Port `original/laya/lang.py` verbatim. Invariants §5 items 54–65.
Go-specific traps:

- `detect_script`'s tie-break depends on Python dict _encounter_ order, with `counts["latin"]` assigned
  **after** the loop. Go map iteration is randomized — use an explicit ordered counter or the result is
  nondeterministic.
- `_STOP`'s insertion order (fr, de, es, pt, it, nl) is the language tie-break. Use an ordered slice.
- Python's `[^\W\d_]+` is Unicode-aware; Go RE2's `\w` is ASCII-only. Split on `unicode.IsLetter` instead.
- `_SCRIPT_RANGES` order is load-bearing (hangul before kana before han), first match wins.
- Tests: port all 14 script cases, 7 `is_english` cases, 6 language-guess cases and 5 flattening cases
  from `original/tests/test_router.py:29-86`.

**Task 2.3: `mailtext`.** Port `original/laya/email.py`. Invariants §5 items 66–74. Note
`email.email_questions` is a **dead byte-identical duplicate** of `presets.email_questions` — port only
the presets one. Currently untested upstream; write the tests this port deserves.

**Task 2.4: `presets`.** Port all five preset constructors from `original/laya/presets.py`. Pure data.
Add the shape test upstream never had: every preset round-trips through question validation.

**Task 2.5: `question.go` + `internal/prompt/render.go`.** `render_options` / `render_criterion`.
Invariants §5 items 14–18. The Go type design (§7) collapses Python's dict-or-list `criteria` into
`[]ChoiceOption`. Port `original/tests/test_criteria.py:33-100` as a table test — but **drop**
`test_criteria.py:103-116`, which uses `inspect.getsource` to assert five string literals are present
in `Agent.__init__`; replace it with a behavioural test on the Go device-fallback policy.

**Commit after each task.**

### M3 — Router (pure, no weights, no network)

**Task 3.1: `RouteDecision` + model registry.** Invariants §5 items 39–47.
Two upstream inconsistencies to handle deliberately:

- `router.py:266` emits `repo` as a raw `(repo, subfolder)` tuple on the auto-workflow branch while every
  other branch emits a string via `_repo_str()`. Through `json.dumps` that surfaces as a JSON array.
  **Decision: always emit the string; document the deviation.**
- `workflow` is `None` on the model/task paths but carries the detected workflow name on the **lang and
  detection** paths even though it did not drive the decision. Reproduce this.

**Task 3.2: `Router` + LRU.** Invariants §5 items 48–53. Python's `Router` is not concurrency-safe;
Go's must be (`sync.Mutex`). Expose the loader as an injectable hook —
`WithLoader(func(ctx, name, ModelSpec) (*Agent, error))` — so the upstream LRU/preload tests, which
monkeypatch `rr.load`, stay expressible without reflection.

**Task 3.3: Port the whole of `original/tests/test_router.py`** (~90 assertions) plus section 1 of
`test_local_e2e.py:46-67`, which is pure routing. Every case ports as-is; the file states outright
"No model weights are loaded: `Router.route` is pure."

**Milestone check:** at this point `lang` + `mailtext` + `presets` + `Route` is a genuinely useful Go
library with zero ML dependencies, and it covers everything the upstream test suite actually tests.

### M4 — Pure-Go tokenizer ⚠️ highest risk

**Task 4.1: The interface first.**

```go
package tokenizer

type Tokenizer interface {
    Encode(text string) []uint32 // add_special_tokens = false
    MaskToken() string
    MaskID() uint32
    CLSID() uint32
    SEPID() uint32
    PADID() uint32
}
```

`build_sequence` then ports 1:1 and stays backend-agnostic, which keeps D2 reversible.

**Task 4.2: Special-token resolution.** ~40 lines: read token _strings_ from `tokenizer_config.json`,
resolve ids from `tokenizer.json`'s `added_tokens` + `model.vocab`. Prefer the tokenizer config over the
encoder config (see §1.2's CLS-id warning). Ignore the Gemma `extra_special_tokens`-as-a-list quirk that
`_fix_tokenizer_config` patches around in Python — Go never needs the workaround.

**Task 4.3: Vendor and fix `gomlx/go-huggingface/tokenizers/hftokenizer`.** Apache-2.0, actively
maintained, and already the pure-Go backend of `hugot`. Its scaffolding is the best available
(normalizers NFC/NFD/NFKC/NFKD/Replace/Prepend/Sequence/BertNormalizer, pre-tokenizers
ByteLevel/Metaspace/Split/Sequence/Punctuation, BPE/WordPiece/Unigram, both merge serializations).
**Three bounded defects must be fixed, all upstreamable:**

1. `pretokenizer.go:342` checks `text[0] != ' '` where HF checks `!starts_with(replacement)`. With the
   Gemma `Replace` normalizer there are no spaces left in the text, so it **always** prepends and
   `"▁foo"` becomes `"▁▁foo"` — drift on every multilingual encode, doubly so on `tok(" " + opt)`.
   Fix to `!strings.HasPrefix(text, replacement)`.
2. `bpeTokenizeWithSpans` parses `ByteFallback`/`FuseUnk` into the struct and then never uses them.
   Unknown symbols go to `unkID` or are **silently dropped** when `unkID < 0`. mmBERT needs both;
   without byte fallback, every emoji and rare CJK character silently becomes `<unk>` (id 3) — on
   precisely the inputs `laya-multilingual` exists for.
3. Added-token matching is "longest-first greedy" rather than HF's `AddedVocabulary`, with no
   `single_word`/`lstrip`/`rstrip`/`normalized` distinction. Both checkpoints have a `lstrip:true` mask
   token and ≥109 `normalized:true` added tokens.

**Do not use `sugarme/tokenizer`:** `pretrained/model.go`'s `createBPE()` has `byte_fallback` and
`fuse_unk` _commented out_ and `bpe.New` has no parameters for them, plus an open panic
(issue #78) on consecutive whitespace in the Metaspace pre-tokenizer — a crash on realistic input.

**Task 4.4: Golden corpus, both checkpoints.** Assert **ids and token strings** — an id diff alone tells
you nothing, `["▁▁","x"]` vs `["▁x"]` tells you exactly which stage broke. Required cases:

- `""`, `" "`, `"  "`, `"\n"`, `"\t"`, `"\r\n"`, `"   \n   "` — consecutive whitespace is the most
  common drift/crash site.
- **`text` and `" " + text` for every case** — this is the `common.py:68` semantic and it differs
  between the two checkpoints (§1.4).
- Compact JSON exactly as `serialize_state` emits it, with nested braces, escapes, non-ASCII, long arrays.
- The mask literal (`[MASK]` / `<mask>`) appearing inside user text.
- Every added token bare and mid-sentence: `<unused0>`, `<start_of_turn>`, `<2mass>`, `[@BOS@]`,
  `|||IP_ADDRESS|||`, and runs of 2–24 spaces (EN ids 50254–50275).
- Normalization: precomposed vs decomposed `é`, fullwidth `Ａ`, ligature `ﬁ`, NBSP, ZWJ/ZWSP, BOM,
  combining marks. EN normalizes NFC; ML does **not** normalize beyond space→`▁`, so these two
  legitimately differ.
- Emoji with ZWJ sequences, skin tones, flags — these exercise byte fallback on ML.
- Devanagari, Arabic, CJK, Thai, Cyrillic, Korean jamo vs precomposed.
- Lone surrogates / invalid UTF-8 (Go tolerates, Python `str` does not — decide and document the
  boundary; sanitize at the API edge).

**Task 4.5: Fuzz + differential.** A Go fuzz target checking invariants only (no panic, ids < vocab_size,
ASCII round-trip). Separately, a one-off differential run of ~100k lines of real multilingual corpus
through both Python and Go, diffing the id streams. That is what actually finds the
metaspace/added-token bug classes.

**Gate: M5 does not start until both golden corpora are 100 % green.**

### M5 — `build_sequence`

**Task 5.1: Port `common.py:49-86` into `internal/prompt/sequence.go`.**
This is 38 lines carrying most of the porting risk, and it has **no test at all upstream**.
Invariants §5 items 1–13.

**Task 5.2: Assert against `testdata/sequence.jsonl` byte-for-byte**, across a matrix of
`(qtype, criteria, instructions, state, max_len, head_max_len, option_order, truncate_left)`.
This catches the truncation arithmetic — `opt_budget`, the `< 16` fallback, `[:48]`, `[-room:]` vs
`[:room]`, the final `ids[:max_len]` and the marker filter — independently of tokenizer drift.

### M6 — Backend + checkpoint loading

**Task 6.1: `internal/backend.Backend` interface.**

```go
type Backend interface {
    Forward(ctx context.Context, in Batch) (logits [][]float32, act [][]float32, err error)
    Close() error
}
type Batch struct {
    InputIDs, AttentionMask [][]int64
    MarkerPos               [][]int64
    MarkerMask              [][]bool
    QType                   []int64
}
```

Test with an in-memory fake replaying `testdata/logits.jsonl` — every M7 test then runs without ORT.

**Task 6.2: `internal/hub` — HF resolve + cache.** `https://huggingface.co/{repo}/resolve/{rev}/{path}`,
optional `Authorization: Bearer $HF_TOKEN`, ETag/sha verification, a local cache under
`$LAYA_CACHE` or `os.UserCacheDir()/laya`, `allow_patterns`-equivalent so a subfolder request downloads
only that subfolder (the bundle repo is 2.4 GB; the English checkpoint alone is 846 MB).
Everything cancellable via `context.Context`.

**Task 6.3: ONNX backend.** Per Spike S2's binding decision. Dynamic batch/seq/k. Device selection
(`cpu`/`cuda`/`coreml`) with a logged fallback, replacing Python's three `print()` warnings with `slog`.

**Task 6.4: Checkpoint validation.** Port `_verify_compatibility`'s intent: require cfg keys `encoder`
and `head_layers`, require the graph's declared inputs/outputs, and fail with a wrapped
`ErrIncompatibleCheckpoint` naming what was wrong.

### M7 — Agent, calibration, end-to-end parity

**Task 7.1: `internal/calib`.** Invariants §5 items 22–29: `temp_bucket`, the
`temperature_by_options` → `temperature[qtype]` lookup, the `max(1e-3, t)` floor, max-subtracted
softmax over exactly the first _k_ logits, entropy confidence, and `Round4`.

**Task 7.2: `Agent.SystemOne`.** Invariants §5 items 19–34. Watch the three answer shapes: a `noul`
answer has **no** `probabilities` and no `legend`, and its confidence is `max(p1, 1-p1)` — **not** the
entropy formula, which for k=2 genuinely disagrees. `score` is `Σ i·p[i]`, an expectation, not an
argmax. `legend` carries the **raw** criterion value, not the rendered option text.

**Task 7.3: Answer-formatting parity** against `testdata/answers.jsonl` — runs against the fake
backend, so it needs no model.

**Task 7.4: End-to-end parity** against `testdata/logits.jsonl`, gated behind `testing.Short()` and a
`LAYA_MODELS` env var. Assert per-option probabilities within 1e-4 **and** that the argmax decision
never flips. Then port sections 2–5 of `original/tests/test_local_e2e.py` (loose directional
thresholds: ≥6/8 land on `billing`, ≥2/3 on the preset checks).

**Task 7.5: README + examples.** Port every README example to Go. Fix the image URLs — upstream's point
at `raw.githubusercontent.com/NandhaKishorM/laya/main/...`, so a fork's README silently renders
upstream's assets. Document the deliberate deviations: no training symbols (D3), `repo` always a string,
`Detection.ScriptProfile` as a map, `instructions` as `string` only.

### M8 — Pure-Go native backend (after 1.0)

Same `Backend` interface, no API change. Implement ModernBERT + mmBERT + the head over safetensors,
lifting `tensor`, `ops` and `safetensors` from `../go-pocket-tts`. Required pieces not already there:
bias-free LayerNorm, GeGLU, sliding-window attention masks, per-layer-type RoPE theta, fused QKV
unpacking, and the **ReLU** head FFN (§1.3). Gate promotion on the same golden vectors plus a
latency benchmark (see Spike S3 — this is where the 12 s/forward estimate has to be beaten).

---

## 5. Invariants a Go test must assert

The full numbered checklist (74 items) is in **`docs/INVARIANTS.md`**. Summary of the sections, with
the Python line ranges each cites:

| Items | Area                                                       | Source              |
| ----- | ---------------------------------------------------------- | ------------------- |
| 1–13  | `build_sequence` token budget, layout, markers, truncation | `common.py:49-86`   |
| 14–18 | `render_options` / `render_criterion` / `serialize_state`  | `common.py:21-46`   |
| 19–34 | `Agent.system_one` numerics and answer formatting          | `agent.py:229-343`  |
| 35–38 | Forward pass, only if the head is reimplemented            | `common.py:105-126` |
| 39–47 | `Router.route` precedence and reason strings               | `router.py:241-290` |
| 48–53 | Router LRU lifecycle                                       | `router.py:144-238` |
| 54–65 | `lang` script/language detection                           | `lang.py`           |
| 66–74 | `mailtext` cleaning                                        | `email.py`          |

The four that cause silent wrong answers rather than loud failures, and therefore need tests first:

- **#18** `serialize_state` byte-parity — Go's `encoding/json` breaks separators, key order _and_
  HTML escaping.
- **#29** `round()` half-to-even vs Go's half-away-from-zero.
- **#56 / #63** Python dict iteration order as a tie-break in `detect_script` and `guess_latin_language`.
- **#26** `noul` confidence is `max(p1, 1-p1)`, not the entropy formula used by the other two types.

---

## 6. Risks

| #   | Risk                                                                                                                                                                                          | Mitigation                                                                                                                                                                                                                                                                 |
| --- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **Pure-Go tokenizer drift.** Marker positions are token indices, so one off-by-one silently corrupts every decision. Both candidate libraries have demonstrable bugs on exactly laya's paths. | The `Tokenizer` interface (Task 4.1) keeps the decision reversible. Golden corpus + fuzz + a 100k-line differential run gate M5. If parity cannot be reached, `daulet/tokenizers` (CGO, wraps the same Rust crate Python uses — parity by construction) is a one-day swap. |
| R2  | **`torch.onnx.export` fails on ModernBERT-large.** transformers#35545 is still open; no ModernBERT-large ONNX is published anywhere.                                                          | Spike S1 up front. `sevenreasons/laya-onnx-fp16` unblocks development while our exporter is fixed.                                                                                                                                                                         |
| R3  | **CPU latency ≫ the README's 33 ms.**                                                                                                                                                         | Spike S3 measures before anything is promised. Levers: int8 dynamic quantization, defaulting the Router to mmBERT-base, GPU execution providers.                                                                                                                           |
| R4  | **int8 quantization wrecks calibration.** This model's whole value is calibrated probabilities.                                                                                               | Measure ECE and Brier, not accuracy. Never quantize by default.                                                                                                                                                                                                            |
| R5  | **`onnxruntime-purego` instability** — `go-pocket-tts` hit a `runtime.AddCleanup` panic in parity tests.                                                                                      | Spike S2 reproduces or clears it; `yalue/onnxruntime_go` is the fallback, at the cost of CGO.                                                                                                                                                                              |
| R6  | **Upstream tokenizer/transformers version drift** silently changes golden vectors.                                                                                                            | `TestGoldenProvenance` (Task 1.4); `scripts/requirements-ref.txt` pinned; regeneration is a reviewed diff.                                                                                                                                                                 |
| R7  | **Supply chain.** laya downloads checkpoints from the Hub; `laya.Open("someone/their-model")` must not be RCE.                                                                                | Carry the upstream security policy over (Task 0.4): verify ONNX/safetensors headers, ETag/sha checks, no `os/exec` or `encoding/gob` on downloaded artifacts.                                                                                                              |
| R8  | **Licensing.** This is a derivative of an Apache-2.0 work.                                                                                                                                    | Preserve `LICENSE`, add `NOTICE` with the original copyright and a statement of modification (Task 0.3). Weights are Apache-2.0 and ungated.                                                                                                                               |

---

## 7. Proposed Go API

Full type definitions — `Obj`/`Probs`/`Questions`/`AnswerSet` as ordered slices, the `State` interface,
`ChoiceOption`, the three `Answer` shapes, `Agent`, `Router`, functional options and sentinel errors —
are in **`docs/API.md`**, together with the exact JSON shapes the Python version emits.

The dynamic-typing decisions Python leaves implicit:

| Python                                                      | Go decision                                                                                                                                                                 |
| ----------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `criteria` on choice: `dict` **or** `list[str]`             | `[]ChoiceOption{Key, Desc any}` + a `Labels("a","b")` helper. One representation, order preserved.                                                                          |
| `criteria` on noul: dict with `"true"`/`"false"`, or absent | Explicit `True`/`False` fields — removes any chance of getting index order wrong (false=0, true=1 always).                                                                  |
| `state`: `str \| dict \| list`                              | A `State` interface with `TextState` / `ObjState` / `ListState`. Rejected `any`+reflection: it makes both the serialization and the detection flattening implicit.          |
| `instructions`: `str` or anything                           | `string` only; callers serialize. Document the dropped edge case (non-str instructions are `json.dumps`'d with `ensure_ascii=True` — the one place laya escapes non-ASCII). |
| dict iteration order                                        | Ordered slices everywhere it is observable; `map` only where it provably is not.                                                                                            |
| `RouteDecision` as a `dict` subclass                        | A struct with JSON tags, plus `Map() Obj` for anyone who wants the dict.                                                                                                    |

---

## 8. Definition of done for 1.0

- [ ] `just check` green; `go test ./... -race` green on linux/amd64 and darwin/arm64.
- [ ] Both tokenizer golden corpora 100 % id- and token-identical to Python.
- [ ] `testdata/sequence.jsonl` byte-identical.
- [ ] End-to-end probabilities within 1e-4 of PyTorch on all three checkpoints; **zero argmax flips**.
- [ ] `lang` + `mailtext` + `presets` + `Route` importable with no ML dependency — verified by a build
      that imports only those packages and links no ONNX symbols.
- [ ] Every README example compiles and runs.
- [ ] Measured latency published in `BENCHMARKS.md` for the hardware actually tested, replacing
      upstream's T4 numbers rather than repeating them. (Upstream's `BENCHMARKS.md` cites
      `research/results/*.json`, and that directory does not exist in the repo — the raw numbers are
      not reproducible from what is checked in. Do not inherit that.)
