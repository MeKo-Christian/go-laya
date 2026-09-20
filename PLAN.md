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

## Status at a glance

Tick a box only when the work is committed and `just check` is green. `[~]` means started but
not finished; keep it rare. A milestone is done when every task box under it is ticked.

| Milestone                                                              | Delivers                                         | Status               |
| ---------------------------------------------------------------------- | ------------------------------------------------ | -------------------- |
| [M0 — Scaffolding](#m0--scaffolding)                                   | Go module, tooling, CI, frozen Python, `Version` | ✅ 5/6 (0.1 skipped) |
| [Spikes S1–S3](#3-spikes--do-these-before-writing-library-code)        | ONNX export, binding choice, latency floor       | ⬜ not started       |
| [M1 — Reference harness](#m1--the-python-reference-harness)            | `testdata/*.jsonl` golden vectors                | ⬜ not started       |
| [M2 — Tier-1 core](#m2--tier-1-core-no-ml-runtime-620-lines-of-python) | `jsonx`, `lang`, `mailtext`, `presets`, render   | ⬜ not started       |
| [M3 — Router](#m3--router-pure-no-weights-no-network)                  | `Route`, model registry, LRU                     | ⬜ not started       |
| [M4 — Tokenizer](#m4--pure-go-tokenizer-highest-risk)                  | pure-Go `tokenizer.json` loader ⚠️               | ⬜ not started       |
| [M5 — `build_sequence`](#m5--build_sequence)                           | prompt assembly + marker positions               | ⬜ not started       |
| [M6 — Backend](#m6--backend--checkpoint-loading)                       | `Backend` iface, hub cache, ONNX impl            | ⬜ not started       |
| [M7 — Agent + parity](#m7--agent-calibration-end-to-end-parity)        | `SystemOne`, calibration, e2e parity, README     | ⬜ not started       |
| [M8 — Native backend](#m8--pure-go-native-backend-after-10)            | safetensors ModernBERT/mmBERT (post-1.0)         | ⬜ deferred          |

**Critical path:** S1 → S2 → M1 → M4 → M5 → M6 → M7. M2 and M3 are independent of every spike and of
the tokenizer, so they can proceed in parallel with the spikes and are the fastest route to something
useful (see the M3 milestone check).

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

These can invalidate the plan. Budget 1–2 days total. **Status:** S1 ⬜ · S2 ⬜ · S3 ⬜

### Spike S1: Does `torch.onnx.export` survive the full `DecisionModel`?

**Files:** create `scripts/export_onnx.py`, `original/` must already exist (Task 0.3).

ModernBERT decorates functions with `torch.compile`, which `torch.onnx.export(dynamo=False)` rejects
(huggingface/transformers#35545, still open). `optimum` works around it with a
`DisableCompileContextManager` and by forcing `attn_implementation="eager"`.

- [ ] **S1.1** Write `scripts/export_onnx.py`; build the model with `reference_compile=False` **and**
      `attn_implementation="eager"`.
- [ ] **S1.2** Export at opset ≥ 17 with dynamic axes `{batch, seq}` for `input_ids`/`attention_mask`
      and `{batch, k}` for `marker_pos`/`marker_mask`.
- [ ] **S1.3** Verify the tail exports: `torch.gather`, `topk(2)`, `masked_fill(-1e4)`. Try
      `dynamo=True` if it does not.
- [ ] **S1.4** Reproduce the manual head loop (§1.3), not `nn.TransformerEncoder.forward`, and confirm
      the head FFN activation in the exported graph is **ReLU**.
- [ ] **S1.5** Repeat for **all three** checkpoints. No ModernBERT-_large_ ONNX exists on
      `onnx-community` — only base variants are published, so large is unproven.
- [ ] **S1.6** Record the working export recipe (flags, opset, `torch`/`transformers`/`onnx` versions)
      in `scripts/export_onnx.py`'s header and in this file.

**Exit criteria**

- [ ] Three `.onnx` files load in ORT and produce `logits`/`act_logits` for a batch.
- [ ] Max abs logit diff vs PyTorch recorded per checkpoint.

**Fallback if export fails:** a third-party export already exists — `sevenreasons/laya-onnx-fp16`
(Apache-2.0, 846 MB, inputs `input_ids i64[B,S]`, `attention_mask i64[B,S]`, `marker_pos i64[B,K]`,
`marker_mask bool[B,K]`, `qtype i64[B]`; outputs `logits[B,K]`, `act_logits[B,2]`; dynamic B/S/K;
claims max logits diff 0.00416 vs PyTorch). Also `Mattepiu/laya-onnx` (fp32 + int8). Use one to unblock
M6 while fixing our own exporter — but **never ship a third-party artifact as the default**: verify it
ourselves or export our own.

- [ ] **S1.F** (only if S1 fails) Pull the third-party export, verify its I/O signature and logit diff
      ourselves, mark it dev-only in code, and open a tracking issue to replace it.

### Spike S2: Which ONNX binding — and does CGO-free hold?

- [ ] **S2.1** Write a throwaway Go program that loads the S1 export and runs one forward pass with
      `shota3506/onnxruntime-purego`, built with `CGO_ENABLED=0`.
- [ ] **S2.2** Record whether the `runtime.AddCleanup` panic `go-pocket-tts` hit reproduces — run it
      under `-race` and in a loop, since it is a finalizer race and will not show on a single pass.
- [ ] **S2.3** If it reproduces, repeat S2.1 with `yalue/onnxruntime_go` and record in this file that
      D2's CGO-free promise covers **the tokenizer only**.
- [ ] **S2.4** Pin the chosen binding _and_ the ORT shared-library version (R7 requires a pinned ORT);
      note how the library is located at runtime.
- [ ] **S2.5** Write the decision, with the evidence, into §0 as D5.

**Exit criteria:** one forward pass green on the chosen binding, with the CGO answer recorded.

### Spike S3: Latency on real hardware

FLOP-derived estimate: ≈360 GFLOP for ModernBERT-large at 512 tokens; ≈256 GFLOP for mmBERT-base at 1024. Measured on comparable hardware, gonum `Sgemm` reaches ~27–32 GFLOPS and 2-thread OpenBLAS
~54 GFLOPS — which puts a _hand-written_ pure-Go forward pass at roughly **12 s/sequence**, against the
README's 33 ms on a T4. ORT-CPU with MLAS should land in the 0.3–5 s range depending on cores.

- [ ] **S3.1** Measure ORT-CPU latency per checkpoint at realistic sequence lengths (512 / 1024) and
      thread counts, on the hardware we actually ship numbers for.
- [ ] **S3.2** Record the numbers, the hardware and the ORT build in `BENCHMARKS.md` — replacing
      upstream's T4 figures rather than repeating them.
- [ ] **S3.3** **Do not write "CPU-first" anywhere in the README until S3.2 exists.**
- [ ] **S3.4** If too slow: evaluate int8 dynamic quantization (typically 2–4×) and/or defaulting the
      Router to mmBERT-base.
- [ ] **S3.5** If int8 is evaluated, measure **ECE and Brier, not just accuracy** — this model's whole
      selling point is calibrated probabilities, and a calibration regression will not show up in an
      argmax test (R4). Never make quantization the default.

---

## 4. Milestones and tasks

Each task is TDD: write the failing test, watch it fail, implement minimally, watch it pass, commit.
Run `just check` before every commit.

### M0 — Scaffolding

**Status: done** (Task 0.1 deliberately skipped — the port is being built on `main`).

(2026-09-20) Landed as nine commits from `fee2c35`. Evidence: `just ci` exit 0 — treefmt 0 changed,
markdownlint clean, `go test -race` ok, golangci-lint 0 issues, check-tidy clean. Cross-build green on
linux/{amd64,arm64}, darwin/{amd64,arm64}, windows/amd64; `govulncheck` reports no vulnerabilities.

**Task 0.1: Branch.**

- [ ] _Skipped._ `git checkout -b feat/go-port` — work is happening directly on `main` instead.
      Revisit only if the port needs to be reviewable as one PR.

**Task 0.2: Go module + house tooling.**

- [x] Create `go.mod` (`module github.com/MeKo-Christian/go-laya`, `go 1.26`).
- [x] `treefmt.toml` based on `../go-pocket-tts/`'s, the broader of the two. **`../algo-fft/.golangci.yml`
      does not exist** — it is `.golangci.**toml**` — so go-laya's was written as YAML against schema v2
      (golangci-lint 2.12.2), keeping algo-fft's portable core and dropping its FFT-kernel exclusions.
- [x] justfile targets from both siblings, with two collisions resolved. They disagree on names —
      algo-fft has `fmt-check`/`check`, go-pocket-tts has `check-formatted`/`ci` — so both survive with
      distinct jobs: `check` is the developer loop (test + lint + cover), `ci` is the pipeline
      (fmt-check + lint-md + test-race + lint + check-tidy). algo-fft has no `test-race` at all (race is
      folded into `test`); go-pocket-tts's split was taken. algo-fft's `check` also chains `check-deps` →
      `scripts/release-guard.sh`, which has no analogue here and was dropped.
- [x] Verify: `just fmt-check && just lint`. `just lint` cannot pass on a genuinely empty module
      (`no go files to analyze`), so this was completed once Task 0.5 added `laya.go`.
- [x] Beyond the task: `lint-md` and `vuln` recipes, and `.gitignore` entries for `.venv-ref/`,
      coverage output and model artefacts.

**Three deliberate departures from the siblings.** (a) **markdownlint is not a treefmt formatter.**
algo-fft's CI pins `prettier@3.8.5` because ≥3.9 and `markdownlint --fix` disagree often enough that
`--fail-on-change` may never converge; the installed prettier is 3.9.6. markdownlint runs as a checker
instead (`just lint-md`). Every formatter is still version-pinned in CI, because `--fail-on-change`
compares bytes. (b) **`gosec` stays enabled** (only G115 excluded), unlike algo-fft — R7's supply-chain
surface is real. (c) **golangci `formatters` enables gofumpt only**, not goimports, which would fight
gci over import grouping. Also `go.mod` says `go 1.26.0`, matching the siblings' `1.25.0` form.

**Task 0.3: Freeze the Python upstream.**

- [x] `git mv laya original/laya`, `git mv tests original/tests`,
      `git mv setup.py pyproject.toml notebooks original/`.
- [x] Keep `LICENSE` at the root.
- [x] Add a `NOTICE` recording that this is a derivative of `NandhaKishorM/laya`
      © Convai Innovations, Apache-2.0, and what was modified.
- [x] Commit: `chore: freeze upstream Python under original/ as the port's parity reference`.

`git log --follow original/laya/agent.py` reaches `e630a68`, so history survived the move. `NOTICE`
pins upstream to 0.3.4 at commit `d113dca` — this repo is a clean mirror, all 53 commits authored
upstream. Unlike go-pocket-tts, `original/` is **committed, not gitignored**: the Apache-2.0
attribution rests on it being visible. It is excluded from treefmt, golangci-lint and the Markdown lint.

**Task 0.4: Replace CI.**

- [x] `.github/workflows/ci.yml` → `go test ./... -race`, `go vet`, `golangci-lint`, `just check-tidy`,
      matrix over `ubuntu-latest`/`macos-latest`, Go 1.26.
- [x] `.github/dependabot.yml` → `gomod` instead of `pip` (plus `github-actions`).
- [x] `.github/workflows/security.yml` — **keep the policy, translate the mechanism.** The upstream
      greps `laya/` for `pickle`, `torch.load(`, `eval(`, `subprocess.` because laya downloads
      checkpoints from the Hub and a pickle path would be RCE on `laya.load("someone/their-model")`.
      The Go equivalents: forbid `encoding/gob` and `os/exec` on downloaded artifacts, and keep both
      gitleaks scans (PR diff **and** working tree — a PyPI token once survived in `main`; see commit
      `31280fc`).
- [x] `.github/workflows/release.yml` → keep the tag-vs-version gate, drop PyPI.
- [ ] **Deferred to M6, when there is code to check:** verify safetensors/ONNX headers before use, pin
      the ORT version, check each download's ETag/sha against Hub metadata (see Task 6.2/6.4).
- [x] `pip-audit` → `govulncheck` (reachable code paths, not a manifest audit); CodeQL → Go with autobuild.
- [x] Beyond the task: a third supply-chain grep rejecting a filesystem `replace` in `go.mod` — it would
      make the build depend on code no checksum covers, which matters once Task 4.3 vendors a forked
      tokenizer — and a cross-build over five GOOS/GOARCH pairs in `release.yml`.

`actionlint` is clean on all three workflows. Each supply-chain grep was checked against a negative
control to confirm it fires rather than silently passing. Built as one `ci.yml` with jobs, not
algo-fft's five-file `workflow_call` fan-out, because §8 wants an OS matrix algo-fft does not have.
golangci-lint is pinned to v2.12.2: `default: all` means a new linter in a new release turns green into
red with no commit here. `go vet` is not a separate step — `golangci-lint` runs `govet` as a linter.

**Task 0.5: Single version constant.** Python keeps `0.3.4` in three files. Go: one
`const Version = "0.1.0"` in `laya.go`, asserted against the tag in `release.yml`.

- [x] `const Version` in `laya.go`, with semver and no-`v`-prefix tests.
- [x] `release.yml` compares it to `$GITHUB_REF_NAME` with the leading `v` stripped, and fails loudly
      if the declaration cannot be parsed rather than comparing an empty string to the tag.

Evidence: `go test ./... -run TestVersion -v` → 2 passed. The tests pin the two properties the release
gate silently depends on; if either broke, the gate would become a no-op that still reports success.

**Task 0.6: `AGENTS.md` + `CLAUDE.md`.** _Discovered during M0; not in the original plan._

- [x] `AGENTS.md` with the house Go conventions plus the three rules specific to a parity port:
      `original/` is never edited, `testdata/` is regenerated only through the pinned reference
      environment and reviewed as a diff, and tokenizer assertions cover token strings as well as ids.
- [x] `CLAUDE.md` = the single line `@AGENTS.md`, matching both sibling repos.

Both siblings carry these; this repo had neither, and every later milestone is worked by agents told to
read the repo's conventions first.

**Task 0.7: repository hygiene found along the way.**

- [x] `BENCHMARKS.md`'s "English vs the rest" table declared four columns while every row supplied
      three, so the renderer dropped a cell from each row.
- [x] `README.md` linked to `#model-routing-three-checkpoints-one-call`, a heading `d113dca` renamed.
      The anchor had pointed at nothing since.
- [x] `check-tidy` as copied from go-pocket-tts aborts with exit 128 before a `go.sum` exists, and
      `git diff` cannot see a `go.sum` that `go mod tidy` has just created. Rewritten on
      `git status --porcelain`, verified in both directions.
- [ ] **README still points at `raw.githubusercontent.com/NandhaKishorM/laya/...`** for every image and
      still documents the Python API. Left for Task 7.5, which rewrites it against the Go API — fixing
      the URLs now would mean documenting an API that does not exist.

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

- [ ] **1.1.1** Create `.venv-ref` with Python 3.12 and the CPU torch wheel.
- [ ] **1.1.2** Install `transformers safetensors huggingface_hub numpy onnx onnxruntime`.
- [ ] **1.1.3** Freeze to `scripts/requirements-ref.txt` and commit it — the versions are the contract.
- [ ] **1.1.4** Add `.venv-ref/` to `.gitignore`.
- [ ] **1.1.5** Document the regeneration command in `scripts/README.md` so it survives the next person.

**Task 1.2: Fetch the checkpoints.** ~2.4 GB, anonymous, no token:

```bash
export LAYA_MODELS=${LAYA_MODELS:-$HOME/laya_models}
python -c "from huggingface_hub import snapshot_download as d; d('convaiinnovations/laya', local_dir='$LAYA_MODELS/laya')"
```

- [ ] **1.2.1** Download all three checkpoints to `$LAYA_MODELS` (549 GB free on `/mnt/projekte`).
- [ ] **1.2.2** Record the resolved commit sha of the Hub repo — golden vectors are only meaningful
      against a known revision.
- [ ] **1.2.3** Confirm the three `temperature` dtypes (F32/F32/F16, §1.1) so the loader never
      hardcodes one.

**Task 1.3: `scripts/dump_python_parity.py`.** Modelled on `../go-pocket-tts/scripts/dump_python_parity.py`.
Emits into `testdata/`, and records `transformers.__version__` / `tokenizers.__version__` in every
file header.

- [ ] **1.3.1** `testdata/tokenizer_{en,ml}.jsonl` — `{text, ids, ids_with_leading_space, tokens}` per
      case, covering the full Task 4.4 case list.
- [ ] **1.3.2** `testdata/sequence.jsonl` —
      `(state, question, max_len, head_max_len, option_order, truncate_left) → (ids, markers)`.
- [ ] **1.3.3** `testdata/render.jsonl` — `render_criterion` / `render_options` / `serialize_state`
      inputs and outputs.
- [ ] **1.3.4** `testdata/answers.jsonl` —
      `(logits, act_logits, temperature, k, qtype, criteria) → answer JSON`.
- [ ] **1.3.5** `testdata/logits.jsonl` — end-to-end `(state, questions) → (logits, act_logits)` for
      ~50 fixed inputs.
- [ ] **1.3.6** Every file carries a header record with the tokenizer/transformers versions and the
      checkpoint sha from 1.2.2.
- [ ] **1.3.7** Deterministic output: fixed seeds, sorted case order, stable float formatting, so
      regeneration is a reviewable diff (R6).

**Task 1.4: Provenance test.** `TestGoldenProvenance` asserts the recorded tokenizer version matches
what the Go implementation claims to target. A silent upstream tokenizer change is the realistic way
parity regresses.

- [ ] **1.4.1** A Go constant naming the targeted `tokenizers` version.
- [ ] **1.4.2** `TestGoldenProvenance` reads every `testdata/*.jsonl` header and fails on a mismatch.
- [ ] **1.4.3** The test also fails if a `testdata` file is missing its header entirely.

**Exit criteria**

- [ ] All five `testdata/*.jsonl` files committed; CI does not need Python to run the suite.
- [ ] Commit: `feat(testdata): golden parity vectors generated from upstream Python`.

### M2 — Tier-1 core (no ML runtime, ~620 lines of Python)

> **Three Python-isms will silently corrupt results if translated naively.** Do M2 in this order.

**Task 2.1: `jsonx` — Python-compatible JSON.** _Do this first; everything downstream depends on it._

`encoding/json` gets all three of these wrong: it emits `{"a":1}` instead of `{"a": 1}`, it **sorts map
keys**, and it **HTML-escapes** `<`, `>`, `&` by default. The Python code feeds
`json.dumps(state, ensure_ascii=False)` straight into the tokenizer, so any of the three changes the
bytes the model sees.

- [ ] **2.1.1** Create `jsonx/jsonx.go`, `jsonx/jsonx_test.go`.
- [ ] **2.1.2** `type Obj []Field` with `Field{Key string; Value any}` — an _ordered_ object.
- [ ] **2.1.3** `Obj.MarshalJSON` reproduces `json.dumps(x, ensure_ascii=False)`: separators `", "` and
      `": "`, no key sorting, no HTML escaping, non-ASCII emitted literally.
- [ ] **2.1.4** `Compact(v any) string` reproduces
      `json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)`, falling back to
      `fmt.Sprintf("%v", v)` for anything `encoding/json` refuses (Python's `default=str`).
- [ ] **2.1.5** `Round4(x float64) float64` — **Python's `round()` is half-to-even on the exact binary
      double; Go's `math.Round` is half-away-from-zero.** Implement as
      `strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)`.
- [ ] **2.1.6** Tie-case tests for `Round4`: `0.00125`, `2.5e-5`, plus negatives and values that are
      exactly representable.
- [ ] **2.1.7** Float formatting matches Python's `repr` for the values that reach JSON (shortest
      round-trip), including `-0`, very small and very large magnitudes.
- [ ] **2.1.8** Table test against `testdata/render.jsonl` — byte-for-byte (invariant #18).

**Task 2.2: `lang`.** Port `original/laya/lang.py` verbatim. Invariants §5 items 54–65.

Go-specific traps:

- [ ] **2.2.1** `detect_script`'s tie-break depends on Python dict _encounter_ order, with
      `counts["latin"]` assigned **after** the loop. Go map iteration is randomized — use an explicit
      ordered counter or the result is nondeterministic (invariant #56).
- [ ] **2.2.2** `_STOP`'s insertion order (fr, de, es, pt, it, nl) is the language tie-break. Use an
      ordered slice (invariant #63).
- [ ] **2.2.3** Python's `[^\W\d_]+` is Unicode-aware; Go RE2's `\w` is ASCII-only. Split on
      `unicode.IsLetter` instead.
- [ ] **2.2.4** `_SCRIPT_RANGES` order is load-bearing (hangul before kana before han), first match
      wins.
- [ ] **2.2.5** Port all 14 script cases, 7 `is_english` cases, 6 language-guess cases and 5 flattening
      cases from `original/tests/test_router.py:29-86`.
- [ ] **2.2.6** A determinism test: the same input run 1000× (or with `-count=10`) gives the same
      script/language — the only way the map-order trap shows up.

**Task 2.3: `mailtext`.** Port `original/laya/email.py`. Invariants §5 items 66–74.

- [ ] **2.3.1** Port the cleaning pipeline into `mailtext/`.
- [ ] **2.3.2** Do **not** port `email.email_questions` — it is a dead byte-identical duplicate of
      `presets.email_questions`; port only the presets one (Task 2.4).
- [ ] **2.3.3** Write the tests this port deserves — the upstream code is currently untested; cover
      quoted replies, signature blocks, forwarded headers, CRLF and unicode whitespace.

**Task 2.4: `presets`.** Port all five preset constructors from `original/laya/presets.py`. Pure data.

- [ ] **2.4.1** Port all five constructors into `presets/`.
- [ ] **2.4.2** Add the shape test upstream never had: every preset round-trips through question
      validation.
- [ ] **2.4.3** Assert option order is preserved exactly — the option index is the answer index.

**Task 2.5: `question.go` + `internal/prompt/render.go`.** `render_options` / `render_criterion`.
Invariants §5 items 14–18.

- [ ] **2.5.1** `Question` interface + `Choice`/`Score`/`Noul` types (§7, `docs/API.md`). The Go type
      design collapses Python's dict-or-list `criteria` into `[]ChoiceOption`.
- [ ] **2.5.2** `render_options` / `render_criterion` / `serialize_state` in `internal/prompt/render.go`,
      on top of `jsonx`.
- [ ] **2.5.3** Port `original/tests/test_criteria.py:33-100` as a table test.
- [ ] **2.5.4** **Drop** `test_criteria.py:103-116`, which uses `inspect.getsource` to assert five
      string literals are present in `Agent.__init__`; replace it with a behavioural test on the Go
      device-fallback policy.

**Exit criteria**

- [ ] `just check` green; commit after each task.

### M3 — Router (pure, no weights, no network)

**Task 3.1: `RouteDecision` + model registry.** Invariants §5 items 39–47.

- [ ] **3.1.1** `RouteDecision` struct with JSON tags plus `Map() Obj` for callers who want the dict.
- [ ] **3.1.2** The model registry (names → `ModelSpec{repo, subfolder, …}`).
- [ ] **3.1.3** Port the route precedence and the exact reason strings.
- [ ] **3.1.4** **Deviation:** `router.py:266` emits `repo` as a raw `(repo, subfolder)` tuple on the
      auto-workflow branch while every other branch emits a string via `_repo_str()`. Through
      `json.dumps` that surfaces as a JSON array. **Always emit the string**, and document the
      deviation in the README (Task 7.5).
- [ ] **3.1.5** **Reproduce faithfully:** `workflow` is `None` on the model/task paths but carries the
      detected workflow name on the **lang and detection** paths even though it did not drive the
      decision.

**Task 3.2: `Router` + LRU.** Invariants §5 items 48–53.

- [ ] **3.2.1** Port the LRU lifecycle (capacity, eviction order, preload).
- [ ] **3.2.2** Python's `Router` is not concurrency-safe; Go's must be — `sync.Mutex`, plus a
      `-race` test that routes and evicts from several goroutines.
- [ ] **3.2.3** Expose the loader as an injectable hook —
      `WithLoader(func(ctx, name, ModelSpec) (*Agent, error))` — so the upstream LRU/preload tests,
      which monkeypatch `rr.load`, stay expressible without reflection.
- [ ] **3.2.4** Eviction closes the evicted agent's backend; assert it, since a leaked ORT session is
      hundreds of MB.

**Task 3.3: Port the upstream router suite.**

- [ ] **3.3.1** Port the whole of `original/tests/test_router.py` (~90 assertions).
- [ ] **3.3.2** Port section 1 of `test_local_e2e.py:46-67`, which is pure routing — the file states
      outright "No model weights are loaded: `Router.route` is pure."

**Milestone check**

- [ ] `lang` + `mailtext` + `presets` + `Route` is a genuinely useful Go library with zero ML
      dependencies, and it covers everything the upstream test suite actually tests.
- [ ] A build importing only those four packages links no ONNX symbols (the §8 check, run early).

### M4 — Pure-Go tokenizer (highest risk)

> ⚠️ This is the single largest correctness risk in the port (R1). Marker positions are token
> indices, so one off-by-one silently corrupts every decision.

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

- [ ] **4.1.1** Define the interface in `tokenizer/`. `build_sequence` then ports 1:1 and stays
      backend-agnostic, which keeps D2 reversible (R1).

**Task 4.2: Special-token resolution.** ~40 lines.

- [ ] **4.2.1** Read token _strings_ from `tokenizer_config.json`; resolve ids from `tokenizer.json`'s
      `added_tokens` + `model.vocab`.
- [ ] **4.2.2** Prefer the tokenizer config over the encoder config — `multilingual/encoder/config.json`
      says `cls_token_id: 1` while its tokenizer says `<bos>` = **2**, and 2 is what the weights were
      trained with (§1.2).
- [ ] **4.2.3** Ignore the Gemma `extra_special_tokens`-as-a-list quirk that `_fix_tokenizer_config`
      patches around in Python — Go never needs the workaround.
- [ ] **4.2.4** Assert the resolved ids for both checkpoints against §1.4's table (EN: UNK 50280,
      CLS 50281, SEP 50282, PAD 50283, MASK 50284 · ML: PAD 0, EOS/SEP 1, BOS/CLS 2, UNK 3, MASK 4).

**Task 4.3: Vendor and fix `gomlx/go-huggingface/tokenizers/hftokenizer`.** Apache-2.0, actively
maintained, and already the pure-Go backend of `hugot`. Its scaffolding is the best available
(normalizers NFC/NFD/NFKC/NFKD/Replace/Prepend/Sequence/BertNormalizer, pre-tokenizers
ByteLevel/Metaspace/Split/Sequence/Punctuation, BPE/WordPiece/Unigram, both merge serializations).

- [ ] **4.3.0** Vendor it under `tokenizer/internal/hftokenizer`, preserving the Apache-2.0 header and
      recording the upstream commit in `NOTICE`.
- [ ] **4.3.1** **Fix Metaspace prepend:** `pretokenizer.go:342` checks `text[0] != ' '` where HF checks
      `!starts_with(replacement)`. With the Gemma `Replace` normalizer there are no spaces left in the
      text, so it **always** prepends and `"▁foo"` becomes `"▁▁foo"` — drift on every multilingual
      encode, doubly so on `tok(" " + opt)`. Fix to `!strings.HasPrefix(text, replacement)`.
- [ ] **4.3.2** **Fix byte fallback / fuse-unk:** `bpeTokenizeWithSpans` parses `ByteFallback`/`FuseUnk`
      into the struct and then never uses them. Unknown symbols go to `unkID` or are **silently
      dropped** when `unkID < 0`. mmBERT needs both; without byte fallback, every emoji and rare CJK
      character silently becomes `<unk>` (id 3) — on precisely the inputs `laya-multilingual` exists
      for. Reuse `../go-pocket-tts/internal/tokenizer/sentencepiece_bytes.go`.
- [ ] **4.3.3** **Fix added-token matching:** it is "longest-first greedy" rather than HF's
      `AddedVocabulary`, with no `single_word`/`lstrip`/`rstrip`/`normalized` distinction. Both
      checkpoints have a `lstrip:true` mask token and ≥ 109 `normalized:true` added tokens.
- [ ] **4.3.4** Added tokens are split out **before** normalizer and pre-tokenizer run — assert that
      ordering directly, since `build_sequence` feeds arbitrary user JSON through the tokenizer and the
      109 whitespace-run added tokens are load-bearing.
- [ ] **4.3.5** Upstream all three fixes to `gomlx/go-huggingface` and link the PRs here.

**Do not use `sugarme/tokenizer`:** `pretrained/model.go`'s `createBPE()` has `byte_fallback` and
`fuse_unk` _commented out_ and `bpe.New` has no parameters for them, plus an open panic
(issue #78) on consecutive whitespace in the Metaspace pre-tokenizer — a crash on realistic input.

**Task 4.4: Golden corpus, both checkpoints.** Assert **ids and token strings** — an id diff alone tells
you nothing, `["▁▁","x"]` vs `["▁x"]` tells you exactly which stage broke.

Required cases:

- [ ] **4.4.1** `""`, `" "`, `"  "`, `"\n"`, `"\t"`, `"\r\n"`, `"   \n   "` — consecutive whitespace is
      the most common drift/crash site.
- [ ] **4.4.2** **`text` and `" " + text` for every case** — this is the `common.py:68` semantic and it
      differs between the two checkpoints: on ML the `Replace` normalizer makes `tok(" x") == tok("x")`,
      on EN the leading space becomes `Ġ` and they differ (§1.4).
- [ ] **4.4.3** Compact JSON exactly as `serialize_state` emits it, with nested braces, escapes,
      non-ASCII, long arrays.
- [ ] **4.4.4** The mask literal (`[MASK]` / `<mask>`) appearing inside user text.
- [ ] **4.4.5** Every added token bare and mid-sentence: `<unused0>`, `<start_of_turn>`, `<2mass>`,
      `[@BOS@]`, `|||IP_ADDRESS|||`, and runs of 2–24 spaces (EN ids 50254–50275).
- [ ] **4.4.6** Normalization: precomposed vs decomposed `é`, fullwidth `Ａ`, ligature `ﬁ`, NBSP,
      ZWJ/ZWSP, BOM, combining marks. EN normalizes NFC; ML does **not** normalize beyond space→`▁`,
      so these two legitimately differ.
- [ ] **4.4.7** Emoji with ZWJ sequences, skin tones, flags — these exercise byte fallback on ML.
- [ ] **4.4.8** Devanagari, Arabic, CJK, Thai, Cyrillic, Korean jamo vs precomposed.
- [ ] **4.4.9** Lone surrogates / invalid UTF-8 (Go tolerates, Python `str` does not) — decide and
      document the boundary; sanitize at the API edge.
- [ ] **4.4.10** Both corpora run green: ids **and** token strings identical to Python.

**Task 4.5: Fuzz + differential.**

- [ ] **4.5.1** A Go fuzz target checking invariants only: no panic, ids < `vocab_size`, ASCII
      round-trip.
- [ ] **4.5.2** Seed the corpus from the Task 4.4 cases and commit the interesting crashers.
- [ ] **4.5.3** A one-off differential run of ~100k lines of real multilingual corpus through both
      Python and Go, diffing the id streams. That is what actually finds the metaspace/added-token bug
      classes.
- [ ] **4.5.4** Record the differential result (corpus, line count, mismatches) in this file.

**Gate**

- [ ] **M5 does not start until both golden corpora are 100 % green.**
- [ ] If parity cannot be reached, execute R1's fallback: swap to `daulet/tokenizers` (CGO, wraps the
      same Rust crate Python uses — parity by construction) behind the Task 4.1 interface, and record
      the CGO consequence against D2.

### M5 — `build_sequence`

**Task 5.1: Port `common.py:49-86` into `internal/prompt/sequence.go`.**
This is 38 lines carrying most of the porting risk, and it has **no test at all upstream**.
Invariants §5 items 1–13.

- [ ] **5.1.1** Port the token-budget arithmetic: `opt_budget`, the `< 16` fallback, `[:48]`.
- [ ] **5.1.2** Port the truncation direction: `[-room:]` when `truncate_left`, else `[:room]`.
- [ ] **5.1.3** Port the final `ids[:max_len]` clamp and the marker filter that drops markers pushed
      past the clamp.
- [ ] **5.1.4** Marker positions are token indices — assert them explicitly, not just the id stream.

**Task 5.2: Assert against `testdata/sequence.jsonl` byte-for-byte.**

- [ ] **5.2.1** A table test across the matrix
      `(qtype, criteria, instructions, state, max_len, head_max_len, option_order, truncate_left)`.
- [ ] **5.2.2** Cases that exercise the truncation arithmetic independently of tokenizer drift —
      feed a stub tokenizer with known ids so a failure names the arithmetic, not the tokenizer.
- [ ] **5.2.3** Boundary cases: exactly `max_len`, one over, an option list that alone exceeds the
      budget, zero options, and a state that truncates to nothing.

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

- [ ] **6.1.1** Define `Backend` and `Batch`.
- [ ] **6.1.2** An in-memory fake replaying `testdata/logits.jsonl` — every M7 test then runs without
      ORT.
- [ ] **6.1.3** The fake fails loudly on an unknown input rather than returning zeros, so a prompt
      regression cannot masquerade as a passing test.

**Task 6.2: `internal/hub` — HF resolve + cache.**

- [ ] **6.2.1** Resolve via `https://huggingface.co/{repo}/resolve/{rev}/{path}`, optional
      `Authorization: Bearer $HF_TOKEN`.
- [ ] **6.2.2** ETag/sha verification on every download (R7).
- [ ] **6.2.3** A local cache under `$LAYA_CACHE` or `os.UserCacheDir()/laya`, with atomic
      write-then-rename so an interrupted download is never served.
- [ ] **6.2.4** An `allow_patterns`-equivalent so a subfolder request downloads only that subfolder —
      the bundle repo is 2.4 GB; the English checkpoint alone is 846 MB.
- [ ] **6.2.5** Everything cancellable via `context.Context`; a cancelled download leaves no partial
      file behind.
- [ ] **6.2.6** Offline mode: an already-cached checkpoint resolves with no network call.
- [ ] **6.2.7** Tests run against an `httptest` server — no network in CI.

**Task 6.3: ONNX backend.** Per Spike S2's binding decision.

- [ ] **6.3.1** Implement `Backend` over the chosen binding; dynamic batch/seq/k.
- [ ] **6.3.2** Device selection (`cpu`/`cuda`/`coreml`) with a logged fallback, replacing Python's
      three `print()` warnings with `slog` — and warn only when a fallback actually happened
      (upstream `e630a68`).
- [ ] **6.3.3** `Close()` releases the session; assert no leak across a load/evict cycle (Task 3.2.4).
- [ ] **6.3.4** Build-tag or separate-package the ONNX backend so the §8 "no ML dependency" check
      stays true for `lang`/`mailtext`/`presets`/`Route`.

**Task 6.4: Checkpoint validation.** Port `_verify_compatibility`'s intent.

- [ ] **6.4.1** Require cfg keys `encoder` and `head_layers`.
- [ ] **6.4.2** Require the graph's declared inputs/outputs (`input_ids`, `attention_mask`,
      `marker_pos`, `marker_mask`, `qtype` → `logits`, `act_logits`).
- [ ] **6.4.3** Fail with a wrapped `ErrIncompatibleCheckpoint` naming what was wrong.
- [ ] **6.4.4** Validate the ONNX/safetensors header **before** handing the bytes to the runtime, and
      never `os/exec` or `encoding/gob` a downloaded artifact (R7, Task 0.4's deferred item).

### M7 — Agent, calibration, end-to-end parity

**Task 7.1: `internal/calib`.** Invariants §5 items 22–29.

- [ ] **7.1.1** `temp_bucket` and the `temperature_by_options` → `temperature[qtype]` lookup.
- [ ] **7.1.2** The `max(1e-3, t)` floor.
- [ ] **7.1.3** Max-subtracted softmax over **exactly the first _k_ logits**.
- [ ] **7.1.4** Entropy confidence.
- [ ] **7.1.5** `Round4` from `jsonx` applied at the same points Python applies `round()`
      (invariant #29).

**Task 7.2: `Agent.SystemOne`.** Invariants §5 items 19–34.

- [ ] **7.2.1** The `choice` answer shape.
- [ ] **7.2.2** The `score` answer: `Σ i·p[i]`, an expectation, **not** an argmax.
- [ ] **7.2.3** The `noul` answer: **no** `probabilities`, no `legend`, and confidence
      `max(p1, 1-p1)` — **not** the entropy formula, which for k=2 genuinely disagrees (invariant #26).
- [ ] **7.2.4** `legend` carries the **raw** criterion value, not the rendered option text.
- [ ] **7.2.5** The act head (`act_logits` → the two-way decision) and its threshold.
- [ ] **7.2.6** Batching: several questions in one forward pass, with per-question `qtype`.

**Task 7.3: Answer-formatting parity.**

- [ ] **7.3.1** Table test against `testdata/answers.jsonl`, running against the fake backend — needs
      no model.
- [ ] **7.3.2** Compare the serialized JSON bytes, not just the parsed struct (invariant #18 again).

**Task 7.4: End-to-end parity.**

- [ ] **7.4.1** Test against `testdata/logits.jsonl`, gated behind `testing.Short()` and a
      `LAYA_MODELS` env var.
- [ ] **7.4.2** Assert per-option probabilities within 1e-4 **and** that the argmax decision never
      flips.
- [ ] **7.4.3** Run it for all three checkpoints.
- [ ] **7.4.4** Port sections 2–5 of `original/tests/test_local_e2e.py` (loose directional thresholds:
      ≥ 6/8 land on `billing`, ≥ 2/3 on the preset checks).

**Task 7.5: README + examples.**

- [ ] **7.5.1** Port every README example to Go, as compiling `Example` functions so CI proves they
      still build (§8).
- [ ] **7.5.2** Fix the image URLs — upstream's point at
      `raw.githubusercontent.com/NandhaKishorM/laya/main/...`, so a fork's README silently renders
      upstream's assets.
- [ ] **7.5.3** Document the deliberate deviations: no training symbols (D3), `repo` always a string
      (Task 3.1.4), `Detection.ScriptProfile` as a map, `instructions` as `string` only.
- [ ] **7.5.4** Replace upstream's T4 latency claims with the Spike S3 numbers in `BENCHMARKS.md`.
- [ ] **7.5.5** State the tokenizer/checkpoint revisions the port is verified against.

### M8 — Pure-Go native backend (after 1.0)

Same `Backend` interface, no API change. Implement ModernBERT + mmBERT + the head over safetensors,
lifting `tensor`, `ops` and `safetensors` from `../go-pocket-tts`.

- [ ] **8.1** Lift `internal/safetensors`, `internal/runtime/tensor` and `internal/runtime/ops` from
      `../go-pocket-tts` (§1.5).
- [ ] **8.2** Bias-free LayerNorm (`norm_bias: false`, eps 1e-5; layer 0 has no `attn_norm`).
- [ ] **8.3** GeGLU with the fused `mlp.Wi [5248,1024]` gate+up split, no MLP bias.
- [ ] **8.4** Fused QKV unpacking (`attn.Wqkv [3072,1024]`, no attention bias).
- [ ] **8.5** Sliding-window attention masks (window 128, ±64) and per-layer-type RoPE theta
      (full layers 0,3,…,27 at 160000; sliding at 10000; mmBERT uses 160000 for both).
- [ ] **8.6** The decision head with a **ReLU** FFN (§1.3) and the manual layer loop.
- [ ] **8.7** fp16 weight loading, including the `temperature` tensor whose dtype differs per
      checkpoint (§1.1).
- [ ] **8.8** Gate promotion on the same golden vectors: probabilities within 1e-4, zero argmax flips.
- [ ] **8.9** A latency benchmark that beats the Spike S3 floor — this is where the 12 s/forward
      estimate has to be beaten.

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
- [ ] `TestGoldenProvenance` green: the tokenizer/transformers versions and the checkpoint revision the
      vectors were generated from are recorded and asserted (R6).
- [ ] The ONNX artefacts are ones we export ourselves, not a third-party upload (S1's fallback is
      development-only).
- [ ] The ORT version is pinned and every downloaded artefact is ETag/sha-verified before use (R7).
- [ ] The deliberate deviations from Python are listed in the README: no training symbols, `repo`
      always a string, `Detection.ScriptProfile` as a map, `instructions` as `string` only.
