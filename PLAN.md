# go-laya — Python → Go Port Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Reimplement the Python `laya` 0.3.4 decision engine as an idiomatic Go library that produces
byte-identical prompts and numerically equivalent decisions from the same Hugging Face checkpoints.

**Architecture:** Three layers, each importable on its own:

1. A dependency-free pure-Go core: question rendering, language routing, email cleaning, presets
   and calibration math.
2. A pure-Go tokenizer that reads `tokenizer.json`.
3. A `Backend` interface for the neural net, implemented over an ONNX export of the whole
   `DecisionModel`. A pure-Go safetensors backend comes later behind the same interface.

**Tech stack:** Go 1.26, ONNX Runtime through `shota3506/onnxruntime-purego` (CGO-free), `just`,
`treefmt` and `golangci-lint`.

**Where things live.** This file holds **open work** and a short record of what is done.
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) holds what the system is: checkpoints, tokenizer
semantics, export recipe, trust model and numerics rules. [`docs/DECISIONS.md`](docs/DECISIONS.md)
holds why, as D1–D28. The dated history of every finished task (evidence, mutation runs,
re-measurements) was condensed out on 2026-09-27 and is in git: `git show 8125f40:PLAN.md`. Ids of
finished tasks cited in code comments refer to that history.

---

## Status at a glance

Tick a box only when the work is committed and `just ci` is green. A milestone is done when every
box under it is ticked.

| Milestone                                                       | Delivers                                                | Status                                   |
| --------------------------------------------------------------- | ------------------------------------------------------- | ---------------------------------------- |
| [M0 — Scaffolding](#m0--scaffolding)                            | Go module, tooling, CI, frozen Python, `Version`        | ✅ done                                  |
| [Spikes S1–S3](#3-spikes)                                       | ONNX export, binding choice, latency floor              | ✅ done                                  |
| [M1 — Reference harness](#m1--the-python-reference-harness)     | `testdata/*.jsonl` golden vectors                       | ✅ done                                  |
| [M2 — Tier-1 core](#m2--tier-1-core)                            | `jsonx`, `lang`, `mailtext`, `presets`, render          | ✅ done                                  |
| [M3 — Router](#m3--router)                                      | `Route`, model registry, LRU                            | ✅ done                                  |
| [M4 — Tokenizer](#m4--pure-go-tokenizer)                        | pure-Go `tokenizer.json` loader                         | ✅ done                                  |
| [M5 — `build_sequence`](#m5--build_sequence)                    | prompt assembly, marker positions, collate              | ✅ done                                  |
| [M6 — Backend](#m6--backend--checkpoint-loading)                | `Backend`, hub cache, ONNX impl, validation, pinned ORT | ✅ done                                  |
| [M7 — Agent + parity](#m7--agent-calibration-end-to-end-parity) | loader, `SystemOne`, calibration, e2e parity, README    | 🟡 7.1–7.5 done bar 7.2.5; 7.7 bar 7.7.7 |
| [Backlog](#backlog--open-work-that-does-not-gate-10)            | tokenizer speed, NFC decision, CUDA, Windows, int8      | 🟡 B.1, B.6 done                         |
| [M8 — Native backend](#m8--pure-go-native-backend-after-10)     | safetensors ModernBERT/mmBERT (post-1.0)                | ⬜ deferred                              |

**Critical path to 1.0:** M7 (loader → `SystemOne` → answer parity → e2e parity → README). The
Backlog and M8 do not gate 1.0.

---

## 0. Decisions already made

Full rationale is in [`docs/DECISIONS.md`](docs/DECISIONS.md). Index:

| #   | Decision                                                              |
| --- | --------------------------------------------------------------------- |
| D1  | ONNX first, pure-Go native backend later, behind one `Backend`        |
| D2  | Pure-Go tokenizer, no CGO                                             |
| D3  | Inference only (`collate_items` is ported)                            |
| D4  | Upstream Python frozen under `original/`                              |
| D5  | `onnxruntime-purego` at `8db8bd7`; every `*Value` closed explicitly   |
| D6  | CPU is a batch deployment; threads = physical cores; no int8          |
| D7  | Reference env on transformers 5.17.0; 4.x mis-loads mmBERT            |
| D8  | M8 is a zero-shared-library backend, not a speed play                 |
| D9  | `backend` is a public leaf package                                    |
| D10 | Purpose-built tokenizer, no vendored fork                             |
| D11 | Question types in leaf `question/`, aliased at the root               |
| D12 | `jsonx.Marshal` is the outer encoder for every parity emitter         |
| D13 | Router caches an `Agent` interface, widened in M7                     |
| D14 | No cost-biased routing option                                         |
| D15 | Invalid UTF-8 sanitised at the tokenizer's API edge                   |
| D16 | Router closes only agents its own loader built                        |
| D17 | Default revision pinned to `1c5edc17…`                                |
| D18 | English checkpoint downloads only the repo root (superseded by D24)   |
| D19 | Shipped export stays `eager`                                          |
| D20 | ORT is a verified download; any 1.x ≥ 1.23 accepted                   |
| D21 | ONNX backend on a positive platform list; stub elsewhere              |
| D22 | `cuda` falls back to CPU until the binding supports it                |
| D23 | Parity emitters reject rather than guess on unordered input           |
| D24 | Forward pass is a local ONNX export; only config + tokenizer download |
| D25 | `state` stays `any`; no `State` interface                             |
| D26 | `act_probability` uses a port of ATen's softmax                       |
| D27 | `Predict` leases its agent; a dropped agent closes after its lease    |
| D28 | The exported struct is `Agent`; the Router's interface is `Predictor` |

## 1. What we are porting — verified facts

See [`docs/ARCHITECTURE.md` §1](docs/ARCHITECTURE.md#1-what-is-being-ported) (checkpoints, encoder,
decision head) and [§3](docs/ARCHITECTURE.md#3-tokenizer) (tokenizers).

## 2. Target repository layout

See [`docs/ARCHITECTURE.md` §2](docs/ARCHITECTURE.md#2-repository-layout), which owns the layout
and the dependency rule.

---

## 3. Spikes

All three are done. Details are in `docs/ARCHITECTURE.md` §5 and `BENCHMARKS.md`.

- **S1 — ONNX export: works for all three checkpoints.** The recipe is dynamo at opset 18 with the
  MHA fastpath off and eager attention. It is recorded in `scripts/export_onnx.py`, and ONNX vs
  PyTorch is per-checkpoint (multilingual is the loosest). The premise was wrong: transformers 5.x
  has no `torch.compile` left. The real blockers were the head's fused encoder-layer op and
  TorchScript silently freezing `seq`.
- **S2 — binding: purego holds (D5).** It is CGO-free and agrees with Python ORT to 3.8e-06.
  `bool` tensors round-trip. The surviving finalizer data race is avoided by closing every value.
  `CGO_ENABLED=0 go test -race` is impossible (race needs cgo), so the two are separate commands.
- **S3 — latency: 0.6–1.9 s per question on CPU at 512 tokens (D6).** Batching does not amortise,
  8 of 12 threads is the peak, and three sessions cost ~3.3 GB. A 2.7× Go-vs-Python gap seen once
  did not reproduce under a counterbalanced rerun (Task 6.8.3).

---

## 4. Milestones and tasks

Each task is TDD: write the failing test, watch it fail, implement minimally, watch it pass, commit.
Run `just ci` before every commit.

### M0 — Scaffolding

**Done.** The module is `github.com/MeKo-Christian/go-laya` (Go 1.26). `just` recipes are split
into `check` (developer loop) and `ci` (fmt-check, lint-md, lint-py, test-race, lint, check-tidy).
treefmt runs gofumpt+gci, prettier, shfmt and ruff format. markdownlint is a checker only, never a
formatter. golangci-lint v2.12.2 runs with `default: all` and gosec on. `original/` holds upstream
and `NOTICE` records the Apache-2.0 derivation. CI runs a linux/macOS matrix, cross-builds five
GOOS/GOARCH pairs, gates the release tag against `const Version`, and runs govulncheck, gitleaks
over the diff and the tree, and supply-chain greps (no `encoding/gob`, `os/exec` or filesystem
`replace`). `AGENTS.md` carries the parity-port rules.

Ledger: 0.2 tooling · 0.3 freeze upstream · 0.4 CI and security · 0.5 `Version` · 0.6 `AGENTS.md` ·
0.7 hygiene.

### M1 — The Python reference harness

**Done.** `.venv-ref` (Python 3.12, torch 2.14.0+cpu, transformers 5.17.0, pinned in
`scripts/requirements-ref.txt`, restore documented in `scripts/README.md`) runs
`scripts/dump_python_parity.py`. It writes the deterministic `testdata/*.jsonl` corpus. Each file
has a provenance header and a `compute` block, and `TestGoldenProvenance` checks every header.
Checkpoints are at `models/laya` (revision `1c5edc17…`, gitignored). The corpus is listed in
`docs/ARCHITECTURE.md` §8.

Ledger: 1.1 reference env · 1.2 checkpoints and revision · 1.3 the generator and fixtures
(1.3.10 `--verify-agent` checks the answer copy against a real `Agent`) · 1.4 provenance test ·
1.5 ruff over `scripts/` · 1.6 transformers 4.x cross-check (D7) · 1.7 collated batches and
`compute` in fixtures.

### M2 — Tier-1 core

**Done.**

- `jsonx`: ordered `Obj`, a Python-compatible encoder, `Compact`, `Round4`, `Repr` and `Decode`.
- `lang`: ordered script and language tie-breaks, `unicode.IsLetter` splitting.
- `mailtext`: `email.py`, with the Unicode `\w`/`\s`/`strip()` traps pinned.
- `presets`: all five constructors, rebuilt per call.
- `question/` with root aliases (D11).
- `internal/prompt` render: `render_options`, `render_criterion`, `serialize_state`, byte-equal to
  `render.jsonl`.

The numerics traps are in `docs/ARCHITECTURE.md` §7.

Ledger: 2.1 `jsonx` · 2.2 `lang` · 2.3 `mailtext` · 2.4 `presets` · 2.5 question types and render.

### M3 — Router

**Done.** `Route` gives upstream's precedence and exact reason strings. `RouteDecision` marshals
through `jsonx` (D12) and always emits `repo` as a string (a deviation, on 7.6.3's list). The
workflow name leaks into later branches, as upstream does. The registry (`ModelSpec`,
`DefaultModels`, `StandaloneModels`, `ModelSpecFromString`, `NormalizeModelName`) is in place, and
`jsonx.ReprString` covers `%r`. The LRU is mutex-guarded, with `Attach`/`Unload`/`Loaded`/`Preload`/
`Close`, `WithLoader` and `ErrNoLoader`, and closes only the agents it built (D16). All 98 checks
of `test_router.py` are ported (32 in `lang/`, 66 in `router_upstream_test.go`), plus section 1 of
`test_local_e2e.py`. One upstream assertion was found vacuous and got a real test.

Ledger: 3.1 `RouteDecision` + registry · 3.2 Router + LRU · 3.3 upstream suite · 3.4 cheaper
checkpoint: documented, no option (D14).

### M4 — Pure-Go tokenizer

**Done.** `tokenizer/` implements both pipelines in pure Go (D10): loader, two-phase added-token
matcher, NFC and `Replace`, a hand-written ByteLevel scanner with an RE2 oracle, Metaspace, and BPE
with byte fallback and `fuse_unk`. Special tokens resolve from the tokenizer config. Both golden
corpora pass with 206 subtests, ids **and** token strings. CI covers the stages through
`pretok_{en,ml}.jsonl`. A fuzz target runs. The 157 281-line differential found exactly one
divergence class, NFC combining classes (`docs/ARCHITECTURE.md` §3).

Ledger: 4.1 interface (`[]int64`, `IDToToken`, `VocabSize`) · 4.2 special tokens · 4.3 the
tokenizer · 4.4 golden corpus v2 and per-stage vectors · 4.5 fuzz and differential.

### M5 — `build_sequence`

**Done.** `internal/prompt.BuildSequence` (`common.py:49-86`) and `Collate` (`common.py:218-251`).
Stub-tokenizer tests pin the budget arithmetic, truncation (including `st[-0:]`), the clamp and
marker filtering. The fixture cannot reach those edges. `TestBuildSequenceGolden` matches all 45
`sequence.jsonl` cases and `TestCollateGolden` matches all 30 batches of `logits.jsonl`, both gated
on `LAYA_MODELS`. `option_order` and `truncate_left` have no public symbol.

Ledger: 5.1 port · 5.2 golden, stub and boundary tests · 5.3 `Collate` · 5.4 internal-only flags.

### M6 — Backend + checkpoint loading

**Done.**

- `backend/`: `Backend`, `Batch` and `ErrIncompatibleCheckpoint`, stdlib only.
- `internal/backend/fake`: replays per checkpoint and rejects unknown batches loudly.
- `internal/hub`: verified resolve, cache, snapshot and offline mode.
- `internal/backend/onnx`: dynamic shapes, header and width validation, device selection with a
  logged fallback, idempotent `Close` checked for leaks, and the platform stub.
- Checkpoint validation: `internal/checkpoint`, `internal/onnxheader`, `internal/safetensors`.
- The pinned, verified ORT download: `internal/ortlib`, `cmd/laya-ort`.

Go-vs-Python ORT parity passes over S1's four shapes with per-checkpoint tolerances. See
`docs/ARCHITECTURE.md` §5–6. `just test-onnx` runs the ORT-gated tests locally, and CI skips them.

Ledger: 6.1 `Backend` + fake · 6.2 hub (D17, D18) · 6.3 ONNX backend (D22) · 6.4 checkpoint
validation · 6.5 attention implementation (D19) · 6.6 ORT acquisition (D20) · 6.7 platforms
(D21) · 6.8 parity matrix · 6.10 onnxspike absorbed.

### M7 — Agent, calibration, end-to-end parity

**Task 7.1: `internal/calib`.** Invariants §5 items 22–29.

Done:

- **7.1.1–7.1.5:** temperatures and buckets, the `1e-3` floor, float32 softmax over the first k
  with numpy's pairwise sum, entropy confidence, and `NoulConfidence`/`Expectation`. The caller
  rounds.
- **7.1.6:** `ECE` and `Brier` against `ece.jsonl`.
- **7.1.8:** numpy's float32 exp/log, bit-exact on all 2^32 inputs, with `f32math.jsonl`.

`TestAnswersNumerics` matches all 33 `answers.jsonl` cases exactly after `Round4`.

- [x] **7.1.7** Make `answers.jsonl` discriminate precision. (2026-09-27) The generator's
      `precision_cases` searches seeded draws for cases that round differently under a wrong port
      and marks each with `discriminates` (the field) and `against` (the port). Against float64
      (`answer_block(dtype=float64)`): one case each for `probabilities`, `confidence`, `score`
      and `noul`. Against a left-to-right sum in place of numpy's pairwise one
      (`sequential=True`): `probabilities` and `confidence` at k = 10, since the two sums first
      diverge at k ≥ 8. The 27 earlier cases are byte-identical.
      `go test -count=1 -run 'TestAnswersNumerics|TestAnswersDiscriminatePrecision' -v ./internal/calib/`
      passes, with `TestAnswersDiscriminatePrecision` requiring each port to miss every field it
      is marked against. Mutations that passed on the 27-case corpus now fail: a float64
      `Softmax` fails `precision/choice-confidence`, and a sequential `numpySum` fails both
      `precision/sum-order-*` cases.

**Task 7.2: The default agent loader.** `NewRouter()` without `WithLoader` loads through it;
`WithLoader(nil)` is the only way to get `ErrNoLoader`.

- [x] **7.2.1** A default loader that builds an Agent from a spec's repo, subfolder, device and
      token, as `router.py:174-178` does. Default revision `1c5edc17…` (D17). It passes
      `checkpoint.Config.ActWidth()` as `onnx.Options.ActWidth`. (2026-09-27) `loader.go`: a
      local directory is used as it is, a missing path-shaped id is `ErrCheckpointNotFound`, anything
      else is a Hub snapshot (the bundle repo at D17's pin, other repos at `main`), then the
      subfolder. Per D24 (user decision), the graph is the local export
      `laya-<name>.onnx` (`WithONNXDir`, `$LAYA_ONNX_DIR`, `<cache>/onnx`; `ErrNoGraph` names the
      export command), so a snapshot fetches only `rl_agent_config.json` and `tokenizer/*`. That
      replaces the item's original `sub + "/*"` and root-only filter, which assumed the weights
      are downloaded; D18 is superseded.
      `go test -count=1 -run 'TestDefaultLoader|TestNewRouterInstalls' -v .` passes 6 tests, with
      the real-weights test skipped in CI. Locally,
      `LAYA_ONNX_DIR=$PWD/build/onnx go test -count=1 -run TestDefaultLoaderReal -v .` loads the
      english checkpoint through ONNX Runtime and passes.
- [x] **7.2.2** `WithRouterDevice` and `WithRouterToken` (`docs/API.md:347-348`), including
      upstream's `token or os.environ["HF_TOKEN"]` fallback (`router.py:159`). (2026-09-27) The
      token fallback is read once by `NewRouter`. The device reaches `onnx.Options.Device` unchanged
      and is validated at load time.
      `go test -count=1 -run 'TestRouterDevice|TestRouterToken' -v .` passes both.
- [x] **7.2.3** Widen `Agent` past `Close() error` only when M7 needs it (D13). The loader's
      `onnxAgent` carries only `Close` until Task 7.7.1 adds `SystemOne`. (2026-09-27) Done by
      7.7.1, when `Router.Predict` needed it: `Agent` is `SystemOne` plus `Close` and nothing else.
- [x] **7.2.4** Offline from the environment: map `HF_HUB_OFFLINE` and/or `LAYA_OFFLINE` onto
      `hub.Client.Offline`. (2026-09-27) `HF_HUB_OFFLINE` is read as huggingface_hub reads it,
      `_is_true(HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE)` (`constants.py:194`), and `LAYA_OFFLINE`
      also switches it on. `go test -count=1 -run 'TestDefaultLoaderOffline' -v .` passes 13 cases.
- [ ] **7.2.5** An opt-in to follow `main` or another revision instead of D17's pin. D17 calls
      following `main` an explicit opt-in, but no option expresses it yet, so the default loader
      always loads the bundle repo at the pin.

**Task 7.3: `Agent.SystemOne`.** Invariants §5 items 19–34.

- [x] **7.3.1** The `choice` answer shape. (2026-09-27) `answer.go`: `Answer`, `Probs`,
      `Action`, `Usage`, `AnswerSet` and `Result` as `docs/API.md` gives them, and
      `formatAnswer`, a port of agent.py:302-337. Choice keys come in option order, and `zip`
      truncates to the shorter of keys and `p`.
- [x] **7.3.2** The `score` answer: `Σ i·p[i]`, an expectation, **not** an argmax. (2026-09-27)
      `Round4(calib.Expectation(p))`. `TestAnswerScoreIsExpectation` uses logits whose argmax and
      expectation differ.
- [x] **7.3.3** The `noul` answer: **no** `probabilities`, no `legend`, and confidence
      `max(p1, 1-p1)`, **not** the entropy formula, which for k=2 genuinely disagrees (#26).
      (2026-09-27) `TestAnswerNoulConfidence`: uniform logits give 0.5, not the entropy's 0.
- [x] **7.3.4** `legend` carries the **raw** criterion value, not the rendered option text.
      (2026-09-27) It covers every level, not just k. `score/dict-legend`
      (`[{"d":"low"},"mid",2]`) and `TestAnswerLegendIsRaw`.
      `go test -count=1 -run 'TestAnswer|TestProbs|TestResult' -v .` passes all 10 tests, covering
      7.3.1–7.3.4, 7.3.7 and 7.3.8.
- [x] **7.3.5** The act head: `act = softmax(act_logits.float(), -1)` (torch float32,
      `agent.py:295`), then `action.act_probability = round(act[r, 0], 4)` (`agent.py:310`).
      Column 0 only. There is no threshold and no decision upstream. The width is
      `len(cfg["act_costs"]) + 1`, validated in 6.4.5. This is torch's exp, not numpy's (7.3.10).
      (2026-09-27) `Round4(calib.ActSoftmax(act[r])[0])` over the whole act row (D26).
      `TestSystemOneBatches` checks that every answer carries the same row's value.
- [x] **7.3.6** Batching: several questions in one forward pass, with per-question `qtype`,
      through `Collate`. (2026-09-27) `systemone.go`: `(*onnxAgent).SystemOne(ctx, state any, qs)`
      and `Predict`. It validates, then `CheckCriteria`, then `BuildSequence` at the config's
      `max_len`/`head_max_len`. A lost marker is an `*OptionBudgetError`. It then runs
      `Collate` and **one** `Forward`, and `formatAnswer` per row with the config's
      temperatures. `input_tokens` is the batch's attention-mask sum. New errors:
      `ErrEmptyQuestions`, `ErrOptionsExceedHeadBudget`, `ErrNoOptions` (an empty choice question,
      rejected before the forward pass; Python fails in the softmax), and a re-exported
      `ErrDuplicateQuestionID`. `go test -count=1 -run 'TestSystemOne|TestOptionBudget|TestPredict' -v .`
      passes 6 tests. `TestSystemOneReplay` runs all 30 `logits.jsonl` cases through
      `SystemOne` against the fake backend, which accepts only the recorded batch, cell for cell.
      Mutating qtype to 0 fails all 30. Locally, `TestDefaultLoaderReal` runs `SystemOne` through
      ONNX Runtime on the english export, and its result is byte-identical to upstream
      `Agent.system_one` for the same state and questions.
- [x] **7.3.7** **Precision and tie-break** (#24a/#30a). Softmax and entropy are numpy float32,
      `score` is float64 over a float32 `p`, `noul` is float64 from a float32 `p[1]`, and the act
      softmax is torch float32. `p.argmax()` is the **first** max. Acceptance: all 33
      `answers.jsonl` cases byte-equal, plus a test with two exactly equal logits that picks the
      first key. (2026-09-27) `TestAnswerGolden` checks all 33 against `answer_json`, byte for
      byte. `TestAnswerTieTakesFirstKey` checks keys `b, a` with logits `[0, 0]`. Mutating the
      tie-break to `>=` fails both.
- [x] **7.3.8** `Answer.MarshalJSON` per type via `jsonx.Obj`, with **no `omitempty`**: a
      criteria key of `""` is legal in Python, and `omitempty` would drop `"choice"` (#30). The
      same applies to empty `legend`/`probabilities`. Acceptance: key sets and order per
      `docs/API.md` for every `answers.jsonl` case, plus an injected `""` key. (2026-09-27)
      `TestAnswerGolden` asserts the key order per type, and `TestAnswerEmptyKeyIsEmitted` and
      `TestAnswerEmptyCollectionsAreEmitted` cover the rest. `json.Marshal` goes through
      `MarshalJSON` and equals the compacted fixture (D12).
- [x] **7.3.9** `Questions.Validate` rejects duplicate IDs with `ErrDuplicateQuestionID`, because
      a Python dict cannot hold them. (2026-09-27) This was already in `question/question.go`, and
      `SystemOne` calls it. `go test -count=1 -run TestQuestionsValidate -v ./question/` passes.
- [x] **7.3.10** Measure whether the act head's torch float32 softmax (`agent.py:295`) agrees with
      `exp32`, or with narrowed `math.Exp`, after `Round4`. If it does not, decide between porting
      ATen's exp and accepting a last-digit gap on `act_probability`. `answers.jsonl`'s 33
      `act_probability` values came from the dumper's **numpy** copy (`answer_block`), not from
      torch, so they cannot tell the two apart. (2026-09-27) Neither agrees. The dumper's
      `--act-softmax` writes `testdata/act_softmax.jsonl`: torch 2.14.0+cpu on AVX2, 4529 rows,
      2800 of them within 3 ulp of a `Round4` tie. Against it, `exp32` with a divide misses 96
      rows after Round4, and narrowed `math.Exp` misses 42. The user chose the port (D26):
      `calib.ActSoftmax` is ATen's short-row `_vec_softmax_lastdim` with SLEEF's `xexpf` and a
      multiply by `1/sum`, and it is bit-exact on all 4529 rows. Review follow-up: the vectorized
      reduction for rows of 8 or more is ported too, bit-exact on 8757 rows of widths 1–20, 32 and
      33, and the generator refuses a torch not on AVX2.
      `go test -count=1 -run TestActSoftmax -v ./internal/calib/` passes. Mutations to a divide,
      or to `exp32`, fail with 975 and 1754 rows off. The 33 `answers.jsonl` values still match
      under `ActSoftmax`.
- [x] **7.3.11** A behavioural test on the Go device-fallback policy, replacing upstream's
      `test_criteria.py:103-116` (which asserted source strings via `inspect.getsource` and is
      not ported). (2026-09-27) The resolve, CPU-retry and warn-once logic moved from `openSession`
      into the pure `placeSession` (`internal/backend/onnx/device.go`). `TestFallbackWarning` runs
      without ORT. It checks exactly one warning with `requested` and `reason` on a real
      fallback, including a failed session, and none on explicit `cpu`, on `auto` settling on
      CPU, or on a working device. `go test -count=1 -run 'TestDevice|TestFallbackWarning' -v ./internal/backend/onnx/`
      passes 4 tests. Locally, `TestOpenDevice` against ORT 1.23.0 asserts the same attributes
      on its 9 cases.
- [x] **7.3.12** The Go `_to_internal` (`agent.py:229-238`) rejects a `crit` of the wrong shape
      where Python raises (`crit.items()` on a non-dict for choice, `crit.get` on a list for noul,
      `common.py:42`). Today the renderer returns an empty or default option list instead, which
      is a silent plausible answer. (2026-09-27) Adds `prompt.CheckCriteria` and
      `ErrCriteriaShape`, and `question.ToInternal` for the root. Choice needs an object. Score
      needs a list, an object or `""`. Noul needs an object or a Python-falsy value. An unknown
      type is rejected. A non-empty string for score is rejected, where Python would render one
      level per character. The typed API cannot produce any of these shapes, so the check guards
      the `Internal` path. `go test -count=1 -run 'TestCheckCriteria|TestToInternal' -v ./internal/prompt/ ./question/`
      passes.
- [x] **7.3.13** Settle `state`'s type. `lang.Analyse`, `lang.IsEnglish` and `Route` take `any`,
      while `docs/API.md` proposes a `State` interface. Either introduce `State` and narrow all
      three, or record `any` as a decision and strike the §7 row. Do not add a third spelling.
      (2026-09-27) The user chose `any` (D25). `SystemOne` takes `any`. The `State` block in
      `docs/API.md` is replaced by a note, its signatures read `state any`, and the §7 row here
      says so.

**Task 7.4: Answer-formatting parity.**

- [x] **7.4.1** Table test against `testdata/answers.jsonl`, running against the fake backend,
      with no model needed. (2026-09-27) `answer_parity_test.go` sends every case through
      `SystemOne` with the mini_en tokenizer, a config carrying the case's temperatures and a
      one-row stub backend serving the recorded logits. Deviation (user decision): the stub
      replaces `internal/backend/fake`, which answers only logits.jsonl batches, and answers.jsonl
      records no state or batch; 7.4.3 adds the real-fake route. The three synthetic noul cases
      with k≠2 (`noul/k03`, `k06`, `k11`) are unreachable through `SystemOne`, since noul always
      renders two options, and stay covered by `TestAnswerGolden` alone; the test asserts exactly
      three skips. `go test -count=1 -run TestAnswerParity -v .` passes 30 cases; with the config's
      temperatures ignored, 23 fail.
- [x] **7.4.2** Compare the serialized JSON bytes, not just the parsed struct (#18 again).
      (2026-09-27) The same test compares `jsonx.Marshal` bytes with `answer_json`, and
      `json.Marshal` with its compacted form, besides the key order.
- [x] **7.4.3** `testdata/logits.jsonl` carries `result_json`, Python's whole `system_one` result,
      and the fake-backend replay compares it byte for byte. (2026-09-27) The generator builds it
      from the recorded forward pass (act head via torch softmax, D26) and refuses unless it equals
      a real `laya.Agent.system_one` run on every state; the existing fields regenerated
      bit-identically. `LAYA_MODELS=$PWD/models go test -count=1 -run TestSystemOneReplay -v .`
      passes 30 cases; a changed model name or a score off by 1e-4 fails all 30. The recorded act
      heads are saturated (every `act_probability` is 1.0), so D26 stays covered by
      `act_softmax.jsonl` alone.

**Task 7.5: End-to-end parity.** — ✅ DONE (2026-09-27)

- [x] **7.5.1** Test against `testdata/logits.jsonl`, gated behind `testing.Short()` and
      `LAYA_MODELS`. (2026-09-27) `e2e_parity_test.go`: `TestE2EParity` runs every case through
      `SystemOne` with the real tokenizer and ONNX Runtime on the dynamo exports, behind `-short`,
      `LAYA_MODELS` and `LAYA_ONNX_DIR`; `go test -short` reports it skipped.
- [x] **7.5.2** Assert per-option probabilities within 1e-4 **and** that the argmax decision never
      flips. The tolerance is per checkpoint, taken from measured numbers rather than a round
      figure. Go ORT at real prompts is already ≤ 9.8e-06 scaled on `logits` (eager export).
      PyTorch's own eager-vs-sdpa gap (absolute, logits / act) is english 1.0e-06 / 4.9e-04,
      multilingual 5.1e-05 / 1.2e-02, typed-decisions 1.5e-06 / 7.3e-04. See
      `docs/ARCHITECTURE.md` §5. (2026-09-27) The unrounded probabilities from Go's logits are
      compared with those from Python's recorded logits at the same temperature. Measured worst:
      english 4.3e-06, multilingual 2.8e-06, typed-decisions 1.4e-06, with no argmax flipping.
      `e2eTol` is 2× each, and the test holds every entry to 1e-4. The formatted answer is also
      checked against `result_json`: the same decision, and rounded probabilities at most one
      step (1e-4) apart. Adding 1e-3 to one logit fails all three checkpoints, and swapping the
      top two logits of a choice row fails them too. Ignoring the config's temperatures fails
      english and typed-decisions; multilingual's temperatures are all 1.0, so nothing changes.
- [x] **7.5.3** Run it for all three checkpoints. (2026-09-27)
      `LAYA_MODELS=$PWD/models LAYA_ONNX_DIR=$PWD/build/onnx go test -count=1 -run TestE2EParity -v .`
      passes english, multilingual and typed-decisions, 10 cases each.
- [x] **7.5.4** Port sections 2–5 of `original/tests/test_local_e2e.py`, with its loose directional
      thresholds (≥ 6/8 land on `billing`, ≥ 2/3 on the preset checks). (2026-09-27)
      `local_e2e_test.go`: `TestLocalE2E`, agents from the default loader, gated like 7.5.1.
      Sections 2–4 pass with multilingual 8/8 on billing and every preset check met; section 5
      (`TestLocalE2E/router`) is a second Router at `max_loaded=1` over all three checkpoints,
      driven only through `Predict`.
      `LAYA_MODELS=$PWD/models LAYA_ONNX_DIR=$PWD/build/onnx go test -count=1 -run 'TestLocalE2E|TestE2EParity' -v .`
      passes all four subtests. The thresholds are weak: the english checkpoint also scores 8/8,
      so they cannot tell the two checkpoints apart on these texts, and moderation's ≥ 2/3 still
      passes with an empty state.
- [x] **7.5.5** Port `test_local_e2e.py:190-211`, gated like 7.5.1: the triage intent lands in
      `{refund, billing_question}`; a language switch at `max_loaded=1` leaves
      `Loaded() == ["multilingual"]` (the only eviction test with real weights); and the `routing`
      payload is present and marshals. (2026-09-27) The triage clause is asserted by
      `TestLocalE2E/presets` (intent `refund`), the rest by `TestLocalE2E/router`: english, then
      hindi routes to multilingual and leaves `Loaded() == [multilingual]` with `dept` on
      `billing`, then `ForModel("typed-decisions")` is honoured and its `routing` marshals
      through jsonx to valid JSON. Same command as 7.5.4; with `WithMaxLoaded(2)` the eviction
      assertion fails (`Loaded() = [english multilingual]`).

**Task 7.6: README + examples.**

- [ ] **7.6.1** Port every README example to Go as compiling `Example` functions, so CI proves
      they build (§8). The single-model examples have a target since 7.7.5 (`Open`,
      `Agent.Predict`).
- [ ] **7.6.2** Fix the image URLs. Upstream's point at
      `raw.githubusercontent.com/NandhaKishorM/laya/main/...`, so a fork's README renders
      upstream's assets.
- [ ] **7.6.3** Document the deliberate deviations: `repo` is always a string (3.1.4);
      `instructions` is `string` only; the pinned default revision (D17); the graph is a local
      export and only the config and tokenizer are downloaded (D24); no ONNX backend off D21's platform list
      (`ErrUnsupportedPlatform`); `cuda` falls back to CPU (D22); no `mps` device; a config with `max_len`/`head_max_len` ≤ 0 or a
      `temperature` list that is not 3 long is rejected, and so is a `SetLimits`/`WithLimits`
      value ≤ 0 (`ErrInvalidLimits`, 7.7.5); an empty choice question is
      `ErrNoOptions` before the forward pass; `$LAYA_CACHE`
      instead of the huggingface_hub cache; and every dropped or renamed export: `proper_reward`,
      `td_lambda_targets` (D3), `ece_score` (internal `calib.ECE`), `load` (→ `Open`), `RLAgent`
      (no alias), `QTYPES`/`QTYPE_NAMES` (→ `QType`), `detect_language` (→ `lang.Analyse`), and
      `confidence_from_probs` and `render_options` (internal).
- [ ] **7.6.4** Replace the README's T4 latency claims ("33 ms", seven times) with the S3 numbers
      already in `BENCHMARKS.md`.
- [ ] **7.6.5** State the tokenizer and checkpoint revisions the port is verified against.
- [ ] **7.6.6** State the **measured** CPU latency and that batching does not amortise it. Never
      describe the port as "CPU-first".
- [ ] **7.6.7** Document the runtime: `go run ./cmd/laya-ort` downloads the pinned ORT 1.23.0,
      `LAYA_ORT_LIB` brings your own, and `Open` accepts any 1.x from 1.23 (D20), though
      everything was measured on 1.23.0.

**Task 7.7: `Router.Predict` / `Router.SystemOne`.** `router.py:293-311`: route, load, run
`system_one`, then add the decision under a `routing` key.

- [x] **7.7.1** Widen `Agent` with `SystemOne` (D13) and have the concrete agent satisfy it.
      (2026-09-27) `laya.go`: `SystemOne(ctx, state, qs) (*Result, error)` beside `Close`;
      `var _ Agent = (*onnxAgent)(nil)` in `predict_test.go`, and the LRU tests' `stubAgent`
      answers as upstream's `_Stub.system_one` does (`test_router.py:171`).
      `go test -count=1 -run 'TestRouterPredict|TestAgentInterface|TestUpstreamLRU' -v .` passes.
- [x] **7.7.2** `Predict`, and `SystemOne` as its alias (#53, `router.py:311`). (2026-09-27)
      `predict.go`: `Route`, `Load(decision.Model)`, `agent.SystemOne`. A failed route loads
      nothing; load and agent errors come back wrapped, and a nil result is an error.
      `TestRouterPredict`, `…Loading`, `…Alias` and `…Errors` pass with stub agents: the route
      options reach `Route`, the routed model loads once, a switch at `max_loaded=1` evicts and
      closes, an attached agent is used. Dropping the route options, or loading a fixed model,
      fails them.
- [x] **7.7.3** `result["routing"] = dict(decision)` (#53). This is where 3.1.4's string `repo`
      becomes visible to a caller. (2026-09-27) `Result.Routing` is a copy of the decision and
      equals `Route`'s for the same request. `TestRouterPredictRouting/workflow-repo-is-a-string`:
      on the auto-workflow path the repo is `DefaultModels()["typed-decisions"].String()`, a JSON
      string. Skipping the assignment fails three subtests.
- [x] **7.7.4** The `routing` block goes through `jsonx.Marshal` (D12), not `encoding/json`.
      (2026-09-27) `TestRouterPredictRouting/marshals-through-jsonx`: `jsonx.Marshal(res.Map())`
      is the bare result's bytes with a `"routing"` key holding `jsonx.Marshal(decision.Map())`
      appended last, in Python's separators, detection block included; `json.Marshal(res)` stays valid, routing after
      usage.
- [x] **7.7.5** A public entry point to a single agent. `docs/API.md` proposes an exported `*Agent`
      with `Open`, `SystemOne`, `Predict`, `MaxLen`/`HeadMaxLen`/`SetLimits`. Today `SystemOne`
      lives on the loader's unexported `onnxAgent`, reachable only through the Router, and no
      task owns the exported surface. `Agent` is already the Router's interface name, so the
      exported type needs another name or the interface does. (2026-10-09) The user chose to
      keep `Agent` for the struct and rename the interface to `Predictor` (D28). `agent.go`:
      `Open(ctx, ref, opts...)` is `laya.load` and builds through the Router's default loader,
      so D17's pin, D24's download and the `$HF_TOKEN` fallback are shared; `ref` `""` is the
      bundle repo. Options mirror `laya.load` (user decision): `WithSubfolder`, `WithDevice`,
      `WithHFToken`, `WithLimits`, plus `WithGraph`. Without `WithGraph`, the export name is the
      checkpoint the spec locates in either registry, and anything else, a local directory
      included, is `ErrNoGraph`. The Router's `WithONNXDir` name was taken, so `Open` reads only
      `$LAYA_ONNX_DIR` and the cache. `SetLimits` replaces `agent.cfg[...]`; `SystemOne` reads
      both limits once under an `RWMutex`, and a value ≤ 0 is `ErrInvalidLimits`.
      `go test -race -count=1 -run 'TestOpen|TestAgent|TestRouter|TestUpstreamLRU|TestDefaultLoader|TestSystemOne' -v .`
      passes 46 tests (the two real-weight ones skipped). Locally,
      `LAYA_MODELS=$PWD/models LAYA_ONNX_DIR=$PWD/build/onnx go test -count=1 -run 'TestOpenReal|TestE2EParity|TestLocalE2E' -v .`
      passes: `TestOpenReal` opens all three checkpoints with `Open` and matches the first
      `logits.jsonl` case's `result_json`. Each of these mutations fails a test:
      dropping `WithSubfolder`, matching the registry by repo alone, running every checkpoint on
      one export, an unlocked limits read (a `-race` report), a no-op `SetLimits`, and an
      ignored `WithLimits`.
- [x] **7.7.6** Eviction during `Predict`. Two goroutines on one Router can evict each other's
      agent between `Load` and `SystemOne` (at `max_loaded=1`, requests in two languages). The
      backend's `Close` waits for a pass already running, so nothing is freed under it, but a pass
      that has not started fails with the ONNX backend's `ErrClosed`, which is internal and not
      reachable with `errors.Is`. Python has the same window and GC to hide it. Decide between
      reference counting resident agents, a retry, or documenting it (the GoDoc does today).
      (2026-09-27) Decided: a lease (D27, user decision, after Codex's P1 on PR #30). Python has no
      such window, since the local reference keeps an evicted agent alive. `Predict` leases its
      agent under the lock that finds it. Eviction, `Unload` and `Close` drop the agent at once,
      and the call that dropped it waits outside the lock for its passes, then closes it and
      reports the error as before. `Load` stays unleased and says so.
      `go test -race -count=1 -run 'TestRouter|TestUpstreamLRU|TestAgentInterface' -v .` passes:
      `TestRouterPredictSurvives{Eviction,Unload,Close}` hold a pass open while its agent is
      dropped, and `…ReportsDeferredCloseError` and `…Concurrent` (8×25 requests) also pass.
      Each of these mutations fails them: dropping the lease, not waiting, never releasing.
- [ ] **7.7.7** The rest of `docs/API.md`'s `Open` options, left out of 7.7.5 (user decision):
      `WithCacheDir` (today only `$LAYA_CACHE`), `WithBackend` (a caller's `backend.Backend`,
      D9) and `WithLogger` (the ONNX device-fallback warning goes to `slog`'s default today).

### Backlog — open work that does not gate 1.0

Take these in any order once M7's parity is green, except B.5, which needs `SystemOne` to produce
the ECE/Brier inputs.

- [x] **B.1** Stream the tokenizer's 34 MB decode. `Open` on multilingual costs 0.6–1.2 s,
      207 MB and 1.54 M allocations, against a < 1 s cold budget. `encoding/json` over 256 000
      vocab entries and 580 604 merges is the whole cost. This is a contained follow-up, not a
      redesign.
      (2026-10-09) `model.vocab` and `model.merges` stay raw, and `tokenizer/decode.go` scans
      them once, resolving every merge to ids from the bytes. encoding/json still validates the
      document, decodes every non-plain string, and handles a repeated or case-variant key. The old
      decode is kept only as the oracle in `decode_test.go`. `TestDecodeMatchesEncodingJSON` gives
      identical tables on both mini fixtures and on english, multilingual and typed-decisions
      (`LAYA_MODELS`). `TestDecodeEdgeCases` passes, and `FuzzDecodeVocabMerges` ran 60 s with no
      failures. Three mutations each fail a test: raw bytes for escapes, later rank wins, and
      first-wins vocab. `just diff-tokenizer` is unchanged: en has 85 lines, all in the NFC
      normalize class (B.2), and ml has 0. Fresh-process `BenchmarkOpen` on multilingual measured
      0.42–0.47 s, 110 MB and 295 k allocs, against 0.51–0.53 s, 207 MB and 1.54 M before. On
      english it measured 34–63 ms and 11.6 MB, against 57 ms and 19.6 MB before. The numbers are
      in `BENCHMARKS.md`. The time that remains is mostly encoding/json's validation pass and map
      lookups, not allocation.
- [ ] **B.2** Decide on NFC combining classes, **in this file, before writing code**. The choice
      is between reproducing `tokenizers` 0.23.2's stale Rust tables (108 codepoints act as
      starters; isolating each occurrence reproduces the Rust result in ~40 lines plus a generated
      table) and staying Unicode-conformant. The first makes go-laya deliberately non-conformant
      and ties it to one crate's data, so `scripts/probe_ccc.py` would have to run on every pin
      bump. The second leaves an 85-line divergence that no natural text in 139 611 lines
      reaches and that cannot touch multilingual.
- [ ] **B.3** Enable CUDA. Device selection works for `cpu`/`auto`/`coreml`, while `cuda` falls
      back (D22). ORT's generic `SessionOptionsAppendExecutionProvider` rejects `CUDA`, and the
      binding at D5's pin neither registers `SessionOptionsAppendExecutionProvider_CUDA_V2` nor
      exposes `OrtSessionOptions`. It needs a binding change (upstream PR, or a fork via a module
      `replace`), which moves D5's pin. The T550 and `/opt/onnxruntime/gpu` on the dev box are
      enough to verify it. Update D22 and 7.6.3 when it lands.
- [ ] **B.4** Establish whether `purego.Dlopen` works on Windows with ORT 1.23, or only that
      nobody has tried. If it works, widen D21's platform list.
- **B.5 int8 dynamic quantization**, with latency and calibration measured together:
  - [ ] **B.5.1** `onnxruntime.quantization.quantize_dynamic` over all three exports in the
        pinned reference environment, as a `scripts/` flag, so the artefact is reproducible.
  - [ ] **B.5.2** Re-run `just bench-onnx` on the quantized graphs and record the speedup in
        `BENCHMARKS.md` beside fp32. At 2–4× it changes the throughput story, **not** the
        "interactive needs a GPU" one. Say so.
  - [ ] **B.5.3** Measure **ECE and Brier** (`calib.ECE`, `calib.Brier`) against fp32 on the same
        inputs, per checkpoint and per question type. This is the acceptance criterion, not a
        follow-up (R4).
  - [ ] **B.5.4** **int8 is never the default.** Ship it, if at all, as an explicit opt-in whose
        documentation carries B.5.3's numbers.
- [x] **B.6** Lint the workflows, TOML and YAML. `actionlint` would earn its place, because
      `ci.yml` is hand-edited on every formatter pin. Add it to `just ci` deliberately.
      (2026-10-09) `just lint-config` runs actionlint (with shellcheck over the `run:` blocks),
      `taplo lint` and `yamllint -s` over git-tracked files only, which keeps `original/` out. It
      is part of `just ci`, and a pinned `config` job runs it in CI. `.yamllint.yml` turns off only
      the rules prettier already settles. yamllint is Python, which the user approved for linters;
      AGENTS.md says so. Each tool fails its own seeded defect, and each defect was reverted: a
      duplicate key in `.golangci.yml` fails yamllint, `${{ matrix.oss }}` in `ci.yml` fails
      actionlint, and a broken table header in `treefmt.toml` fails taplo. The clean tree passes.

### M8 — Pure-Go native backend (after 1.0)

Same `backend.Backend` interface, no API change. It implements ModernBERT, mmBERT and the head over
safetensors, lifting `tensor`, `ops` and `safetensors` from `../go-pocket-tts`. This is a
zero-shared-library backend, not a speed play (D8). The facts it needs are in
`docs/ARCHITECTURE.md` §1.

- [ ] **8.1** Lift `internal/runtime/tensor` and `internal/runtime/ops` from `../go-pocket-tts`, and
      reconcile with the existing `internal/safetensors` header validator. go-pocket-tts has no
      `LICENSE`, so record the provenance in `NOTICE`.
- [ ] **8.2** Bias-free LayerNorm (eps 1e-5; layer 0 has no `attn_norm`).
- [ ] **8.3** GeGLU with the fused `mlp.Wi [5248,1024]` gate+up split, no MLP bias.
- [ ] **8.4** Fused QKV unpacking (`attn.Wqkv [3072,1024]`, no attention bias).
- [ ] **8.5** Sliding-window attention masks (window 128, ±64) and per-layer-type RoPE theta.
- [ ] **8.6** The decision head with a **ReLU** FFN and the manual layer loop.
- [ ] **8.7** fp16 weight loading. The loader tolerates the per-checkpoint `temperature` dtype,
      which is never read.
- [ ] **8.8** Gate promotion on the same golden vectors: probabilities within 1e-4 and zero argmax
      flips. Head intermediates for invariants #35–38 are generated here, as a reviewed
      regeneration.
- [ ] **8.9** Latency within 10× of ORT-CPU at 512 tokens on the same hardware, recorded in
      `BENCHMARKS.md` beside the S3 numbers.
- [ ] **8.10** The `backend` interface is unchanged. The native backend is selected by option, and
      `TestNoMLDependency` stays green.

---

## 5. Invariants a Go test must assert

The full numbered checklist (74 items) is in **`docs/INVARIANTS.md`**:

| Items | Area                                                       | Source              |
| ----- | ---------------------------------------------------------- | ------------------- |
| 1–13  | `build_sequence` token budget, layout, markers, truncation | `common.py:49-86`   |
| 14–18 | `serialize_state` / `render_options` / `render_criterion`  | `common.py:15-46`   |
| 19–34 | `_to_internal` + `system_one` numerics and answer format   | `agent.py:229-343`  |
| 35–38 | Forward pass, only if the head is reimplemented            | `common.py:105-126` |
| 39–47 | `Router.route` precedence and reason strings               | `router.py:241-290` |
| 48–53 | Router LRU lifecycle                                       | `router.py:144-238` |
| 54–65 | `lang` script/language detection                           | `lang.py`           |
| 66–74 | `mailtext` cleaning                                        | `email.py`          |

These four cause silent wrong answers rather than loud failures:

- **#18:** `serialize_state` byte parity (separators, key order, HTML escaping).
- **#29:** `round()` half-to-even vs Go's half-away-from-zero.
- **#56 / #63:** dict iteration order as a tie-break in `detect_script` and
  `guess_latin_language`.
- **#26:** `noul` confidence is `max(p1, 1-p1)`, not the entropy formula.

---

## 6. Risks

| #   | Risk                                                                                                     | Status and mitigation                                                                                                                                                                                                                                                   |
| --- | -------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **Pure-Go tokenizer drift.** Marker positions are token indices; one off-by-one corrupts every decision. | Mitigated. Both corpora are 100 % green, a fuzz target runs, and a 157 281-line differential found one class (NFC tables, 85 lines, no natural text): Backlog B.2. The `Tokenizer` interface keeps `daulet/tokenizers` (CGO) as a one-day swap, which is not indicated. |
| R2  | **ONNX export fails on ModernBERT-large** (transformers#35545).                                          | Retired by S1: all three export and validate.                                                                                                                                                                                                                           |
| R3  | **CPU latency ≫ upstream's 33 ms on a T4.**                                                              | Confirmed: 0.6–1.9 s at 512 tokens (D6). Published in `BENCHMARKS.md`. The levers left are int8 (B.5, gated on calibration), a GPU provider (B.3) and the correct thread count (done).                                                                                  |
| R4  | **int8 wrecks calibration.**                                                                             | B.5 measures ECE and Brier, not accuracy. int8 is never the default.                                                                                                                                                                                                    |
| R5  | **`onnxruntime-purego` instability**: an untagged 31-star HEAD with a finalizer data race.               | Every `*Value` is closed (D5), and the reproduction stays behind `LAYA_ORT_FINALIZER=1`. `yalue/onnxruntime_go` is the fallback behind the `Backend` seam. CUDA needs a binding change (B.3), which is the likeliest reason the pin moves.                              |
| R6  | **Upstream tokenizer/transformers drift** silently changes golden vectors.                               | Real: 4.57.6 moves multilingual by 6.96 (D7). `TestGoldenProvenance` checks every header, requirements are pinned, regeneration is byte-identical and reviewed, and `scripts/crosscheck_transformers.py` checks the next bump.                                          |
| R7  | **Supply chain**: `laya.Open("someone/their-model")` must not be RCE.                                    | Mitigated in M6: hash-verified downloads that fail closed, ONNX and safetensors headers validated before the runtime sees them, pinned and verified ORT, no `os/exec`/`encoding/gob` (CI grep). See `docs/ARCHITECTURE.md` §6.                                          |
| R8  | **Licensing**: derivative of an Apache-2.0 work.                                                         | `LICENSE` is kept and `NOTICE` records the derivation. Weights are Apache-2.0 and ungated. M8's lift from go-pocket-tts gets a `NOTICE` line.                                                                                                                           |

---

## 7. Proposed Go API

Full type definitions are in **`docs/API.md`**, together with the exact JSON shapes Python emits.
The dynamic-typing decisions:

| Python                                                      | Go decision                                                                                                            |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- |
| `criteria` on choice: `dict` **or** `list[str]`             | `[]ChoiceOption{Key, Desc any}` + a `Labels("a","b")` helper. One representation, order preserved.                     |
| `criteria` on noul: dict with `"true"`/`"false"`, or absent | Explicit `True`/`False` fields (false=0, true=1 always).                                                               |
| `state`: `str \| dict \| list`                              | `any` everywhere (`lang.Analyse`, `lang.IsEnglish`, `Route`, `SystemOne`); no `State` interface (D25).                 |
| `instructions`: `str` or anything                           | `string` only. The dropped edge case (non-str instructions are `json.dumps`'d with `ensure_ascii=True`) is documented. |
| dict iteration order                                        | Ordered slices wherever it is observable; `map` only where it provably is not.                                         |
| `RouteDecision` as a `dict` subclass                        | A struct with JSON tags, plus `Map() Obj`.                                                                             |

---

## 8. Definition of done for 1.0

- [ ] `just check` green; `go test ./... -race` green on linux/amd64 and darwin/arm64.
- [x] Both tokenizer golden corpora 100 % id- and token-identical to Python: 206 subtests
      (103 cases × 2 checkpoints, bare and with a leading space). This is a local gate
      (`LAYA_MODELS`). CI covers the stages via `testdata/pretok_{en,ml}.jsonl`.
- [x] `testdata/sequence.jsonl` byte-identical. (2026-09-27)
      `LAYA_MODELS=$PWD/models go test -count=1 -run TestBuildSequenceGolden -v ./internal/prompt/`
      gives 45 of 45 subtests passing and 0 skipped. It is a local gate, like the corpora above.
- [ ] End-to-end probabilities within 1e-4 of the fp32, CPU, single-thread, `sdpa` PyTorch run
      recorded in the fixture headers, on all three checkpoints; **zero argmax flips**.
- [x] `lang` + `mailtext` + `presets` + `backend` importable with no ML dependency. (2026-09-27)
      `go test -count=1 -run 'TestNoMLDependency|TestRuntimeImportedOnlyByBackend' -v .` passes
      both. It runs `go list -deps` over those four packages in every CI `go test`.
- [ ] Every README example compiles and runs.
- [ ] Measured latency published in `BENCHMARKS.md` for the hardware actually tested, replacing
      upstream's T4 numbers rather than repeating them. Upstream cites a `research/results/`
      directory that does not exist; do not inherit that.
- [ ] `TestGoldenProvenance` green, **and** it asserts the checkpoint revision the vectors came
      from. Today it checks library versions and `compute`, not the Hub sha.
- [ ] The ONNX artefacts are ones we export ourselves, not a third-party upload.
- [ ] The ORT version is pinned and every downloaded artefact is ETag/sha-verified before use (R7).
- [ ] The deliberate deviations from Python are listed in the README (Task 7.6.3).
