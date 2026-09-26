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
holds why, as D1–D23. The dated history of every finished task (evidence, mutation runs,
re-measurements) was condensed out on 2026-09-27 and is in git: `git show 8125f40:PLAN.md`. Ids of
finished tasks cited in code comments refer to that history.

---

## Status at a glance

Tick a box only when the work is committed and `just ci` is green. A milestone is done when every
box under it is ticked.

| Milestone                                                       | Delivers                                                | Status                |
| --------------------------------------------------------------- | ------------------------------------------------------- | --------------------- |
| [M0 — Scaffolding](#m0--scaffolding)                            | Go module, tooling, CI, frozen Python, `Version`        | ✅ done               |
| [Spikes S1–S3](#3-spikes)                                       | ONNX export, binding choice, latency floor              | ✅ done               |
| [M1 — Reference harness](#m1--the-python-reference-harness)     | `testdata/*.jsonl` golden vectors                       | ✅ done               |
| [M2 — Tier-1 core](#m2--tier-1-core)                            | `jsonx`, `lang`, `mailtext`, `presets`, render          | ✅ done               |
| [M3 — Router](#m3--router)                                      | `Route`, model registry, LRU                            | ✅ done               |
| [M4 — Tokenizer](#m4--pure-go-tokenizer)                        | pure-Go `tokenizer.json` loader                         | ✅ done               |
| [M5 — `build_sequence`](#m5--build_sequence)                    | prompt assembly, marker positions, collate              | ✅ done               |
| [M6 — Backend](#m6--backend--checkpoint-loading)                | `Backend`, hub cache, ONNX impl, validation, pinned ORT | ✅ done               |
| [M7 — Agent + parity](#m7--agent-calibration-end-to-end-parity) | loader, `SystemOne`, calibration, e2e parity, README    | 🟡 7.1 done bar 7.1.7 |
| [Backlog](#backlog--open-work-that-does-not-gate-10)            | tokenizer speed, NFC decision, CUDA, Windows, int8      | ⬜ open               |
| [M8 — Native backend](#m8--pure-go-native-backend-after-10)     | safetensors ModernBERT/mmBERT (post-1.0)                | ⬜ deferred           |

**Critical path to 1.0:** M7 (loader → `SystemOne` → answer parity → e2e parity → README). The
Backlog and M8 do not gate 1.0.

---

## 0. Decisions already made

Full rationale is in [`docs/DECISIONS.md`](docs/DECISIONS.md). Index:

| #   | Decision                                                            |
| --- | ------------------------------------------------------------------- |
| D1  | ONNX first, pure-Go native backend later, behind one `Backend`      |
| D2  | Pure-Go tokenizer, no CGO                                           |
| D3  | Inference only (`collate_items` is ported)                          |
| D4  | Upstream Python frozen under `original/`                            |
| D5  | `onnxruntime-purego` at `8db8bd7`; every `*Value` closed explicitly |
| D6  | CPU is a batch deployment; threads = physical cores; no int8        |
| D7  | Reference env on transformers 5.17.0; 4.x mis-loads mmBERT          |
| D8  | M8 is a zero-shared-library backend, not a speed play               |
| D9  | `backend` is a public leaf package                                  |
| D10 | Purpose-built tokenizer, no vendored fork                           |
| D11 | Question types in leaf `question/`, aliased at the root             |
| D12 | `jsonx.Marshal` is the outer encoder for every parity emitter       |
| D13 | Router caches an `Agent` interface, widened in M7                   |
| D14 | No cost-biased routing option                                       |
| D15 | Invalid UTF-8 sanitised at the tokenizer's API edge                 |
| D16 | Router closes only agents its own loader built                      |
| D17 | Default revision pinned to `1c5edc17…`                              |
| D18 | English checkpoint downloads only the repo root                     |
| D19 | Shipped export stays `eager`                                        |
| D20 | ORT is a verified download; any 1.x ≥ 1.23 accepted                 |
| D21 | ONNX backend on a positive platform list; stub elsewhere            |
| D22 | `cuda` falls back to CPU until the binding supports it              |
| D23 | Parity emitters reject rather than guess on unordered input         |

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

`TestAnswersNumerics` matches all 27 `answers.jsonl` cases exactly after `Round4`.

- [ ] **7.1.7** Make `answers.jsonl` discriminate precision. All 27 cases still pass with the
      softmax and entropy in float64, or with a different summation order. At four decimals the
      corpus cannot tell #24a's float32 path from the obvious one. Add generator cases whose
      rounded output differs between float32 and float64 (a probability or confidence within
      ~1e-7 of a `.00005` boundary), selected by the generator in the pinned environment, not by
      hand. This is a reviewed `testdata/` regeneration. 7.3.7's acceptance has the same blind
      spot. Now unblocked: 7.1.8 made exp/log bit-exact, so a near-boundary case cannot flip on
      the kernel.

**Task 7.2: The default agent loader.** Until it exists, `NewRouter()` without `WithLoader`
returns `ErrNoLoader`.

- [ ] **7.2.1** A default loader that builds an Agent from a spec's repo, subfolder, device and
      token, as `router.py:174-178` does. Default revision `1c5edc17…` (D17). A subfolder maps to
      `hub.Client.Snapshot(…, []string{sub + "/*"})`. The root (English) checkpoint downloads
      root files only (D18), which `Snapshot`'s allow patterns cannot express, so it needs a
      root-only filter. It passes `checkpoint.Config.ActWidth()` as `onnx.Options.ActWidth`.
- [ ] **7.2.2** `WithRouterDevice` and `WithRouterToken` (`docs/API.md:347-348`), including
      upstream's `token or os.environ["HF_TOKEN"]` fallback (`router.py:159`).
- [ ] **7.2.3** Widen `Agent` past `Close() error` only when M7 needs it (D13).
- [ ] **7.2.4** Offline from the environment: map `HF_HUB_OFFLINE` and/or `LAYA_OFFLINE` onto
      `hub.Client.Offline`.

**Task 7.3: `Agent.SystemOne`.** Invariants §5 items 19–34.

- [ ] **7.3.1** The `choice` answer shape.
- [ ] **7.3.2** The `score` answer: `Σ i·p[i]`, an expectation, **not** an argmax.
- [ ] **7.3.3** The `noul` answer: **no** `probabilities`, no `legend`, and confidence
      `max(p1, 1-p1)`, **not** the entropy formula, which for k=2 genuinely disagrees (#26).
- [ ] **7.3.4** `legend` carries the **raw** criterion value, not the rendered option text.
- [ ] **7.3.5** The act head: `act = softmax(act_logits.float(), -1)` (torch float32,
      `agent.py:295`), then `action.act_probability = round(act[r, 0], 4)` (`agent.py:310`).
      Column 0 only. There is no threshold and no decision upstream. The width is
      `len(cfg["act_costs"]) + 1`, validated in 6.4.5. This is torch's exp, not numpy's (7.3.10).
- [ ] **7.3.6** Batching: several questions in one forward pass, with per-question `qtype`,
      through `Collate`.
- [ ] **7.3.7** **Precision and tie-break** (#24a/#30a). Softmax and entropy are numpy float32,
      `score` is float64 over a float32 `p`, `noul` is float64 from a float32 `p[1]`, and the act
      softmax is torch float32. `p.argmax()` is the **first** max. Acceptance: all 27
      `answers.jsonl` cases byte-equal, plus a test with two exactly equal logits that picks the
      first key.
- [ ] **7.3.8** `Answer.MarshalJSON` per type via `jsonx.Obj`, with **no `omitempty`**: a
      criteria key of `""` is legal in Python, and `omitempty` would drop `"choice"` (#30). The
      same applies to empty `legend`/`probabilities`. Acceptance: key sets and order per
      `docs/API.md` for every `answers.jsonl` case, plus an injected `""` key.
- [ ] **7.3.9** `Questions.Validate` rejects duplicate IDs with `ErrDuplicateQuestionID`, because
      a Python dict cannot hold them.
- [ ] **7.3.10** Measure whether the act head's torch float32 softmax (`agent.py:295`) agrees with
      `exp32`, or with narrowed `math.Exp`, after `Round4`. If it does not, decide between porting
      ATen's exp and accepting a last-digit gap on `act_probability`. `answers.jsonl`'s 27
      `act_probability` values came from the dumper's **numpy** copy (`answer_block`), not from
      torch, so they cannot tell the two apart.
- [ ] **7.3.11** A behavioural test on the Go device-fallback policy, replacing upstream's
      `test_criteria.py:103-116` (which asserted source strings via `inspect.getsource` and is
      not ported).
- [ ] **7.3.12** The Go `_to_internal` (`agent.py:229-238`) rejects a `crit` of the wrong shape
      where Python raises (`crit.items()` on a non-dict for choice, `crit.get` on a list for noul,
      `common.py:42`). Today the renderer returns an empty or default option list instead, which
      is a silent plausible answer.
- [ ] **7.3.13** Settle `state`'s type. `lang.Analyse`, `lang.IsEnglish` and `Route` take `any`,
      while `docs/API.md` proposes a `State` interface. Either introduce `State` and narrow all
      three, or record `any` as a decision and strike the §7 row. Do not add a third spelling.

**Task 7.4: Answer-formatting parity.**

- [ ] **7.4.1** Table test against `testdata/answers.jsonl`, running against the fake backend,
      with no model needed.
- [ ] **7.4.2** Compare the serialized JSON bytes, not just the parsed struct (#18 again).

**Task 7.5: End-to-end parity.**

- [ ] **7.5.1** Test against `testdata/logits.jsonl`, gated behind `testing.Short()` and
      `LAYA_MODELS`.
- [ ] **7.5.2** Assert per-option probabilities within 1e-4 **and** that the argmax decision never
      flips. The tolerance is per checkpoint, taken from measured numbers rather than a round
      figure. Go ORT at real prompts is already ≤ 9.8e-06 scaled on `logits` (eager export).
      PyTorch's own eager-vs-sdpa gap (absolute, logits / act) is english 1.0e-06 / 4.9e-04,
      multilingual 5.1e-05 / 1.2e-02, typed-decisions 1.5e-06 / 7.3e-04. See
      `docs/ARCHITECTURE.md` §5.
- [ ] **7.5.3** Run it for all three checkpoints.
- [ ] **7.5.4** Port sections 2–5 of `original/tests/test_local_e2e.py`, with its loose directional
      thresholds (≥ 6/8 land on `billing`, ≥ 2/3 on the preset checks).
- [ ] **7.5.5** Port `test_local_e2e.py:190-211`, gated like 7.5.1: the triage intent lands in
      `{refund, billing_question}`; a language switch at `max_loaded=1` leaves
      `Loaded() == ["multilingual"]` (the only eviction test with real weights); and the `routing`
      payload is present and marshals.

**Task 7.6: README + examples.**

- [ ] **7.6.1** Port every README example to Go as compiling `Example` functions, so CI proves
      they build (§8).
- [ ] **7.6.2** Fix the image URLs. Upstream's point at
      `raw.githubusercontent.com/NandhaKishorM/laya/main/...`, so a fork's README renders
      upstream's assets.
- [ ] **7.6.3** Document the deliberate deviations: `repo` is always a string (3.1.4);
      `instructions` is `string` only; the pinned default revision (D17); the English checkpoint
      downloads the root only (D18); no ONNX backend off D21's platform list
      (`ErrUnsupportedPlatform`); `cuda` falls back to CPU (D22); no `mps` device; `$LAYA_CACHE`
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

- [ ] **7.7.1** Widen `Agent` with `SystemOne` (D13) and have the concrete agent satisfy it.
- [ ] **7.7.2** `Predict`, and `SystemOne` as its alias (#53, `router.py:311`).
- [ ] **7.7.3** `result["routing"] = dict(decision)` (#53). This is where 3.1.4's string `repo`
      becomes visible to a caller.
- [ ] **7.7.4** The `routing` block goes through `jsonx.Marshal` (D12), not `encoding/json`.

### Backlog — open work that does not gate 1.0

Take these in any order once M7's parity is green, except B.5, which needs `SystemOne` to produce
the ECE/Brier inputs.

- [ ] **B.1** Stream the tokenizer's 34 MB decode. `Open` on multilingual costs 0.6–1.2 s,
      207 MB and 1.54 M allocations, against a < 1 s cold budget. `encoding/json` over 256 000
      vocab entries and 580 604 merges is the whole cost. This is a contained follow-up, not a
      redesign.
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
- [ ] **B.6** Lint the workflows, TOML and YAML. `actionlint` would earn its place, because
      `ci.yml` is hand-edited on every formatter pin. Add it to `just ci` deliberately.

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

| Python                                                      | Go decision                                                                                                                   |
| ----------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------- |
| `criteria` on choice: `dict` **or** `list[str]`             | `[]ChoiceOption{Key, Desc any}` + a `Labels("a","b")` helper. One representation, order preserved.                            |
| `criteria` on noul: dict with `"true"`/`"false"`, or absent | Explicit `True`/`False` fields (false=0, true=1 always).                                                                      |
| `state`: `str \| dict \| list`                              | Proposed: a `State` interface. Shipped so far: `any` in `lang.Analyse`, `lang.IsEnglish` and `Route`. Settled by Task 7.3.13. |
| `instructions`: `str` or anything                           | `string` only. The dropped edge case (non-str instructions are `json.dumps`'d with `ensure_ascii=True`) is documented.        |
| dict iteration order                                        | Ordered slices wherever it is observable; `map` only where it provably is not.                                                |
| `RouteDecision` as a `dict` subclass                        | A struct with JSON tags, plus `Map() Obj`.                                                                                    |

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
