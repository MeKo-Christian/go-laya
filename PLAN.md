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

| Milestone                                                              | Delivers                                         | Status                            |
| ---------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------- |
| [M0 — Scaffolding](#m0--scaffolding)                                   | Go module, tooling, CI, frozen Python, `Version` | ✅ 5/6 (0.1 skipped)              |
| [Spikes S1–S3](#3-spikes--do-these-before-writing-library-code)        | ONNX export, binding choice, latency floor       | 🟢 S1–S3 done                     |
| [M1 — Reference harness](#m1--the-python-reference-harness)            | `testdata/*.jsonl` golden vectors                | ✅ done                           |
| [M2 — Tier-1 core](#m2--tier-1-core-no-ml-runtime-620-lines-of-python) | `jsonx`, `lang`, `mailtext`, `presets`, render   | 🟢 2.1–2.4 done; 2.5.4 partial    |
| [M3 — Router](#m3--router-pure-no-weights-no-network)                  | `Route`, model registry, LRU                     | ✅ done                           |
| [M4 — Tokenizer](#m4--pure-go-tokenizer-highest-risk)                  | pure-Go `tokenizer.json` loader ⚠️               | 🟢 4.1–4.5 done; 4.3.9/4.5.5 open |
| [M5 — `build_sequence`](#m5--build_sequence)                           | prompt assembly + marker positions               | ✅ done                           |
| [M6 — Backend](#m6--backend--checkpoint-loading)                       | `Backend` iface, hub cache, ONNX impl            | 🟡 6.1 done; 6.2 open: 6.2.9 only |
| [M7 — Agent + parity](#m7--agent-calibration-end-to-end-parity)        | `SystemOne`, calibration, e2e parity, README     | ⬜ not started                    |
| [M8 — Native backend](#m8--pure-go-native-backend-after-10)            | safetensors ModernBERT/mmBERT (post-1.0)         | ⬜ deferred                       |

**Critical path:** M1 ✅ → M2 (`jsonx` first) → M4 → M5 → M6 → M7, with M3 off the path. _(Reordered
on the 2026-09-20 review.)_ _(2026-09-20: M4 was in the event run **in parallel** with M2 from `jsonx`
onwards. The order above is a scheduling preference, not a dependency — M4 imports nothing from M2
and the two meet only at M5.)_ M2 goes before M4 because M5 needs `jsonx`, M3 needs `lang`, and M2 has
no unknowns left — it does not get cheaper by waiting, and M4's shape (D10) is now settled. Tasks 1.7
and 4.4.12 were taken first for that reason and are **done** (2026-09-20): both rewrite `testdata/`,
so no Go work that reads it could run alongside them. M3
depends only on `lang` and fills the idle time during M4's differential runs.

---

## 0. Decisions already made

| #   | Decision                                                                                                  | Rationale                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| --- | --------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| D1  | **ONNX first, pure-Go native backend later**, both behind a `Backend` interface                           | The entire `DecisionModel.forward` is one static graph — no KV cache, no loop, no control flow — so it exports to a single ONNX file and there is exactly _one_ numerics-parity surface to test. Reimplementing ModernBERT (RoPE, alternating local/global attention, GeGLU) up front is weeks of work before anything runs. In-house precedent: `yalue/onnxruntime_go` in `pogo`, `FlashSR`, `go-autoresearch`, `Emanetics`.                                                                                                                                                                                                                                                                                                                                                                         |
| D2  | **Pure-Go tokenizer, no CGO**                                                                             | User decision. Keeps the build CGO-free and cross-compilable. This is the single largest correctness risk in the port (see §6 R1) and is therefore front-loaded: golden corpus before implementation.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| D3  | **Inference only**                                                                                        | `proper_reward` and `td_lambda_targets` are RLCD training math a Go port cannot use. **`collate_items` is not** _(corrected 2026-09-20)_: `agent.py:266` calls it on every `system_one`, and its padding (`common.py:218-251`) is what the graph sees — it is ported in Task 5.3. Deliberate API narrowing; Task 7.5.3 carries the full list of dropped exports for the README.                                                                                                                                                                                                                                                                                                                                                                                                                       |
| D4  | **Python moves to `original/`**                                                                           | Same pattern as `go-pocket-tts`. Upstream Python is frozen as the parity reference and drives `scripts/dump_python_parity.py`. Also keeps the Apache-2.0 derivative-work attribution honest.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| D5  | **`shota3506/onnxruntime-purego` for ONNX, pinned to `8db8bd7`; every `*Value` closed explicitly**        | Spike S2. The CGO-free path holds: `CGO_ENABLED=0 go build ./...` passes and the S1 export runs through the binding, agreeing with the Python ORT run to 3.8e-06. R5's `runtime.AddCleanup` panic is fixed upstream (PR #11, merged 2026-03-15); `go-pocket-tts` is pinned two commits short of it. The cost is stated rather than hidden: the binding carries **no tags at all**, so the pin is a pseudo-version of the untagged HEAD of a 31-star library whose README says "APIs may change without notice", and a live data race remains on the finalizer path. `yalue/onnxruntime_go` (719 stars, v1.36.0, cgo) stays the fallback, and D1's `Backend` seam is what keeps the swap contained.                                                                                                    |
| D6  | **CPU is a batch deployment; no quantization by default; `IntraOpNumThreads` is the physical core count** | Spike S3. Measured through the shipped binding on a 15 W laptop: 0.6–1.9 s per question at the default 512-token `max_len`, 9–43× upstream's 33 ms on a T4. Two measured surprises drive the decision rather than the headline number: **batching does not amortise on CPU** (per-question cost is flat, where the T4 falls 4.6×), and **all hardware threads is the wrong setting** (every checkpoint peaks at 8 of 12 and then loses 12–26%). So go-laya ships CPU as the default, documents it as a background workload, and treats a GPU execution provider as the answer for interactive use. int8 stays off the default path until Task 6.9 measures ECE and Brier, not accuracy (R4).                                                                                                          |
| D7  | **The reference environment stays on transformers 5.17.0; 4.x is unsafe for this port**                   | Task 1.6, measured rather than assumed. The two ModernBERT-large checkpoints are bit-identical across the major, but 4.57.6 silently mis-parses mmBERT's `rope_parameters.sliding_attention.rope_theta` (160000) and substitutes its own default of 10000.0, so every sliding-attention layer runs on the wrong RoPE frequencies — logits off by 6.96, act_logits by 1879. Correcting that one value makes 4.57.6 bit-identical, so this is a parsing bug, not a numerics difference. English is unaffected only because its sliding `rope_theta` genuinely is 10000.0. This **inverts** Task 1.6.2, which presumed 4.x was the safer pin. Upstream's declared `transformers>=4.45.0` floor is therefore wrong twice over: it predates ModernBERT (4.48) and it admits versions that mis-load mmBERT. |
| D10 | **Purpose-built pure-Go tokenizer; no vendored fork**                                                     | Review of 2026-09-20. Task 4.3 had planned to vendor `gomlx/go-huggingface`'s `hftokenizer` and fix three bugs. The stages it gets wrong are exactly the ones laya needs (added-token semantics, Metaspace prepend, byte fallback); the parts it gets right (JSON loader, merge loop, byte table) are the easy ~250 lines; and the in-house byte-fallback code §1.5 pointed at does not exist. Two pipelines, ~750 production lines, every semantic pinned by golden corpus before code (Task 4.4.12). gomlx stays a reference to read, never to vendor; `daulet/tokenizers` stays R1's CGO fallback. D2 holds: the build remains `CGO_ENABLED=0` end to end.                                                                                                                                         |
| D8  | **M8 is a zero-shared-library backend, not a speed play**                                                 | Review of 2026-09-20. S3 measured ORT-CPU as bandwidth-bound (1→8 threads = 2.1×, batching flat) and the FLOP estimate puts a pure-Go forward at ~12 s against ORT's 1.7 s, so the old Task 8.9 ("beat the S3 floor") was unreachable by construction. What a pure-Go backend does buy is deployment with no `.so` at all: it solves Tasks 6.6 and 6.7 (runtime pinning, Windows, wasm) by construction. Post-1.0, behind the same `backend` interface, and it no longer shapes near-term wording (the `temperature` dtype, invariants #35–38).                                                                                                                                                                                                                                                       |
| D9  | **`backend` is a public leaf package**                                                                    | Review of 2026-09-20. D5's "swap to `yalue/onnxruntime_go`" seam is only real for users if `Backend` and `Batch` are importable, and `docs/API.md`'s `WithBackend` took an internal type. `backend/` holds the interface and the batch struct with zero dependencies; `internal/backend/onnx/` holds the ORT implementation. Under purego nothing "links ONNX symbols" — the binding `dlopen`s at `Open` — so the §8 no-ML-dependency check is `go list -deps ./lang ./mailtext ./presets ./backend` containing no `onnxruntime-purego`. The root package is allowed to depend on the binding.                                                                                                                                                                                                        |
| D11 | **The question types live in a leaf `question/`, re-exported at the root**                                | Review of 2026-09-20 (M2). §2 put `question.go` at the root, but Task 2.4.2 makes `presets` import those types and D9's own check requires `go list -deps ./presets` to stay free of the ONNX binding — which D9 equally allows the **root** package to pull in. Root-resident types put every preset one import edge from the runtime, so the check would start failing the moment M6 lands, for a reason nothing in M2 would explain. The types move to `question/`; the root file holds **type aliases only**, so `laya.ChoiceQuestion` is the same type under the documented name and the public API is unchanged.                                                                                                                                                                                |
| D12 | **`jsonx.Marshal` is the outer encoder wherever Python parity matters**                                   | Review of 2026-09-20 (M2). `encoding/json` runs `compact()` over whatever a `MarshalJSON` returns, so passing a `jsonx.Obj` through `json.Marshal` — directly, or by embedding it in a struct `json.Marshal` handles — strips the `", "` and `": "` separators again and silently undoes `jsonx`. A `MarshalJSON` that only ever reaches `encoding/json` is decoration. This is why Task 2.2.7's acceptance ("`json.Marshal(RouteDecision)` byte-equal to Python") is unreachable as written, and it constrains every later emitter: Task 7.3's byte comparison, `Answer.MarshalJSON`, `Probs`, `RouteDecision`. Pinned by `TestEncodingJSONCompactsMarshalerOutput`.                                                                                                                                 |
| D13 | **The Router caches an `Agent` interface, widened in M7**                                                 | M3. `docs/API.md` declares `Router.Load`, `Attach` and `WithLoader` in terms of `*Agent`, and `Agent` is an M6/M7 type — so M3 could not be built as written. What the Router actually asks of an agent is _release_: it caches them, evicts the least recently used, and must close what it drops. So `Agent` is an interface carrying `Close() error`, and M7 widens it with `SystemOne`. Upstream's own LRU tests never build a real agent either — `_Stub` at `test_router.py:167-172` is a bare object — so an interface is also what makes them portable, through `WithLoader`. The cost is stated rather than hidden: widening an interface breaks any third-party implementor, which is acceptable pre-1.0 and is why the widening is named here rather than discovered in M7.                |

**D2 has a consequence that needs resolving in Spike S2:** `yalue/onnxruntime_go` requires CGO (it
`dlopen`s the shared library _through_ cgo). A CGO-free tokenizer paired with a CGO ONNX binding
gives up the benefit. `go-pocket-tts` already uses `github.com/shota3506/onnxruntime-purego`, which
is CGO-free — but its own PLAN.md records a real defect: _"local ONNX-backed native parity tests can
panic inside `onnxruntime-purego` with `runtime.AddCleanup`"_. S2 decides between the two; default to
purego for consistency with D2, fall back to `yalue/onnxruntime_go` if purego proves unstable.

> **(2026-09-20) — answered, and the recorded defect was only half the story.** The panic
> `go-pocket-tts` hit is upstream PR #11, and it is fixed. Its pin
> (`v0.0.0-20251207004809-1c85186598a5`) is the last commit before the fix, and at that pin the
> **first** `NewTensorValue` call panics outright — so its ONNX path is broken on Go 1.24+ rather
> than flaky. A _different_ defect survives at the fixed HEAD: the package contains no
> synchronisation at all, and `Runtime.Close` writes `r.apiFuncs = nil` (`runtime.go:178`) while
> the GC's cleanup goroutine reads it in `releaseValuePtr` (`value.go:152`). The `!= nil` guard
> there is a data race on a multi-word struct, not a safety net. It is avoidable — see D5 — but it
> is a standing reason to keep the `Backend` seam honest.

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
The `temperature` tensor is F32 in two checkpoints and F16 in the third. It is **dead at inference**
_(review, 2026-09-20)_: `common.py:102` registers the buffer and `forward` never reads it; calibration
comes from `cfg["temperature"]` / `cfg["temperature_by_options"]` in `rl_agent_config.json`
(`agent.py:194-195,304`). Its dtype matters only so a strict safetensors loader (M8) does not assume
one dtype per file.

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
act_head.2.{weight,bias} [n_act, 256]   # n_act = len(cfg["act_costs"]) + 1 (common.py:137); 2 on all three shipped checkpoints
temperature              [3]            # registered buffer, never read at inference (§1.1)
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
116 added tokens, 109 of them non-special with `normalized:true` — 23 runs of **2–24** spaces at ids
**50254–50276** (the 24-space run is 50254), `|||IP_ADDRESS|||` at id 0, `|||EMAIL_ADDRESS|||` at 50277,
`|||PHONE_NUMBER|||` at 50278, and **83** `[unusedN]` (`[unused0]`..`[unused82]`, ids 50285–50367)
_(corrected again 2026-09-20 from the file: the earlier "85" double-counted the two extra
placeholders; 23 + 83 + 7 + 3 = 116 ✓)_ ·
**UNK 50280, CLS 50281, SEP 50282, PAD 50283, MASK 50284** (mask has `lstrip:true`).

**Multilingual (Gemma/mmBERT):** normalizer `Replace{" " → "▁"}` · pre-tokenizer
`Metaspace{replacement:"▁", prepend_scheme:"always", split:true}` · BPE, vocab 256000, 580604 merges,
**`byte_fallback:true`, `fuse_unk:true`** · 249 added tokens ·
**PAD 0, EOS/SEP 1, BOS/CLS 2, UNK 3, MASK 4**. All 249 added tokens are `normalized:false` and include
runs of `\n`×1–31, `\t`×2–31 and `▁`×2–31 — matched on the raw text, before `Replace` turns spaces
into `▁`. `typed-decisions/tokenizer/tokenizer.json` is byte-identical to the English one, so there are
exactly two pipelines.

**Two semantics the Go tokenizer must get right, from config and not by convention:**

1. `tok(" " + text)` — which `laya/common.py:68` does for every option — behaves _oppositely_ in the two
   checkpoints. On multilingual, the `Replace` normalizer turns `" "` into `"▁"` before Metaspace runs,
   so the text already starts with `▁` and HF's `if !starts_with(replacement)` guard **suppresses the
   prepend**: `tok(" x") == tok("x")`. On English (ByteLevel, `add_prefix_space:false`) the leading
   space is real and becomes `Ġ`: `tok(" x") != tok("x")`.

   > ⚠️ _(corrected 2026-09-20, M4)_ The multilingual half holds only when the text does **not** begin
   > with an added token. When it does, the added-token pre-split isolates the leading `" "` as its own
   > segment, which `Replace` + Metaspace turn into `▁` = id 235248: `"<2mass>"` → `[5]` but
   > `" <2mass>"` → `[235248, 5]`. Measured over the corpus, **9 of 103** multilingual cases differ
   > with a leading space, not zero. The same mechanism gives a mid-word continuation its own `▁`:
   > `"ab<mask>cd"` → `['▁ab', '<mask>', '▁cd']`.

2. Added tokens are pre-split out of the input **before** normalizer and pre-tokenizer run. Since
   `build_sequence` feeds `serialize_state(state)` — arbitrary user JSON — through the tokenizer,
   indentation runs are realistic input and the 109 whitespace-run added tokens are load-bearing.
3. _(review, 2026-09-20)_ Byte coverage is a load-time assertion, not a runtime path. The ML vocab has
   255 of the 256 `<0xXX>` byte tokens — `<0x09>` is missing, harmless because U+0009 is itself a vocab
   entry. The EN byte-level alphabet lacks 13 byte-chars (0xC0, 0xC1, 0xF5–0xFF), which never occur in
   valid UTF-8; with `unk:null` an invalid byte would be **silently dropped**, which is why `Encode`
   sanitises to valid UTF-8 first (Task 4.1.2).

### 1.5 Prior art that can be reused in-house

- `../go-pocket-tts/internal/safetensors/` — reader/store/writer.
- `../go-pocket-tts/internal/runtime/tensor/` — `MatMul`, `Linear`, `LayerNorm`, `Softmax`, plus
  hand-written amd64/arm64 assembly for `dot` and `axpy`.
- `../go-pocket-tts/internal/runtime/ops/` — `Attention` (incl. fused 4D, parallel), `RoPE`, `MLP`.
- `../go-pocket-tts/scripts/dump_python_parity.py` + `internal/*/parity*.go` — the env-gated parity-test
  idiom was borrowed from here. The JSONL-with-header-record design (Task 1.3) is go-laya's own: the
  sibling emits one JSON object with no version header.

Struck on review (2026-09-20): `internal/tokenizer/sentencepiece_bytes.go` was listed here as
"byte-fallback handling". It is a temp-file wrapper around a SentencePiece UNIGRAM library and contains
no byte-fallback logic; go-pocket-tts has no `tokenizer.json` loader, no BPE and no normalizers at all,
so M4 starts from zero in-house code (D10). Note also that go-pocket-tts carries no `LICENSE` file — same
author, so lifting is fine in practice, but record the provenance in `NOTICE` when M8 does it.

These make Milestone M8 (the pure-Go native backend) far more tractable than a from-scratch estimate suggests.

---

## 2. Target repository layout

```
go-laya/
  go.mod                       module github.com/MeKo-Christian/go-laya
  justfile  treefmt.toml  .golangci.yml
  PLAN.md  README.md  LICENSE  NOTICE
  laya.go        Agent, Open, SystemOne, Result/Answer
  question.go    type aliases re-exporting question/ (D11); no types of its own
  router.go      Router, RouteDecision, model registry, LRU
  options.go     functional options
  errors.go      sentinel errors
  lang/          script + language detection      (public, zero deps)
  mailtext/      email cleaning                   (public, zero deps)
  presets/       the five question presets        (public, zero deps)
  jsonx/         ordered JSON with Python byte-parity -- the outer encoder (D12)
  question/      Question interface + Choice/Score/Noul, a leaf so presets stays ONNX-free (D11)
  tokenizer/     Tokenizer interface + purpose-built pure-Go tokenizer.json loader (D10)
  backend/       Backend interface + Batch -- public leaf, zero deps (D9)
  internal/prompt/   render_options, render_criterion, build_sequence, collate
  internal/golden/   the testdata/ corpus loader, shared by every package that asserts against it
  internal/calib/    softmax, entropy confidence, temperature, py-round, ECE
  internal/hub/      HF resolve + local cache
  internal/backend/fake/  replays testdata/logits.jsonl per checkpoint, so tests above the backend need no ORT (Task 6.1)
  internal/backend/onnx/  ONNX Runtime implementation; absorbs internal/onnxspike in M6 (Task 6.10)
  internal/onnxspike/ Spikes S2/S3; M6 moves it into internal/backend/onnx rather than deleting it
  cmd/laya/          optional CLI
  scripts/           dump_python_parity.py, export_onnx.py, crosscheck_transformers.py (scripts/README.md)
  testdata/          golden vectors (checked in; CI never needs Python)
  docs/              API.md (type appendix; PLAN.md wins on conflict), INVARIANTS.md (the 74 assertions)
  original/          frozen upstream Python, reference only
```

`lang`, `mailtext`, `presets` and `backend` have **no dependency on the runtime**, so a user who only
wants email cleaning or language detection never loads ONNX Runtime. _(2026-09-20, M3: this sentence
used to say "routing", and routing is the one thing it does not cover. `router.go` is at the root by
this very layout, and D9 allows the root to depend on the binding, so from M6 a routing-only importer
gains a build-graph edge to `onnxruntime-purego`. It still never `dlopen`s the library — that happens
in `Open` — and §8's check never included the root. The claim is narrowed, not the layout.)_ That is the biggest structural win over the
Python layout, where `import laya` drags in torch. Under purego nothing is _linked_ — the binding
`dlopen`s the library inside `Open` — so the testable form of the claim is
`go list -deps ./lang ./mailtext ./presets ./backend` containing no `onnxruntime-purego` (D9) — in
practice also `./jsonx` and `./question`, which M2 added below `presets` (D11); the root
package may depend on the binding.

---

## 3. Spikes — do these before writing library code

These can invalidate the plan. Budget 1–2 days total. **Status:** S1 ✅ · S2 ✅ · S3 ✅

### Spike S1: Does `torch.onnx.export` survive the full `DecisionModel`?

**Files:** create `scripts/export_onnx.py`, `original/` must already exist (Task 0.3).

ModernBERT decorates functions with `torch.compile`, which `torch.onnx.export(dynamo=False)` rejects
(huggingface/transformers#35545, still open). `optimum` works around it with a
`DisableCompileContextManager` and by forcing `attn_implementation="eager"`.

> **(2026-09-20) — done. The premise above turned out to be wrong, twice over.**
> transformers 5.17.0 has no `torch.compile` left in `modeling_modernbert.py` at all —
> `reference_compile` survives only as a key `ModernBertConfig.to_dict` pops for backwards
> compatibility — so #35545 never fires and the **encoder traces straight through**. What
> actually blocks the export is the **decision head**: `nn.TransformerEncoderLayer` dispatches
> to the fused `aten::_transformer_encoder_layer_fwd` in eval mode, and that op has no ONNX
> symbolic. `norm_first=True` does not disable it (the fused path takes `norm_first` as an
> argument); `torch.backends.mha.set_fastpath_enabled(False)` does, and is the first condition
> `why_not_sparsity_fast_path` tests.
>
> The second blocker is worse because it is silent: the legacy TorchScript exporter **freezes
> the sequence length**. transformers' mask builder resolves a Python `bool` on
> `attention_mask.shape[-1]` (`masking_utils.py:213`), emits a `TracerWarning`, and carries on.
> The resulting file loads and runs perfectly at the traced length and fails in ORT at every
> other one. `dynamo=True` at **opset 18** keeps the axis dynamic. Opset 17 under dynamo
> produces a graph ORT refuses to load — the 18→17 downconversion leaves `Split` carrying its
> opset-18 `num_outputs` attribute.

- [x] **S1.1** Write `scripts/export_onnx.py`; build the model with `reference_compile=False` **and**
      `attn_implementation="eager"`. (2026-09-20) — the exporter replicates `build_model`
      rather than calling it, because `common.py:129` hardcodes `attn_implementation="sdpa"`.
      `reference_compile` is set only when the attribute exists, so the script still works
      against a 4.x reference environment.
- [x] **S1.2** Export at opset ≥ 17 with dynamic axes `{batch, seq}` for `input_ids`/`attention_mask`
      and `{batch, k}` for `marker_pos`/`marker_mask`. (2026-09-20) — **opset 18, `dynamo=True`.**
      Under TorchScript the `seq` axis silently freezes; see the note above.
- [x] **S1.3** Verify the tail exports: `torch.gather`, `topk(2)`, `masked_fill(-1e4)`. Try
      `dynamo=True` if it does not. (2026-09-20) — the whole tail exports on both exporters;
      none of those three ops was ever the problem. `dynamo=True` was needed anyway, for `seq`.
      It requires `onnxscript`, which was missing from Task 1.1.2's install list.
- [x] **S1.4** Reproduce the manual head loop (§1.3), not `nn.TransformerEncoder.forward`, and confirm
      the head FFN activation in the exported graph is **ReLU**. (2026-09-20) — the manual loop
      comes for free by exporting `DecisionModel.forward` itself. Op counts in the exported
      graph: `Relu` 2 (the two head layers), `Gelu` 0, `Erf` 30 / 24 (encoder GeGLU + `scorer` + `act_head`), `Tanh` 0. Confirmed on all three.
- [x] **S1.5** Repeat for **all three** checkpoints. No ModernBERT-_large_ ONNX exists on
      `onnx-community` — only base variants are published, so large is unproven. (2026-09-20) —
      all three export and validate; ModernBERT-large is no longer unproven.
- [x] **S1.6** Record the working export recipe (flags, opset, `torch`/`transformers`/`onnx` versions)
      in `scripts/export_onnx.py`'s header and in this file. (2026-09-20) — recipe in the script's
      module docstring and in `scripts/README.md`; measured numbers in the table below.

**(2026-09-20) Measured — `scripts/export_onnx.py --all --dynamo`.** Reference environment
`python 3.12.4 · torch 2.14.0+cpu · transformers 5.17.0 · onnx 1.23.0 · onnxruntime 1.30.0 ·
onnxscript 0.7.2`, checkpoints `convaiinnovations/laya @ 1c5edc17a7acd8701df6fc341c0d179f1c62c982`.
Max abs difference vs PyTorch, fp32, over four input shapes (traced; then a different `seq`,
`k` and `batch` in turn):

| Checkpoint      | Params (excl. buffers) | File    | `logits` worst | `act_logits` worst | eager vs sdpa (`logits`) |
| --------------- | ---------------------- | ------- | -------------- | ------------------ | ------------------------ |
| english         | 421,293,827            | 1688 MB | 6.4e-05        | 2.0e-03            | 1.0e-06                  |
| multilingual    | 321,908,995            | 1290 MB | **4.9e-04**    | **8.4e-02**        | **5.1e-05**              |
| typed-decisions | 421,293,827            | 1688 MB | 6.1e-06        | 4.2e-03            | 1.5e-06                  |

Three things in that table are load-bearing later:

1. **`multilingual` is an order of magnitude looser than the other two** on `logits` and two
   orders on `act_logits`. Any M6/M7 parity tolerance has to be per-checkpoint, not global.
2. **`act_logits` is always far looser than `logits`.** It sits downstream of a softmax, an
   entropy and a `topk(2)` difference (`common.py:119-123`), so small logit differences are
   amplified before `act_head` ever sees them. The `logits` figure is the one that matters for
   answer parity; `act_logits` only drives the act/abstain head.
3. **Upstream runs `sdpa`; this exports `eager`** because S1.1 says so. On `multilingual` the
   two differ by 5.1e-05, which sets a floor on how close the Go port can get to golden vectors
   generated from the Python. See the new Task 6.5.

**Exit criteria**

- [x] Three `.onnx` files load in ORT and produce `logits`/`act_logits` for a batch.
      (2026-09-20) — and at three shapes beyond the traced one. A same-shape check passes on
      the TorchScript export too, which is exactly how a frozen `seq` axis would have reached M6.
- [x] Max abs logit diff vs PyTorch recorded per checkpoint. (2026-09-20) — see the table above.

**Fallback if export fails:** a third-party export already exists — `sevenreasons/laya-onnx-fp16`
(Apache-2.0, 846 MB, inputs `input_ids i64[B,S]`, `attention_mask i64[B,S]`, `marker_pos i64[B,K]`,
`marker_mask bool[B,K]`, `qtype i64[B]`; outputs `logits[B,K]`, `act_logits[B,2]`; dynamic B/S/K;
claims max logits diff 0.00416 vs PyTorch). Also `Mattepiu/laya-onnx` (fp32 + int8). Use one to unblock
M6 while fixing our own exporter — but **never ship a third-party artifact as the default**: verify it
ourselves or export our own.

- [ ] **S1.F** (only if S1 fails) Pull the third-party export, verify its I/O signature and logit diff
      ourselves, mark it dev-only in code, and open a tracking issue to replace it.
      **(2026-09-20) — not needed; S1 succeeded.** Left open rather than struck out: both
      artefacts were confirmed to exist, ungated and Apache-2.0, but **both cover only the
      English checkpoint**, so neither would have unblocked `multilingual` or `typed-decisions`.
      Our own export is also tighter than `sevenreasons/laya-onnx-fp16`'s claimed 0.00416.

### Spike S2: Which ONNX binding — and does CGO-free hold?

> **(2026-09-20) — done. Purego holds, but not for the reason the risk register gave.** R5's
> `runtime.AddCleanup` panic is fixed upstream and was never going to be the deciding factor. The
> defect that _is_ still live is one nobody had written down: the binding uses **no synchronisation
> anywhere**, so a `*Value` left to the garbage collector races `Runtime.Close`. Closing every value
> explicitly removes it entirely, which is what D5 records as a rule rather than a hope.
>
> The second surprise is a toolchain one: **`CGO_ENABLED=0 go test -race` is refused** — the race
> detector needs cgo. The two halves of D2's claim are therefore two separate commands, and S2.1's
> wording ("built with `CGO_ENABLED=0`") cannot be satisfied in the same run as S2.2's.

- [x] **S2.1** Write a throwaway Go program that loads the S1 export and runs one forward pass with
      `shota3506/onnxruntime-purego`, built with `CGO_ENABLED=0`. (2026-09-20) — a test package,
      `internal/onnxspike/`, not a `main`: `-race` in a loop is the only way S2.2's defect shows,
      and the repo's env-gated test idiom keeps it out of CI. M6 absorbs the package into
      `internal/backend/onnx` (Task 6.10) — "deletes" was the original wording, corrected on review.
      `CGO_ENABLED=0 go test -count=1 -run TestForwardPass ./internal/onnxspike/` → `PASS (1.25s)`,
      `ONNX Runtime 1.23.0 (C API 23) from /usr/local/lib/libonnxruntime.so`.
- [x] **S2.2** Record whether the `runtime.AddCleanup` panic `go-pocket-tts` hit reproduces — run it
      under `-race` and in a loop, since it is a finalizer race and will not show on a single pass.
      (2026-09-20) — **it reproduces at `go-pocket-tts`'s pin and is gone at ours**, but a data race
      takes its place. See the table below.
- [ ] **S2.3** If it reproduces, repeat S2.1 with `yalue/onnxruntime_go` and record in this file that
      D2's CGO-free promise covers **the tokenizer only**.
      **(2026-09-20) — not needed; the condition did not fire.** Left open rather than struck out:
      the surviving race is a real reason this may still be revisited, and `yalue/onnxruntime_go`
      was confirmed to be the healthier project on every non-CGO axis — MIT, 719 stars, tagged
      through v1.36.0, commits in August and September 2026, against an untagged 31-star repo last
      pushed 2026-03-15. If the swap ever happens, D2's promise narrows to the tokenizer.
- [x] **S2.4** Pin the chosen binding _and_ the ORT shared-library version (R7 requires a pinned ORT);
      note how the library is located at runtime. (2026-09-20) — `go list -m all`:
      `github.com/shota3506/onnxruntime-purego v0.0.0-20260315223538-8db8bd7424b2` and
      `github.com/ebitengine/purego v0.9.0`. The binding has no tags, so a pseudo-version is the only
      pin there is. ORT is pinned by C API version, not by file: `supportedAPIVersions = []uint32{23}`
      is the whole list the binding implements, so it is **ONNX Runtime 1.23.x or nothing** — measured
      here against `libonnxruntime.so.1.23.0`. Resolution order is `LAYA_ORT_LIB`, then
      `ORT_LIBRARY_PATH`, then `/usr/local/lib`, `/usr/lib`, `/usr/lib/x86_64-linux-gnu` and the
      Homebrew paths; a variable that is set but points nowhere is an error, not a fallback.
      **Acquiring** a pinned `.so` is not solved and is new Task 6.6.
- [x] **S2.5** Write the decision, with the evidence, into §0 as D5. (2026-09-20)

**(2026-09-20) Measured — `internal/onnxspike`, `english` checkpoint, ORT 1.23.0, fp32,
`batch=1 seq=16 k=3`.** The fixture holds one input with both the Python-ORT and the PyTorch
outputs, so the two error sources stay separable. Scaled difference is `|got−want| / max(1, |want|)`,
because `logits` carries the `-1e4` `masked_fill` sentinel while `act_logits` comes out in the
thousands.

| Output       | vs Python ORT 1.30.0 | vs PyTorch |
| ------------ | -------------------- | ---------- |
| `logits`     | 3.8e-06              | 3.2e-06    |
| `act_logits` | 2.4e-07              | 1.9e-07    |

Three consequences:

1. **The ORT version skew is a non-event.** The graph was validated in Python under onnxruntime
   1.30.0 and executed here under 1.23.0; IR 10 / opset 18 sit far inside both. The
   Go-vs-Python-ORT difference is the same order as the ONNX-vs-PyTorch difference S1 measured.
2. **`marker_mask` — a `bool` tensor — round-trips correctly.** Nothing in-house had ever passed one
   through this binding; `go-pocket-tts` only ever used `float32` and `int64`. The check is
   discriminating rather than decorative: flipping the fixture's one `false` to `true` moves
   `logits[2]` from `-10000` to `0.688`, a scaled difference of **1.0** against a tolerance of 1e-4.
3. **The session must be created from a path.** The exports are split into a graph file plus an
   `.onnx.data` blob that ORT resolves relative to the model path, so `NewSessionFromReader` cannot
   load them.

**(2026-09-20) The finalizer question, S2.2, in full.**

| Binding commit                   | Scenario                           | Result                                                                                                                     |
| -------------------------------- | ---------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| `1c85186598a5` (go-pocket-tts's) | any `NewTensorValue`               | `panic: runtime.AddCleanup: cleanup function closes over ptr, cleanup will never run` on the **first** call, `value.go:35` |
| `8db8bd7424b2` (ours)            | values closed, `-race -count=500`  | green                                                                                                                      |
| `8db8bd7424b2` (ours)            | values abandoned, `-race -count=1` | `WARNING: DATA RACE` — `releaseValuePtr` (`value.go:152`) against `Runtime.Close` (`runtime.go:178`)                       |

The panic is deterministic, not intermittent, which means **`go-pocket-tts`'s ONNX path does not
work at all on Go 1.24+**, not merely in parity tests — its `-skip 'TestParity_.*_VsONNX'` workaround
hides a hard failure. Worth telling them; it is a one-line `go get` away from being fixed.

The surviving race needs a value that is dropped without `Close`, because `Close` calls
`cleanup.Stop()` and takes the finalizer out of play. That is why D5 makes explicit closing a rule
and not a recommendation, and why the reproduction is checked in behind `LAYA_ONNXSPIKE_FINALIZER=1`
rather than deleted: M6 has to keep satisfying it.

**Exit criteria**

- [x] One forward pass green on the chosen binding, with the CGO answer recorded. (2026-09-20) — and
      compared against both Python runtimes rather than merely returning two tensors. The CGO answer:
      `CGO_ENABLED=0 go build ./...` and `CGO_ENABLED=0 go test ./internal/onnxspike/` both pass, so
      **D2's CGO-free promise survives the ONNX backend**. The one caveat is the toolchain's, not the
      binding's — `-race` implies cgo, so the race runs are a separate command.

### Spike S3: Latency on real hardware

> **(2026-09-20) — measured. The estimate was right about the order of magnitude and wrong about
> which lever matters.** ORT-CPU landed at the fast end of the predicted 0.3–5 s band, so the
> pure-Go native backend (M8) is not urgent. But two of the things the plan assumed about how to get
> there turned out to be false, and both are cheap to act on: **all twelve hardware threads is the
> wrong setting** (every checkpoint peaks at 8 and then loses 12–26%), and **batching does not
> amortise on CPU at all** — the per-question cost is flat, where on a T4 it falls 4.6×. The
> deployment shape that follows is not the one upstream's README implies.

FLOP-derived estimate: ≈360 GFLOP for ModernBERT-large at 512 tokens; ≈256 GFLOP for mmBERT-base at 1024. Measured on comparable hardware, gonum `Sgemm` reaches ~27–32 GFLOPS and 2-thread OpenBLAS
~54 GFLOPS — which puts a _hand-written_ pure-Go forward pass at roughly **12 s/sequence**, against the
README's 33 ms on a T4. ORT-CPU with MLAS should land in the 0.3–5 s range depending on cores.

- [x] **S3.1** Measure ORT-CPU latency per checkpoint at realistic sequence lengths (512 / 1024) and
      thread counts, on the hardware we actually ship numbers for. (2026-09-20) — `BenchmarkForward`
      and `BenchmarkSessionLoad` in `internal/onnxspike/bench_test.go`, driven by `just bench-onnx`,
      **through the Go binding rather than Python**: the number a caller experiences is the one that
      counts. Three full sweeps at 5 timed iterations per cell; tables below.
- [x] **S3.2** Record the numbers, the hardware and the ORT build in `BENCHMARKS.md` — replacing
      upstream's T4 figures rather than repeating them. (2026-09-20) — new section
      **"Speed — CPU, measured here"**, naming the CPU, its core layout, the `powersave` governor,
      ORT 1.23.0 and the binding. Upstream's T4 table is kept but demoted to _"upstream's published
      figures, not measured here"_, the same convention the file already applies to the Jev columns;
      the Headline row now carries both numbers, and `## Limits` gains the CPU line.
- [x] **S3.3** **Do not write "CPU-first" anywhere in the README until S3.2 exists.** (2026-09-20) —
      `grep -rn -i 'cpu[- ]first' README.md docs/ PLAN.md BENCHMARKS.md AGENTS.md` matches only this
      line. The prohibition never fired because go-laya has not written its own README yet; it stays
      a standing constraint on **Task 7.5**, where a sub-item now carries it, and S3.2 has now made
      the honest claim available.
- [ ] **S3.4** If too slow: evaluate int8 dynamic quantization (typically 2–4×) and/or defaulting the
      Router to mmBERT-base.
      **(2026-09-20) — the condition fires, and the work moved rather than shrank.** Neither lever
      is a quick win, and both are mis-scoped as spike items.
      _int8_ is inseparable from S3.5's calibration requirement, and that needs a labelled eval set
      and `internal/calib`, neither of which exists before M1 and Task 7.1 — carried to **Task 6.9**.
      _"Defaulting the Router to mmBERT-base"_ **conflicts with the port's own mandate**: upstream
      routes by language, not by cost, and M7 asserts end-to-end parity against Python. Measured,
      mmBERT-base is worth 2.2–2.7× — real, and only ever available as an **opt-in**. Carried to
      **Task 3.4**.
- [ ] **S3.5** If int8 is evaluated, measure **ECE and Brier, not just accuracy** — this model's whole
      selling point is calibrated probabilities, and a calibration regression will not show up in an
      argmax test (R4). Never make quantization the default.
      **(2026-09-20) — not reachable here; inlined into Task 6.9 as its acceptance criterion** rather
      than reinterpreted into something this spike could have claimed.

**(2026-09-20) Measured** — `internal/onnxspike`, ORT 1.23.0 through
`shota3506/onnxruntime-purego`, fp32, `CGO_ENABLED=0`, on a 12th Gen Intel Core i7-1255U (2 P + 8 E
cores, 12 threads, 15 W, `powersave`). p50 in ms, **minimum across three sweeps** — run-to-run
variation reached 2× on the worst cell, so nothing here is good to better than about ±20%.

One question, `batch=1`, `k=4`, 8 threads:

| tokens | `english` | `multilingual` | `typed-decisions` |
| ------ | --------- | -------------- | ----------------- |
| 128    | 374       | 160            | 406               |
| 512    | 1691      | 645            | 1856              |
| 1024   | —         | 1484           | 3962              |

Thread scaling at `batch=1`, 512 tokens:

| threads | `english` | `multilingual` | `typed-decisions` |
| ------- | --------- | -------------- | ----------------- |
| 1       | 3660      | 1332           | 3730              |
| 4       | 1871      | 706            | 1937              |
| 8       | **1691**  | **631**        | **1720**          |
| 12      | 2269      | 845            | 2130              |

Four consequences:

1. **CPU is a batch workload, not an interactive one.** 0.6–1.9 s per question at the default
   `max_len` of 512 (`agent.py:256`), against upstream's 33 ms — 9–43× depending on which row you
   compare, since upstream never states the sequence length behind its figure. D6 records this.
2. **`IntraOpNumThreads` must be the physical core count, not the thread count.** All 12 costs
   12–26% against 8 on a part where only 2 of 10 cores have SMT siblings. Task 6.3 owns the default;
   it must not be `runtime.NumCPU()`.
3. **Batching buys nothing.** `8×512` costs 1856 ms/question against `1×512`'s 1691 ms for `english`
   — flat to within noise, where the T4 table shows 32.8 → 7.2 ms/question over the same move. One
   question at 512 tokens already saturates the cores. 1 → 8 threads is only 2.1–2.2×, which says the
   same thing: this is bandwidth-bound well before it is core-bound.
4. **Holding all three checkpoints costs ~3.3 GB.** Measured resident growth per session: 1423 MiB
   (`english`), 488 MiB (`multilingual`), 1423 MiB (`typed-decisions`), at 1.5–3.2 s to build a
   session from an already-downloaded file. That is the real constraint on raising the Router's
   `max_loaded` (Task 3.2.1), and it is cheaper than upstream's "7.4 s median reload on CPU".

**(2026-09-20) The binding costs nothing — and something is unexplained.** Run back-to-back at the
same thread count and the same thermal state, `batch=1`, 512 tokens, p50:

| checkpoint     | Go + ORT 1.23.0 | Python + ORT 1.30.0 |
| -------------- | --------------- | ------------------- |
| `english`      | 1901 ms         | 5064 ms             |
| `multilingual` | 751 ms          | 849 ms              |

purego's marshalling is therefore not a cost worth worrying about, which is the question this
cross-check was for. The 2.7× on `english` is **not** a claim that Go is faster than Python — the
same ONNX Runtime does the arithmetic in both — and it is not explained: the only known difference
is the runtime version, since the binding speaks C API 23 only while the reference environment has
1.30.0. It favours our path, so it threatens no claim here, but a 2.7× gap on ModernBERT-large and
1.1× on mmBERT-base is a shape worth understanding. **Task 6.8** is where it gets chased.

**Exit criteria**

- [x] Latency measured on real hardware and written down where a reader will find it, before any
      performance claim is made in go-laya's own name. (2026-09-20)

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
- [ ] **Nothing lints the workflows, the TOML or the YAML.** _(new, 2026-09-20)_ Dropping Trunk
      (Task 1.5.3) removed a config that _enabled_ `actionlint`, `taplo` and `yamllint` — but that
      config never ran, so nothing regressed and nothing was ever covered. Worth adding to
      `just ci` deliberately rather than inheriting from a dormant tool. `actionlint` is the one
      that would earn its place: `ci.yml` is hand-edited every time a formatter version is pinned.

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

- [x] **1.1.1** Create `.venv-ref` with Python 3.12 and the CPU torch wheel. (2026-09-20) —
      python 3.12.4, torch 2.14.0+cpu.
- [x] **1.1.2** Install `transformers safetensors huggingface_hub numpy onnx onnxruntime`.
      (2026-09-20) — **plus `onnxscript`**, which this list is missing and which
      `torch.onnx.export(dynamo=True)` imports. Spike S1 cannot run without it.
- [x] **1.1.3** Freeze to `scripts/requirements-ref.txt` and commit it — the versions are the contract.
      (2026-09-20) — 41 pins. A bare `uv pip freeze` does not produce a restorable lock here:
      `torch==2.14.0+cpu` is not on PyPI, and uv's default first-index-wins finds plain `torch`
      there and stops rather than consulting the extra index. The file carries a hand-written
      header with the extra index, and `--index-strategy unsafe-best-match` has to go on the
      command line because uv rejects it inside a requirements file. Restore verified into a
      clean venv, not assumed.
- [x] **1.1.4** Add `.venv-ref/` to `.gitignore`. (2026-09-20) — already done in M0 Task 0.7.
- [x] **1.1.5** Document the regeneration command in `scripts/README.md` so it survives the next person.
      (2026-09-20) — `scripts/README.md`. No template to copy: `../go-pocket-tts/scripts/` has
      neither a README nor a pinned requirements file.

**Task 1.2: Fetch the checkpoints.** ~2.4 GB, anonymous, no token:

```bash
export LAYA_MODELS=${LAYA_MODELS:-$HOME/laya_models}
python -c "from huggingface_hub import snapshot_download as d; d('convaiinnovations/laya', local_dir='$LAYA_MODELS/laya')"
```

- [x] **1.2.1** Download all three checkpoints to `$LAYA_MODELS` (549 GB free on `/mnt/projekte`).
      (2026-09-20) — 2.3 GB into `./models/laya` (user's call; `/models/` is already gitignored).
      `model.safetensors` 843 MB / 644 MB / 843 MB.
- [x] **1.2.2** Record the resolved commit sha of the Hub repo — golden vectors are only meaningful
      against a known revision. (2026-09-20) — `1c5edc17a7acd8701df6fc341c0d179f1c62c982`, written
      to `models/PROVENANCE.json` **at download time**. That timing is not incidental:
      `laya.Agent.__init__` calls `_fix_tokenizer_config` (`original/laya/agent.py:21-46`), which
      rewrites `tokenizer/tokenizer_config.json` in place, so the local tree stops matching the
      recorded revision the first time the Python loads a model.
- [x] **1.2.3** Confirm the three `temperature` dtypes (F32/F32/F16, §1.1) so the loader never
      hardcodes one. (2026-09-20) — confirmed from the safetensors headers, no torch needed:
      english F32, multilingual F32, typed-decisions F16. In the first two it is the **only**
      F32 tensor in the file (1 of 206 / 1 of 170). **(2026-09-20, review)** — and it does not
      matter to inference: the buffer is never read (§1.1). It stays an M8 loader fact only.
- [x] **1.2.4** _(new)_ Keep the downloaded tree out of the tooling walks. (2026-09-20) —
      `markdownlint` globs the filesystem, not git, so `just lint-md` started failing on
      `models/laya/README.md`, which is upstream's file and not ours to fix. `--ignore models`
      in the `justfile`, `"models/**"` in `treefmt.toml`.
- [x] **1.2.5** _(new)_ Cover ONNX external data in `.gitignore`. (2026-09-20) — `*.onnx` does
      **not** match `*.onnx.data`, and the dynamo export of ModernBERT-large is over the 2 GB
      protobuf limit and splits exactly that way, leaving a 1.7 GB blob untracked and addable.
      Added `*.onnx.data`, `*.onnx_data` and `/build/`.

**Task 1.3: `scripts/dump_python_parity.py`.** Modelled on `../go-pocket-tts/scripts/dump_python_parity.py`.
Emits into `testdata/`, and records `transformers.__version__` / `tokenizers.__version__` in every
file header.

> **(2026-09-20) — done.** `scripts/dump_python_parity.py`, **eight** files not five (1.3.1 emits
> two, and `mailtext` and `round4` were added — see below). JSONL with a header record on line 1,
> because `treefmt`'s prettier owns `*.json` and would reflow a pretty fixture on every `just fmt`,
> and a line-per-case diff is reviewable where a reflowed blob is not. `testdata/*.jsonl` is
> excluded from `treefmt` and verified byte-stable across `just fmt`. Seven of the eight need no
> weights at all: only `logits.jsonl` loads a checkpoint.

- [x] **1.3.1** `testdata/tokenizer_{en,ml}.jsonl` — 53 cases each, covering Task 4.4.1–4.4.8.
      (2026-09-20) — the record gained **`tokens_with_leading_space`**: the schema as written had a
      single `tokens` field, so 4.4.10's "ids **and** token strings" was only half-assertable on the
      `" " + text` variant that `common.py:68` actually uses. The corpus confirms §1.4 item 1 rather
      than assuming it — EN ids differ with a leading space, ML ids do not — and §1.4's special-token
      table exactly (EN UNK 50280/CLS 50281/SEP 50282/PAD 50283/MASK 50284; ML 3/2/1/0/4), which is
      Task 4.2.4's assertion data. **Task 4.4.9 is deliberately excluded**: a lone surrogate cannot
      round-trip through JSON (Python writes `\ud800`, Go's decoder yields U+FFFD), so a vector would
      assert against a corrupted input. 4.4.9 says "decide and document the boundary" — this is the
      decision; Go tests it at the API edge instead. New Task 4.4.11 records that.
- [x] **1.3.2** `testdata/sequence.jsonl` — 45 cases, all three checkpoints. (2026-09-20) — carries
      **both** the public `qdef` and the internal `q` (`{t,ins,crit}`), plus the checkpoint, because
      the budgets differ (512/192 english, 1024/256 the other two) and a Go failure has to name
      which. Also `n_options` and `markers_lost`, which is where `agent.py:262-263`'s `ValueError`
      guard is pinned. Invariant #6 reproduces exactly: 77 options at `head_max_len=192` gives a
      marker stride of 4, i.e. `per=4`.
- [x] **1.3.3** `testdata/render.jsonl` — 41 cases. (2026-09-20) — `original/tests/test_criteria.py:33-100`
      ported, plus #17's named golden cases and `serialize_state`'s key-order / HTML / non-ASCII
      traps. Output is always carried as a **string**, never a nested object, or #17's
      `{"d": "münchen"}` → literal-`ü` case would be unverifiable. The `default=str` cases use
      `date`, `Decimal` and `complex` rather than `object()`, whose `str()` embeds a heap address and
      would differ on every run; their inputs are recorded post-substitution because Go cannot
      construct a Python `Decimal` anyway.
- [x] **1.3.4** `testdata/answers.jsonl` — 27 cases. (2026-09-20) — carries
      **`temperature_by_options`** as well as `temperature`: the 3-vector alone cannot exercise the
      `temp_bucket` lookup (#23, Task 7.1.1), and `k` sweeps every bucket boundary (2, 3–5, 6–10,
      11+) against both english's populated table and multilingual's empty one. `answer_json` is the
      exact serialized string, because 7.3.1/7.3.2 compare bytes. Injected by hand because no sample
      produces them: a `score` of exactly **0.0** (`docs/API.md:203-205`), and k=2 uniform under both
      confidence formulas — which come out **0.0 for choice and 0.5 for noul**, the clearest possible
      demonstration that #25 and #26 must not be unified.
- [x] **1.3.5** `testdata/logits.jsonl` — 10 states × 3 questions × 3 checkpoints, batch-level.
      (2026-09-20) — batch-level is not a convenience: `collate_items` derives `L` and `kmax` from
      the whole batch and the act features soft-max over the padded `kmax` (`common.py:119-125`), so
      one question in isolation is a different computation. Tagged per checkpoint and qtype for Task
      6.9.3's reuse. **Final outputs only** — head intermediates for #35–#38 are M8's, and M8 is
      deferred; Task 8.8 carries the regeneration.
- [x] **1.3.6** (2026-09-20) — uniform header on line 1 of all eight files: the `transformers` /
      `tokenizers` / `numpy` / `safetensors` / `python` versions, the Hub sha `1c5edc17…`, **and a
      sha256 of each `tokenizer_config.json`**. That last one turns 1.2.2's `_fix_tokenizer_config`
      hazard from a warning into a check. The hazard is currently **dormant**: all three local
      configs were sha256-compared against the Hub at the pinned revision and are byte-identical,
      already carrying the shapes the function would write.
- [x] **1.3.7** (2026-09-20) — `np.random.default_rng` seeded per case from a **sha256** of the case
      name, not `hash()`, which Python randomises per process; sorted, de-duplicated case order,
      `torch.set_num_threads(1)`, CPU only. Verified by regenerating and diffing.
- [x] **1.3.8** _(new)_ `testdata/mailtext.jsonl` — 26 cases over `original/laya/email.py`.
      (2026-09-20) — #66–#74 is the only invariant block with **neither** an upstream test nor golden
      data, and Task 2.3.3 would otherwise port 90 lines of regex by reading them. It already paid:
      the signature-cut window is widest for a _short_ message, not narrowest —
      `min(int(n*0.6), n-8)` goes negative below nine lines and `max(1, …)` pins the start at 1 — the
      opposite of what the code reads like.
- [x] **1.3.9** _(new)_ `testdata/round4.jsonl` — 26 cases. (2026-09-20) — #29 is one of the four
      §5 singles out as causing silent wrong answers, and no sampled softmax output lands on a tie,
      so the values have to be chosen. Serves Task 2.1.5/2.1.6 directly, long before an answer exists
      to apply `Round4` to. Includes `-2.5e-05 → -0.0`, which is a second trap: Go must emit negative
      zero rather than normalising it.
- [x] **1.3.10** _(new)_ `--verify-agent`. (2026-09-20) — `answers.jsonl` is generated by a
      hand-inlined copy of `agent.py:294-343`, because the arithmetic has no seam; it lives inside
      the batch loop. A copy that drifts would produce a corpus that is internally consistent and
      wrong, so the generator runs a real `laya.Agent.system_one` over the same forward pass and
      compares every field.

**Task 1.4: Provenance test.** `TestGoldenProvenance` asserts the recorded tokenizer version matches
what the Go implementation claims to target. A silent upstream tokenizer change is the realistic way
parity regresses.

- [x] **1.4.1** (2026-09-20) — `targetTokenizersVersion` and `targetTransformersVersion` in
      `provenance_test.go`. Test-only: nothing in the library reads them yet, and an exported
      constant the public API does not need is harder to remove than to add. M4 promotes them if the
      pure-Go tokenizer needs a runtime target.
- [x] **1.4.2** (2026-09-20) — globs rather than taking a file list, so a fixture added later is
      covered without anyone remembering to register it. An empty glob is a failure, not a skip.
- [x] **1.4.3** (2026-09-20) — all three failure modes exercised against a corrupted copy and
      confirmed to fire: a bumped version, a header removed, and an empty file. A provenance test
      that cannot fail is worse than none.

**Task 1.5: Lint the harness.** _(new, 2026-09-20)_ `original/**` is excluded from `treefmt`
and the repo has no Python formatter, so `scripts/*.py` is the one part of the tree nothing
checks. That was acceptable when `scripts/` was empty and is not now.

- [x] **1.5.1** (2026-09-20) — `ruff.toml` at the root (`line-length = 100`, `E/F/B/I/UP`,
      `laya` declared first-party for isort since it is reached through a `sys.path.insert` rather
      than an install). **`ruff format` under treefmt; `ruff check` deliberately not** — it is a
      checker, and `treefmt.toml`'s own header already explains why chaining one as a formatter can
      fail to converge under `--fail-on-change`. It gets `just lint-py`, wired into `just ci`.
      `original/**` stays excluded.
- [x] **1.5.2** (2026-09-20) — `RUFF_VERSION: "0.16.8"` in `ci.yml`'s `format` job `env:` block,
      same curl-tarball idiom as `TREEFMT_VERSION`, plus a `Lint Python` step. The sweep landed as
      its own commit ahead of the wiring so both commits stay green: the file was hand-wrapped
      between 88 and 107 columns, so no line length makes `ruff format` a no-op.
- [x] **1.5.3** _(new)_ Drop Trunk. (2026-09-20) — `.trunk/` enabled 15 linters including a
      **second** Python formatter stack (`black` + `isort` alongside `ruff`), and none of them ran:
      no workflow invoked `trunk check` and both git hooks sat in `actions.disabled`. It was never
      tracked either — excluded via `.git/info/exclude`, so this is a local removal with no repo
      diff. One stack, not two.

**Task 1.6: Cross-check the transformers major version.** _(new, 2026-09-20)_ The reference
environment resolved to **transformers 5.17.0**. Upstream declares `transformers>=4.45.0` with no
upper bound, so that is legal, but laya 0.3.4 predates 5.x. `load_state_dict(strict=True)` passing
proves the _parameter set_ matches; it says nothing about numerics.

- [x] **1.6.1** (2026-09-20) — `scripts/crosscheck_transformers.py`, `.venv-ref4` with
      transformers **4.57.6** (the last 4.x; note upstream's declared `>=4.45.0` floor predates
      ModernBERT itself, which landed in 4.48). All three checkpoints, fp32 CPU, single-threaded,
      run **before** any vector was frozen. Tokenizer ids were diffed too — the two environments
      resolve `tokenizers` 0.22.2 and 0.23.2 — and agree exactly, so R6 is clean on that axis.
- [x] **1.6.2** **Does not fire, and the task's presumption is backwards.** (2026-09-20) — the two
      ModernBERT-large checkpoints are **bit-identical** across the major. `multilingual` diverges
      enormously (logits by 6.96, act_logits by 1879) and the cause is a **parsing bug in 4.x**, not
      a numerics difference: mmBERT's config sets
      `rope_parameters.sliding_attention.rope_theta = 160000`, and transformers 4.57.6 loads its own
      default of `10000.0` instead, so every sliding-attention layer runs on the wrong RoPE
      frequencies. Setting that one value by hand makes 4.57.6 bit-identical — that is the whole
      difference. English survives only because its sliding `rope_theta` genuinely **is** 10000.0
      and so is unharmed by being ignored. Pinning to 4.x, which 1.6.2 assumes is the safe
      direction, would have silently corrupted the multilingual checkpoint. **The reference
      environment stays at transformers 5.17.0**, now on evidence rather than by default.

**Exit criteria**

- [x] All **eight** `testdata/*.jsonl` files committed; CI does not need Python to run the suite.
      (2026-09-20) — the criterion said five; 1.3.1 emits two files and 1.3.8/1.3.9 added two more.
      `just ci` is green with no Python interpreter involved in the Go suite.
- [x] Commit: `feat(testdata): golden parity vectors generated from upstream Python`. (2026-09-20)

**Task 1.7: Carry the collated batch and the compute provenance in the fixtures.** _(new, 2026-09-20,
review)_ `logits.jsonl` records the outputs and `input_tokens` but **not** the collated inputs, so
Task 5.3 (`Collate`) has nothing to assert against short of a full forward pass, and the fake backend
(6.1.2) cannot key on tensors. And no header says which device / dtype / attention produced the
numbers, although §8's "within 1e-4 of PyTorch" depends on exactly that.

- [x] **1.7.1** Regenerate `logits.jsonl` with the collated batch per case — `input_ids`,
      `attention_mask`, `marker_pos`, `marker_mask`, `qtype` — exactly what `collate_items` returns.
      (2026-09-20) — the five tensors the forward consumes, through the existing `tensor_rec`.
      `collate_items` returns two more that are left out on purpose: `label` is a constant `[-1,
-1, -1]` because no item here carries one, and `meta` after its key filter restates `qtype`.
      37 KB to 110 KB, and purely additive — no pre-existing value moved.
- [x] **1.7.2** Every header gains a `compute` block (`dump_python_parity.py:1006,1014`):
      `{device: cpu, dtype: float32, attn: sdpa, torch_threads: 1}`.
      (2026-09-20) — in `header()`, so all eight files carry it, including the six that never
      import torch: a header whose shape depends on which fixture it heads is one no single test
      can check. `:1006` and `:1014` turned out to be the _source_ of the facts
      (`set_num_threads(1)`, `attn="sdpa"`) rather than insertion points; `dtype` comes from
      `export_onnx.py:154`'s `model.float().eval()`.
- [x] **1.7.3** `TestGoldenProvenance` asserts `compute`; the mismatch path is exercised against a
      corrupted copy, as 1.4.3 did.
      (2026-09-20) — and the corrupted-copy test 1.4.3 refers to **did not exist** — it was done by
      hand against a scratch copy and discarded. So `checkProvenance` now returns its violations
      instead of calling `t.Error`, and six corruptions drive it: bumped version, changed `attn`,
      changed `device`, missing block, removed header, empty file. Each asserts the failure _names
      the cause_.
- [x] **1.7.4** One reviewed regeneration diff (R6); nothing else in the eight files may change.
      (2026-09-20) — the seven non-logits files changed on line 1 only, and no pre-existing value
      in `logits.jsonl` moved. Two independent full regenerations are byte-identical on all eight,
      so 1.3.7's determinism holds over the new tensors too.

### M2 — Tier-1 core (no ML runtime, ~620 lines of Python)

> **Three Python-isms will silently corrupt results if translated naively.** Do M2 in this order.

**Task 2.1: `jsonx` — Python-compatible JSON.** _Do this first; everything downstream depends on it._

`encoding/json` gets all three of these wrong: it emits `{"a":1}` instead of `{"a": 1}`, it **sorts map
keys**, and it **HTML-escapes** `<`, `>`, `&` by default. The Python code feeds
`json.dumps(state, ensure_ascii=False)` straight into the tokenizer, so any of the three changes the
bytes the model sees.

- [x] **2.1.1** Create `jsonx/jsonx.go`, `jsonx/jsonx_test.go`.
      (2026-09-20) — plus `jsonx/number.go` for `Round4`/`Repr`; same package, split for
      readability.
- [x] **2.1.2** `type Obj []Field` with `Field{Key string; Value any}` — an _ordered_ object.
      (2026-09-20) — with `Get`, and an `UnmarshalJSON` the box list does not have — see the new
      2.1.9.
- [x] **2.1.3** `Obj.MarshalJSON` reproduces `json.dumps(x, ensure_ascii=False)`: separators `", "` and
      `": "`, no key sorting, no HTML escaping, non-ASCII emitted literally.
      (2026-09-20) — a hand-written encoder, not a wrapper: post-processing stdlib output would
      corrupt any string containing a comma or a colon. Two divergences beyond the three in #18
      turned up and are in neither the invariant nor the task: Go escapes **U+2028/U+2029** for
      JSONP safety where Python emits them literally, and Go writes `\u0008` where Python writes
      `\b`.
- [x] **2.1.4** `Compact(v any) string` reproduces
      `json.dumps(v, ensure_ascii=False, separators=(", ", ": "), default=str)`, falling back to
      `fmt.Sprintf("%v", v)` for anything `encoding/json` refuses (Python's `default=str`).
      (2026-09-20) — `Compact` and `Marshal` share one encoder and differ only in the fallback,
      which is the real relationship: `render_criterion` passes the separators explicitly and
      `serialize_state` relies on Python's defaults being the same two. A Go **map is refused**
      (`ErrUnorderedMap`) rather than sorted — sorting would emit bytes that look right and are
      not, which is the failure this package exists to prevent.
- [x] **2.1.5** `Round4(x float64) float64` — **Python's `round()` is half-to-even on the exact binary
      double; Go's `math.Round` is half-away-from-zero.** Implement as
      `strconv.ParseFloat(strconv.FormatFloat(x, 'f', 4, 64), 64)`.
      (2026-09-20) — format-and-reparse, as the box specifies.
- [x] **2.1.6** Tie-case tests for `Round4`: `0.00125`, `2.5e-5`, plus negatives and values that are
      exactly representable.
      (2026-09-20) — all 26 cases of `testdata/round4.jsonl`, including `-2.5e-05 → -0.0`; `==` is
      true for `-0.0` against `0.0`, so the sign has its own `math.Signbit` assertion.
- [x] **2.1.7** Float formatting matches Python's `repr` for the values that reach JSON (shortest
      round-trip), including `-0`, very small and very large magnitudes.
      (2026-09-20) — bigger than expected. Go and Python disagree on **four of five** sampled
      values: Python writes `1.0`, `1000000000000000.0`, `1e+16`, `1e-05`, `-0.0` where
      `encoding/json` writes `1`, `1000000000000000`, `10000000000000000`, `0.00001`, `0`. `Repr`
      reimplements CPython's rule — shortest round-trip digits, exponential only when the decimal
      point falls at or before -4 or past 16, always a fractional part — and is asserted against
      `round4.jsonl`'s `input_repr`/`output_repr` columns.
- [x] **2.1.8** Table test against `testdata/render.jsonl` — byte-for-byte (invariant #18).
      (2026-09-20) — 25 of the 41 cases (`render_criterion` and `serialize_state`); the 16
      `render_options` cases are Task 2.5's. 62 subtests, and they were confirmed to discriminate
      by sabotaging the `.0` rule and watching the golden cases fail.
- [x] **2.1.9** _(new, 2026-09-20)_ `Obj.UnmarshalJSON` and `Decode`, order-preserving, numbers as
      `json.Number`. Not in the box list and load-bearing: `render.jsonl` records its **inputs** as
      objects whose key order _is_ the assertion (`state/dict-unsorted-keys` is
      `{"z": 1, "a": 2, "m": 3}`), so a loader going through `map[string]any` destroys exactly what
      those cases check, and one through `float64` turns every `3` in the corpus into `3.0`.
      `internal/golden` is the shared loader; the corpus lives once at the repo root and several
      packages assert against the same file.

**Task 2.2: `lang`.** Port `original/laya/lang.py` verbatim. Invariants §5 items 54–65.

Go-specific traps:

- [x] **2.2.1** `detect_script`'s tie-break depends on Python dict _encounter_ order, with
      `counts["latin"]` assigned **after** the loop. Go map iteration is randomized — use an explicit
      ordered counter or the result is nondeterministic (invariant #56).
      (2026-09-20) — an ordered counter, with `latin` appended after the loop so an exact tie still
      goes to the first non-Latin script encountered.
- [x] **2.2.2** `_STOP`'s insertion order (fr, de, es, pt, it, nl) is the language tie-break. Use an
      ordered slice (invariant #63).
      (2026-09-20) — an ordered slice.
- [x] **2.2.3** Python's `[^\W\d_]+` is Unicode-aware; Go RE2's `\w` is ASCII-only. Split on
      `unicode.IsLetter` instead.
      (2026-09-20) — `unicode.IsLetter`. Documented edge, not fixed: `[^\W\d_]+` also admits the
      non-decimal numerics Python's `\w` covers (`½`, `Ⅷ`), where splitting on `IsLetter` breaks a
      run Python joins. Nothing in laya's corpus reaches it.
- [x] **2.2.4** `_SCRIPT_RANGES` order is load-bearing (hangul before kana before han), first match
      wins.
      (2026-09-20) — first match wins, order preserved.
- [x] **2.2.5** Port all 14 script cases, 7 `is_english` cases, 6 language-guess cases and 5 flattening
      cases from `original/tests/test_router.py:29-86`.
      (2026-09-20) — all 14 script, 7 `is_english`, 6 language-guess and 5 flattening cases.
- [x] **2.2.6** A determinism test: the same input run 1000× (or with `-count=10`) gives the same
      script/language — the only way the map-order trap shows up.
      (2026-09-20) — `-count=10`, and mutation-checked: swapping the profile order and the
      stop-word tie-break produced 12 failures, so the ordering assertions bite.
- [x] **2.2.7** _(new, 2026-09-20, review)_ `Detection.ScriptProfile` is **ordered** (`jsonx.Obj` or a
      `[]ScriptShare`), in encounter order with `latin` last — the same order #56 already forces
      `detect_script` to track. It is emitted inside `routing.detection`, where a sorted Go map would be
      observable (`test_local_e2e.py:211` asserts the payload serialises), so it is not a deviation and
      comes off the 7.5.3 list. Acceptance: `json.Marshal(RouteDecision)` byte-equal to Python for the
      detection-path router cases.
      (2026-09-20) — done, but **the box states the wrong function's ordering** and is corrected
      above: the emitted profile comes from `script_profile`, which seeds `counts = {"latin": 0}`
      (`lang.py:116`), so `latin` is **first** and vanishes at zero. `latin` last is
      `detect_script`'s order (#56). Both are asserted separately because they genuinely differ.
      The stated acceptance is also unreachable as written — `encoding/json` compacts a
      `MarshalJSON`'s output and strips Python's separators — so parity runs through
      `jsonx.Marshal(d.Map())` (D12); the `RouteDecision` half stays M3's.

**Task 2.3: `mailtext`.** Port `original/laya/email.py`. Invariants §5 items 66–74.

- [x] **2.3.1** Port the cleaning pipeline into `mailtext/`.
      (2026-09-20) — `original/laya/email.py:23-53` only.
- [x] **2.3.2** Do **not** port `email.email_questions` — it is a dead byte-identical duplicate of
      `presets.email_questions`; port only the presets one (Task 2.4).
      (2026-09-20) — not ported; its two fixture cases are asserted by Task 2.4 instead, which is
      what makes them useful.
- [x] **2.3.3** Write the tests this port deserves — the upstream code is currently untested; cover
      quoted replies, signature blocks, forwarded headers, CRLF and unicode whitespace.
      (2026-09-20) — all 19 `clean_email_body` and 5 `email_state` golden cases, plus the three
      concrete instances of #74's generic `\w`/`\s` hazard, each pinned by a named test and
      mutation-checked: the signature marker's `[\w ,!.]*` matches `Beste Grüße` in Python and
      would not under RE2's ASCII `\w`; `str.strip()` strips **U+001C–U+001F**, which Go's
      `unicode.IsSpace` does not; and truncation counts code points. Also recorded against #73:
      `state.update()` **replaces in place**, so an extra `from` overrides the sender's value but
      keeps its position.

**Task 2.4: `presets`.** Port all five preset constructors from `original/laya/presets.py`. Pure data.

- [x] **2.4.1** Port all five constructors into `presets/`.
      (2026-09-20) — verbatim, including `categories or {...}` — an empty taxonomy is falsy in
      Python and selects the defaults, so "a choice question with no options" is not expressible
      upstream and is not made expressible here.
- [x] **2.4.2** Add the shape test upstream never had: every preset round-trips through question
      validation.
      (2026-09-20) — and better than a shape test for one of the five: `testdata/mailtext.jsonl`
      carries `email_questions`' two recorded cases, and 2.3.2 establishes the copy they came from
      is byte-identical to the presets one, so the email preset is asserted **byte-for-byte against
      the Python**. The other four are shape-tested as the box intends.
- [x] **2.4.3** Assert option order is preserved exactly — the option index is the answer index.
      (2026-09-20) — four presets asserted positionally. Also `All()` rebuilds per call:
      `Questions` is a slice, so a shared one would let one caller mutate what the next sees — a
      trap Python's fresh dict does not have.

**Task 2.5: `question.go` + `internal/prompt/render.go`.** `render_options` / `render_criterion`.
Invariants §5 items 14–18.

- [x] **2.5.1** `Question` interface + `Choice`/`Score`/`Noul` types (§7, `docs/API.md`). The Go type
      design collapses Python's dict-or-list `criteria` into `[]ChoiceOption`.
      (2026-09-20) — in `question/` with root aliases rather than at the root, per D11.
- [x] **2.5.2** `render_options` / `render_criterion` / `serialize_state` in `internal/prompt/render.go`,
      on top of `jsonx`.
      (2026-09-20) — on `jsonx.Compact` (#17) and `jsonx.Marshal` (#18); no encoding reimplemented.
      Two layers — see the note under this task.
- [x] **2.5.3** Port `original/tests/test_criteria.py:33-100` as a table test.
      (2026-09-20) — ported as a table, plus the `:96-100` round-trip.
- [ ] **2.5.4** **Drop** `test_criteria.py:103-116`, which uses `inspect.getsource` to assert five
      string literals are present in `Agent.__init__`; replace it with a behavioural test on the Go
      device-fallback policy.
      (2026-09-20) — **partial: the drop is done; the behavioural replacement is not.** There is no
      Go device-fallback policy to test — `Agent` is M7 — and inventing one just to have something
      to assert would be worse than the test that was removed. M7 owns the replacement.

- [ ] **2.5.5** _(new, 2026-09-20)_ **Deviation to close, not a bug in 2.5.** `render_options` is
      ported over Python's loose internal `{t, ins, crit}` shape (that is what the corpus records,
      and what `options/score/dict-criteria` needs — `enumerate` on a dict walks its keys, which
      `ScoreQuestion{Levels}` cannot express). Where upstream **raises** on a `crit` of the wrong
      shape — `crit.items()` on a non-dict for choice, `crit.get` on a list for noul
      (`common.py:42`, the `AttributeError` already noted under 1.3) — the Go renderer returns the
      empty or default option list instead. Unreachable through the public types, but M5 and M7
      build `prompt.Internal` directly, and a silently wrong option list is exactly the
      plausible-answer failure §M1 exists to prevent. The Go port of `_to_internal`
      (`agent.py:229-238`) must reject those shapes at the boundary where Python raises.

**Exit criteria**

- [x] `just check` green; commit after each task. (2026-09-20) — `just ci` exit 0 (it is the
      superset: `fmt-check lint-md lint-py test-race lint check-tidy`), plus
      `go list -deps ./lang ./mailtext ./presets ./jsonx ./question | grep -c onnxruntime` = 0,
      all five cross-builds, and the suite green with Python stripped from `PATH`.

### M3 — Router (pure, no weights, no network)

**Status: done** (2026-09-20). Landed as six commits from `3d7fe08`. Evidence: `just ci` exit 0
(treefmt clean, markdownlint clean, `go test -race ./...` ok on all ten packages, golangci-lint 0
issues, check-tidy clean); `go test -race -count=10 .` ok;
`go list -deps ./lang ./mailtext ./presets ./jsonx ./question` names no `onnxruntime-purego`;
cross-builds green on linux/{amd64,arm64}, darwin/arm64, windows/amd64 and under `CGO_ENABLED=0`;
the package's tests pass under `env -i`, so nothing here needs Python or weights. Two pieces of the
declared router surface are deferred with reasons, to Task 6.11 and Task 7.6.

**Task 3.1: `RouteDecision` + model registry.** Invariants §5 items 39–47.

- [x] **3.1.1** `RouteDecision` struct with JSON tags plus `Map() Obj` for callers who want the dict.
      (2026-09-20) — five keys always present in Python's insertion order; `MarshalJSON` goes through
      `jsonx.Marshal` per D12, and `TestRouteDecisionMarshalledByEncodingJSONIsCompacted` pins why it
      must.
- [x] **3.1.2** The model registry (names → `ModelSpec{repo, subfolder, …}`). (2026-09-20) —
      `ModelSpec{Repo, Subfolder}` collapses upstream's tuple-or-string union, so `_split` has no
      counterpart; `DefaultModels()`/`StandaloneModels()` hand out copies.
- [x] **3.1.3** Port the route precedence and the exact reason strings. (2026-09-20) — all eight
      reasons asserted verbatim, and five whole decisions asserted byte-for-byte against
      `json.dumps` output from the reference environment.
- [x] **3.1.4** **Deviation:** `router.py:266` emits `repo` as a raw `(repo, subfolder)` tuple on the
      auto-workflow branch while every other branch emits a string via `_repo_str()`. Through
      `json.dumps` that surfaces as a JSON array. **Always emit the string**, and document the
      deviation in the README (Task 7.5). (2026-09-20) — `decideByWorkflow` uses `ModelSpec.String()`
      like every other branch; the deviation is recorded in its GoDoc and is on Task 7.5's list.
- [x] **3.1.5** **Reproduce faithfully:** `workflow` is `None` on the model/task paths but carries the
      detected workflow name on the **lang and detection** paths even though it did not drive the
      decision. (2026-09-20) — reproduced, and asserted in both directions by
      `TestRouteWorkflowLeaksIntoLaterBranches`.
- [x] **3.1.6** _(new, 2026-09-20, review)_ Expose what `original/tests/test_router.py` asserts on, or
      3.3.1 cannot port "the whole file" without unexported access: `NormalizeModelName` (upstream
      `normalise_name`, including the error on an unknown name), `MatchTypedDecisionsWorkflow`,
      `DefaultModels()` / `StandaloneModels()` (copies), `ModelSpec.String()` for `_repo_str`, and
      `Router.SetMaxLoaded` / `MaxLoaded()` — the upstream tests mutate `max_loaded` after construction
      (`test_router.py:241,249,266`) and `docs/API.md` is constructor-only. Acceptance:
      `test_router.py:97-110, 176-266, 213-228` port as they are. (2026-09-20) — all exported; the
      three cited spans port unchanged in `router_upstream_test.go`.
      **Deviation:** `NormalizeModelName`'s error text is Go's own. Upstream interpolates Python list
      reprs into the message and nothing asserts on it (`test_router.py:109-113` only checks that it
      raises), so reproducing the formatting would be unpinned cosplay. The error wraps
      `ErrUnknownModel` and names the valid names and aliases.
- [x] **3.1.7** _(new, 2026-09-20)_ `jsonx.ReprString`: Python's `str.__repr__`, because six reason
      strings interpolate `%r` over caller-supplied text (`router.py:256, 261, 267, 272, 285`) and Go
      has no equivalent. (2026-09-20) — `strconv.Quote` disagrees three ways at once (always
      double-quotes, emits Go's `\a\b\f\v` where Python emits `\x07\x08\x0c\x0b`, and would escape
      printable non-ASCII), and each disagreement still reads like a plausible reason. 33 cases
      pinned against real CPython output, plus the invalid-UTF-8 case Python cannot represent.
- [x] **3.1.8** _(new, 2026-09-20)_ `ModelSpecFromString`, which `docs/API.md:402`'s decision table
      promises and its signature block omits. (2026-09-20) — needed by `test_router.py:229-231` and
      by `test_local_e2e.py:47`, both of which override the registry with local directories and
      expect the path back unchanged.

**Task 3.2: `Router` + LRU.** Invariants §5 items 48–53.

- [x] **3.2.1** Port the LRU lifecycle (capacity, eviction order, preload). (2026-09-20) — including
      that `load` appends before it evicts, so at a cap of 1 the newcomer survives; that `attach` and
      `preload` both raise the cap; and that the cap floors at 1.
      Upstream's `_evict` reconciliation pass (`router.py:191-195`) has no counterpart: it exists
      because two Python dicts can drift apart, and one map and one slice maintained together cannot.
      **Deviation:** `Preload` with no names walks a declared order rather than the registry map.
      Python walks its models dict, whose order is the `DEFAULT_MODELS` literal; a Go map has none,
      and `Loaded()` would come back differently on every run. Pinned at `-count=10`.
- [x] **3.2.2** Python's `Router` is not concurrency-safe; Go's must be — `sync.Mutex`, plus a
      `-race` test that routes and evicts from several goroutines. (2026-09-20) — the mutex guards
      the agent cache alone; `models`, `defaultModel` and `autoTaskDetection` are fixed at
      construction, so `Route` takes no lock and stays as cheap as upstream promises it is.
      `go test -race -count=10 .` green.
- [x] **3.2.3** Expose the loader as an injectable hook —
      `WithLoader(func(ctx, name, ModelSpec) (*Agent, error))` — so the upstream LRU/preload tests,
      which monkeypatch `rr.load`, stay expressible without reflection. (2026-09-20) — the signature
      returns `Agent`, the interface of D13, not `*Agent`.
- [x] **3.2.4** Eviction closes the evicted agent's backend; assert it, since a leaked ORT session is
      hundreds of MB. (2026-09-20) — **a deviation, and it needs the line:** upstream drops the victim
      from both dicts and lets refcounting free it, which in Go frees nothing.
      Releasing is bounded by ownership: the Router closes agents its own loader built and only
      forgets an attached one. Without that, attaching one checkpoint to two routers would be a
      double free, and `test_router.py:269` — which asserts the attached sentinel survives — would be
      asserting a use-after-close.
- [x] **3.2.5** _(new, 2026-09-20)_ `Attach`, `Unload` and `Loaded`, which 3.2.1 does not name and
      `test_router.py:206-209, 258-270` asserts on (invariants #50, #52). (2026-09-20) —
      **Deviation:** `Unload` returns an error where Python returns nothing, for the same reason
      eviction closes at all: here freeing is a call that can fail.
- [x] **3.2.6** _(new, 2026-09-20)_ `Router.Close`, which `docs/API.md:342` declares and upstream has
      no counterpart for. (2026-09-20) — releases every agent the Router built, forgets the rest, and
      is idempotent.
- [x] **3.2.7** _(new, 2026-09-20)_ `ErrNoLoader`. Until M6 supplies the default loader there is
      nothing to build an agent from, and caching a nil agent that panics at first use would be the
      worst available answer. (2026-09-20) — every upstream LRU test injects a loader anyway.

**Task 3.3: Port the upstream router suite.**

- [x] **3.3.1** Port the whole of `original/tests/test_router.py` (~90 assertions). (2026-09-20) —
      98 checks, counted by instrumenting the Python file's own `check()`. **32 of them were already
      ported by M2** into `lang/lang_test.go` (`detect_script`, `is_english`, `guess_latin_language`,
      `state_text` — upstream keeps its language tests in the router file); the remaining 66 are in
      `router_upstream_test.go`. Two have no Go counterpart and say so in place: `decision/is dict`
      and `lru/cap 1 agents match order` both assert properties of Python's containers that one
      struct, and one map/slice pair, cannot lose.
      **Finding: one upstream assertion is vacuous.** `test_router.py:264` checks
      `ra.max_loaded >= 1` on a cap-1 router holding one attached agent — true however `attach`
      behaves. Deleting the cap raise left the whole ported suite green, and the sabotage pass is
      what caught it; invariant #50's actual content now has its own test
      (`TestRouterAttachRaisesTheCapEnoughToHoldEverything`).
- [x] **3.3.2** Port section 1 of `test_local_e2e.py:46-67`, which is pure routing — the file states
      outright "No model weights are loaded: `Router.route` is pure." (2026-09-20) — eleven languages
      through a local-path registry override, plus an assertion that the router used has no loader at
      all, so "no weights are loaded" is enforced rather than asserted by comment.

**Task 3.4: An opt-in cheaper-checkpoint path for CPU deployments.** _(new, 2026-09-20)_
Spike S3 suggested "defaulting the Router to mmBERT-base" as a latency lever, and the measurement
backs the size of it: `laya-multilingual` is **2.2–2.7× faster** than either ModernBERT-large
checkpoint at every shape, and the only one that answers in under a second. But upstream routes by
language, not by cost, and M7 asserts end-to-end parity against Python — so **changing the default
is not available**. This task is the honest version of the lever.

> **(2026-09-20) The 2.2–2.7× above understates it.** Re-derived from `BENCHMARKS.md`'s own tables
> while citing them for 3.4.3: against `laya-typed-decisions` the ratio is 2.88× at 1×512 and 3.15×
> at 8×512. The range is right for `laya`; it is low for the typed-decisions checkpoint. The GoDoc
> therefore cites the measured milliseconds rather than repeating a ratio.

- [x] **3.4.1** Decide whether go-laya offers a cost-biased routing option at all, or simply
      documents the measured numbers and lets the caller pass `model=` themselves. Cheapest correct
      answer wins; do not build an option nobody asked for. **(2026-09-20) Answered: document only,
      no option.** `ForModel("multilingual")` already expresses exactly the request, so an option
      would add a second spelling of one thing and a second switch the parity suite has to hold off.
- [x] **3.4.2** If it is offered: an explicit functional option, never a changed default, and the
      parity suite runs with it **off** so Task 7.4 keeps meaning what it says. (2026-09-20) — not
      applicable: 3.4.1 decided against offering it.
- [x] **3.4.3** Whatever is decided, `BENCHMARKS.md`'s per-checkpoint numbers are the justification
      and must be cited, not re-derived. (2026-09-20) — cited in `Router`'s GoDoc: 645 ms against
      1691 ms and 1856 ms for one question at the default 512-token `max_len`.

**Deferred out of M3, with the reason.** Two pieces of the router surface need types M3 cannot
define, so they are filed where they can actually be built: the default loader and its
`WithRouterDevice` / `WithRouterToken` options under **Task 6.11**, and `Router.Predict` /
`Router.SystemOne` under **Task 7.6**. Neither is reachable from anything upstream tests without
weights — `test_router.py` never calls `predict`.

**Milestone check**

- [x] `lang` + `mailtext` + `presets` + `Route` is a genuinely useful Go library with zero ML
      dependencies, and it covers everything the upstream test suite actually tests. (2026-09-20) —
      covers it: all 98 checks of `test_router.py` are accounted for, 32 in `lang/` and 66 in the
      root package. **The "zero ML dependencies" half needed correcting**, see the note below.
- [x] `go list -deps ./lang ./mailtext ./presets ./backend` contains no `onnxruntime-purego` (the §8
      check as D9 words it, run early). Under purego nothing links ORT symbols, so "links no ONNX
      symbols" was never a testable claim. (2026-09-20) — run as
      `go list -deps ./lang ./mailtext ./presets ./jsonx ./question`, `backend/` not existing yet:
      zero matches.

> **(2026-09-20) The zero-ML-dependency claim above named the wrong thing, and §2's prose repeated
> it.** `Route` lives in `router.go` at the **root** (§2's layout, `docs/API.md`, and D9 all say so),
> and D9 equally allows the root package to depend on the binding — so from M6 onwards a
> routing-only importer pulls `onnxruntime-purego` into its module graph. The same collision D11
> resolved for the question types, one milestone later.
>
> **Resolved: the root keeps the Router, and the claim is narrowed to what is true and testable.**
> Under purego nothing is `dlopen`ed until `Open`, so a caller who only routes still never _loads_
> ONNX Runtime; what it gains is a build-graph edge. §8's definition of done already scopes the
> check to `lang`, `mailtext`, `presets` and `backend` — the root was never in it — so §8 was right
> and the M3 wording and §2's prose were loose. Both corrected.

### M4 — Pure-Go tokenizer (highest risk)

> ⚠️ This is the single largest correctness risk in the port (R1). Marker positions are token
> indices, so one off-by-one silently corrupts every decision.

**Task 4.1: The interface first.**

```go
package tokenizer

type Tokenizer interface {
    Encode(text string) []int64 // add_special_tokens = false
    MaskToken() string
    MaskID() int64
    CLSID() int64
    SEPID() int64
    PADID() int64
}
```

- [x] **4.1.1** Define the interface in `tokenizer/`. `build_sequence` then ports 1:1 and stays
      backend-agnostic, which keeps D2 reversible (R1).
- [x] **4.1.2** _(new, 2026-09-20, review)_ `[]int64`, not `[]uint32`: the only consumer is
      `backend.Batch.InputIDs [][]int64` for a graph whose `input_ids` are int64
      (`scripts/export_onnx.py`, `internal/onnxspike/testdata/forward_pass.json`), `build_sequence` is
      pure slice concatenation and should port without width changes, and the golden JSON decodes as
      int64. Vocab and merge tables stay `int32` internally. `Encode` first applies
      `strings.ToValidUTF8(text, "�")` — the 4.4.11 boundary; without it, bytes 0xF5–0xFF would map
      to the 13 EN byte-chars that have no vocab entry and be silently dropped (`unk:null`, §1.4 item 3).
- [x] **4.1.3** _(new, 2026-09-20, review)_ The concrete type — not the interface — exposes
      `IDToToken(id int64) string` and `VocabSize() int64`, for the golden test (which prints id **and**
      token-string diffs) and the 4.5.1 fuzz invariant. No decode path: nothing in inference decodes.

**Task 4.2: Special-token resolution.** ~40 lines.

- [x] **4.2.1** Read token _strings_ from `tokenizer_config.json`; resolve ids from `tokenizer.json`'s
      `added_tokens` + `model.vocab`.
- [x] **4.2.2** Prefer the tokenizer config over the encoder config — `multilingual/encoder/config.json`
      says `cls_token_id: 1` while its tokenizer says `<bos>` = **2**, and 2 is what the weights were
      trained with (§1.2).
- [x] **4.2.3** Ignore the Gemma `extra_special_tokens`-as-a-list quirk that `_fix_tokenizer_config`
      patches around in Python — Go never needs the workaround.
- [x] **4.2.4** Assert the resolved ids for both checkpoints against §1.4's table (EN: UNK 50280,
      CLS 50281, SEP 50282, PAD 50283, MASK 50284 · ML: PAD 0, EOS/SEP 1, BOS/CLS 2, UNK 3, MASK 4).

**Task 4.3: Purpose-built pure-Go tokenizer, `tokenizer/`.** _(rewritten 2026-09-20 on review; D10)_
Two pipelines only — `typed-decisions`' `tokenizer.json` is byte-identical to English's. Estimate
~750 production lines (loader 200, added-token matcher 150, normalizers 40, ByteLevel scanner + table
120, Metaspace 40, BPE 150, glue 50) plus ~500 test lines.

> **Why not vendor.** The original Task 4.3 planned to vendor `gomlx/go-huggingface`'s `hftokenizer`
> and fix three bugs: Metaspace prepend checking `text[0] != ' '` instead of `!starts_with(replacement)`,
> `byte_fallback`/`fuse_unk` parsed and never used, and "longest-first greedy" added-token matching
> with no `lstrip`/`normalized` semantics. Those are exactly the stages laya depends on; what it gets
> right (JSON loader, merge loop, byte table) is the easy ~250 lines. The module is not in the local
> cache and the bug citations were never pinned to a commit. The in-house byte-fallback code §1.5 had
> pointed at does not exist. Vendoring 3k lines under `default: all` golangci-lint also costs a `NOTICE`
> entry, a lint exclusion path and a fork to maintain. So: gomlx is a **reference to read**, never to
> vendor; `sugarme/tokenizer` stays rejected (`byte_fallback`/`fuse_unk` commented out, open panic on
> consecutive whitespace); `daulet/tokenizers` stays R1's CGO fallback. The Python oracle is local
> (`.venv-ref`, tokenizers 0.23.2), so every semantic below is pinned by corpus **before** code
> (Task 4.4.12 is the first M4 commit).

- [x] **4.3.1** **Loader.** Decode `tokenizer.json` (34 MB for ML; budget < 1 s cold — measure it),
      build `vocab map[string]int32` and `merges map[[2]int32]{rank, newID}`; fail at load if a merge's
      concatenation is missing from the vocab (HF's `MergeTokenOutOfVocabulary`); assert the two
      byte-coverage facts from §1.4 item 3.
- [x] **4.3.2** **AddedVocabulary, HF's two phases.** Phase 1 matches `normalized:false` tokens on the
      raw string (all 249 ML tokens, 7 EN specials); phase 2 normalizes each remaining segment and
      matches `normalized:true` tokens (109 EN). Leftmost-longest at each position (a per-position trie
      walk; ≤ 249 patterns, max length 31). `lstrip` extends the match start left over `unicode.IsSpace`
      chars but never past the previous match's end; `rstrip`/`single_word` implemented though unset on
      both checkpoints. The order is load-bearing on both: on EN `[MASK]` (phase 1, lstrip) must swallow
      spaces before the space-run tokens (phase 2) see them; on ML the `▁`-run tokens must match on raw
      text before `Replace` turns spaces into `▁`. Acceptance: `'foo  [MASK]'` → `[foo, 50284]`;
      `'a\n<mask>'` ML → `[476, 108, 4]`; `'a [MASK]'` → `[66, 50284]`; `'a​[MASK]'` →
      `[66, 12882, 50284]` (Unicode `White_Space`, not Cf); `'a▁▁b'` ML → `['▁a', '▁▁', '▁b']`;
      `'a' + ' '*49 + 'b'` EN → `[a, 50254, 50254, Ġb]`.
- [x] **4.3.3** **Normalizers.** NFC via `golang.org/x/text/unicode/norm` (pure Go, per segment) and
      `Replace{String}`. Nothing else is needed by either pipeline; do not port the rest of HF's list.
- [x] **4.3.4** **ByteLevel.** A **hand-written rune scanner**, not a regex: HF's pattern
      `'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+` has a negative
      lookahead RE2 cannot express, and Go's `\s` is ASCII while Oniguruma's is Unicode `White_Space`
      (`'a\xa0\xa0b'` → two separate NBSP tokens). The scanner (~80 lines): try the seven `'x` literals;
      optional U+0020 + `IsLetter` run; + `IsNumber` run; + other run; else a whitespace run that gives
      back its last char when the run has length ≥ 2 and does not reach the segment end — that is
      `\s+(?!\S)` followed by `\s+`. Plus the `bytes_to_unicode` table. Acceptance:
      `'   x'` EN → `['ĠâĢ', 'ĥ', 'âĢĥ', 'x']`; `"DON'T"` → `[DON, ', T]`;
      `'a' + '\t'*26 + 'b'` → `[a, ĉ×8, ĉ×8, ĉ×4, ĉ×5, ĉ, b]`.
- [x] **4.3.5** **Metaspace** `prepend_scheme:always`, `split:true`: per segment, `Replace` then
      prepend `▁` iff `!strings.HasPrefix(seg, "▁")`, then split `MergedWithNext`. Acceptance:
      `'   \n   '` → seven tokens, and `'<start_of_turn>user\nhi<end_of_turn>'` →
      `['<start_of_turn>', '▁user', '\n', '▁hi', '<end_of_turn>']`.
- [x] **4.3.6** **BPE.** Port `merge_word` — per-char vocab lookup; byte fallback emits `<0xXX>` per
      char **before** merging, and only if every byte is in the vocab; `fuse_unk` — and `merge_all` (a
      heap ordered by rank then position, with stale-entry skip). Every merge maps to a `newID` resolved
      at load, so a merge can never produce a missing entry. `fuse_unk` is unreachable on both real
      checkpoints (ML's only missing byte token is `<0x09>`, needed only by U+0009, itself in the vocab),
      so it is tested on a synthetic ten-token `tokenizer.json` under `tokenizer/testdata/`. Acceptance:
      `'a\x00b'` ML → `['▁a', '<0x00>', 'b']`; the byte-fallback and tab-run cases above.
- [x] **4.3.7** A tests-only RE2 oracle for the scanner (the pattern with `\s` spelled out as the
      `White_Space` set and the same give-back post-fix) plus a fuzz cross-check scanner-vs-oracle. Two
      implementations agreeing, plus the Python differential, is as close to Oniguruma as Go gets.
- [x] **4.3.8** Load time and allocation of `Encode` measured and recorded here; if the 34 MB decode is
      slower than an ONNX session load, a streaming decoder is a contained follow-up, not a redesign.

  Measured 2026-09-20 on the S3 laptop, `go test -bench . -benchtime 5x`:

  | Benchmark             |    time | alloc/op | allocs/op |
  | --------------------- | ------: | -------: | --------: |
  | `Open/english`        |   68 ms |  19.6 MB |   187,150 |
  | `Open/multilingual`   | 1178 ms | 207.4 MB | 1,539,697 |
  | `Encode/english`      |  191 µs |  24.0 kB |       427 |
  | `Encode/multilingual` |  456 µs |  22.9 kB |       235 |

  `Encode` is a non-issue — `build_sequence` calls it three times per question, so under 2 ms against
  S3's 0.6–1.9 s forward pass. The multilingual `Open` straddles the budget: 0.6–0.7 s for a single
  cold open, 1.18 s under repeated load where 207 MB per iteration pressures the allocator. Both are
  recorded rather than the flattering one. New Task 4.3.9 carries the remedy.

**Task 4.4: Golden corpus, both checkpoints.** Assert **ids and token strings** — an id diff alone tells
you nothing, `["▁▁","x"]` vs `["▁x"]` tells you exactly which stage broke.

Required cases:

- [x] **4.4.1** `""`, `" "`, `"  "`, `"\n"`, `"\t"`, `"\r\n"`, `"   \n   "` — consecutive whitespace is
      the most common drift/crash site.
- [x] **4.4.2** **`text` and `" " + text` for every case** — this is the `common.py:68` semantic and it
      differs between the two checkpoints: on ML the `Replace` normalizer makes `tok(" x") == tok("x")`,
      on EN the leading space becomes `Ġ` and they differ (§1.4).
- [x] **4.4.3** Compact JSON exactly as `serialize_state` emits it, with nested braces, escapes,
      non-ASCII, long arrays.
- [x] **4.4.4** The mask literal (`[MASK]` / `<mask>`) appearing inside user text.
- [x] **4.4.5** Every added token bare and mid-sentence: `<unused0>`, `<start_of_turn>`, `<2mass>`,
      `[@BOS@]`, `|||IP_ADDRESS|||`, and runs of 2–24 spaces (EN ids 50254–50276; 24 spaces = 50254).
- [x] **4.4.6** Normalization: precomposed vs decomposed `é`, fullwidth `Ａ`, ligature `ﬁ`, NBSP,
      ZWJ/ZWSP, BOM, combining marks. EN normalizes NFC; ML does **not** normalize beyond space→`▁`,
      so these two legitimately differ.
- [x] **4.4.7** Emoji with ZWJ sequences, skin tones, flags — these exercise byte fallback on ML.
- [x] **4.4.8** Devanagari, Arabic, CJK, Thai, Cyrillic, Korean jamo vs precomposed.
- [x] **4.4.9** Lone surrogates / invalid UTF-8 (Go tolerates, Python `str` does not) — decide and
      document the boundary; sanitize at the API edge.
- [x] **4.4.10** Both corpora run green: ids **and** token strings identical to Python.
- [x] **4.4.11** _(new, 2026-09-20)_ Lone surrogates are **not** in `testdata/tokenizer_*.jsonl`
      and cannot be: Python writes `\ud800` and Go's decoder yields U+FFFD, so the vector would
      assert against a corrupted input. 4.4.9's "decide and document the boundary" is therefore
      settled the other way — sanitize at the API edge and test that Go-side, without a vector.
- [x] **4.4.12** _(new, 2026-09-20, review)_ **Corpus v2 — the first M4 commit, before any Go code.**
      The 53 cases per checkpoint contain **zero byte-fallback cases**: every ML emoji/script case is a
      single-char vocab entry, while `'a𠀋b'` → `['▁a', '<0xF0>', '<0xA0>', '<0x80>', '<0x8B>', 'b']`
      is pinned by nothing. Regenerate `tokenizer_{en,ml}.jsonl` (reviewed diff, R6) adding ~45 probes:
      byte-fallback chars (U+2000B, U+1FAE8, U+1D518, U+17000); Unicode whitespace runs (NBSP×2, U+3000,
      EM SPACE×2 with and without a leading space, `\v`, U+0085); contractions including `DON'T` and
      curly `’`; added tokens adjacent to text; `[MASK]`/`<mask>` lstrip variants; 48/49-space runs;
      tab runs of 2 and 26; newline runs of 2/3/32; literal `▁▁`; NUL/SOH/DEL; digits. In the dumper,
      token strings come from `convert_ids_to_tokens`, never `Encoding.tokens` (which returns the
      lstripped slice, e.g. `'  [MASK]'`).
      (2026-09-20) — 42 probes, 53 to 95 cases per checkpoint, nothing removed, and a second
      regeneration is byte-identical. The case the box was written for reproduces exactly: `'a𠀋b'`
      on multilingual gives `['▁a', '<0xF0>', '<0xA0>', '<0x80>', '<0x8B>', '▁b']`. The
      `convert_ids_to_tokens` clause was **already satisfied** at `dump_python_parity.py:275,277` —
      a constraint to preserve, not a bug to fix.

**Task 4.4.13** _(new, 2026-09-20, M4)_ **Per-stage vectors so CI tests something.**
`models/` is gitignored, so the corpora above are a local gate; without this, CI ran no tokenizer
parity assertion at all.

- [x] **4.4.13** `dump_pretok` emits `testdata/pretok_{en,ml}.jsonl`: the normalizer's output and the
      pre-tokenizer's pieces, from `backend_tokenizer.normalizer.normalize_str` and
      `.pre_tokenizer.pre_tokenize_str`. Both are pure `str -> str` / `str -> list[str]`, so the Go
      normalizer, ByteLevel scanner and Metaspace are tested against real Oniguruma output with no
      checkpoint present. `pre_tokenize_str` is fed the **normalized** text, which is the order
      `encode` uses. 103 cases each, all green.

**Task 4.4.14** _(new, 2026-09-20, M4)_ **The give-back at a segment boundary.**

- [x] **4.4.14** `\s+(?!\S)`'s lookahead is evaluated inside the added-token segment, so
      `'a\t\t[unused0]'` keeps its tab run whole while `'a\t\tX'` splits it 1 + 1. Nothing in 4.4.12
      reached this, and a one-token shift moves every marker after it. Eight `giveback/` probes added,
      in both bracket spellings: `[unused0]` is an added token on English and plain text on
      multilingual, `<unused0>` the reverse.

**Task 4.3.9** _(new, 2026-09-20, M4)_ **Streaming the 34 MB decode.** Deferred, not done.

- [ ] **4.3.9** `Open` on multilingual costs 1.18 s, 207 MB and 1.54 M allocations under repeated
      load (0.6–0.7 s for a single cold open), against 4.3.8's "budget < 1 s cold". `encoding/json`
      over 256000 vocabulary entries and 580604 merge pairs is the whole cost. 4.3.8 already scopes the
      remedy as "a contained follow-up, not a redesign"; it is not done inside a parity milestone.

**Task 4.5: Fuzz + differential.**

- [x] **4.5.1** A Go fuzz target checking invariants only: no panic, ids < `vocab_size`, ASCII
      round-trip.
- [x] **4.5.2** Seed the corpus from the Task 4.4 cases and commit the interesting crashers.
- [x] **4.5.3** A one-off differential run of ~100k lines of real multilingual corpus through both
      Python and Go, diffing the id streams. That is what actually finds the metaspace/added-token bug
      classes. Oracle: `.venv-ref/bin/python` (tokenizers 0.23.2, the version every fixture header
      records), both checkpoints; record mismatches **per stage** (added-token / normalizer /
      pre-tokenizer / BPE / byte fallback), not just a count. Unicode-version skew between Go's
      `unicode` tables and Oniguruma's is the one defect only this run can find (R1).
      (2026-09-20) — `scripts/build_corpus.py` + `scripts/dump_stages.py` +
      `tokenizer/differential_test.go`, driven by `just corpus && just dump-stages && just diff-tokenizer`.
      157281 lines × 2 tokenizers, per-stage attribution, **85 mismatches, all in the normalizer**
      (4.5.4). The oracle records the added-token split, the normalizer's output and the
      pre-tokenizer's pieces per segment; the Go side replays each stage in `Encode`'s order and
      reports the first that disagrees.
- [x] **4.5.4** Record the differential result (corpus, line count, mismatches) in this file.
      (2026-09-20) — the record is the block below.

**The differential result** _(2026-09-20, task 4.5.4)_

Corpus: 157281 unique lines, `build/corpus/MANIFEST.json` (gitignored; rebuild with `just corpus`).
Five streams, because prose alone does not reach three of the five stages:

| stream    |  lines | what it covers                                                                |
| --------- | -----: | ----------------------------------------------------------------------------- |
| `tatoeba` | 99 813 | real sentences, 427 languages, water-filled across the per-language exports   |
| `probes`  | 17 670 | every Unicode category boundary, combining marks, UCD `NormalizationTest.txt` |
| `vocab`   | 19 812 | the multilingual checkpoint's own vocabulary, U+2581 decoded back to space    |
| `locale`  | 14 359 | gettext catalogues: format specifiers, markup, punctuation runs               |
| `added`   |  5 627 | every added token of **both** checkpoints in 16 whitespace contexts           |

Result, per stage, both tokenizers (english also serves typed-decisions — their `tokenizer.json`
files are byte-identical, 3583228 bytes, same sha256, which is why `testdata/` has two and not three):

| stage         | english | multilingual |
| ------------- | ------: | -----------: |
| added-split   |       0 |            0 |
| normalizer    |  **85** |            0 |
| pre-tokenizer |       0 |            0 |
| BPE           |       0 |            0 |
| byte fallback |       0 |            0 |

**The 85 are one defect class, and it is the one R1 predicted.** Go's `golang.org/x/text` v0.41.0
and the Rust normaliser behind `tokenizers` 0.23.2 disagree on the canonical combining class of
**108 codepoints** — combining marks assigned in Unicode 11.0 through 15.0 (U+07FD, U+0898-089F,
U+08CA-08D3, U+1AC0-1ACE, U+1DF6-1DFA, U+10D24-10D27, U+10EFD-10EFF and more). The Rust crate's
tables predate them, so it treats each as a starter and does not reorder across it; Go does.
Enumerated by `scripts/probe_ccc.py`, which reads the table out of the implementation by
normalising each codepoint beside U+0334 (ccc 1) and watching for a swap.

Three facts that bound it:

- **Go is not the outlier.** CPython 3.12's `unicodedata` (Unicode 15.0.0) agrees with Go on all 108
  and produces the same 85 divergent lines against the Rust normaliser. Cross-checked independently
  of the Go harness.
- **It cannot affect multilingual.** That checkpoint's normalizer is `Replace(" " → U+2581)`, not
  NFC, so the combining-class tables are never consulted. Only english and typed-decisions are
  exposed.
- **No natural text reached it.** All 85 lines come from the UCD `NormalizationTest.txt` stream.
  Zero of 99 813 real sentences across 427 languages, zero of 19 812 vocabulary entries, zero of
  14 359 gettext strings and zero of 5 627 added-token contexts contain a sequence that diverges.
  A disputed codepoint occurs at all in 539 probe lines and exactly 1 vocabulary entry.

**The harness was sabotage-tested, and the first attempt was vacuous.** Five defects were injected
one at a time and the run re-executed: NFC→NFD (85+ → `normalizer`), the ByteLevel give-back
threshold (750 → `pre-tokenizer`), added-token `lstrip` (8 en / 6 ml → `added-split`), the BPE
merge loop's left-neighbour re-push (94 588 / 95 511 → `bpe`), and the byte-fallback hex case
(13 684 ml, 0 en → `bpe`, correct: english sets `byte_fallback:false`). Each was attributed to the
right stage. **Two of the five were caught by nothing on the first, prose-only corpus** — natural
text contains no `[MASK]`, so the added-token stage was tested exactly zero times, and two-space
runs before text were too rare to reach the give-back. The `added` stream and the whitespace-run
probes exist because of that result, not because of a hunch.

**Gate**

**Task 4.5.5** _(new, 2026-09-20, from 4.5.3)_ **Reconcile the combining-class disagreement — a
decision, not yet a fix.**

- [ ] **4.5.5** Decide between reproducing the Rust tables and staying Unicode-conformant, and
      implement the choice. R1 already anticipated "the fix is a pinned table, not a different
      strategy", and the mechanism is bounded: the 108 codepoints act as starters upstream, and a
      starter blocks canonical reordering and composition across it, so isolating each occurrence and
      normalising the spans between them reproduces the Rust result exactly — roughly 40 lines plus a
      generated table. The cost is stated rather than hidden: it makes go-laya's NFC deliberately
      **non-conformant**, pinned to one crate's stale data, and a `tokenizers` bump moves the set, so
      `scripts/probe_ccc.py` has to run as part of any pin change. Against that, the deviation it
      closes is unreachable from 139 611 lines of natural text, vocabulary and UI strings, and cannot
      touch multilingual at all. **Whoever takes this decides it in this file before writing code.**

**Gate**

- [~] **M5 does not start until both golden corpora are 100 % green and `CGO_ENABLED=0 go build ./...`
  still passes.** _(2026-09-20) — partial:_ both corpora **are** 100 % green (206 subtests: 103
  cases × 2 checkpoints, ids and token strings, bare and with the leading space) and
  `CGO_ENABLED=0 go build ./...` passes. Task 4.5.3's differential run is **done** and its result is
  recorded above. What remains is Task 4.5.5: the run found one real divergence class, 85 lines out
  of 157 281, confined to the NFC normalizer on english/typed-decisions and to codepoints no natural
  text in the corpus contains. That is a bounded, named deviation rather than an unknown, so it is
  recorded here rather than silently accepted — but it is the last thing standing in this gate, and
  it does not block `build_sequence`, whose arithmetic is tokenizer-independent and whose
  stub-tokenizer tests (5.2.2) exist precisely for this.
- [ ] If parity cannot be reached, execute R1's fallback: swap to `daulet/tokenizers` (CGO, wraps the
      same Rust crate Python uses — parity by construction) behind the Task 4.1 interface, and record
      the CGO consequence against D2.

### M5 — `build_sequence`

**Task 5.1: Port `common.py:49-86` into `internal/prompt/sequence.go`.**
This is 38 lines carrying most of the porting risk, and it has **no test at all upstream**.
Invariants §5 items 1–13.

- [x] **5.1.1** Port the token-budget arithmetic: `opt_budget`, the `< 16` fallback, `[:48]`.
      (2026-09-25) — `internal/prompt.BuildSequence`; `TestBuildSequenceBudget` pins the 49-id
      option cap, invariant #6's own 77-option example (`per=4`, budget −116, head at 8, 320 ids),
      a fallback that does truncate (mask included), `head_max_len < 16`, and a head that takes the
      whole budget. Python's `//` is spelled as Go's `/`: they differ only on a negative numerator,
      which `max(4, …)` absorbs, and the comment says so.
- [x] **5.1.2** Port the truncation direction: `[-room:]` when `truncate_left`, else `[:room]`.
      (2026-09-25) — `pyHead`/`pyTail` carry Python's slice semantics, including `st[-0:]` being the
      **whole** state. `TestBuildSequenceTruncation` pins both directions, room beyond the state, and
      `room == 0` in both: under `truncate_left` the last slot goes to the state's first id, not to
      `[SEP]` — so the 5.2.3 note's "agrees only by way of the clamp" is true only when the options
      end short of `max_len - 1`.
- [x] **5.1.3** Port the final `ids[:max_len]` clamp and the marker filter that drops markers pushed
      past the clamp.
      (2026-09-25) — `TestBuildSequenceClamp`: a marker at `max_len` is dropped, one at `max_len - 1`
      survives, and the trailing `[SEP]` is clamped away. Invariant #13's `ValueError` stays with the
      caller (M7), as it does upstream.
- [x] **5.1.4** Marker positions are token indices — assert them explicitly, not just the id stream.
      (2026-09-25) — every case asserts the marker slice exactly and that each surviving marker
      points at a mask id; `TestBuildSequenceLayout` adds `option_order`, the mask scrub in all three
      sources, noul/score heads, a structured state and zero options. All of it runs on a
      one-id-per-word stub tokenizer, so a failure names the arithmetic; 5.2.2 builds on that stub.
      Three injected defects (`st[-0:]` as empty, `<=` in the marker filter, head budget +1) each
      failed a case. The fixture test is still 5.2.1's.

**Task 5.2: Assert against `testdata/sequence.jsonl` byte-for-byte.**

- [x] **5.2.1** A table test across the matrix
      `(qtype, criteria, instructions, state, max_len, head_max_len, option_order, truncate_left)`.
      (2026-09-25) — `TestBuildSequenceGolden` (`internal/prompt/sequence_golden_test.go`): all 45
      cases on the three real tokenizers, ids and markers exact plus `markers_lost`; a mismatch
      prints ids and token strings around the first divergence. Gated by `golden.SkipWithoutModels`,
      so CI skips it. A `room` off by one fails 7 cases — but a head budget +1, a head floor of 9
      and `<=` in the marker filter **fail none**: no fixture head sits on its budget edge and no
      marker lands on `max_len`. Only the stub tests (5.2.2, 5.2.3) guard those three.
- [x] **5.2.2** Cases that exercise the truncation arithmetic independently of tokenizer drift —
      feed a stub tokenizer with known ids so a failure names the arithmetic, not the tokenizer.
      (2026-09-25) — delivered with 5.1's stub tests: `TestBuildSequenceBudget`, `…Truncation` and
      `…Clamp` run on a one-id-per-word stub; no duplicates added.
- [x] **5.2.3** Boundary cases: exactly `max_len`, one over, an option list that alone exceeds the
      budget, zero options, and a state that truncates to nothing.
      (2026-09-25) — `TestBuildSequenceBoundaries`, one row per boundary on the stub: one over is
      split into state-cut (both directions) and trailing-`[SEP]`-cut; options past `max_len` lose a
      marker (invariant #13's precondition); `room == 0` in both directions. `<=` in the marker
      filter and `st[-0:]` read as empty each fail a row.
      _(2026-09-20, measured while scoping M5)_ **Two of these cannot come from the fixture and must
      come from 5.2.2's stub tokenizer.** `markers_lost` is 0 in all 45 cases, so nothing in
      `sequence.jsonl` trips invariant #13's `ValueError`; and the smallest `room` across the 45 is
      **15**, with the three `truncate_left` cases at 498/1010/1010, so none of them reaches
      `room == 0` — which is the case that matters, because Python's `st[-0:]` returns the **whole**
      list rather than none, and a Go port written as `st[len(st)-room:]` agrees only by way of the
      final `ids[:max_len]` clamp. Regenerating `sequence.jsonl` to add them is a reviewed diff
      (R6), so the stub is the cheaper route.

**Task 5.3: Port `collate_items` (`common.py:218-251`).** _(new, 2026-09-20, review)_ D3 had filed it
as training math; `agent.py:266` calls it on every `system_one`, and its padding is what the graph sees.

- [x] **5.3.1** `internal/prompt.Collate(items, padID) backend.Batch`: `L = max len(ids)`,
      `kmax = max len(markers)`; ids right-padded with `pad_id`, `attention_mask` 0 on padding,
      `marker_pos` 0-filled, `marker_mask` false beyond each row's markers, `qtype` per row. The
      `target`/`label`/`meta` fields are training-only and are not ported.
      (2026-09-25) — `internal/prompt.Collate` over a flat `[]Item` (agent.py:266 always passes one
      group); rows are copies. `TestCollate` pins ragged ids and markers with a non-zero pad, a
      single row, `kmax == 0` (n empty rows, as `torch.zeros((n, 0))`) and no items, which returns
      the zero `Batch` where Python returns `None` — unreachable through the Agent. Padding with 0
      and `kmax` from row 0 each fail a case. Needed 6.1.1's `backend.Batch`, done alongside.
- [x] **5.3.2** Acceptance: equals the collated tensors in `logits.jsonl` (after Task 1.7) for all 30
      batches, and `Σ attention_mask == input_tokens` for every case (invariant #33).
      (2026-09-25) — `TestCollateGolden`: each batch rebuilt as `system_one` does (a test-local
      `_to_internal`, `BuildSequence` under the checkpoint's own `max_len`/`head_max_len` from
      `rl_agent_config.json`, `Collate` with `PADID()`); 30/30 equal on all five tensors and their
      shapes, plus `qtypes` and `input_tokens`. Gated by `golden.SkipWithoutModels`. Padding with 0
      fails 20 of the 30 batches.

**Task 5.4: `option_order` / `truncate_left` are internal parameters.** _(new, 2026-09-20, review)_
Invariants #4 and #11 say the public API never sets them, and `docs/API.md` exposes neither — so
they live on `internal/prompt`'s function signature only, never on `Agent` or `Question`.

- [x] **5.4.1** Both flags exercised by 5.2.1 (the fixture carries them in all 45 cases) and by 5.2.2's
      stub-tokenizer tests; no public symbol.
      (2026-09-26) — no code change; verified. `TestBuildSequenceGolden` passes both from all 45 cases
      (3 set `option_order`, 3 `truncate_left`). Forcing `truncateLeft = false` fails 2 golden cases
      — `typed-decisions/choice/truncate-left` passes either way — and 4 stub rows; forcing
      `optionOrder = nil` fails all 3 `option-order` golden cases and the `option_order` layout row.
      `go doc -all` over the 8 non-`internal` packages matches `optionorder|truncateleft` 0 times, and
      `docs/API.md` names neither.

### M6 — Backend + checkpoint loading

**Task 6.1: `backend.Backend` interface — a public leaf package (D9).**

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

- [x] **6.1.1** Define `Backend` and `Batch` in `backend/`, importing nothing beyond the standard
      library; `go list -deps ./backend` names no third-party module.
      (2026-09-25) — pulled forward as the gate of 5.3, exactly as specified above.
      `go list -deps -f '{{if not .Standard}}{{.ImportPath}}{{end}}' ./backend` prints only
      `…/go-laya/backend` itself. §8's CI-level `go list -deps` check is still open.
- [x] **6.1.2** An in-memory fake replaying `testdata/logits.jsonl`, keyed on the collated tensors Task
      1.7 adds — every M7 test then runs without ORT.
      (2026-09-26) — `internal/backend/fake`: `New(tb, checkpoint)` loads that checkpoint's records,
      `Forward` matches all five tensors exactly and returns copies of the recorded `logits` and
      `act_logits`. **Per checkpoint, not global:** `english` and `typed-decisions` record
      byte-identical batches with different logits (shared tokenizer and budgets, different weights),
      so tensors alone cannot key the fixture. `TestFakeReplaysEveryRecord` replays all 30 bit for
      bit and `TestFakeIsPerCheckpoint` pins the ambiguity; both pass with `LAYA_MODELS` unset.
      Forcing a match on any input fails both; dropping the checkpoint filter fails both as well.
      The tensor decoders moved from `internal/prompt`'s test into `internal/golden`
      (`Matrix`/`Vector`/`CollatedBatch`) so the fake and `TestCollateGolden` share them.
- [x] **6.1.3** The fake fails loudly on an unknown input rather than returning zeros, so a prompt
      regression cannot masquerade as a passing test.
      (2026-09-26) — a miss returns no outputs and an error wrapping `ErrUnknownBatch` that names
      the nearest same-shape recording and its first differing cell, e.g.
      `first at input_ids[1][5]: got 21008, recorded 21007`.
      `TestFakeRejectsUnknownBatch` covers one changed cell in each of the five tensors, a dropped
      row, a trimmed column and all 10 `multilingual` batches fed to the `english` fake; returning
      zeros on a miss fails it. An unknown checkpoint name fails `New` rather than yielding a fake
      that rejects everything.

**Task 6.2: `internal/hub` — HF resolve + cache.**

- [x] **6.2.1** Resolve via `https://huggingface.co/{repo}/resolve/{rev}/{path}`, optional
      `Authorization: Bearer $HF_TOKEN`.
      (2026-09-26) — `hub.Client.Fetch`: a HEAD for the metadata, then a GET, both on the resolve
      URL, the revision escaped whole (`refs%2Fpr%2F1`) as huggingface_hub quotes it. Redirects are
      followed by hand so the token goes only to the Hub's host: a relative 307 keeps it, a 302 to
      the CDN drops it. `TestFetchResolvesURL` and `TestFetchAuth` pass; forwarding the token on
      every redirect fails `TestFetchAuth`. Repo, revision and path are validated before any
      request (`ErrInvalidPath`, `TestFetchRejectsUnsafeNames`), since from 6.2.4 on the path comes
      from the Hub's own listing.
- [x] **6.2.2** ETag/sha verification on every download (R7).
      (2026-09-26) — checked against the live Hub first: the resolve URL's `X-Linked-Etag` is the
      **git blob sha1** of a regular file (`README.md`: `git hash-object` gives the header's value)
      and the **sha256** of an LFS file (`model.safetensors`: equal to the API's `lfs.sha256`), and
      only the first hop carries it — the CDN does not. A 404 carries a **weak** `W/"…"` ETag, so a
      weak, missing or wrong-length tag is `ErrUnverifiable` (fail closed), never a skipped check.
      `TestFetchVerifies` covers both hash kinds, match and one-byte mismatch
      (`ErrHashMismatch`), direct and via the CDN, with no file cached on any failure. Dropping the
      comparison fails it, and so does stripping `W/` and trusting the rest.
- [x] **6.2.3** A local cache under `$LAYA_CACHE` or `os.UserCacheDir()/laya`, with atomic
      write-then-rename so an interrupted download is never served.
      (2026-09-26) — `DefaultDir` (`TestDefaultDir`); files land at
      `Dir/owner/name/<commit>/path`, keyed on `X-Repo-Commit` (40 hex or `ErrInvalidPath`, since it
      becomes a directory), written to `.partial-*` beside it and renamed only after the hash
      matched. `TestFetchCache`: one file and nothing else, and a second `Fetch` downloads nothing.
- [x] **6.2.4** An `allow_patterns`-equivalent so a subfolder request downloads only that subfolder —
      the bundle repo is 2.4 GB; the English checkpoint alone is 846 MB.
      (2026-09-26) — `hub.Client.Snapshot(ctx, repo, rev, allow)`: lists the repo via
      `GET /api/models/{repo}/revision/{rev}` (checked live: `sha` + `siblings[].rfilename`), keeps
      the names matching `allow` with fnmatch semantics as `filter_repo_objects` does (`*` crosses
      `/`, a trailing `/` gets `*`; `TestMatch`), and `Fetch`es each at the listing's **commit**, not
      at `rev`: a file served from another commit is `ErrCommitMismatch` (`TestSnapshotPinsCommit`).
      `TestSnapshotAllowPatterns`: `multilingual/*` downloads exactly the two `multilingual/` files —
      not `multilingualx/a` — and a second call downloads nothing. Listed names are remote input and
      go through the path validation before any download (`TestSnapshotRejectsUnsafeListing`).
      **Deviation:** patterns matching nothing are `ErrNotFound` naming them
      (`TestSnapshotNoMatch`), where upstream downloads nothing and fails later on the missing
      subfolder. Mutations — no filter, fetching at `rev` without the commit check, no name
      validation — each fail at least one of these.
- [x] **6.2.5** Everything cancellable via `context.Context`; a cancelled download leaves no partial
      file behind.
      (2026-09-26) — `TestFetchCancel`: a download cancelled halfway returns `context.Canceled`, and
      one whose connection drops halfway fails; neither leaves a file. While stalled, the cache path
      must not exist yet — a crash runs no cleanup — and writing straight to it fails the test.
- [x] **6.2.6** Offline mode: an already-cached checkpoint resolves with no network call.
      (2026-09-26) — `Client.Offline`. A complete `Snapshot` records `refs/<rev>` → commit and
      `listings/<commit>.json` under `Dir/owner/name/`, both by rename; an offline `Snapshot` or
      `Fetch` resolves the revision (a commit id is its own), filters the cached listing and requires
      every match as a regular file, else `ErrNotCached`. `TestOffline` runs the offline client over a
      transport that fails the test on any request: by `main` and by commit it returns the online
      directory; an uncached subfolder, an unknown revision or commit, a deleted file and one swapped
      for a symlink are all `ErrNotCached`. Skipping the offline branch or the file check fails it.
      Wiring `HF_HUB_OFFLINE`/`LAYA_OFFLINE` belongs to the loader (6.11.4).
- [x] **6.2.7** Tests run against an `httptest` server — no network in CI.
      (2026-09-26) — every `internal/hub` test talks to `httptest` servers only: `fakeHub` (Hub and
      CDN) for `Fetch`, `fakeRepo` (listing plus resolve) for `Snapshot`; `grep -L httptest
internal/hub/*_test.go` prints nothing and no test names a real host.
- [x] **6.2.8** _(new, 2026-09-26)_ Decide the default revision. The Hub's `main` moved from
      `1c5edc17…` (the revision every golden vector describes, Task 1.2.2) to `55cf4c4…`, so
      `Open("convaiinnovations/laya")` at `main` no longer loads what the fixtures were made from.
      Pinning a default in the default loader (Task 6.11.1) or following `main` is a user decision.
      (2026-09-26) — **decided: pin** `1c5edc17a7acd8701df6fc341c0d179f1c62c982`; following `main`
      is an explicit opt-in. Carried into 6.11.1 and onto 7.5.3's deviation list; no code until the
      loader exists.
- [ ] **6.2.9** _(new, 2026-09-26)_ The English checkpoint is the bundle repo's **root**, so upstream
      passes no `allow_patterns` for it and downloads all 2.4 GB (`agent.py:125-128`). Narrowing that
      to the root checkpoint's files is a deviation and needs a decision before 6.11.1 builds on it.

**Task 6.3: ONNX backend.** Per Spike S2's binding decision.

- [ ] **6.3.1** Implement `Backend` over the chosen binding; dynamic batch/seq/k.
- [ ] **6.3.2** Device selection (`cpu`/`cuda`/`coreml`) with a logged fallback, replacing Python's
      three `print()` warnings with `slog` — and warn only when a fallback actually happened
      (upstream `e630a68`).
- [ ] **6.3.3** `Close()` releases the session; assert no leak across a load/evict cycle (Task 3.2.4).
- [ ] **6.3.5** _(new, 2026-09-26, read-only recon)_ The binding's `Session.Run` takes a `ctx` but
      never uses it (it passes NULL `RunOptions`), so a started forward pass cannot be cancelled;
      `Forward` checks `ctx` before and after, and says so in its doc comment.
- [ ] **6.3.4** The ONNX backend lives in `internal/backend/onnx`; the §8 check is D9's
      `go list -deps` assertion over `lang`/`mailtext`/`presets`/`backend`. `Route` lives in the root
      package, which is allowed to depend on the binding — `dlopen` happens only in `Open`.

**Task 6.4: Checkpoint validation.** Port `_verify_compatibility`'s intent.

- [ ] **6.4.1** Require cfg keys `encoder` and `head_layers`.
- [ ] **6.4.2** Require the graph's declared inputs/outputs (`input_ids`, `attention_mask`,
      `marker_pos`, `marker_mask`, `qtype` → `logits`, `act_logits`).
- [ ] **6.4.3** Fail with a wrapped `ErrIncompatibleCheckpoint` naming what was wrong.
- [ ] **6.4.4** Validate the ONNX/safetensors header **before** handing the bytes to the runtime, and
      never `os/exec` or `encoding/gob` a downloaded artifact (R7, Task 0.4's deferred item).
- [ ] **6.4.5** _(new, 2026-09-20, review)_ Validate the graph's `act_logits` width against
      `len(cfg["act_costs"]) + 1` (§1.3) and `logits` width against `kmax`; a mismatch fails with
      `ErrIncompatibleCheckpoint` naming the shape. All three shipped checkpoints have width 2, which is
      exactly why a hardcoded 2 would pass every test and break on the first fine-tune.

**Task 6.5: Decide which attention implementation the shipped export uses.** _(new, 2026-09-20)_
Spike S1 exports with `attn_implementation="eager"` because S1.1 says to. Upstream runs `sdpa`
(`original/laya/common.py:134`), and on `multilingual` the two differ by 5.1e-05 in `logits` —
which is a floor on how close the Go port can get to golden vectors generated from the Python,
and is larger than the ONNX-vs-PyTorch error on the other two checkpoints.

- [ ] **6.5.1** Try `dynamo=True` with `attn_implementation="sdpa"`; the dynamo exporter may
      handle `scaled_dot_product_attention` where TorchScript would not.
- [ ] **6.5.2** If sdpa exports, make it the default and re-measure — the export should match the
      reference path, not merely be close to it.
- [ ] **6.5.3** If it does not, set the M7 parity tolerance from the measured eager-vs-sdpa gap
      **per checkpoint** and say so in the test, rather than picking a round number.

**Task 6.6: Acquire and pin the ONNX Runtime shared library.** _(new, 2026-09-20)_
R7 requires a pinned runtime, and S2.4 could only pin half of it: `go.mod` pins the binding, and the
binding pins the _C API version_ (`supportedAPIVersions = []uint32{23}` — 1.23.x or nothing), but
nothing pins the `.so` itself. No in-house repo solves this; `go-pocket-tts` is bring-your-own
(`docs/INSTALL.md:9-33`), with the version appearing only in a developer's local settings file.

- [ ] **6.6.1** Decide between documenting a bring-your-own library and shipping a verified
      download — `pockettts-tools model download-onnx --sha256` is the in-house precedent.
- [ ] **6.6.2** Verify the resolved library's version at startup rather than inferring it from the
      filename, and fail with a named error when the C API version does not match.
- [ ] **6.6.3** Reuse `internal/onnxspike`'s resolution chain: `LAYA_ORT_LIB`, `ORT_LIBRARY_PATH`,
      then platform candidates; a variable that is set but points nowhere is an error, not a
      fallback.

**Task 6.7: Decide the Windows and js/wasm story for the ONNX backend.** _(new, 2026-09-20)_
`release.yml` builds `windows/amd64`, but `go-pocket-tts` stubs purego out on Windows and js/wasm
entirely (`internal/onnx/runner_windows.go`, `runner_wasm.go`). Spike S2 inherited that constraint
rather than testing it, so today the honest claim is "untested", not "unsupported".

- [ ] **6.7.1** Establish whether `purego.Dlopen` actually works on Windows with ORT 1.23, or only
      that nobody has tried.
- [ ] **6.7.2** Either support it or ship a stub that fails with a named error, and say which in the
      README. Task 6.3.4's build-tag split is where it belongs.
      _(2026-09-26, read-only recon)_ — **a prerequisite of 6.3.1, not a follow-up:** the binding calls
      `purego.Dlopen` untagged, and purego v0.9.0 defines it only on darwin/freebsd/linux/netbsd, so the
      first non-test import of the binding breaks `release.yml`'s `GOOS=windows go build ./...` unless
      a tagged stub lands with it.

**Task 6.8: Go-vs-Python ONNX Runtime parity across the whole matrix.** _(new, 2026-09-20)_
Spike S2 proved one forward pass, on one checkpoint, at one shape. M6 owes the rest.

- [ ] **6.8.1** All three checkpoints at S1's four shapes, against fixtures generated by
      `scripts/export_onnx.py --fixture`, using the fixture schema and the `maxScaledDiff` metric
      `internal/onnxspike/spike_test.go` already implements (Task 6.10 moves them; do not re-derive).
- [ ] **6.8.2** Set the tolerance **per checkpoint** from the measured numbers — S1 recorded
      `multilingual` as an order of magnitude looser than the other two — rather than picking one
      global constant.
- [ ] **6.8.3** _(new, 2026-09-20)_ Explain the S3 timing gap: at a matched thread count and thermal
      state the same graph ran 2.7× faster under ORT 1.23.0 from Go than under ORT 1.30.0 from
      Python on `english`, but only 1.1× faster on `multilingual`. Numerics agreed to 3.8e-06, so
      this is a kernel-selection or version difference, not a correctness one — but it is the kind of
      difference that turns into a tolerance surprise when the pinned runtime moves (Task 6.6).

**Task 6.9: int8 dynamic quantization — latency and calibration in one task.** _(new, 2026-09-20)_
Carries Spike S3.4 and S3.5 forward. S3 measured 0.6–1.9 s per question on CPU and the condition for
"if too slow" fired, but int8 cannot be evaluated on latency alone: **calibrated probabilities are
the product**, and a calibration regression does not show up in an argmax test (R4). Splitting the
two halves is how quantization silently becomes the default somewhere, so they stay one task.

**Gated on M1's harness and Task 7.1 (`internal/calib`)** — a labelled eval set and an ECE/Brier
implementation must exist first. Do not start it before then, and do not narrow it to the latency
half when they are late.

- [ ] **6.9.1** `onnxruntime.quantization.quantize_dynamic` over all three exports, in the pinned
      reference environment (`scripts/requirements-ref.txt`), as a `scripts/` flag rather than a
      one-off — the artefact must be reproducible like every other export.
- [ ] **6.9.2** Re-run `just bench-onnx` against the quantized graphs and record the speedup in
      `BENCHMARKS.md` beside the fp32 numbers, on the same hardware. The plan predicts 2–4×; at
      0.6–1.9 s fp32 that lands at 0.2–0.9 s, so it changes the throughput story and **not** the
      "interactive needs a GPU" one. Say so rather than letting the multiple imply otherwise.
- [ ] **6.9.3** Measure **ECE and Brier** against fp32 on the same inputs, per checkpoint and per
      question type. This is the acceptance criterion, not a follow-up.
- [ ] **6.9.4** **int8 is never the default.** Ship it, if at all, as an explicit opt-in whose
      documentation carries 6.9.3's numbers.

**Task 6.10: Absorb `internal/onnxspike` into `internal/backend/onnx`.** _(new, 2026-09-20, review)_
"M6 deletes it" was wrong for roughly 600 of the package's 834 lines. What moves, by `git mv` where
possible: the `findORTLibrary` chain and its set-but-missing-is-an-error rule (Task 6.6.3); the
`.onnx.data` sibling check and the from-a-path-never-a-reader constraint; `maxScaledDiff` and the
two-tolerance split; the fixture schema and `TestForwardPass` (6.8.1's harness); `TestValueCleanup`
(D5's regression, kept behind `LAYA_ONNXSPIKE_FINALIZER=1` under a new name); both benchmarks and
`just bench-onnx` (Task 6.9.2 needs the **same** harness for comparability). What dies: the package
name, the "throwaway" doc comment, `requireORTLibraryB`, the bare `ortAPIVersion` const (6.6.2 verifies
it instead), and `findModel`'s `../../build/onnx` default.

- [ ] **6.10.1** `just bench-onnx` and the finalizer reproduction run from the new package unchanged.
- [ ] **6.10.2** `internal/onnxspike/` is gone; `internal/onnxspike/README.md`'s content moves with it.

**Task 6.11: The Router's default agent loader.** _(new, 2026-09-20, from M3)_
M3 shipped `WithLoader` and `ErrNoLoader`: until a checkpoint can actually be built, a Router has no
way to make an agent and says so rather than caching a nil one.

- [ ] **6.11.1** A default loader, so `NewRouter()` with no `WithLoader` can load. It is what
      `router.py:174-178` does inline: build an Agent from the spec's repo, subfolder, device and
      token. Its default revision is `1c5edc17a7acd8701df6fc341c0d179f1c62c982` (6.2.8), and a
      subfolder maps to `hub.Client.Snapshot(…, []string{sub + "/*"})`.
- [ ] **6.11.2** `WithRouterDevice` and `WithRouterToken` (`docs/API.md:347-348`), including
      upstream's `token or os.environ["HF_TOKEN"]` fallback (`router.py:159`). Deferred out of M3
      because they configure a builder that did not exist; adding them there would have stored two
      values nothing read.
- [ ] **6.11.3** Widen `Agent` past `Close() error` only when M7 needs it (D13), not here.
- [ ] **6.11.4** _(new, 2026-09-26)_ Offline from the environment: map `HF_HUB_OFFLINE` (upstream's
      switch, honoured by `snapshot_download`) and/or a `LAYA_OFFLINE` onto `hub.Client.Offline`
      (6.2.6).

### M7 — Agent, calibration, end-to-end parity

**Task 7.1: `internal/calib`.** Invariants §5 items 22–29.

- [ ] **7.1.1** `temp_bucket` and the `temperature_by_options` → `temperature[qtype]` lookup.
- [ ] **7.1.2** The `max(1e-3, t)` floor.
- [ ] **7.1.3** Max-subtracted softmax over **exactly the first _k_ logits**.
- [ ] **7.1.4** Entropy confidence.
- [ ] **7.1.5** `Round4` from `jsonx` applied at the same points Python applies `round()`
      (invariant #29).
- [ ] **7.1.6** _(new, 2026-09-20, review)_ Port `ece_score` (`common.py:187-197`, a public upstream
      export the plan had neither ported nor dropped) as `internal/calib.ECE`, and define a multi-class
      Brier score (mean `Σ(p−y)²`) beside it — Task 6.9.3 is gated on both existing and no task
      created them. Fixture `testdata/ece.jsonl` from the generator. Acceptance: matches the fixture,
      including `conf == 0` falling in no bin and the empty input yielding NaN.

**Task 7.2: `Agent.SystemOne`.** Invariants §5 items 19–34.

- [ ] **7.2.1** The `choice` answer shape.
- [ ] **7.2.2** The `score` answer: `Σ i·p[i]`, an expectation, **not** an argmax.
- [ ] **7.2.3** The `noul` answer: **no** `probabilities`, no `legend`, and confidence
      `max(p1, 1-p1)` — **not** the entropy formula, which for k=2 genuinely disagrees (invariant #26).
- [ ] **7.2.4** `legend` carries the **raw** criterion value, not the rendered option text.
- [ ] **7.2.5** The act head: `act = softmax(act_logits.float(), -1)` (torch float32, `agent.py:295`),
      then `action.act_probability = round(act[r, 0], 4)` (`agent.py:310`). **Column 0 only; there is
      no threshold and no decision upstream** _(corrected 2026-09-20 — the task used to ask for "its
      threshold")_. The width is `len(cfg["act_costs"]) + 1`, validated in 6.4.5.
- [ ] **7.2.6** Batching: several questions in one forward pass, with per-question `qtype`, through
      Task 5.3's `Collate`.
- [ ] **7.2.7** _(new, 2026-09-20, review)_ **Precision and tie-break** (invariants #24a/#30a). Softmax
      and entropy confidence are numpy **float32** (`agent.py:294,305-307`; the dumper mirrors this at
      `dump_python_parity.py:747-751`); `score` is float64 over a float32 `p`; `noul` is float64 from a
      float32 `p[1]`; the act softmax is torch float32. `p.argmax()` is **first max**. Go does the
      arithmetic in `float32` where numpy does. Acceptance: all 27 `answers.jsonl` cases byte-equal; a
      test with two exactly equal logits picks the first key.
- [ ] **7.2.8** _(new, 2026-09-20, review)_ `Answer.MarshalJSON` per type via `jsonx.Obj`, with **no
      `omitempty`**: a criteria key of `""` is legal in Python and `omitempty` would drop `"choice"`,
      breaking invariant #30; the same class of bug hits `legend`/`probabilities` when empty.
      Acceptance: key sets and order per `docs/API.md` for every `answers.jsonl` case, plus an injected
      `""` key.
- [ ] **7.2.9** _(new, 2026-09-20, review)_ `Router.SystemOne` as an alias of `Router.Predict`
      (invariant #53, `router.py:311`); `Questions.Validate` rejects duplicate IDs with
      `ErrDuplicateQuestionID` — a Python dict cannot hold them, so this is an error, not a last-wins.

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
- [ ] **7.4.5** _(new, 2026-09-20, review)_ Port `test_local_e2e.py:190-211`, which 7.4.4's summary
      omitted: the triage intent lands in `{refund, billing_question}`; a language switch at
      `max_loaded=1` leaves `Loaded() == ["multilingual"]` (the only eviction test with real weights);
      and the `routing` payload is present and marshals. Gated like 7.4.1.

**Task 7.5: README + examples.**

- [ ] **7.5.1** Port every README example to Go, as compiling `Example` functions so CI proves they
      still build (§8).
- [ ] **7.5.2** Fix the image URLs — upstream's point at
      `raw.githubusercontent.com/NandhaKishorM/laya/main/...`, so a fork's README silently renders
      upstream's assets.
- [ ] **7.5.3** Document the deliberate deviations _(list completed on the 2026-09-20 review)_: `repo`
      always a string (Task 3.1.4); `instructions` as `string` only; the default revision pinned
      to `1c5edc17` instead of following `main` (6.2.8); no `mps` device; `$LAYA_CACHE`
      instead of the huggingface_hub cache; and every dropped or renamed public export —
      `proper_reward`, `td_lambda_targets` (D3), `ece_score` (internal `calib.ECE`), `load` (→ `Open`),
      `RLAgent` (no alias), `QTYPES`/`QTYPE_NAMES` (→ `QType`), `detect_language` (→ `lang.Analyse`),
      `confidence_from_probs` and `render_options` (internal). `Detection.ScriptProfile` is **not** on
      the list any more: Task 2.2.7 made it ordered.
- [ ] **7.5.4** Replace upstream's T4 latency claims with the Spike S3 numbers in `BENCHMARKS.md`.
      **(2026-09-20) — the `BENCHMARKS.md` half is done by S3.2**; what remains here is the README,
      which still opens with "33 ms" and repeats it six more times.
- [ ] **7.5.6** Carry S3.3's standing prohibition: go-laya's own README states the **measured** CPU
      latency and the fact that batching does not amortise it, and does not describe the port as
      "CPU-first". S3.2 made the honest claim available; this is where it gets made.
- [ ] **7.5.5** State the tokenizer/checkpoint revisions the port is verified against.

**Task 7.6: `Router.Predict` / `Router.SystemOne`.** _(new, 2026-09-20, from M3)_
`router.py:293-311`: route, load the chosen checkpoint, run `system_one`, then add the decision under
a `routing` key. Deferred out of M3 because it needs the widened `Agent` of D13 and the `Result`
type, neither of which existed there.

- [ ] **7.6.1** Widen `Agent` with `SystemOne`, as D13 says M7 would, and have the concrete agent
      satisfy it.
- [ ] **7.6.2** `Predict`, and `SystemOne` as its alias (`router.py:311`).
- [ ] **7.6.3** `result["routing"] = dict(decision)` (invariant #53). Note the consequence of task
      3.1.4: upstream's payload carries the raw tuple on the auto-workflow branch, and go-laya's
      carries the string, so this is where that deviation becomes visible to a caller.
- [ ] **7.6.4** The `routing` block goes through `jsonx.Marshal` (D12), not `encoding/json`, or the
      separators are compacted back out of the detection object inside it.

### M8 — Pure-Go native backend (after 1.0)

Same `backend.Backend` interface, no API change. Implement ModernBERT + mmBERT + the head over
safetensors, lifting `tensor`, `ops` and `safetensors` from `../go-pocket-tts`.

> **(2026-09-20, review — D8.)** This milestone is a **zero-shared-library** deployment story, not a
> speed play. S3 measured ORT-CPU as bandwidth-bound and the FLOP estimate puts pure Go at ~12 s per
> forward against ORT's 1.7 s, so the old 8.9 ("beat the S3 floor") could never have been ticked. What
> a native backend buys is a binary with no `.so` to acquire, pin or `dlopen` — Tasks 6.6 and 6.7
> solved by construction, Windows and js/wasm included.

- [ ] **8.1** Lift `internal/safetensors`, `internal/runtime/tensor` and `internal/runtime/ops` from
      `../go-pocket-tts` (§1.5).
- [ ] **8.2** Bias-free LayerNorm (`norm_bias: false`, eps 1e-5; layer 0 has no `attn_norm`).
- [ ] **8.3** GeGLU with the fused `mlp.Wi [5248,1024]` gate+up split, no MLP bias.
- [ ] **8.4** Fused QKV unpacking (`attn.Wqkv [3072,1024]`, no attention bias).
- [ ] **8.5** Sliding-window attention masks (window 128, ±64) and per-layer-type RoPE theta
      (full layers 0,3,…,27 at 160000; sliding at 10000; mmBERT uses 160000 for both).
- [ ] **8.6** The decision head with a **ReLU** FFN (§1.3) and the manual layer loop.
- [ ] **8.7** fp16 weight loading. The `temperature` tensor's dtype differs per checkpoint (§1.1) and
      the loader must tolerate that, but it is never read: calibration comes from the config JSON.
- [ ] **8.8** Gate promotion on the same golden vectors: probabilities within 1e-4, zero argmax flips.
      Head intermediates for invariants #35–38 are generated here, not before.
- [ ] **8.9** _(rewritten, D8)_ Latency within 10× of ORT-CPU at 512 tokens on the same hardware,
      recorded in `BENCHMARKS.md` beside the S3 numbers — a ceiling that keeps the backend usable as a
      background workload, not a race against MLAS.
- [ ] **8.10** _(new, D8)_ The `backend` interface is unchanged; the native backend is selected by
      option, and `go list -deps` for the four zero-dependency packages stays clean.

---

## 5. Invariants a Go test must assert

The full numbered checklist (74 items) is in **`docs/INVARIANTS.md`**. Summary of the sections, with
the Python line ranges each cites:

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

The four that cause silent wrong answers rather than loud failures, and therefore need tests first:

- **#18** `serialize_state` byte-parity — Go's `encoding/json` breaks separators, key order _and_
  HTML escaping.
- **#29** `round()` half-to-even vs Go's half-away-from-zero.
- **#56 / #63** Python dict iteration order as a tie-break in `detect_script` and `guess_latin_language`.
- **#26** `noul` confidence is `max(p1, 1-p1)`, not the entropy formula used by the other two types.

---

## 6. Risks

| #   | Risk                                                                                                                                                                                                                                                                                                             | Mitigation                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| --- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **Pure-Go tokenizer drift.** Marker positions are token indices, so one off-by-one silently corrupts every decision. Both candidate libraries had demonstrable bugs on exactly laya's paths, which is why D10 uses neither.                                                                                      | The `Tokenizer` interface (Task 4.1) keeps the decision reversible. Golden corpus v2 (4.4.12) + fuzz + a 100k-line differential run gate M5. If parity cannot be reached, `daulet/tokenizers` (CGO, wraps the same Rust crate Python uses — parity by construction) is a one-day swap. _(2026-09-20)_ With D10 the residual risk is Unicode-version skew between Go's `unicode` tables / `x/text` NFC and Oniguruma's; only 4.5.3 can find it, and the fix is a pinned table, not a different strategy. If Task 4.3 exceeds ~1.5k lines or a week, re-read gomlx for the dragging stage rather than vendoring it. _(2026-09-20, 4.5.3)_ **Found, and it is exactly this.** 157281 lines through both implementations: one divergence class, 85 lines, all in NFC, caused by `tokenizers` 0.23.2's Rust tables lacking the canonical combining class of 108 codepoints assigned in Unicode 11.0-15.0. Go and CPython agree; the oracle is the outlier. Unreachable from 139611 lines of natural text, vocabulary and UI strings, and impossible on multilingual, whose normalizer is Replace and not NFC. The pinned table this cell predicted is scoped as task 4.5.5; the `daulet/tokenizers` fallback is **not** indicated, since the defect is 108 codepoints of table data and not a strategy failure. |
| R2  | **`torch.onnx.export` fails on ModernBERT-large.** transformers#35545 is still open; no ModernBERT-large ONNX is published anywhere.                                                                                                                                                                             | Spike S1 up front. `sevenreasons/laya-onnx-fp16` unblocks development while our exporter is fixed.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| R3  | **CPU latency ≫ the README's 33 ms.** _(2026-09-20, S3)_ Confirmed and quantified: 0.6–1.9 s per question at the default 512-token `max_len`, 9–43× the T4 figure. Not fatal — it is the fast end of the predicted band — but it rules out the interactive framing upstream's README uses.                       | Measured before anything was promised; `BENCHMARKS.md` now carries the numbers and the hardware, and D6 records the decision. Of the levers listed here, **batching is not one** (measured flat on CPU) and **routing to mmBERT-base cannot be a default** without breaking M7's parity — it becomes opt-in Task 3.4. What is left: int8 (Task 6.9, gated on calibration), a GPU execution provider (Task 6.3.2), and setting `IntraOpNumThreads` correctly, which is free.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                |
| R4  | **int8 quantization wrecks calibration.** This model's whole value is calibrated probabilities.                                                                                                                                                                                                                  | Measure ECE and Brier, not accuracy. Never quantize by default.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| R5  | **`onnxruntime-purego` instability.** _(2026-09-20, S2)_ The recorded `AddCleanup` panic is **fixed** upstream. What remains: an unsynchronised `Runtime.Close` racing the GC cleanup path, and a dependency on the untagged HEAD of a 31-star library that says its API may change without notice.              | Close every `*Value` explicitly — that removes the race, and `internal/onnxspike` keeps a reproduction under `LAYA_ONNXSPIKE_FINALIZER=1`. `yalue/onnxruntime_go` remains the fallback behind D1's `Backend` seam, at the cost of CGO. New Task 6.6 pins the runtime itself.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| R6  | **Upstream tokenizer/transformers version drift** silently changes golden vectors. _(2026-09-20, M1)_ Real, and now measured: transformers 4.57.6 moves `multilingual`'s logits by 6.96 against the pin (D7). The tokenizer half is clean — `tokenizers` 0.22.2 and 0.23.2 produce identical ids on every probe. | `TestGoldenProvenance` (Task 1.4) reads the header of every `testdata/*.jsonl` and fails the suite on a mismatch; all three of its failure modes were exercised rather than assumed. `scripts/requirements-ref.txt` stays pinned, regeneration is byte-identical, and the diff is reviewed. `scripts/crosscheck_transformers.py` is how the next pin bump gets checked instead of hoped about.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| R7  | **Supply chain.** laya downloads checkpoints from the Hub; `laya.Open("someone/their-model")` must not be RCE.                                                                                                                                                                                                   | Carry the upstream security policy over (Task 0.4): verify ONNX/safetensors headers, ETag/sha checks, no `os/exec` or `encoding/gob` on downloaded artifacts.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| R8  | **Licensing.** This is a derivative of an Apache-2.0 work.                                                                                                                                                                                                                                                       | Preserve `LICENSE`, add `NOTICE` with the original copyright and a statement of modification (Task 0.3). Weights are Apache-2.0 and ungated.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |

---

## 7. Proposed Go API

Full type definitions — `Obj`/`Probs`/`Questions`/`AnswerSet` as ordered slices, the `State` interface,
`ChoiceOption`, the three `Answer` shapes, `Agent`, `Router`, functional options and sentinel errors —
are in **`docs/API.md`**, together with the exact JSON shapes the Python version emits.

The dynamic-typing decisions Python leaves implicit:

| Python                                                      | Go decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `criteria` on choice: `dict` **or** `list[str]`             | `[]ChoiceOption{Key, Desc any}` + a `Labels("a","b")` helper. One representation, order preserved.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `criteria` on noul: dict with `"true"`/`"false"`, or absent | Explicit `True`/`False` fields — removes any chance of getting index order wrong (false=0, true=1 always).                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| `state`: `str \| dict \| list`                              | A `State` interface with `TextState` / `ObjState` / `ListState`. Rejected `any`+reflection: it makes both the serialization and the detection flattening implicit. _(2026-09-20, M3: **out of step with the shipped code.** M2 shipped `lang.Analyse(state any)` and `lang.IsEnglish(state any)` in the public API, and M3's `Route(state any, …)` follows them — so the `any` road was taken twice before this row was revisited. Settle it in M7, where `Agent.SystemOne` is the last and largest caller: either introduce `State` and narrow all three, or record `any` as the decision and strike this row. Do not add a third spelling.)_ |
| `instructions`: `str` or anything                           | `string` only; callers serialize. Document the dropped edge case (non-str instructions are `json.dumps`'d with `ensure_ascii=True` — the one place laya escapes non-ASCII).                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| dict iteration order                                        | Ordered slices everywhere it is observable; `map` only where it provably is not.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| `RouteDecision` as a `dict` subclass                        | A struct with JSON tags, plus `Map() Obj` for anyone who wants the dict.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |

---

## 8. Definition of done for 1.0

- [ ] `just check` green; `go test ./... -race` green on linux/amd64 and darwin/arm64.
- [x] Both tokenizer golden corpora 100 % id- and token-identical to Python — 206 subtests green
      (103 cases × 2 checkpoints, each asserting the bare and leading-space encoding).
      **This is a local gate, not a CI one:** `models/` is gitignored and `tokenizer.json` is
      3.5 MB / 34 MB, so the runner skips unless `LAYA_MODELS` names a tree containing `laya/`.
      CI covers the stages instead, against `testdata/pretok_{en,ml}.jsonl` (task 4.4.13).
- [ ] `testdata/sequence.jsonl` byte-identical.
- [ ] End-to-end probabilities within 1e-4 of the fp32, CPU, single-thread, `sdpa` PyTorch run recorded
      in the fixture headers (Task 1.7's `compute` block) on all three checkpoints; **zero argmax flips**.
- [ ] `lang` + `mailtext` + `presets` + `backend` importable with no ML dependency — verified by
      `go list -deps` over those packages naming no `onnxruntime-purego` (D9; "links no ONNX symbols"
      was never testable under purego).
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
- [ ] The deliberate deviations from Python are listed in the README, per Task 7.5.3's completed list:
      dropped/renamed exports, `repo` always a string, `instructions` as `string` only, no `mps`.
