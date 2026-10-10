<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/logo-lockup-dark.png" />
    <img src="assets/logo-lockup.png" alt="Laya" width="330" />
  </picture>
</p>

**Multilingual, non-autoregressive System 1 decision engine, in Go.** Typed decisions over 100+ languages in a single forward pass, trained with reinforcement learning against strictly proper scoring rules (RLCD), with a router that picks the right checkpoint per request.

<div align="center">

[![Go Reference](https://pkg.go.dev/badge/github.com/MeKo-Christian/go-laya.svg)](https://pkg.go.dev/github.com/MeKo-Christian/go-laya)
[![CI](https://github.com/MeKo-Christian/go-laya/actions/workflows/ci.yml/badge.svg)](https://github.com/MeKo-Christian/go-laya/actions/workflows/ci.yml)
[![Hugging Face Model](https://img.shields.io/badge/%F0%9F%A4%97%20Model-convaiinnovations%2Flaya-blue)](https://huggingface.co/convaiinnovations/laya)
[![Multilingual](https://img.shields.io/badge/%F0%9F%A4%97%20Model-laya--multilingual-blue)](https://huggingface.co/convaiinnovations/laya-multilingual)
[![Hugging Face Space](https://img.shields.io/badge/%F0%9F%A4%97%20Space-laya--demo-orange)](https://huggingface.co/spaces/convaiinnovations/laya-demo)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

</div>

**go-laya** is a Go port of [`laya`](https://github.com/NandhaKishorM/laya) 0.3.4 by Convai
Innovations (see [`NOTICE`](NOTICE)). It runs the same three checkpoints and is built to answer
exactly as the Python package does; the model, the training and every accuracy and calibration
figure below are upstream's. Where it deliberately differs, and which revisions it is verified
against, is in [Differences from Python laya](#differences-from-python-laya).

<p align="center">
  <img src="assets/laya_vs_jev_full.png" alt="Laya versus TypeSafe Jev: accuracy on shared public datasets, every application workflow, all 51 languages, speed, calibration, and the cost of not preloading" width="100%" />
</p>
<p align="center"><em>Upstream's comparison, measured with the Python package. Its speed panel is a T4 GPU; this port runs on CPU (see <a href="#speed-cpu-measured-here">Speed</a>).</em></p>

Laya evaluates typed questions (`choice`, `score`, `noul`) over any state (text, email, ticket or JSON document) in **a single forward pass**. No text generation, so nothing to parse and nothing to hallucinate. On CPU, measured here, one question takes **160 ms** on `laya-multilingual` at 128 tokens and **1.7 s** on `laya` at its 512-token default ([`BENCHMARKS.md`](BENCHMARKS.md)).

Three checkpoints, and a `Router` that picks between them per request:

|                                                                                         | encoder          | params | context | use it for                    |
| --------------------------------------------------------------------------------------- | ---------------- | ------ | ------- | ----------------------------- |
| [`laya`](https://huggingface.co/convaiinnovations/laya)                                 | ModernBERT-large | 421M   | 512     | English                       |
| [`laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual)       | mmBERT-base      | 322M   | 1024    | 100+ languages, 2-3x faster   |
| [`laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large | 421M   | 1024    | the typed-decisions workflows |

---

## Installation

```bash
go get github.com/MeKo-Christian/go-laya
```

Go 1.26 or later. The forward pass runs on [ONNX Runtime](https://onnxruntime.ai) through a cgo-free
binding, so a build needs no C toolchain but a run needs two things on disk:

1. **The ONNX Runtime shared library.** `go run github.com/MeKo-Christian/go-laya/cmd/laya-ort`
   downloads the pinned ORT **1.23.0** for linux and darwin on amd64 and arm64, verifies its sha256,
   and stores it in the laya cache (`$LAYA_CACHE`, else the user cache directory + `/laya`), where
   `Open` finds it on its own. To bring your own library instead, set `LAYA_ORT_LIB` (or
   `ORT_LIBRARY_PATH`) to its path. `Open` accepts any ORT 1.x from 1.23 on and rejects anything
   else with `ErrRuntimeVersion`, but every number in this README was measured on 1.23.0.
2. **An ONNX export of each checkpoint you use.** The Hub ships safetensors, which only PyTorch
   runs, so the graph is exported once, locally, with
   [`scripts/export_onnx.py`](scripts/export_onnx.py) `--all --dynamo --out <dir>` (a Python step,
   pinned in [`scripts/requirements-ref.txt`](scripts/requirements-ref.txt)). Point
   `$LAYA_ONNX_DIR`, `WithONNXDir` or `WithGraph` at it, or put it in `onnx/` under the laya cache.
   `Open` downloads only the config and the tokenizer from the Hub.

`WithRuntime(RuntimeNative)` (for `Router`, `WithRouterRuntime`) needs neither: it runs the
checkpoint's own `model.safetensors` in pure Go on the CPU, and `Open` downloads the weights and
`encoder/config.json` with the config and the tokenizer ([D30][later]). Its full parity gate and its
latency are still open (PLAN Tasks 8.8 and 8.9).

---

## Quickstart: Route Mode (Recommended)

Laya ships three checkpoints. The built-in **`Router`** is the recommended entry point: it evaluates any state in any language, detects the script and language before the forward pass, and dispatches to the right checkpoint.

Every Go snippet in this README is taken from an `Example` in [`example_test.go`](example_test.go), so CI proves it compiles.

```go
// state is the README's support e-mail. An Obj keeps its key order, which is
// the order the model reads the fields in.
var state = laya.Obj{
	{Key: "from", Value: "user@acme.com"},
	{Key: "subject", Value: "Duplicate charge on invoice #4411"},
	{Key: "body", Value: "Hi, we were billed twice for March. Please refund the duplicate today or we will cancel our plan."},
}

// questions are the README's typed questions. Questions is a slice, not a
// map, because the order is the order the answers come back in.
var questions = laya.Questions{
	{ID: "department", Q: laya.ChoiceQuestion{
		Ins: "Which department should handle this request?",
		Opts: []laya.ChoiceOption{
			{Key: "billing", Desc: "invoices, payments, refunds"},
			{Key: "technical", Desc: "bugs, outages, system errors"},
			{Key: "sales", Desc: "pricing, new contracts"},
			{Key: "other", Desc: "everything else"},
		},
	}},
	{ID: "urgency", Q: laya.ScoreQuestion{
		Ins:    "How urgent is this request?",
		Levels: []laya.Criterion{"not urgent", "soon", "critical deadline or blocking issue"},
	}},
	{ID: "churn_risk", Q: laya.NoulQuestion{Ins: "Does the user threaten to cancel or leave?"}},
	{ID: "refund_requested", Q: laya.NoulQuestion{Ins: "Does the user explicitly request a refund?"}},
}
```

```go
ctx := context.Background()
router, err := laya.NewRouter()
if err != nil {
	log.Fatal(err)
}
defer router.Close()

// Build every checkpoint up front, so no request pays a model load.
if err := router.Preload(ctx); err != nil {
	log.Panic(err)
}

// English state: routed to laya (ModernBERT-large).
res, err := router.Predict(ctx, state, questions)
if err != nil {
	log.Panic(err)
}
dept, _ := res.Answers.Get("department")
fmt.Printf("Department: %s (confidence %.2f)\n", dept.Choice, dept.Confidence) // billing
fmt.Println("Routing   :", res.Routing.Model)                                  // english

// Hindi state: routed to laya-multilingual (mmBERT-base).
hindi := laya.Obj{{Key: "body", Value: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}}
res, err = router.Predict(ctx, hindi, questions)
if err != nil {
	log.Panic(err)
}
dept, _ = res.Answers.Get("department")
fmt.Printf("Department: %s (confidence %.2f)\n", dept.Choice, dept.Confidence) // billing
fmt.Println("Routing   :", res.Routing.Model)                                  // multilingual
fmt.Println("Repo      :", res.Routing.Repo)                                   // convaiinnovations/laya/multilingual
fmt.Println("Reason    :", res.Routing.Reason)                                 // non-Latin script (devanagari, ...)

// An explicit override when you want a specific checkpoint.
if _, err := router.Predict(ctx, state, questions, laya.ForModel(laya.ModelTypedDecisions)); err != nil {
	log.Panic(err)
}
```

Every result carries the routing decision that explains the choice: `res.Routing.Model`, `.Repo`, `.Reason`, and the script and language `.Detection`.

Inspect a routing decision without running any forward pass. This one needs no weights, so `go test` runs it and checks the output:

```go
d, err := router.Route(laya.Obj{{Key: "body", Value: "Der Kunde wurde zweimal belastet"}}, questions)
if err != nil {
	log.Panic(err)
}
fmt.Println(d.Model)
fmt.Println(d.Reason)
// Output:
// multilingual
// Latin script but language looks like 'de', not English
```

### Why Route: The Evidence

Upstream's shared benchmark (17,416 questions, one T4 GPU, identical questions per model):

| Benchmark / Task                   | English (`laya`) | Multilingual (`laya-multilingual`) | `Router` (Routed) |
| ---------------------------------- | ---------------- | ---------------------------------- | ----------------- |
| MASSIVE intent, English            | **0.783**        | 0.657                              | **0.783**         |
| MASSIVE intent, 13 other languages | 0.306            | **0.451**                          | **0.451**         |
| XNLI, English                      | **0.860**        | 0.843                              | **0.860**         |
| XNLI, 14 other languages           | 0.521            | **0.731**                          | **0.731**         |
| Languages usable (>3x random)      | 23 / 51          | 45 / 51                            | **45 / 51**       |

The English checkpoint collapses on non-Latin scripts (Khmer scores **0.000 accuracy at 0.952 confidence**). Because the model stays confident while being wrong, confidence gating cannot save you. `Router` detects the script before the forward pass, without loading anything.

### Production Preload & Memory

A cold checkpoint load costs seconds (an ORT session takes 1.5–3.2 s to build, measured here); language detection does not load anything. At the default `WithMaxLoaded(1)`, traffic that alternates languages rebuilds a model on _every_ request.

For a server, preload, and size `WithMaxLoaded` to what you keep resident:

```go
ctx := context.Background()
// Keep two checkpoints resident; past that the least recently used is
// evicted. The default is one.
router, err := laya.NewRouter(laya.WithMaxLoaded(2))
if err != nil {
	log.Fatal(err)
}
defer router.Close()

// Preload only the checkpoints you serve. With no names, Preload builds
// all of them.
if err := router.Preload(ctx, laya.ModelEnglish, laya.ModelMultilingual); err != nil {
	log.Panic(err)
}
fmt.Println(router.Loaded()) // [english multilingual]

// Free the memory again.
if err := router.Unload(); err != nil {
	log.Panic(err)
}
```

If your app already opened an agent, attach it instead of loading a second copy:

```go
// Hand the router the agent you already built instead of loading a
// second copy. The router never closes an attached agent.
if err := router.Attach(laya.ModelEnglish, agent); err != nil {
	log.Panic(err)
}
```

| Deployment Mode                    | Per-request cost on CPU                           | Model reloads |
| ---------------------------------- | ------------------------------------------------- | ------------- |
| `NewRouter()` (`WithMaxLoaded(1)`) | a 1.5–3.2 s session load on every language switch | 1 per switch  |
| `Preload`                          | the forward pass only: **0.16–4 s**, see [Speed]  | **none**      |

Resident memory is the constraint on preloading: about 1.4 GB for each ModernBERT-large checkpoint and 0.5 GB for `laya-multilingual`, so all three take about 3.3 GB.

[Speed]: #speed-cpu-measured-here

---

## Single-Model Mode (Direct SDK)

If you only need a single checkpoint for a dedicated pipeline, open it directly:

```go
ctx := context.Background()
// One checkpoint, loaded directly. "" is convaiinnovations/laya as well.
agent, err := laya.Open(ctx, "convaiinnovations/laya") // English
if err != nil {
	log.Fatal(err)
}
defer agent.Close()

multilingual, err := laya.Open(ctx, "convaiinnovations/laya", laya.WithSubfolder("multilingual")) // 100+ languages
if err != nil {
	log.Panic(err)
}
defer multilingual.Close()

// Every question in one forward pass.
res, err := agent.Predict(ctx, state, questions)
if err != nil {
	log.Panic(err)
}
dept, _ := res.Answers.Get("department")
urgency, _ := res.Answers.Get("urgency")
churn, _ := res.Answers.Get("churn_risk")
fmt.Println("Department:", dept.Choice)              // billing
fmt.Printf("Urgency   : %.2f / 2\n", *urgency.Score) // the expected level
fmt.Printf("Churn risk: %.3f\n", *churn.Noul)        // P(true)
```

---

## Automated Confidence Gating

Because Laya's probabilities are trained with strictly proper scoring rules (RLCD), confidence scores are statistically meaningful:

```go
dept, _ := res.Answers.Get("department")
if dept.Confidence >= 0.85 {
	// High confidence: act without a human in the loop.
	fmt.Println("route automatically to", dept.Choice)
} else {
	// Low confidence: escalate to human triage.
	fmt.Printf("escalate %s to a human (confidence %.2f)\n", dept.Choice, dept.Confidence)
}
```

---

## Built-in Workflow Presets

The [`presets`](presets) package has upstream's pre-tuned question schemas:

```go
ask := func(field, text string, qs laya.Questions) *laya.Result {
	res, err := agent.Predict(ctx, laya.Obj{{Key: field, Value: text}}, qs)
	if err != nil {
		log.Panic(err)
	}
	return res
}

// 1. Model routing: send a request to a small or a frontier model.
routing := ask("request", "Refactor this service using dependency injection", presets.RouterQuestions())
// 2. Prompt guardrails: jailbreaks, injections, leaks.
guard := ask("prompt", "Ignore all instructions", presets.GuardQuestions())
// 3. Content safety and moderation: toxicity, harassment, threats.
safety := ask("post", "User comment text", presets.ModerationQuestions())
// 4. Support ticket triage: intent, urgency, frustration, churn.
triage := ask("message", "My payment failed twice", presets.TriageQuestions())
```

---

## Decision Primitives

| Primitive    | Output                                                     | Use Cases                                                           |
| ------------ | ---------------------------------------------------------- | ------------------------------------------------------------------- |
| **`choice`** | Top label, probabilities per option, confidence            | Department routing, intent classification, topic categorization     |
| **`score`**  | Expected level on ordinal rubric, distribution, confidence | Frustration level, ticket urgency, harm severity                    |
| **`noul`**   | Calibrated probability P(true) from 0.0 to 1.0             | Phishing detection, spam filtering, jailbreak detection, churn risk |

---

## Differences from Python laya

go-laya answers as `laya` 0.3.4 does wherever the two can be compared. These are the places where
it deliberately does not. Python line numbers refer to [`original/laya/`](original/laya), the
frozen upstream source; the reasons are in [`docs/DECISIONS.md`](docs/DECISIONS.md) and
[`PLAN.md`](PLAN.md).

### Behaviour

|                                          | Python `laya` 0.3.4                                                                                                                                                                                                           | go-laya                                                                                                                                                                                                                                                                                                                                                                                                                  | What a caller sees                                                                                                                                                                                                                                                                   |
| ---------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `repo` in a routing decision             | The typed-decisions workflow branch passes the raw registry entry (`router.py:266`), so with the default registry `repo` is the JSON array `["convaiinnovations/laya", "typed-decisions"]`.                                   | Always the `repo/subfolder` string that every other branch writes (`ModelSpec.String`, PLAN Task 3.1.4).                                                                                                                                                                                                                                                                                                                 | `RouteDecision.Repo` is a `string`, here `"convaiinnovations/laya/typed-decisions"`.                                                                                                                                                                                                 |
| Question instructions                    | Anything. A non-string is `json.dumps`'d with `ensure_ascii=True` (`agent.py:235-237`).                                                                                                                                       | `string` only.                                                                                                                                                                                                                                                                                                                                                                                                           | The `Ins string` field of `ChoiceQuestion`, `ScoreQuestion` and `NoulQuestion`. A non-string does not compile.                                                                                                                                                                       |
| Hub revision                             | `snapshot_download` with no `revision`, so the Hub's `main` (`agent.py:124-128`).                                                                                                                                             | The bundle repo `convaiinnovations/laya`, which holds all three default checkpoints, is pinned to the revision every golden vector describes ([D17][later]). The pin goes by repo name, so it also holds for `StandaloneModels()`' English entry, which is that same repo. Any other repo, including the standalone `convaiinnovations/laya-multilingual` and `convaiinnovations/laya-typed-decisions`, loads at `main`. | `WithRevision` (for `Open`) or `WithRouterRevision` (for `NewRouter`) is the opt-in: `"main"`, a tag or a commit sha, used for every Hub repo, the bundle repo included ([D29][later]). The pin is the unexported `pinnedRevision` in `loader.go`; see [Verified against][verified]. |
| What is downloaded, and what runs        | `<subfolder>/*`, or the whole repo for the root checkpoint (`agent.py:124-128`). The forward pass runs `model.safetensors` on PyTorch.                                                                                        | Only `rl_agent_config.json` and `tokenizer/*` ([D24][later]). The forward pass runs a local ONNX export, `laya-<name>.onnx`, from `WithGraph`, the Router's `WithONNXDir`, `$LAYA_ONNX_DIR` or `onnx/` under the laya cache (see [Installation](#installation)). With `WithRuntime(RuntimeNative)`, also `model.safetensors` and `encoder/config.json`, run in pure Go ([D30][later]).                                   | `ErrNoGraph` when there is no export. Its message names the path, and for the default lookup also the `scripts/export_onnx.py` command.                                                                                                                                              |
| Platforms                                | Wherever PyTorch runs.                                                                                                                                                                                                        | The ONNX backend builds on linux (amd64, arm64, loong64) and on darwin and netbsd (amd64, arm64) ([D21][later]). Everywhere else a stub takes its place.                                                                                                                                                                                                                                                                 | `Open` fails with `ErrUnsupportedPlatform` (`onnx backend: not supported on this platform`).                                                                                                                                                                                         |
| `cuda`                                   | Runs on CUDA when it is available. An explicit `cuda` without it prints a warning and runs on the CPU (`agent.py:158-160`).                                                                                                   | No GPU execution provider: the ONNX Runtime binding cannot enable CUDA ([D22][later], PLAN Task B.3). An explicit `cuda` or `cuda:N` runs on the CPU, and `auto` skips CUDA without a word.                                                                                                                                                                                                                              | `WithDevice("cuda")` or `WithRouterDevice("cuda")` succeeds and logs one `slog.Default()` warning, `onnx backend: requested device unavailable, running on CPU`.                                                                                                                     |
| `mps`                                    | A device. `auto` prefers it after CUDA (`agent.py:161-170`).                                                                                                                                                                  | Not a device. `coreml`, ONNX Runtime's CoreML provider, takes its place in `auto`'s order.                                                                                                                                                                                                                                                                                                                               | `WithDevice("mps")` fails `Open` with `ErrUnknownDevice` (`onnx backend: unknown device "mps"`).                                                                                                                                                                                     |
| Config limits and temperatures           | Read unchecked: `max_len` and `head_max_len` with `cfg.get` (`agent.py:256-257`), and `temperature` indexed by question type (`agent.py:304`), so a short list fails only once that type is asked and a long one is accepted. | A config whose `max_len` or `head_max_len` is not a positive integer, or whose `temperature` list does not have exactly 3 elements, is rejected when the checkpoint loads.                                                                                                                                                                                                                                               | `ErrIncompatibleCheckpoint` (not `ErrInvalidLimits`), with the config path and key.                                                                                                                                                                                                  |
| `SetLimits` and `WithLimits`             | `agent.cfg["max_len"] = ...` takes any value.                                                                                                                                                                                 | A value ≤ 0 is rejected, and `SetLimits` leaves the limits as they were (PLAN Task 7.7.5).                                                                                                                                                                                                                                                                                                                               | `ErrInvalidLimits`.                                                                                                                                                                                                                                                                  |
| A choice question with no options        | Passes the marker check (`agent.py:262`) and fails later: in the forward pass when no question in the batch has two options, otherwise at the softmax over an empty row (`agent.py:305-306`).                                 | Rejected before that question is tokenised, so the forward pass never runs.                                                                                                                                                                                                                                                                                                                                              | `ErrNoOptions`, wrapped with the question id.                                                                                                                                                                                                                                        |
| NFC on `laya` and `laya-typed-decisions` | `tokenizers` 0.23.2 normalises with Rust tables that predate Unicode 11–15, so 108 combining marks assigned since (U+07FD, U+0898–089F, U+1AC0–1ACE and others) count as starters and are never reordered.                    | Unicode-conformant NFC (`golang.org/x/text`), which agrees with CPython's `unicodedata` (PLAN Task B.2).                                                                                                                                                                                                                                                                                                                 | Different token ids only for text that puts one of those marks in a run with other combining marks: 85 of the 157 281 lines in the tokenizer differential, none of them natural text. `laya-multilingual` has no NFC step and is unaffected.                                         |
| Cache                                    | huggingface_hub's cache (`HF_HUB_CACHE`).                                                                                                                                                                                     | Its own: `WithCacheDir` (or `WithRouterCacheDir`), else `$LAYA_CACHE`, else the user cache directory + `/laya`, laid out as `owner/name/<commit>/path`. The verified ORT library and the default `onnx/` directory live there too.                                                                                                                                                                                       | A huggingface_hub cache is not read.                                                                                                                                                                                                                                                 |

### Renamed and dropped exports

These names in upstream's `__all__` (`__init__.py:26-51`) are renamed, internal or dropped in
go-laya:

| Python                                                      | go-laya                                                                                                                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `load` (`agent.py:351`)                                     | `Open`.                                                                                                                                          |
| `RLAgent` (`agent.py:348`, an alias of `Agent`)             | No alias: the type is `Agent`. The Router's interface is `Predictor` ([D28][later]).                                                             |
| `detect_language` (`lang.analyse`, `lang.py:159`)           | `lang.Analyse`, in the public [`lang`](lang) package.                                                                                            |
| `QTYPES`, `QTYPE_NAMES` (`common.py:11-12`)                 | The type `QType`, with the constants `Choice`, `Score` and `Noul` (0, 1, 2) and `QType.String`. There is no map from a name to a `QType`.        |
| `render_options` (`common.py:33`)                           | The `RenderOptions()` method of every question type (`Question.RenderOptions`). The free function is internal (`internal/prompt.RenderOptions`). |
| `confidence_from_probs` (`common.py:200`)                   | Internal (`internal/calib.Confidence`). Its value reaches callers as `Answer.Confidence`.                                                        |
| `ece_score` (`common.py:187`)                               | Internal (`internal/calib.ECE`), checked against `testdata/ece.jsonl`; nothing public calls it.                                                  |
| `proper_reward`, `td_lambda_targets` (`common.py:140, 169`) | Not ported. They are RLCD training math, and go-laya is inference only ([D3][core]).                                                             |

### Verified against

The port is checked against these versions. Each value is copied from the file next to it.

| What                                                    | Version                                                 | Defined in                                                                                                                                |
| ------------------------------------------------------- | ------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| Checkpoints, `convaiinnovations/laya` (all three)       | Hub revision `1c5edc17a7acd8701df6fc341c0d179f1c62c982` | `pinnedRevision` in [`loader.go`](loader.go); `checkpoint_sha` in the header of every `testdata/*.jsonl`                                  |
| `tokenizers`                                            | 0.23.2                                                  | [`scripts/requirements-ref.txt`](scripts/requirements-ref.txt); `targetTokenizersVersion` in [`provenance_test.go`](provenance_test.go)   |
| `transformers`                                          | 5.17.0                                                  | [`scripts/requirements-ref.txt`](scripts/requirements-ref.txt); `targetTransformersVersion` in [`provenance_test.go`](provenance_test.go) |
| PyTorch, for the reference logits                       | 2.14.0+cpu, float32, CPU, one thread, `sdpa`            | [`scripts/requirements-ref.txt`](scripts/requirements-ref.txt); the `compute` block of every golden header                                |
| ONNX Runtime, which go-laya runs on                     | 1.23.0                                                  | `Version` in [`internal/ortlib/ortlib.go`](internal/ortlib/ortlib.go)                                                                     |
| onnxruntime, which `scripts/export_onnx.py` checks with | 1.30.0                                                  | [`scripts/requirements-ref.txt`](scripts/requirements-ref.txt)                                                                            |

`TestGoldenProvenance` fails if any golden header records another `tokenizers` or `transformers`
version or another `compute` block. It also fails if a header's `checkpoint_sha` is not
`pinnedRevision`.

[core]: docs/DECISIONS.md#core-decisions
[later]: docs/DECISIONS.md#later-decisions
[verified]: #verified-against

---

## Benchmarks

**Full report: [`BENCHMARKS.md`](BENCHMARKS.md)**: the CPU latency measured here, and upstream's results for every language and theme.

The speed figures are this port's own. **Every accuracy and calibration figure is upstream's**,
measured with the Python package on the same checkpoints and reproduced here for reference; this
port did not re-run them.

### Speed, CPU, measured here

One question, batch 1, 4 options, 8 threads, ONNX Runtime 1.23.0, fp32, on a 12th Gen Intel Core
i7-1255U laptop. Take these as a floor, not as what a server does, and as good to about ±20%
([method](BENCHMARKS.md#speed--cpu-measured-here)):

| tokens | `laya` (english) | `laya-multilingual` | `laya-typed-decisions` |
| ------ | ---------------- | ------------------- | ---------------------- |
| 128    | 374 ms           | **160 ms**          | 406 ms                 |
| 256    | 914 ms           | **355 ms**          | 862 ms                 |
| 512    | 1691 ms          | **645 ms**          | 1856 ms                |
| 1024   | —                | **1484 ms**         | 3962 ms                |

**Batching does not amortise on CPU.** One question at 512 tokens already saturates the cores, so
per-question cost stays flat:

| questions × tokens | `laya` (english)  | `laya-multilingual` | `laya-typed-decisions` |
| ------------------ | ----------------- | ------------------- | ---------------------- |
| 1 × 128            | 374 ms (374/q)    | 160 ms (160/q)      | 406 ms (406/q)         |
| 8 × 128            | 2975 ms (372/q)   | 1129 ms (141/q)     | 3289 ms (411/q)        |
| 1 × 512            | 1691 ms (1691/q)  | 645 ms (645/q)      | 1856 ms (1856/q)       |
| 8 × 512            | 14846 ms (1856/q) | 4988 ms (624/q)     | 15689 ms (1961/q)      |

So, plainly: sub-second per question is reachable only on `laya-multilingual` at short sequences,
and on CPU this is a batch and background workload, not an interactive one. Anything with a human
waiting on it wants a GPU, and this port has no GPU execution provider yet: `WithDevice("cuda")`
falls back to CPU with a warning. GPU latency is not measured here, and stays unmeasured until CUDA
support (PLAN Task B.3) lands.

### Laya (with routing) vs Jev

Upstream's comparison: every Laya figure is what upstream's `Router().predict(...)` returned — the
checkpoint the router selects for that input, not a hand-picked best of three. Jev figures are
**third-party published, never measured here** (no TypeSafe API access), so sample sizes and prompts
differ.

|                                                 | Jev 1.13.0               | Laya (routed)      |                              |
| ----------------------------------------------- | ------------------------ | ------------------ | ---------------------------- |
| typed-decisions, 2,000 decisions                | 0.727                    | **0.766**          | +0.039                       |
| AG News, 4 labels                               | 0.910                    | **0.950**          | +0.040                       |
| DAIR Emotion, 6 labels                          | 0.480                    | **0.595**          | +0.115                       |
| Banking77 (72 vs 77 labels)                     | **0.870**                | 0.425              | Jev leads on >20 options     |
| ECE _(lower better)_                            | 0.246                    | **0.081**          | 3× better (post-temperature) |
| p50 latency, 1 question — T4 _(upstream)_       | 236–276 ms               | **32.8 ms**        | not measured here            |
| p50 latency, 1 question — CPU _(here, 512 tok)_ | —                        | 0.63–1.9 s         | see [Speed]                  |
| Languages usable                                | _no published benchmark_ | **45 of 51**       | —                            |
| Weights                                         | closed API               | **Apache 2.0**     | —                            |
| Cost                                            | $0.042 / 1M tokens       | **$0 self-hosted** | —                            |

On DAIR Emotion, Jev assigned **zero probability to the true label on 16% of examples** — a hard
failure for anything branching on confidence.

#### Where Jev leads

- **High-cardinality label spaces (>20 options at default settings):** On Banking77, Jev scores 0.870 (on 72 labels) while Laya scores 0.425 (on 77 labels at default 256-token head budget). This is an architectural token-budget constraint: options share a fixed `head_max_len` budget (192 tokens on English, 256 on multilingual), so 77 options receive only ~3 to 4 tokens per label, causing text to become indistinguishable. Jev supports up to 255 options out-of-the-box. While `laya-multilingual` supports 1,024 context (and up to 8,192 in the encoder) and you can raise the budget at runtime with `agent.SetLimits(1024, 512)`, Jev is currently better suited for 50+ options in a single prompt without tuning.
- **Soft distribution matching:** On typed-decisions, while Laya achieves higher argmax accuracy (0.766 vs 0.727), Jev achieves higher soft accuracy (0.580 vs 0.471) against the teacher's full probability distributions.
- **Out-of-the-box raw calibration:** Before temperature scaling, the base checkpoint has higher raw ECE (0.213 vs 0.144). Laya achieves its 0.081 ECE after domain temperature fitting.

### typed-decisions, measured on all three checkpoints

Upstream's run: 400 cases, 2,000 decisions, four workflows.

| model                            | accuracy  | soft acc | Brier     | ECE     | score MAE |
| -------------------------------- | --------- | -------- | --------- | ------- | --------- |
| **`laya-typed-decisions`**       | **0.766** | 0.471    | **0.062** | 0.213   | **0.242** |
| `laya`                           | 0.362     | 0.332    | 0.316     | 0.175   | 0.694     |
| `laya-multilingual`              | 0.342     | 0.326    | 0.439     | 0.285   | 0.687     |
| _Jev 1.13.0 (published)_         | _0.727_   | _0.580_  | _0.148_   | _0.144_ | _0.391_   |
| _teacher self-agreement ceiling_ | _0.735_   |          |           |         |           |
| _per-question majority class_    | _0.461_   |          |           |         |           |
| _random guess_                   | _0.318_   |          |           |         |           |

The fine-tuned checkpoint beats Jev by 3.9 points and clears the teacher ceiling, with 2.4x
better Brier and 1.6x better score MAE. It wins on all four workflows: invoice processing
0.804, security incidents 0.766, customer service 0.764, agent-trace observability 0.730.
By primitive: `noul` 0.857, `choice` 0.733, `score` 0.723.

Two places it still trails Jev: **soft accuracy** (0.471 vs 0.580 — its argmax is better but
its distributions match the teacher less well) and **ECE** (0.213 vs 0.144), which temperature
fitting addresses.

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461).
All of the capability on this benchmark comes from fine-tuning.

### Multilingual (51 languages, MASSIVE intent, 20 options, random = 0.050)

<p align="center">
  <img src="assets/laya_benchmark.png" alt="Per-language accuracy for both checkpoints across 51 languages" width="100%" />
</p>

|                          | `laya`    | `laya-multilingual` |
| ------------------------ | --------- | ------------------- |
| English                  | **0.783** | 0.657               |
| 13 other languages       | 0.306     | **0.451**           |
| XNLI, English            | **0.860** | 0.843               |
| XNLI, 14 other languages | 0.521     | **0.731**           |

Across all 51 languages the English checkpoint macro-averages **0.227** with macro ECE
**0.733**, and only 23 of 51 languages clear 3x random. Khmer scores **0.000 at 95.2%
confidence**. This is why [`Router`](#quickstart-route-mode-recommended) exists: the
model's own confidence gives no warning, so the routing decision has to be made before the
forward pass.

### English tasks

| task              | `laya`    | `laya-multilingual` | note            |
| ----------------- | --------- | ------------------- | --------------- |
| AG News           | **0.947** | 0.937               | in training mix |
| BoolQ             | **0.830** | 0.787               | in training mix |
| DAIR Emotion      | **0.573** | 0.513               | held out        |
| prompt-injections | **0.698** | 0.578               | held out, n=116 |
| SST-5 (ordinal)   | 0.372     | 0.282               | held out        |

### Calibration

Both checkpoints are over-confident as shipped. Refitting one temperature per (question type,
option count) on held-out data moves mean ECE **0.466 -> 0.081** (`laya`) and
**0.314 -> 0.106** (`laya-multilingual`). `laya-multilingual` ships with no fitted
temperatures at all, so fit them before relying on its probabilities.

### Honest limits

- **The base checkpoints are near chance on typed-decisions zero-shot** -- 0.362 and 0.352
  against a 0.318 random baseline and a 0.461 majority-class baseline. The 0.766 figure comes
  from the checkpoint fine-tuned on that benchmark's own training split. Laya is a fast base to
  specialise, not a zero-shot decision engine.
- **On CPU, latency is seconds, not milliseconds**, for both ModernBERT-large checkpoints at
  their default lengths (see [Speed]).
- **High-cardinality choice questions and token budgets:** Sequences split into an option prompt budget (`head_max_len`) and the remaining document/state budget (`max_len - head_max_len`):

  - `laya` (English) defaults to 512 context (`head_max_len = 192`, ~320 tokens for state).
  - `laya-multilingual` and `laya-typed-decisions` default to 1,024 context (`head_max_len = 256`, ~768 tokens for state; mmBERT-base encoder supports up to 8,192 with RoPE).
    At default settings, a 77-option question like Banking77 allocates only (256 − 16) / 77 ≈ 3 tokens per label, which causes accuracy to fall off sharply (0.425 vs Jev's 0.870). If evaluating 50+ options in a single question:
  1. Raise both limits so every option has enough tokens to remain distinct (or `max_len` up to 2048 / 4096 / 8192):

     ```go
     // Many options share the head_max_len budget. Raise it, and max_len with
     // it, so each of 50+ options keeps enough tokens to stay distinct.
     if err := agent.SetLimits(1024, 512); err != nil {
     	log.Panic(err)
     }
     fmt.Println(agent.MaxLen(), agent.HeadMaxLen()) // 1024 512
     ```

  2. Or split large option sets into a two-step coarse-to-fine hierarchical choice.

- Ordinal `score` questions are the weakest primitive (SST-5 0.372).
- `laya` collapses outside English; `laya-multilingual` is weaker on English. Route, or pick
  deliberately.

---

## Live Demo & Resources

- **Upstream (Python):** [NandhaKishorM/laya](https://github.com/NandhaKishorM/laya)
- **Hugging Face Model:** [convaiinnovations/laya](https://huggingface.co/convaiinnovations/laya)
- **Interactive Web Demo:** [convaiinnovations/laya-demo](https://huggingface.co/spaces/convaiinnovations/laya-demo)
- **Engineering Writeup:** [Read the full story on Dev.to](https://dev.to/nandakishor_m_6cc0adfde9f/i-built-non-autoregressive-decision-models-a-year-ago-then-a-frontier-lab-called-it-a-18me)

---

## Fine-Tuning

Fine-tuning is Python and lives upstream: the
[`laya_finetune_typed_decisions_2xT4_kaggle.ipynb`](https://github.com/NandhaKishorM/laya/blob/main/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb)
notebook builds the dataset, trains with RLCD, fits calibration temperatures and pushes the result
to the Hub. A fine-tuned checkpoint runs here like the published ones: export it with
`scripts/export_onnx.py`, then open its directory with `laya.Open` and point `WithGraph` at the
export.

Fine-tuning is where most of the value is: the base checkpoints score near chance on
typed-decisions zero-shot, while the fine-tuned one reaches **0.766** on the same 2,000 decisions.

---

## Support the Project

Laya is independent research by its upstream author. If it helps your research or products,
consider supporting them:

<p align="left">
  <a href="https://www.buymeacoffee.com/nandakishorm" target="_blank">
    <img src="https://img.buymeacoffee.com/button-api/?text=Buy%20me%20a%20coffee&emoji=&slug=nandakishorm&button_colour=FFDD00&font_colour=000000&font_family=Cookie&outline_colour=000000&coffee_colour=ffffff" alt="Buy Me A Coffee" />
  </a>
</p>

---

## License

Apache 2.0. Laya is developed by Convai Innovations; this Go port is a derivative work, and
[`NOTICE`](NOTICE) carries the attribution.
