# Laya benchmarks

Two kinds of numbers live here, and they have different standing.

- **Latency and load costs are measured here**, on CPU, by this port, with their hardware, method
  and raw transcripts: [Speed — CPU, measured here](#speed--cpu-measured-here). So is the int8
  versus fp32 calibration comparison:
  [int8 dynamic quantization](#int8-dynamic-quantization-planmd-b5--measured-not-shipped).
- **Accuracy, calibration and option-order figures are upstream's**, copied from upstream's
  `BENCHMARKS.md` at 0.3.4 and not re-measured. That is commit `d113dca`, the one `NOTICE` pins
  (`git show d113dca:BENCHMARKS.md`). Upstream ran them with the Python package and never published
  the raw results, so they cannot be reproduced from either repository. In those runs every
  checkpoint answered **byte-identical questions** (fixed seed).

Jev figures are **third-party published, measured neither here nor by upstream** (no TypeSafe API
access), so sample sizes and prompts differ; treat them as indicative.

---

## Headline

|                                                             | Laya       | Jev (published) |
| ----------------------------------------------------------- | ---------- | --------------- |
| typed-decisions (2,000 decisions) _(upstream)_              | **0.766**  | 0.727           |
| AG News (4 labels) _(upstream)_                             | **0.953**  | 0.910           |
| DAIR Emotion (6 labels) _(upstream)_                        | **0.600**  | 0.480           |
| ECE after temperature fitting _(upstream)_                  | **0.081**  | 0.246           |
| p50 latency, 1 question — CPU _(measured here, 512 tokens)_ | 0.63–1.9 s | —               |

---

## Languages

### All 51 MASSIVE languages — intent, 20 options (random = 0.050)

|                              | laya    | laya-multilingual |
| ---------------------------- | ------- | ----------------- |
| macro accuracy               | 0.2269  | **0.3661**        |
| macro ECE _(lower better)_   | 0.7331  | **0.3869**        |
| languages clearing 3× random | 23 / 51 | **45 / 51**       |

<details><summary><b>Per language (51)</b> — sorted by how much routing gains</summary>

| lang    | laya  | laya-multilingual | Δ      | laya ECE | multilingual ECE |
| ------- | ----- | ----------------- | ------ | -------- | ---------------- |
| `th`    | 0.080 | 0.480             | +0.400 | 0.881    | 0.336            |
| `ko`    | 0.110 | 0.450             | +0.340 | 0.850    | 0.329            |
| `he`    | 0.060 | 0.400             | +0.340 | 0.911    | 0.350            |
| `ur`    | 0.070 | 0.400             | +0.330 | 0.883    | 0.311            |
| `hi`    | 0.100 | 0.430             | +0.330 | 0.850    | 0.321            |
| `ar`    | 0.110 | 0.400             | +0.290 | 0.800    | 0.341            |
| `pl`    | 0.240 | 0.510             | +0.270 | 0.713    | 0.350            |
| `el`    | 0.130 | 0.380             | +0.250 | 0.839    | 0.383            |
| `fa`    | 0.140 | 0.390             | +0.250 | 0.820    | 0.399            |
| `ru`    | 0.310 | 0.540             | +0.230 | 0.668    | 0.316            |
| `tr`    | 0.140 | 0.370             | +0.230 | 0.788    | 0.417            |
| `lv`    | 0.100 | 0.320             | +0.220 | 0.847    | 0.480            |
| `bn`    | 0.080 | 0.290             | +0.210 | 0.865    | 0.408            |
| `nb`    | 0.330 | 0.530             | +0.200 | 0.648    | 0.327            |
| `vi`    | 0.060 | 0.260             | +0.200 | 0.891    | 0.521            |
| `az`    | 0.100 | 0.300             | +0.200 | 0.825    | 0.368            |
| `hu`    | 0.090 | 0.290             | +0.200 | 0.857    | 0.422            |
| `is`    | 0.110 | 0.300             | +0.190 | 0.835    | 0.469            |
| `sv`    | 0.380 | 0.570             | +0.190 | 0.596    | 0.276            |
| `km`    | 0.000 | 0.180             | +0.180 | 0.952    | 0.412            |
| `ml`    | 0.070 | 0.240             | +0.170 | 0.857    | 0.414            |
| `it`    | 0.340 | 0.500             | +0.160 | 0.647    | 0.302            |
| `fi`    | 0.130 | 0.290             | +0.160 | 0.849    | 0.436            |
| `ms`    | 0.270 | 0.430             | +0.160 | 0.688    | 0.392            |
| `da`    | 0.350 | 0.500             | +0.150 | 0.626    | 0.263            |
| `id`    | 0.360 | 0.510             | +0.150 | 0.613    | 0.305            |
| `te`    | 0.090 | 0.220             | +0.130 | 0.858    | 0.370            |
| `sl`    | 0.200 | 0.330             | +0.130 | 0.756    | 0.433            |
| `jv`    | 0.160 | 0.270             | +0.110 | 0.803    | 0.506            |
| `ta`    | 0.120 | 0.230             | +0.110 | 0.822    | 0.397            |
| `ja`    | 0.530 | 0.640             | +0.110 | 0.460    | 0.228            |
| `hy`    | 0.050 | 0.150             | +0.100 | 0.835    | 0.506            |
| `zh-TW` | 0.460 | 0.540             | +0.080 | 0.520    | 0.327            |
| `de`    | 0.420 | 0.500             | +0.080 | 0.558    | 0.301            |
| `tl`    | 0.290 | 0.360             | +0.070 | 0.676    | 0.374            |
| `nl`    | 0.390 | 0.450             | +0.060 | 0.591    | 0.378            |
| `af`    | 0.290 | 0.350             | +0.060 | 0.687    | 0.484            |
| `my`    | 0.060 | 0.120             | +0.060 | 0.861    | 0.455            |
| `sq`    | 0.210 | 0.260             | +0.050 | 0.755    | 0.476            |
| `sw`    | 0.130 | 0.180             | +0.050 | 0.828    | 0.549            |
| `cy`    | 0.120 | 0.160             | +0.040 | 0.841    | 0.591            |
| `kn`    | 0.110 | 0.150             | +0.040 | 0.842    | 0.437            |
| `es`    | 0.510 | 0.530             | +0.020 | 0.480    | 0.275            |
| `ka`    | 0.090 | 0.110             | +0.020 | 0.845    | 0.528            |
| `ro`    | 0.330 | 0.350             | +0.020 | 0.658    | 0.404            |
| `zh-CN` | 0.620 | 0.630             | +0.010 | 0.376    | 0.212            |
| `am`    | 0.120 | 0.110             | -0.010 | 0.825    | 0.463            |
| `pt`    | 0.470 | 0.450             | -0.020 | 0.512    | 0.342            |
| `mn`    | 0.130 | 0.100             | -0.030 | 0.837    | 0.558            |
| `fr`    | 0.590 | 0.540             | -0.050 | 0.388    | 0.277            |
| `en`    | 0.820 | 0.680             | -0.140 | 0.179    | 0.209            |

</details>

### English vs the rest

| task                               | laya      | laya-multilingual |
| ---------------------------------- | --------- | ----------------- |
| MASSIVE intent — English           | **0.783** | 0.657             |
| MASSIVE intent — other languages   | 0.306     | **0.451**         |
| MASSIVE scenario — English         | **0.603** | 0.560             |
| MASSIVE scenario — other languages | 0.281     | **0.439**         |
| XNLI — English                     | **0.860** | 0.843             |
| XNLI — other languages             | 0.521     | **0.731**         |

The English checkpoint does not degrade gracefully outside English — it collapses, and stays confident doing so. Khmer: **0.000 accuracy at 0.952 confidence**. Its mean confidence never drops below 0.885 at any accuracy level, so confidence gating cannot catch it — which is why routing happens _before_ the forward pass.

---

## Themes — the application workflows

Each is real labelled data, 400 cases, all three checkpoints. _held out_ means the source was **not** in Laya's training mix.

| theme                         | laya      | laya-multilingual | laya-typed-decisions | data         |
| ----------------------------- | --------- | ----------------- | -------------------- | ------------ |
| Email spam                    | **0.993** | 0.993             | 0.958                | in training  |
| Phishing                      | 0.980     | **0.993**         | 0.940                | in training  |
| LLM guardrails (jailbreak)    | 0.708     | 0.755             | **0.762**            | **held out** |
| Moderation (toxicity)         | **0.530** | 0.525             | 0.530                | **held out** |
| RAG passage relevance         | 0.625     | **0.657**         | 0.625                | in training  |
| Support triage (10-way queue) | 0.502     | **0.522**         | 0.505                | in training  |
| Model routing (domain)        | 0.639     | 0.123             | **0.659**            | held out     |

**Where it is strong:** email spam 0.993 and phishing 0.993, both with ECE around 0.01 — production-grade, though both were in the training mix.

**Where it is weak:** moderation on held-out toxic-chat is 0.530 with macro-F1 0.400 — barely above chance on a balanced split. The demo Space has a Moderation tab; hand-picked examples work, real traffic does not. Guardrails at 0.708–0.762 is the honest jailbreak-detection number, consistent across two unrelated datasets (deepset prompt-injections measured 0.698 separately).

### On the public datasets where Jev numbers exist

| dataset                 | laya  | laya-multilingual | laya-typed-decisions | Jev (published) |
| ----------------------- | ----- | ----------------- | -------------------- | --------------- |
| AG News (4 labels)      | 0.950 | 0.930             | **0.953**            | 0.910           |
| DAIR Emotion (6 labels) | 0.595 | 0.530             | **0.600**            | 0.480           |
| banking77 (77 labels)   | 0.425 | 0.425             | **0.492**            | 0.870           |

banking77 is the one clear loss, and it is architectural: a choice question's options share a fixed `head_max_len` budget, so 77 labels get roughly 4 tokens each and stop being distinguishable. Both checkpoints score **exactly 0.425**, which is what you would expect from a budget ceiling rather than a capability gap. Keep choice questions under ~20 options.

---

## typed-decisions — 400 cases, 2,000 decisions

| model                    | accuracy  | soft acc | Brier   | ECE     | score MAE |
| ------------------------ | --------- | -------- | ------- | ------- | --------- |
| `laya-typed-decisions`   | **0.766** | 0.471    | 0.061   | 0.213   | 0.242     |
| `laya`                   | 0.361     | 0.332    | 0.316   | 0.175   | 0.694     |
| `laya-multilingual`      | 0.342     | 0.326    | 0.439   | 0.285   | 0.687     |
| _Jev 1.13.0 (published)_ | _0.727_   | _0.580_  | _0.148_ | _0.144_ | _0.391_   |
| _teacher ceiling_        | _0.735_   | _—_      | _—_     | _—_     | _—_       |
| _majority class_         | _0.461_   | _—_      | _—_     | _—_     | _—_       |
| _random guess_           | _0.318_   | _—_      | _—_     | _—_     | _—_       |

| workflow                  | laya-typed-decisions |
| ------------------------- | -------------------- |
| agent trace observability | 0.730                |
| customer service          | 0.764                |
| invoice processing        | 0.804                |
| security incidents        | 0.766                |

**The base checkpoints sit below the majority-class baseline** (0.362 and 0.342 against 0.461). All of the capability on this benchmark comes from fine-tuning.

---

## Speed — CPU, measured here

These numbers are ours, and they are the ones that matter for a Go port whose whole premise (D2) is a
dependency-free binary you can drop on a server.

**GPU latency is not measured here**, and stays unmeasured until CUDA support (PLAN.md Backlog B.3)
lands.

**Hardware.** 12th Gen Intel Core i7-1255U — 2 P-cores + 8 E-cores, 12 hardware threads, 15 W
nominal, `powersave` governor, 31 GB RAM, Linux 6.8. This is a laptop, and a thermally limited one:
treat every number here as the **floor**, not as what a server does.

**Stack.** ONNX Runtime 1.23.0 via `github.com/shota3506/onnxruntime-purego`, `CPUExecutionProvider`,
fp32, `CGO_ENABLED=0`, Go 1.26. The graphs are `scripts/export_onnx.py --all --dynamo` output
(PLAN.md S1). Reproduce with `just bench-onnx`.

**Method.** Three full sweeps, five timed iterations per cell after one discarded warm-up, p50 and
p90 recorded per cell. The tables report the **minimum p50 across the three sweeps** — the
least-throttled estimate, and the one that is hardest on the conclusion drawn below. That is not a
formality: run-to-run variation reached **2×** on the worst cell, so no single figure here is good to
better than about ±20%.

**Raw transcripts.** The `go test -bench` output behind these tables is checked in under
`docs/benchmarks/raw/`, so the numbers are reproducible from the repository rather than from one
laptop's `build/` directory (PLAN.md §8 forbids inheriting upstream's unpublished result files).
`bench-onnx-sweep-a.txt` and `bench-onnx-sweep-b.txt` are two full sweeps with the shape axis at
8 threads; `bench-onnx-threads12.txt` is an earlier sweep with the shape axis at 12 threads, which is
where the "all hardware threads is the wrong setting" finding first showed. The third 8-thread
sweep's transcript was overwritten by the recipe's default output path before it was copied, so only
its p50s survive, in the tables.

### One question — batch 1, 4 options, 8 threads

| tokens | `laya` (english) | `laya-multilingual` | `laya-typed-decisions` |
| ------ | ---------------- | ------------------- | ---------------------- |
| 128    | 374 ms           | **160 ms**          | 406 ms                 |
| 256    | 914 ms           | **355 ms**          | 862 ms                 |
| 512    | 1691 ms          | **645 ms**          | 1856 ms                |
| 1024   | —                | **1484 ms**         | 3962 ms                |

512 is `agent.py:256`'s default `max_len`, so the 512 row is what a caller who does not tune anything
will see. `laya` has no 1024 row because 512 is its trained context.

### Thread scaling — batch 1, 512 tokens, 4 options

| `IntraOpNumThreads` | `laya`      | `laya-multilingual` | `laya-typed-decisions` |
| ------------------- | ----------- | ------------------- | ---------------------- |
| 1                   | 3660 ms     | 1332 ms             | 3730 ms                |
| 2                   | 2052 ms     | 813 ms              | 2226 ms                |
| 4                   | 1871 ms     | 706 ms              | 1937 ms                |
| 6                   | 1866 ms     | 681 ms              | 2030 ms                |
| 8                   | **1691 ms** | **631 ms**          | **1720 ms**            |
| 12                  | 2269 ms     | 845 ms              | 2130 ms                |

Two things to take from this. **All twelve threads is the wrong setting** — every checkpoint peaks at
8 and then loses 12–26%, because only 2 of the 10 cores here have SMT siblings and ORT contends for
them. And **1 → 8 threads buys 2.1–2.2×, not 8×**: this is memory-bandwidth-bound long before it runs
out of cores, so throwing a bigger core count at it will not close the gap to a GPU.

### Batching does not amortise on CPU

| shape (questions × tokens) | `laya`            | `laya-multilingual` | `laya-typed-decisions` |
| -------------------------- | ----------------- | ------------------- | ---------------------- |
| 1 × 128                    | 374 ms (374/q)    | 160 ms (160/q)      | 406 ms (406/q)         |
| 8 × 128                    | 2975 ms (372/q)   | 1129 ms (141/q)     | 3289 ms (411/q)        |
| 1 × 512                    | 1691 ms (1691/q)  | 645 ms (645/q)      | 1856 ms (1856/q)       |
| 8 × 512                    | 14846 ms (1856/q) | 4988 ms (624/q)     | 15689 ms (1961/q)      |

Batching helps when one question leaves the hardware underused. **On CPU it does not**: per-question
cost is flat to within noise, because one question at 512 tokens already saturates the cores. Batch
for throughput accounting if you like; do not batch expecting latency per answer to improve.

The option count is a non-factor: at `k=20` instead of `k=4`, 512 tokens, the three checkpoints
measured 1718 / 583 / 1649 ms — inside the run-to-run noise of the `k=4` cells. The encoder runs over
tokens; the head runs over options, and the head is small.

### Loading a checkpoint

| checkpoint             | session load | resident after load |
| ---------------------- | ------------ | ------------------- |
| `laya`                 | 2.1 s        | 1423 MiB            |
| `laya-multilingual`    | 1.5 s        | 488 MiB             |
| `laya-typed-decisions` | 3.2 s        | 1423 MiB            |

Upstream measures a "7.4 s median reload on CPU" and warns that `max_loaded=1` rebuilds a model on
every language switch. Building an ORT session from an already-downloaded ONNX file is cheaper than
that — but at 1.4 GB resident apiece, holding all three costs about 3.3 GB, which is the real
constraint on raising `max_loaded`.

### Loading the tokenizer

`tokenizer.Open` alone, which every checkpoint load pays before the session is built. One fresh
process per run at `-test.benchtime=1x`, plus `5x` runs: five before, ten after across two sessions;
go1.26.8, same laptop. Reproduce with `go test -c` and
`-test.bench 'BenchmarkOpen/<checkpoint>' -test.benchmem`.

| checkpoint     | before (`encoding/json`)             | after (streaming vocab and merges)  |
| -------------- | ------------------------------------ | ----------------------------------- |
| `english`      | 57 ms · 19.6 MB · 187 k allocs       | 34–63 ms · 11.6 MB · 57 k allocs    |
| `multilingual` | 0.51–0.53 s · 207 MB · 1.54 M allocs | 0.42–0.47 s · 110 MB · 295 k allocs |

Multilingual's `tokenizer.json` is 34 MB: 256 000 vocabulary entries and 580 604 merges. Decoding the
merges into `[][2]string` allocated two strings per merge only to look them up and drop them, so
scanning the two values directly (PLAN.md Task B.1) roughly halves the bytes and cuts allocations by 81%.
Time moves less, because most of what remains is not allocation: about a third is `encoding/json`
still validating the whole document, and most of the rest is the 2.3 million map lookups that resolve
every merge to ids.

### Go or Python, ORT 1.23 or 1.30 — no difference beyond the noise

An early cross-check recorded `english` at 1901 ms from Go on ORT 1.23.0 against 5064 ms from Python
on ORT 1.30.0, a 2.7× gap, but only 1.1× on `multilingual`. It did not survive a controlled rerun
(PLAN.md Task 6.8.3). The rerun ran both sides on both runtime versions at batch 1, 512 tokens and
8 threads. It used four rounds per checkpoint, ordered as a balanced Latin square, so every
configuration ran once in every position. Each cell waited for the package to cool, and the
temperature and clock were logged before and after. Median p50 over the four rounds:

| checkpoint     | Go + 1.23.0 | Python + 1.23.0 | Go + 1.30.0 | Python + 1.30.0 |
| -------------- | ----------- | --------------- | ----------- | --------------- |
| `english`      | 1975 ms     | 1828 ms         | 1879 ms     | 1727 ms         |
| `multilingual` | 679 ms      | 649 ms          | 620 ms      | 626 ms          |

The widest spread is 1.14× (`english`) and 1.09× (`multilingual`). Run order alone moves
`english` by as much: the median of the cells run first in their round is 1706 ms, against 1960 ms
for those run third. What is left between configurations is therefore not separable from the
machine's noise. Nothing here comes near 2.7×, so neither the binding nor the runtime version costs
anything worth chasing. The measurement cannot rank the four configurations beyond that.

A single back-to-back pair is not a comparison on this laptop. In the first, uncontrolled run,
Python on 1.30.0 took 3800 ms on `english` against 2174 ms for Go on 1.30.0 in the same round.
The Python side is `scripts/bench_ort.py`, which builds `BenchmarkForward`'s exact inputs. The
transcripts are `docs/benchmarks/raw/ort-version-gap-balanced.txt` (the evidence) and
`ort-version-gap.txt` (the first, fixed-order run).

### int8 dynamic quantization (PLAN.md B.5) — measured, not shipped

> **Verdict: int8 is not shipped.** It is faster, 2.0–2.3× per question at 512 tokens, but it fails the calibration
> thresholds fixed before it was measured, on all three checkpoints. 9–19.5% of the choice and
> score answers change their argmax. CPU stays fp32 (D6, D33). `scripts/export_onnx.py
--int8` stays, so that anyone can reproduce the measurement.

PLAN.md R4 makes calibration, not speed, the acceptance criterion: calibrated probabilities are the
product. B.5 therefore measured both on the same int8 graphs, and the calibration verdict decided.

**The artefact (B.5.1).** `scripts/export_onnx.py --int8 --from build/onnx --from-suffix=-dynamo
--out build/onnx-int8` quantizes the fp32 dynamo exports above with ORT 1.30.0's `quantize_dynamic`
in the pinned reference environment. The recipe quantizes every `MatMul` with a constant weight
(int8 per output channel) and quantizes activations to uint8 at run time. The quantizer also
rewrites the head's four `Gemm` nodes as `MatMul` and quantizes them. Attention's
activation-by-activation products and the embedding `Gather` stay fp32. Every graph keeps IR 10 and
opset 18, so Go's pinned ORT 1.23.0 loads it, and `TestInt8Exports` runs every batch of
`testdata/logits.jsonl` through each one there.

| checkpoint             | fp32 graph + data | int8 graph + data | quantized `MatMul` | fp32 `MatMul` left |
| ---------------------- | ----------------- | ----------------- | ------------------ | ------------------ |
| `laya` (english)       | 1688 MB           | 581 MB            | 124                | 62                 |
| `laya-multilingual`    | 1290 MB           | 915 MB            | 100                | 49                 |
| `laya-typed-decisions` | 1688 MB           | 581 MB            | 124                | 62                 |

`laya-multilingual` shrinks least because about 197 M of its 322 M weights are the 256 000-token
embedding table, which stays fp32.

#### Calibration (B.5.3), the acceptance criterion

**Data.** Fixed, seeded subsets of 200 rows each, drawn by `go run ./cmd/laya-evalfetch` from the
Hugging Face datasets-server rows API on 2026-10-11 (seed 20261011). The subsets are not checked
in. [`eval-manifest.json`](docs/benchmarks/eval-manifest.json) is: it records the dataset ids,
splits, row offsets and each file's sha256, and the harness refuses a file whose sha differs.

| set     | dataset                         | question                                                                    |
| ------- | ------------------------------- | --------------------------------------------------------------------------- |
| ag_news | `fancyzhx/ag_news`, test        | choice, 4 options: "What is the topic of this news article?"                |
| emotion | `dair-ai/emotion` (split), test | choice, 6 options: "Which emotion does the author express?"                 |
| sst2    | `stanfordnlp/sst2`, validation  | noul: "Is the sentiment of this movie review positive?"                     |
| yelp    | `Yelp/yelp_review_full`, test   | score, 5 levels: "How many stars does this review give?", 1 star to 5 stars |

**Method.** `TestInt8Calibration` (`internal/evalset`) asks every case through the public API, as a
caller would opt in: a `Router` over the local checkpoint with `WithONNXDir` set to the fp32
exports, then to the int8 exports. The inputs are the same for both. The fp32 side is the export
each int8 graph was quantized from, as the graph's `laya.int8` stamp names it
(`laya-<name>-dynamo.onnx`), so the two sides are one graph at two precisions; the test fails if
`LAYA_ONNX_DIR` does not hold that file. It scores the answers as
upstream's own evaluation notebook does
(`original/notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb`). It reads the probabilities
after the checkpoint's temperatures and the four-decimal rounding, exactly as a caller gets them,
and renormalizes them. The ECE confidence is the largest probability for choice and score, and
max(p, 1−p) for noul (`calib.ECE`, 15 bins). Brier is over the full distribution (`calib.Brier`).
A score's prediction is its most probable level. A flip is a case whose prediction differs between
fp32 and int8. Max Δp is the largest per-option probability difference over every case.

**Thresholds**, fixed in `internal/evalset/metrics.go` and committed before the harness had
produced a number. They apply per checkpoint and question type, and a cell passes only if all three
hold:

- int8's ECE is at most **0.02** worse than fp32's;
- int8's Brier is at most **0.02** worse than fp32's;
- at most **5%** of the argmaxes flip.

Being better never fails.

| checkpoint             | type (sets)                | n   | ECE fp32 → int8        | Brier fp32 → int8      | accuracy fp32 → int8 | flips      | max Δp | verdict                                               |
| ---------------------- | -------------------------- | --- | ---------------------- | ---------------------- | -------------------- | ---------- | ------ | ----------------------------------------------------- |
| `laya`                 | choice (AG News + Emotion) | 400 | 0.181 → 0.173 (-0.008) | 0.475 → 0.493 (+0.018) | 0.715 → 0.708        | 62 (15.5%) | 1.00   | **fail**: flip rate 0.155                             |
| `laya`                 | score (Yelp)               | 200 | 0.549 → 0.527 (-0.022) | 1.174 → 1.128 (-0.046) | 0.300 → 0.280        | 21 (10.5%) | 0.69   | **fail**: flip rate 0.105                             |
| `laya`                 | noul (SST-2)               | 200 | 0.500 → 0.478 (-0.022) | 1.000 → 0.953 (-0.046) | 0.500 → 0.515        | 3 (1.5%)   | 0.97   | pass                                                  |
| `laya-multilingual`    | choice (AG News + Emotion) | 400 | 0.208 → 0.220 (+0.012) | 0.508 → 0.537 (+0.028) | 0.703 → 0.667        | 53 (13.2%) | 0.98   | **fail**: Brier +0.0281; flip rate 0.133              |
| `laya-multilingual`    | score (Yelp)               | 200 | 0.603 → 0.479 (-0.124) | 1.271 → 1.177 (-0.094) | 0.290 → 0.295        | 18 (9.0%)  | 0.59   | **fail**: flip rate 0.090                             |
| `laya-multilingual`    | noul (SST-2)               | 200 | 0.255 → 0.419 (+0.164) | 0.543 → 0.792 (+0.248) | 0.635 → 0.510        | 27 (13.5%) | 0.83   | **fail**: ECE +0.1637; Brier +0.2485; flip rate 0.135 |
| `laya-typed-decisions` | choice (AG News + Emotion) | 400 | 0.091 → 0.099 (+0.008) | 0.422 → 0.424 (+0.002) | 0.725 → 0.723        | 52 (13.0%) | 0.90   | **fail**: flip rate 0.130                             |
| `laya-typed-decisions` | score (Yelp)               | 200 | 0.337 → 0.406 (+0.069) | 0.910 → 0.947 (+0.037) | 0.375 → 0.285        | 39 (19.5%) | 0.64   | **fail**: ECE +0.0686; Brier +0.0367; flip rate 0.195 |
| `laya-typed-decisions` | noul (SST-2)               | 200 | 0.487 → 0.448 (-0.039) | 0.966 → 0.860 (-0.105) | 0.500 → 0.505        | 1 (0.5%)   | 0.49   | pass                                                  |

Per dataset:

| checkpoint             | dataset | ECE fp32 → int8 | Brier fp32 → int8 | accuracy fp32 → int8 | score MAE fp32 → int8 | flips      |
| ---------------------- | ------- | --------------- | ----------------- | -------------------- | --------------------- | ---------- |
| `laya`                 | ag_news | 0.051 → 0.064   | 0.133 → 0.144     | 0.910 → 0.905        | —                     | 11 (5.5%)  |
| `laya`                 | emotion | 0.361 → 0.369   | 0.818 → 0.842     | 0.520 → 0.510        | —                     | 51 (25.5%) |
| `laya`                 | sst2    | 0.500 → 0.478   | 1.000 → 0.953     | 0.500 → 0.515        | —                     | 3 (1.5%)   |
| `laya`                 | yelp    | 0.549 → 0.527   | 1.174 → 1.128     | 0.300 → 0.280        | 1.353 → 1.363         | 21 (10.5%) |
| `laya-multilingual`    | ag_news | 0.049 → 0.059   | 0.121 → 0.139     | 0.925 → 0.925        | —                     | 8 (4.0%)   |
| `laya-multilingual`    | emotion | 0.389 → 0.408   | 0.896 → 0.934     | 0.480 → 0.410        | —                     | 45 (22.5%) |
| `laya-multilingual`    | sst2    | 0.255 → 0.419   | 0.543 → 0.792     | 0.635 → 0.510        | —                     | 27 (13.5%) |
| `laya-multilingual`    | yelp    | 0.603 → 0.479   | 1.271 → 1.177     | 0.290 → 0.295        | 1.414 → 1.498         | 18 (9.0%)  |
| `laya-typed-decisions` | ag_news | 0.143 → 0.143   | 0.152 → 0.154     | 0.915 → 0.910        | —                     | 9 (4.5%)   |
| `laya-typed-decisions` | emotion | 0.246 → 0.212   | 0.692 → 0.694     | 0.535 → 0.535        | —                     | 43 (21.5%) |
| `laya-typed-decisions` | sst2    | 0.487 → 0.448   | 0.966 → 0.860     | 0.500 → 0.505        | —                     | 1 (0.5%)   |
| `laya-typed-decisions` | yelp    | 0.337 → 0.406   | 0.910 → 0.947     | 0.375 → 0.285        | 1.077 → 1.167         | 39 (19.5%) |

What the numbers say:

- **The aggregates hide the damage.** On `laya` and `laya-typed-decisions`, ECE and Brier for
  choice move by at most 0.018. But 13–15.5% of the choice answers change, and on Emotion it is
  21.5–25.5%. int8 moves probability mass between close options, and in aggregate the errors it
  introduces and the errors it removes roughly cancel. A caller gating on a confidence threshold
  sees different answers on one case in seven.
- **`laya-multilingual` degrades outright.** Its SST-2 noul loses 0.125 accuracy, with ECE +0.164
  and Brier +0.248.
- **The two noul passes measure little.** On SST-2, `laya` and `laya-typed-decisions` already
  answer P(true) ≈ 0 for almost every review in fp32, positive or not. Accuracy 0.500 is the share
  of negative reviews. With nothing to lose, int8 cannot lose it. In a spot check on `laya`, eight
  of these reviews asked as a two-option choice all came out right, while a reworded noul still
  answered P(true) ≈ 0. So this is the checkpoint's noul head on sentiment, not the harness.
- **The weight settings are not the cause.** Before B.5.3 ran, four `quantize_dynamic` variants
  were scored against the fp32 export on the golden batches (`testdata/logits.jsonl`):
  per-channel or per-tensor weights, with and without `reduce_range`. All four kept only 18–21 of
  30 argmaxes on `laya` (`docs/benchmarks/raw/int8-recipe-variants-english.txt`). That points at
  the dynamic quantization of these encoders' activations. Recovering the loss would take a
  different method, such as static quantization calibrated on real data or keeping the sensitive
  layers in fp32, and that method would have to pass the same harness.

#### Speed (B.5.2)

**Hardware and method.** These are the S3 laptop and harness above: `just bench-onnx` with five
timed iterations per cell after one warm-up, ORT 1.23.0, `CGO_ENABLED=0`, one full sweep per
precision. The int8 sweep ran first (05:49–05:55), and the fp32 sweep ran right after it
(05:55–06:07). `nproc` was 12, and no other test session ran. The 1-minute load average was 3.6
at the int8 sweep's start, and the sweeps' own threads raised it to 9–11. The "vs S3" column
divides S3's idle fp32 p50 from the tables above by the int8 p50.

One question, 512 tokens, 4 options, 8 threads:

| checkpoint             | fp32 p50 | int8 p50 | speedup | vs S3 fp32 (idle) | session load, fp32 → int8 | resident after load, fp32 → int8 |
| ---------------------- | -------- | -------- | ------- | ----------------- | ------------------------- | -------------------------------- |
| `laya` (english)       | 1947 ms  | 844 ms   | 2.3×    | 1691 ms → 2.0×    | 1.32 s → 0.58 s           | 1423 → 371 MiB                   |
| `laya-multilingual`    | 668 ms   | 318 ms   | 2.1×    | 645 ms → 2.0×     | 0.50 s → 0.25 s           | 488 → 133 MiB                    |
| `laya-typed-decisions` | 1660 ms  | 825 ms   | 2.0×    | 1856 ms → 2.2×    | 1.24 s → 0.64 s           | 1423 → 371 MiB                   |

Other cells of the same two sweeps:

| cell                         | `laya`                | `laya-multilingual`  | `laya-typed-decisions` |
| ---------------------------- | --------------------- | -------------------- | ---------------------- |
| 1 thread, 512 tokens         | 3657 → 1512 ms, 2.4×  | 1394 → 637 ms, 2.2×  | 3811 → 1499 ms, 2.5×   |
| 8 threads, 128 tokens        | 368 → 164 ms, 2.2×    | 225 → 64 ms, 3.5×    | 377 → 180 ms, 2.1×     |
| 8 threads, 8 questions × 512 | 13668 → 6745 ms, 2.0× | 4924 → 2728 ms, 1.8× | 14067 → 7198 ms, 2.0×  |

The 3.5× on `laya-multilingual` at 128 tokens is a slow fp32 cell (p90 302 ms); S3 measured it at
143 ms, which gives 2.2×. "Resident after load" is the RSS growth across the first session
build in a fresh process, before any forward pass, as in S3. ORT keeps the int8 graphs'
external data mapped, so pages it has not touched yet, such as most of `laya-multilingual`'s fp32
embedding table, do not count. Treat that column as load cost, not as the working set.

**A first run under load** (04:54–05:14, while two other test sessions shared the machine) gave
1.5–2.8×; the fp32 sweep ran under more load than the int8 one, which inflated the upper end. Its
transcripts are kept as `docs/benchmarks/raw/bench-onnx-{int8,fp32}-load.txt`; the numbers above
are from `bench-onnx-{int8,fp32}-quiet.txt`.

**The speedup changes the throughput story, not the interactive one.** Even if int8 had passed,
`laya` would still take about 840 ms per question at 512 tokens on this laptop. A batch workload
would get through about twice as many questions per CPU-hour. Anything with a human waiting on
it still wants a GPU (B.3).

#### Reproduce

```bash
# B.5.1: the int8 graphs (reference environment, about 25 s and 7.3 GB peak per checkpoint)
systemd-run --user --scope -q -p MemoryMax=10G -p MemorySwapMax=0 \
    .venv-ref/bin/python -I scripts/export_onnx.py --all --int8 \
    --from build/onnx --from-suffix=-dynamo --out build/onnx-int8
LAYA_ONNX_INT8_DIR=build/onnx-int8 go test -run TestInt8Exports -v ./internal/backend/onnx/

# B.5.3: the manifest's subsets (network, paced at one request a second), then one process per checkpoint
go run ./cmd/laya-evalfetch -out build/eval
LAYA_MODELS=$PWD/models LAYA_ONNX_DIR=build/onnx LAYA_ONNX_INT8_DIR=build/onnx-int8 \
    LAYA_EVAL_CHECKPOINTS=english systemd-run --user --scope -q -p MemoryMax=8G -p MemorySwapMax=0 \
    go test -run TestInt8Calibration -v -timeout 4h ./internal/evalset/

# B.5.2: the same sweep over both
LAYA_ONNX_DIR=build/onnx-int8 just bench-onnx 5 build/bench-onnx-int8.txt ""
LAYA_ONNX_DIR=build/onnx      just bench-onnx 5 build/bench-onnx-fp32.txt
```

With the manifest present, `laya-evalfetch` reads the rows it records again, at its offsets, and
fails if a file's sha256 differs. That would mean the Hub's rows changed and the numbers above no
longer describe them. Only `-update` draws a new sample and rewrites the manifest.
`TestInt8Calibration` fails on every cell that misses a threshold, so today it fails on all three
checkpoints. That failure is B.5.4's verdict.

**Raw transcripts** are in `docs/benchmarks/raw/`:

- `int8-export.json`: the export reports, with op counts, sizes and versions.
- `int8-go-load.txt`: `TestInt8Exports` on ORT 1.23.0.
- `int8-calib-{english,multilingual,typed-decisions}.txt` and `int8-calib-results.json`: the
  calibration runs.
- `int8-recipe-variants-english.txt`: the recipe comparison.
- `bench-onnx-int8-quiet.txt` and `bench-onnx-fp32-quiet.txt`: the two sweeps the speed tables
  are from; `bench-onnx-int8-load.txt` and `bench-onnx-fp32-load.txt`: the first run, under load.

### Native backend (M8)

> **Idle-machine rerun, 2026-10-10 21:38–21:47.** The headline table is from three sweeps run with
> no other test session on the machine. The 1-minute load average of 2.8–6.3 in those transcripts
> is the benchmark's own 8 workers. The first measurement, taken under load from two parallel
> test sessions, is kept further down because it shows how the ratio swings under contention.

PLAN.md Task 8.9 holds the pure-Go backend (`internal/backend/native`, D8) to **within 10× of ORT-CPU
at 512 tokens on the same hardware**. D8 makes it a zero-shared-library backend, not a speed play, so
the budget is a ceiling, not a target.

**Hardware.** The S3 laptop above: i7-1255U, 12 hardware threads, `powersave`, Linux 6.8.

**Stack.** go1.26.9, no cgo on the backend's path, `GOMAXPROCS=11` (Go's own choice here, the same as
in every S3 transcript). The weights are decoded from the checkpoints' fp16 `model.safetensors` to
float32. The kernels are pure Go, plus an AVX2/FMA dot product, and run on 8 workers
(`tensor.SetWorkers(8)`, the counterpart of S3's `IntraOpNumThreads=8`). These measurements ran on
the repository's own copy of the kernels; since B.19 they come from `github.com/cwbudde/algo-tensor`
with every output bit unchanged (D35).

**Method.** S3's reference cell, with S3's inputs value for value: one question, 512 tokens, 4
options, every position attended, ids `1 + (i·7919) mod 997`. `BenchmarkForward` in
`internal/backend/native/bench_test.go` mirrors S3's harness: one untimed pass, then five timed
ones, and p50 and p90 per cell. `TestBenchHarness` pins the cell and the inputs to S3's in CI. Three
sweeps of `just bench-native`, one process per checkpoint under an 8 GB memory cap; the table takes
the **minimum p50 across the sweeps**, as S3 does. Right after the three native sweeps, S3's own
ORT harness ran the same cell (`-bench 'BenchmarkForward/<checkpoint>$/threads=8$/b1_s512_k4$'`,
ORT 1.23.0) once, on the same idle machine. That gives the "ORT p50, here" column, which agrees
with S3's figures to within 3–10%.

| checkpoint             | native p50 | native p90 | ORT p50, S3 | ORT p50, here | ratio (vs S3 / vs here) | 10× budget | met |
| ---------------------- | ---------- | ---------- | ----------- | ------------- | ----------------------- | ---------- | --- |
| `laya` (english)       | 9721 ms    | 11623 ms   | 1691 ms     | 1729 ms       | 5.7× / 5.6×             | 16.9 s     | yes |
| `laya-multilingual`    | 3593 ms    | 3690 ms    | 645 ms      | 623 ms        | 5.6× / 5.8×             | 6.45 s     | yes |
| `laya-typed-decisions` | 9986 ms    | 12014 ms   | 1856 ms     | 1662 ms       | 5.4× / 6.0×             | 18.6 s     | yes |

The p90 is from the sweep that gave the minimum p50. The three idle sweeps agree to within 5% at
p50:

- `laya`: 9721–9821 ms;
- `laya-multilingual`: 3593–3718 ms;
- `laya-typed-decisions`: 9986–10485 ms.

Every checkpoint is inside the budget on both ORT columns. The ratio is 5.4–6.0×, which leaves
1.7–1.8× of headroom. The under-load numbers were close, at 9958, 3640 and 10458 ms.

**Under load the margin does not hold.** The first measurement ran while two other test sessions
were active. Each native sweep was paired with the ORT run that followed it:

| sweep | 1-min load, start of each native run | `laya`         | `laya-multilingual` | `laya-typed-decisions` |
| ----- | ------------------------------------ | -------------- | ------------------- | ---------------------- |
| a     | 1.0–4.7                              | 9958 ms, 6.2×  | 3640 ms, 3.4×       | 12076 ms, 4.6×         |
| b     | 11.7–12.5                            | 21473 ms, 13×  | 7423 ms, 11.8×      | 16917 ms, 6.3×         |
| c     | 9.7–12.4                             | 18280 ms, 5.8× | 7019 ms, 5.0×       | 10458 ms, 4.1×         |

Sweep b broke the budget on two checkpoints. Both sides slowed down by up to about 2× under this
load (native `laya` 9958 to 21473 ms, ORT `laya` 1613 to 3174 ms), but not in the same minutes, so a
single pair swings from 3.4× to 13×. The load samples are `uptime` taken before each process, not
during it. The idle rerun confirms the margin above. A busy shared machine can still put a single
measurement over 10×.

A CPU profile of one typed-decisions run under that load (`bench-native-pprof-typed-decisions.txt`;
it covers `Open` and the warm-up pass too) puts **87%** of the CPU time in `tensor.Linear`, almost
all of it in the AVX2 dot product. On amd64, `Linear` computes one dot product per output with no
cache blocking, so every token streams the whole weight matrix again. Attention is 5%, on one
goroutine, and decoding the weights about 2%.

**Loading.** `BenchmarkLoad` is `Open` from a warm page cache, timed over five loads per sweep, with
the resident growth of the first load in a fresh process. The native column is the idle rerun; the
ORT "here" column is from the run under load:

| checkpoint             | native load | resident after load | ORT session load, S3 | ORT session load, here |
| ---------------------- | ----------- | ------------------- | -------------------- | ---------------------- |
| `laya`                 | 1.39–1.41 s | 1966–1978 MiB       | 2.1 s, 1423 MiB      | 2.2 s, 1423 MiB        |
| `laya-multilingual`    | 1.19–1.21 s | 1575–1584 MiB       | 1.5 s, 488 MiB       | 0.9 s, 488 MiB         |
| `laya-typed-decisions` | 1.37–1.49 s | 1965–1983 MiB       | 3.2 s, 1423 MiB      | 2.2 s, 1423 MiB        |

All three load at least as fast as an ORT session did in S3. Memory is the larger cost: the
native backend holds every weight as float32 on the Go heap, so all three loaded at once need about
4.7 GB since B.16 (5.5 GB in the table above), against ORT's 3.3 GB.

**Peak memory at load (PLAN.md B.16).** `load` used to copy each decoded weight into its tensor, so
every weight existed twice for a moment; it now takes the decoded slice over. `BenchmarkLoad` also
reports `peak_MiB`, the process's `VmHWM` right after the first load in a fresh process. One test
binary per commit, before (`2566863`) and after (`753ad96`), interleaved per checkpoint over two
rounds, each run its own process under an 8 GB cap:

| checkpoint             | peak, before  | peak, after | resident after load, before → after |
| ---------------------- | ------------- | ----------- | ----------------------------------- |
| `laya`                 | 1972–1984 MiB | 1676 MiB    | 1967–1978 → 1670 MiB                |
| `laya-multilingual`    | 1581–1589 MiB | 1334 MiB    | 1576–1584 → 1328 MiB                |
| `laya-typed-decisions` | 1971–2006 MiB | 1676 MiB    | 1965–2001 → 1670 MiB                |

The peak drops by 250–330 MiB, about 15%. The resident size after load drops by as much, because
the runtime has not returned the freed copies to the OS when `Open` returns; the table above, from
before B.16, counts them. Other test sessions shared the machine (1-minute load average 10–14), so
the load times in that transcript, `docs/benchmarks/raw/bench-native-load-b16.txt`, do not compare.

**The default worker count (PLAN.md B.18).** The table above runs at `tensor.SetWorkers(8)`, S3's
setting. A caller of `laya.Open` could not call it, since the package was internal, and the kernels
used to default to one worker, so every real caller ran single-threaded. They now default to
`GOMAXPROCS` at start-up (D34). `BenchmarkForwardDefaultWorkers` is the reference cell at whatever
the default is. One test binary without the change and one with it, interleaved per checkpoint, three
timed passes each under an 8 GB cap, `GOMAXPROCS=12`:

| checkpoint             | default before: 1 worker | default after: 12 workers | speedup |
| ---------------------- | ------------------------ | ------------------------- | ------- |
| `laya`                 | 35254 ms                 | 10822 ms                  | 3.3×    |
| `laya-multilingual`    | 10192 ms                 | 3939 ms                   | 2.6×    |
| `laya-typed-decisions` | 31378 ms                 | 9076 ms                   | 3.5×    |

These are p50s **under uncontrolled, unequal load**: other sessions shared the machine, and the
1-minute load average at the start of the before/after runs was 10.18/4.11 for `laya`, 10.75/10.29
for `laya-multilingual` and 7.60/3.48 for `laya-typed-decisions`. The ratio is indicative, not a
bound either way: contention can slow a 12-worker run more than a 1-worker run, and the lighter
load in the "after" runs favours them. `laya-multilingual`, where both runs started at about the
same load, still shows 2.6×. The "after" column sits within 12% of the 8-worker idle table above.
No result changes: every output of `Linear` and `MatMul` is one dot product however the work is
split (`TestKernelsSameBitsAnyWorkers`, now in `internal/backend/native/kernels_test.go`).

**The kernels from algo-tensor (PLAN.md B.19).** The kernels now come from
`github.com/cwbudde/algo-tensor` v0.2.0 (D35), whose `MatMulTransB`, the attention products, splits
over the workers where go-laya's copy ran it on one. `BenchmarkForwardDefaultWorkers` without the
switch and with it: one test binary each, interleaved per checkpoint, three timed passes each under
an 8 GB cap, `GOMAXPROCS` unset (12). Two sweeps, the second with each pair's order reversed:

| checkpoint             | sweep 1: before → after | speedup | sweep 2: before → after | speedup |
| ---------------------- | ----------------------- | ------- | ----------------------- | ------- |
| `laya`                 | 8453 → 9637 ms          | 0.88×   | 9507 → 9519 ms          | 1.00×   |
| `laya-multilingual`    | 3482 → 3607 ms          | 0.97×   | 3486 → 3625 ms          | 0.96×   |
| `laya-typed-decisions` | 9662 → 10136 ms         | 0.95×   | 9985 → 9828 ms          | 1.02×   |

These are p50s **under uncontrolled, unequal load**. The 1-minute load average at the start of the
before/after runs was 2.29/5.75, 5.86/5.13 and 5.40/4.64 in sweep 1, and 7.51/6.33, 8.74/9.35 and
10.29/8.11 in sweep 2. **No speedup shows.** The pairs differ by −12% to +2%, about as much as the
same binary differs between sweeps (`laya` before: 8453 and 9507 ms), and the largest gap, `laya` in
sweep 1, has the least-loaded "before" run of all. `laya-multilingual` is 3–4% slower in both
sweeps, which this setup cannot tell from noise. In the profile above `MatMulTransB` took 5% of the
CPU time and `Linear` 87%, so the parallel attention products had little to win; B.20's GEMM is
where the time is. The outputs are the same bits (`TestForwardDigest`).

**Raw transcripts.**

- **Idle rerun:**
  - `docs/benchmarks/raw/bench-native-quiet-{a,b,c}.txt` are the three `just bench-native` sweeps.
  - `bench-onnx-refcell-quiet.txt` is the ORT run after them.
- **The first run, under load:**
  - `bench-native-sweep-{a,b,c}.txt` are the native sweeps.
  - `bench-onnx-refcell-{a,b,c}.txt` are the ORT runs, one after each. ORT run a also holds the
    fresh `BenchmarkSessionLoad`.
- **B.16, peak memory at load:** `bench-native-load-b16.txt`, `BenchmarkLoad` before and after,
  interleaved.
- **B.18, the default worker count:** `bench-native-default-workers-b18.txt`,
  `BenchmarkForwardDefaultWorkers` before and after, interleaved.
- **B.19, the kernels from algo-tensor:** `bench-native-algo-tensor-b19.txt`,
  `BenchmarkForwardDefaultWorkers` before and after, interleaved, in two sweeps.

Every process is headed by its `uptime`.
`bench-native-pprof-typed-decisions.txt` is the profile's top 15.

### What this means

**Sequence length and checkpoint decide the cost:** at 128 tokens `laya-multilingual` answers in
160 ms, while at the default 512-token `max_len` `laya` takes 1691 ms.

So, plainly:

- **Sub-second per question is reachable only on `laya-multilingual` (mmBERT-base) at short
  sequences.** Both ModernBERT-large checkpoints are 1.7 s and up at default settings.
- **This is a batch and background workload on CPU**, not an interactive one. Anything with a human
  waiting on it wants a GPU execution provider.
- **Set `IntraOpNumThreads` to the physical core count, not the thread count.** It is free, and it is
  worth 12–26%.
- **int8 dynamic quantization was measured and is not shipped** (PLAN.md B.5, D33). It is 2.0–2.3×
  faster at 512 tokens, but 9–19.5% of the choice and score answers change, which fails the
  calibration thresholds on every checkpoint
  ([details](#int8-dynamic-quantization-planmd-b5--measured-not-shipped)). The lever left is
  offering the cheaper checkpoint deliberately (Task 3.4), and it is not applied by default.

---

## Calibration

|                     | as shipped | temperature refit |
| ------------------- | ---------- | ----------------- |
| `laya`              | 0.466      | **0.081**         |
| `laya-multilingual` | 0.314      | **0.106**         |

Both ship over-confident; `laya-multilingual` ships with no fitted temperatures at all. Refitting one temperature per (question type, option count) on held-out data is the single highest-value fix available, and takes ECE below Jev's measured 0.246.

The table above is upstream's. This port's own calibration measurement, int8 against fp32 on public datasets, is under [int8 dynamic quantization](#int8-dynamic-quantization-planmd-b5--measured-not-shipped).

## Option-order robustness

How often the answer changes when the options are permuted. Jev measured at 0.13.

| suite             | laya  | laya-multilingual |
| ----------------- | ----- | ----------------- |
| massive_intent.en | 0.150 | 0.230             |
| en.emotion        | 0.040 | 0.090             |
| xnli.en           | 0.000 | 0.015             |

At 20 options both are less order-stable than Jev — worth fixing with more aggressive option-order shuffling during training.

---

## Limits, stated plainly

- **CPU takes 0.6–1.9 s per question** at the default 512-token `max_len`, measured here on a 15 W laptop. Batching does not amortise it.
- **Near chance on typed-decisions zero-shot** — the 0.766 belongs to the fine-tuned checkpoint, on that benchmark's own training split.
- **Moderation does not hold up on held-out data** (0.530, macro-F1 0.400).
- **Keep `choice` questions under ~20 options.**
- **Both checkpoints ship over-confident.** Fit temperatures on your own data.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- `laya` collapses outside English; `laya-multilingual` is weaker on English. Route.
