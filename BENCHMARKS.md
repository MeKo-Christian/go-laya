# Laya benchmarks

Every checkpoint answered **byte-identical questions** in each run (fixed seed). Jev figures are **third-party published, never measured here** — no TypeSafe API access — so sample sizes and prompts differ; treat them as indicative.

| run          | what                                                                                                                       | where                                         |
| ------------ | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------- |
| T4 Colab     | typed-decisions, MASSIVE (14 langs), XNLI (15 langs), English suites, latency, option-order robustness, calibration repair | `research/results/t4_colab_benchmark.json`    |
| CPU sweep    | MASSIVE intent across **all 51 languages**, typed-decisions on all three checkpoints                                       | `research/results/cpu_51_language_sweep.json` |
| Applications | the six workflow themes + the datasets where Jev numbers exist, all three checkpoints                                      | `research/results/app_benchmark.json`         |

---

## Headline

|                                                              | Laya        | Jev (published) |
| ------------------------------------------------------------ | ----------- | --------------- |
| typed-decisions (2,000 decisions)                            | **0.766**   | 0.727           |
| AG News (4 labels)                                           | **0.953**   | 0.910           |
| DAIR Emotion (6 labels)                                      | **0.600**   | 0.480           |
| ECE after temperature fitting                                | **0.081**   | 0.246           |
| p50 latency, 1 question — T4 _(upstream, not measured here)_ | **32.8 ms** | 236-276 ms      |
| p50 latency, 1 question — CPU _(measured here, 512 tokens)_  | 0.63–1.9 s  | —               |

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

The T4 table below is upstream's. This one is ours, and it is the number that matters for a Go port
whose whole premise (D2) is a dependency-free binary you can drop on a server.

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
laptop's `build/` directory (PLAN.md §8 forbids inheriting upstream's unreproducible
`research/results/*.json`). `bench-onnx-sweep-a.txt` and `bench-onnx-sweep-b.txt` are two full sweeps
with the shape axis at 8 threads; `bench-onnx-threads12.txt` is an earlier sweep with the shape axis at
12 threads, which is where the "all hardware threads is the wrong setting" finding first showed. The
third 8-thread sweep's transcript was overwritten by the recipe's default output path before it was
copied, so only its p50s survive, in the tables.

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

On a T4, going from 1 question to 10 takes `laya-multilingual` from 32.8 ms to 7.2 ms per question —
a 4.6× win from filling the device. **On CPU that win does not exist**: per-question cost is flat to
within noise, because one question at 512 tokens already saturates the cores. Batch for throughput
accounting if you like; do not batch expecting latency per answer to improve.

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

### What this means

**CPU inference is 9–43× slower than the T4 figure, and that range is honest rather than evasive:**
upstream never states the sequence length behind its 33 ms, so the multiple depends on which row you
compare. At 128 tokens `laya-multilingual` is 160 ms against 32.8 ms (4.9×); at its default 512-token
`max_len`, `laya` is 1691 ms against 39.5 ms (43×).

So, plainly:

- **Sub-second per question is reachable only on `laya-multilingual` (mmBERT-base) at short
  sequences.** Both ModernBERT-large checkpoints are 1.7 s and up at default settings.
- **This is a batch and background workload on CPU**, not an interactive one. Anything with a human
  waiting on it wants a GPU execution provider.
- **Set `IntraOpNumThreads` to the physical core count, not the thread count.** It is free, and it is
  worth 12–26%.
- The levers left — int8 dynamic quantization, and offering the cheaper checkpoint deliberately — are
  tracked as PLAN.md Tasks 6.9 and 3.4. Neither is applied by default, and int8 will not be until its
  effect on ECE and Brier is measured: calibrated probabilities are the product.

---

## Speed (Tesla T4) — upstream's published figures, not measured here

| questions per call | laya     | laya-multilingual |
| ------------------ | -------- | ----------------- |
| 1                  | 39.5 ms  | **32.8 ms**       |
| 5                  | 84.5 ms  | **40.1 ms**       |
| 10                 | 158.6 ms | **72.3 ms**       |
| 50                 | 771.3 ms | **337.4 ms**      |

103–332 questions/sec batched. Jev independently measured at 236-276 ms p50, so Laya answers one question roughly **6–7× faster**.

## Calibration

|                     | as shipped | temperature refit |
| ------------------- | ---------- | ----------------- |
| `laya`              | 0.466      | **0.081**         |
| `laya-multilingual` | 0.314      | **0.106**         |

Both ship over-confident; `laya-multilingual` ships with no fitted temperatures at all. Refitting one temperature per (question type, option count) on held-out data is the single highest-value fix available, and takes ECE below Jev's measured 0.246.

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

- **CPU is 9–43× slower than the headline 33 ms** — 0.6–1.9 s per question at the default 512-token `max_len`, measured on a 15 W laptop. Batching does not amortise it.
- **Near chance on typed-decisions zero-shot** — the 0.766 belongs to the fine-tuned checkpoint, on that benchmark's own training split.
- **Moderation does not hold up on held-out data** (0.530, macro-F1 0.400).
- **Keep `choice` questions under ~20 options.**
- **Both checkpoints ship over-confident.** Fit temperatures on your own data.
- **Ordinal `score` is the weakest primitive** (SST-5 0.372).
- `laya` collapses outside English; `laya-multilingual` is weaker on English. Route.
