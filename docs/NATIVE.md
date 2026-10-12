# Native backend

> The pure-Go `backend.Backend`: a laya `DecisionModel` run from the checkpoint's
> `model.safetensors` with no shared library. It is a zero-shared-library backend, not a speed play
> (D8), and `laya.WithRuntime(laya.RuntimeNative)` selects it (D30). The checkpoint facts it rests
> on are in [`ARCHITECTURE.md`](ARCHITECTURE.md) §1, measured latency and memory in
> [`../BENCHMARKS.md`](../BENCHMARKS.md) ("Native backend"), and how it is tested in
> [`TESTING.md`](TESTING.md).

## 1. Packages

| Package                   | Holds                                                                                   |
| ------------------------- | --------------------------------------------------------------------------------------- |
| `internal/backend/native` | `Open`, weight loading by `state_dict` name, `Forward` (encoder, then head)             |
| `internal/modernbert`     | `Norm`, `MLP`, `Attention`, `Layer`, `Encoder`, and `Config`, the layer plan            |
| `internal/head`           | `Head`: everything `DecisionModel.forward` does after the encoder (`common.py:108-126`) |
| `internal/safetensors`    | header validation and F16/F32 decoding                                                  |
| `algo-tensor/tensor`      | the kernels: `Linear`, `MatMul`, `MatMulTransB`, `LayerNorm`, … (D35)                   |

`internal/modernbert` and `internal/head` do not load weights; `internal/backend/native` decodes them
and hands them to the constructors. The package doc of `internal/backend/native` lists every
tensor name it reads.

## 2. Loading

- **Download set.** With `RuntimeNative` the snapshot adds `<sub>/model.safetensors` and
  `<sub>/encoder/config.json` through the same hash-verified, cancellable Hub path (D24 as amended by
  D30). A checkpoint without `encoder/config.json` is `ErrIncompatibleCheckpoint`, a recorded
  deviation: upstream would build the encoder from the config's Hub id.
- **The file.** `safetensors.Open` validates through `ReadHeader`'s parser, with header and data read
  through one descriptor opened by `regularfile.Open`. Every tensor must be F16 or F32, except
  `temperature`, which may have any dtype and is never decoded (it is dead at inference). BF16 is
  refused; no checkpoint uses it.
- **Decoding.** `File.Float32` reads one tensor with `ReadAt` in 1 MiB chunks inside its validated
  span, checks the context per chunk, and decodes F16 exactly (subnormals included,
  `TestF16ToF32Exhaustive` covers all 65536 bit patterns).
- **The head shape before any weight.** `Options.Head` carries `rl_agent_config.json`'s
  `head_layers` and `len(act_costs)+1`. They are checked against the header of the file that is
  then decoded, not of a separate open, so a file replaced in between cannot slip past. `head_layers`
  follows Python: a bool is 0/1, a value that is not positive means no head layers.
- **Strict.** The tensor set must be exactly the expected names at the shapes `encoder/config.json`
  implies; anything else is `ErrIncompatibleCheckpoint`.
- **One copy.** `tensor.FromOwned` takes the decoded slice over instead of copying it: `Float32`
  allocates its result per call, and `safetensors.File` keeps no reference to it. The shape is still
  copied. Peak RSS at `Open` is about 1.68 GiB (ModernBERT-large) and 1.33 GiB (mmBERT-base); all
  three resident need about 4.7 GB, against ORT's 3.3 GB. A load takes 1.2–1.5 s from a warm page
  cache.

## 3. Encoder (ModernBERT and mmBERT)

- **Norm.** Bias-free LayerNorm, eps 1e-5 (`NormEps`). Layer 0's `attn_norm` is `nn.Identity`,
  modelled by `IdentityNorm()`; a nil weight or the zero `Norm` is an error, never the identity.
- **MLP (GeGLU).** One `Linear` over the fused `Wi [2I, H]`; the first `I` of each row goes through
  the exact erf GELU (not tanh), is multiplied by the last `I`, then `Wo [H, I]`. No bias; dropout
  is not modelled. `NewMLP` rejects an odd, empty or zero-sized `Wi`, and a `Wo` given as `[I, H]`.
- **Attention.** `Wqkv [3H, H]` splits as `(3, heads, head_dim)`: q, k, v, each head-major.
  `Wo [H, H]`, no biases. The constructor rejects Wqkv rows ≠ 3H, H not divisible by the heads, an
  odd head_dim and a `Wo` other than `[H, H]`.
- **Sliding window.** Allowed is `|q − k| ≤ sliding_window = local_attention // 2`, i.e. **64**.
  `ModernBertAttention` stores `sliding_window + 1 = 65` (`modeling_modernbert.py:253`), but the
  sdpa path never reads it (`masking_utils.py:142-152`).
- **A padded query with no allowed key** gets sdpa's zeros, not eager's average over every value.
- **RoPE.** `rotate_half` (not interleaved), at each layer type's θ: `FullAttention(θ)`,
  `SlidingAttention(θ, localAttention)`. `inv_freq` is computed in float32 and is bit-exact with
  `ModernBertRotaryEmbedding`; a float64 `inv_freq` is measurably wrong. A few cos/sin values are
  one ulp off out to position 8191.
- **Assembly.** Token embeddings → embeddings norm → layers in plan order → `final_norm`. Each
  layer adds both residuals to its un-normed input.
- **Config.** The layer plan resolves as `ModernBertConfig` does: explicit `layer_types`, else every
  `global_attn_every_n_layers`-th (3) layer is full attention. θ comes from `rope_parameters`, else
  `global/local_rope_theta`, else 160000/10000, with JSON `null` kept apart from a missing key. A RoPE
  type transformers would run but Go does not implement is an error. The two shipped configs give
  28 and 22 layers, full at 0, 3, ….

## 4. Decision head

- `type_emb` is added at **every** position, not only at [CLS].
- The two layers run as upstream's manual loop: `norm_first`, a **ReLU** 4d FFN, LayerNorm eps
  1e-5 with bias, packed `in_proj` with bias, `nhead = max(1, d // 64)`, and no final norm.
- The key-padding mask is `~attention_mask.bool()`: any nonzero mask value counts as a token.
- Marker gather uses `marker_pos.clamp(min=0)`. The scorer is LayerNorm → Linear → erf GELU →
  Linear, and marker columns outside a row's real markers are filled with **−1e4** (`MaskFill`), not
  −inf.
- The act features are float32: top1, top1 − top2, the normalized entropy (with upstream's clamped
  denominator) and k / 255, with the softmax through `calib.ActSoftmax` (D26). `act_head` runs on
  `h[:, 0]`.
- Fewer than two marker columns is an error, as upstream's `topk(2)` raises. `SystemOne` refuses
  such a batch before any backend runs (D32); the head's check stays as defence in depth.
- torch runs the checkpoints' 16- and 12-head layers on its fused fast path and a one-head layer on
  the slow path; the oracle (`scripts/dump_head_ops.py`) records both, and the Go head matches both.

## 5. Kernels and threads

- The kernels are `github.com/cwbudde/algo-tensor` v0.4.0 (D35, D36). `tensor.Workers()` defaults
  to `GOMAXPROCS` at start-up (D34); `SetWorkers(n ≤ 1)` means one worker. `GOMAXPROCS` is the
  supported knob; `SetWorkers` is not go-laya API.
- **Results do not depend on the worker count.** Each output of `Linear` and `MatMul` is one dot
  product however the work is split. `Linear` splits over `tensor.Parallel` from 2^18 multiply-adds,
  by outputs or by tokens. `MatMulTransB` (the attention products) splits over it from 2^20, in whole
  4-row blocks. Both take their helper goroutines from one budget of `Workers() − 1` shared by every
  caller.
- **Bits are pinned per build.** `TestForwardDigest` hashes every output bit at Task 8.9's cell and
  on a padded batch of three, at one worker and at the default. The pins hold on amd64 with the
  assembly and without `GOAMD64=v3` (`digestBuildPinned`). A `purego` build runs the pure-Go
  fallbacks and adds in another order; `GOAMD64=v3` lets the compiler fuse multiply-adds.
- **Concurrency.** Encoder and head only read their weights and allocate every intermediate per
  call, so concurrent `Forward` calls share one model. An `RWMutex` guards only the lifecycle
  against `Close`.
- **Where the time goes.** Before B.20, 87% of CPU time was `tensor.Linear`, almost all in the AVX2
  dot product: one dot product per output with no cache blocking, so every token streamed the whole
  weight matrix again. Attention was about 5%, weight decoding about 2%. At 512 tokens and 8 workers
  the backend ran at 5.4–6.0× ORT-CPU's latency, inside Task 8.9's 10× budget on an idle machine.
- **Cache-blocked `Linear` (B.20, D36).** Since algo-tensor v0.3.1, every token runs over 256 KiB
  panels of weight rows on every architecture. On amd64 with AVX2 and FMA, `linearDot3AVX2` runs
  three tokens over one weight row and replays `dotF32AVX2` exactly, so no bit moved. Task 8.9's
  cell at the default worker count gained 1.4–1.8× on the i7-1255U (`BENCHMARKS.md`). arm64 gets
  only the panels; on an M5 Pro the gain could not be told from noise. arm64 has no multi-output
  NEON kernel for `Linear`: the 4×4 NEON block is `MatMulTransB`'s, and it adds in another order
  than `dotF32NEON`. Batch 1 has no group of three tokens and gains nothing.
- **`MatMulTransB` on amd64 (algo-tensor v0.4.0).** With AVX2 and FMA, the attention products run
  the same kernel and panels as `Linear`, three rows of A over one row of B, from three rows and
  k ≥ 8 (algo-tensor's PLAN 1.4, CWBudde/algo-tensor#12). Every output is still one `dotF32`, so no
  bit moved: `TestForwardDigest` gives the pinned digests on v0.4.0. v0.4.0's numerics change, AVX2
  kernels for `MatMulTransBBias`, reaches nothing here; arm64 and `purego` builds run the same code
  as on v0.3.1.

## 6. Parity

- **Per block.** `internal/modernbert/testdata/ops.json` (real transformers modules of a tiny
  `ModernBertModel`) and `internal/head/testdata/head.json` (upstream's own `DecisionModel` at
  d = 8). Regeneration: `scripts/README.md`, "The native backend's block fixtures".
- **Head intermediates** (invariants #35–38): `testdata/head_intermediates.jsonl`, 33 batches from
  the real checkpoints. Worst diffs: h after `type_emb` ≤ 1.84e-06, after the head layers and gather
  ≤ 1.02e-06 (bounds 1e-5); logits ≤ 1.14e-05 and act features ≤ 6.97e-06 (bounds 5e-5). Every fill
  column is exactly −1e4.
- **End to end.** `TestE2EParity/native` runs every `logits.jsonl` case through `SystemOne` on the
  native backend. Worst probability diff: english 2.15e-06, multilingual 3.49e-06, typed-decisions
  9.24e-07; relative act-logit diff 1.43e-06, 4.74e-07, 4.13e-07; zero argmax flips. ONNX on the same
  cases: 4.29e-06, 2.8e-06, 1.4e-06.
