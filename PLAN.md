# go-laya — Plan

> **For Claude:** REQUIRED SUB-SKILL: use `superpowers:executing-plans` to implement this plan task-by-task.

**Goal:** Reimplement the Python `laya` 0.3.4 decision engine as an idiomatic Go library that produces
byte-identical prompts and numerically equivalent decisions from the same Hugging Face checkpoints.

**This file holds open work only.** What the system is, and why, lives in `docs/`:

| File                                           | Holds                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------ |
| [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) | checkpoints, layout, tokenizer, prompt, ONNX backend, trust, numerics, agent   |
| [`docs/NATIVE.md`](docs/NATIVE.md)             | the pure-Go native backend: loading, encoder, head, kernels, parity            |
| [`docs/TESTING.md`](docs/TESTING.md)           | test tiers, gating variables, tolerances, mutation practice, known blind spots |
| [`docs/DECISIONS.md`](docs/DECISIONS.md)       | D1–D35 with their evidence                                                     |
| [`docs/INVARIANTS.md`](docs/INVARIANTS.md)     | the 74 behaviours a test must assert                                           |
| [`docs/API.md`](docs/API.md)                   | the Go types and the JSON shapes Python emits                                  |
| [`BENCHMARKS.md`](BENCHMARKS.md)               | measured latency, memory and int8 calibration                                  |

**Ids of finished work.** Code comments, scripts and `docs/` cite finished items by id: Spikes S1–S3,
Tasks 0.x–8.x, Backlog B.1–B.19 and the old sections §5–§8. Their text, evidence and mutation runs
are in git: `git show af1a331:PLAN.md` (M7, the Backlog and M8), and `git show 8125f40:PLAN.md` for
the detail of M0–M6. The ids of the open items below are kept, so citations of them stay valid.

---

## Status

Milestones M0–M8 and the 1.0 definition of done are complete. Two Backlog items remain, B.3 and
B.20; neither gates a release, and they share no code.

Each task is TDD: write the failing test, watch it fail, implement minimally, watch it pass, commit.
Run `just ci` before every commit. Tick a box only against a command you ran and can quote
(`AGENTS.md`, "Keeping `PLAN.md` honest").

## Backlog

- [ ] **B.3** Enable CUDA. Device selection works for `cpu`/`auto`/`coreml`, while `cuda` falls
      back (D22). ORT's generic `SessionOptionsAppendExecutionProvider` rejects `CUDA`, and the
      binding at D5's pin neither registers `SessionOptionsAppendExecutionProvider_CUDA_V2` nor
      exposes `OrtSessionOptions`. It needs a binding change (upstream PR, or a fork via a module
      `replace`), which moves D5's pin. The T550 and `/opt/onnxruntime/gpu` on the dev box are
      enough to verify it. Update D22 and the README's "Differences from Python laya" (the `cuda`
      row, formerly Task 7.6.3's list) when it lands.
- [ ] **B.20** A cache-blocked AVX2 GEMM behind `tensor.Linear` on amd64, written in algo-tensor so
      all three consumers gain. A profile puts 87% of native CPU time in `Linear`, which computes
      one dot product per output and streams the whole weight matrix again for every token
      (`BENCHMARKS.md`, Task 8.9). arm64 already has a 4×4 NEON block. It must keep every bit, or
      be a recorded numerics change in algo-tensor and a new golden run here. Measure against
      Task 8.9's cell. Its dependency, B.19, is done: the kernels are
      `github.com/cwbudde/algo-tensor` v0.2.0's (D35), and `TestForwardDigest` is the check that no
      bit moved before go.mod raises the required version.
- [x] **B.21** Read `internal/hub`'s cached refs and listings (`readCached`,
      `internal/hub/snapshot.go`) through `regularfile.Open` with a cap, and open the cached ORT
      library in `ortlib.Cached` the same way. Both `Lstat` the path and then plainly open it, so a
      named pipe swapped in between blocks an offline `Open`, and `readCached` reads without a size
      limit; refs and listings are kilobytes. `purego.Dlopen` reopens the library path afterwards
      anyway, the same unavoidable second open as ORT's for the graph. Exploiting either needs
      write access to the user's cache directory, so it is low severity. A FIFO test per reader,
      modelled on `TestReadFIFO`. Found in B.17's audit, PR #61 (2026-10-11).
      (2026-10-12, PR #69) `readCached` opens through `regularfile.Open` and reads a ref with a 1 KiB
      cap and a listing with a 96 MiB one, six times the 16 MiB accepted from the Hub, so every
      listing `Snapshot` caches reads back even after `json.Marshal` escapes its names (the bundle's
      is 1091 bytes for 38 files). A file that is not regular at the open is a miss, so an offline
      `Snapshot` is `ErrNotCached`, not a wait; one over its cap is an error naming it.
      `ortlib.Cached` opens through `regularfile.Open`, and a non-regular file there is
      `ErrHashMismatch`; `regularfile.ErrNotRegular` lets both map the case. `TestReadCachedFIFO`
      (ref and listing) and `TestCachedFIFO` swap a FIFO in through a hook between the check and the
      open; `TestReadCachedCap` pins the caps, and `TestSnapshotEscapedListing` reads back a listing
      that escaping grew past 80 MiB (it fails with the cap at 16 MiB).
      `go test -count=1 -run 'FIFO|Cap|Escaped|ReadCached' -v ./internal/{hub,ortlib,regularfile}/`
      passes; with both opens put back to `os.Open`, `TestReadCachedFIFO/ref`, `/listing` and
      `TestCachedFIFO` fail through the 5 s guard ("blocked on a FIFO with no writer").
      `purego.Dlopen` still reopens the library path (`docs/ARCHITECTURE.md` §6).

## Risks

Code and scripts cite these ids. Each line is the risk's current state.

| #   | Risk                                                                                                     | State                                                                                                                                                                                                                                                                                                                           |
| --- | -------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| R1  | **Pure-Go tokenizer drift.** Marker positions are token indices; one off-by-one corrupts every decision. | Mitigated. Golden corpora, a fuzz target and `just diff-tokenizer`, which accepts only the documented NFC class (`docs/ARCHITECTURE.md` §3) and fails on any other divergence. `daulet/tokenizers` (CGO) stays a fallback behind the `Tokenizer` interface.                                                                     |
| R2  | **ONNX export fails on ModernBERT-large** (transformers#35545).                                          | Retired: all three checkpoints export (`docs/ARCHITECTURE.md` §5).                                                                                                                                                                                                                                                              |
| R3  | **CPU latency ≫ upstream's 33 ms on a T4.**                                                              | Real: 0.6–1.9 s at 512 tokens (D6, `BENCHMARKS.md`). int8 is not shipped (D33). The lever left is a GPU provider (B.3).                                                                                                                                                                                                         |
| R4  | **int8 wrecks calibration.**                                                                             | Real: int8 fails the calibration thresholds on all three checkpoints and is never the default (D33).                                                                                                                                                                                                                            |
| R5  | **`onnxruntime-purego` instability**: an untagged HEAD with a finalizer data race.                       | Every `*Value` is closed (D5); `LAYA_ORT_FINALIZER=1` keeps the reproduction. `yalue/onnxruntime_go` is the fallback behind the `Backend` seam. B.3 is the likeliest reason the pin moves.                                                                                                                                      |
| R6  | **Upstream tokenizer/transformers drift** silently changes golden vectors.                               | Real (D7). `TestGoldenProvenance` and the fixture provenance tests check the pins, regeneration is a reviewed diff, and `scripts/crosscheck_transformers.py` checks the next bump.                                                                                                                                              |
| R7  | **Supply chain**: `laya.Open("someone/their-model")` must not be RCE.                                    | Mitigated: verified downloads that fail closed, headers validated before use, pinned and verified ORT, regular-file opens of every checkpoint and cache file (B.21), CI greps (`docs/ARCHITECTURE.md` §6). ORT reopens the graph and `purego.Dlopen` the library by path after the checks; both need write access to the files. |
| R8  | **Licensing**: derivative of an Apache-2.0 work.                                                         | `LICENSE` is kept and `NOTICE` records the derivation and the kernels' origin (D35).                                                                                                                                                                                                                                            |
