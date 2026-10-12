# Testing and verification

> How go-laya proves it agrees with `original/`. The rules for writing tests are in
> [`../AGENTS.md`](../AGENTS.md); the golden corpus is described in [`ARCHITECTURE.md`](ARCHITECTURE.md)
> §8, and how to regenerate it in [`../scripts/README.md`](../scripts/README.md).

## 1. Tiers

**CI** runs with no Python, no weights and no network. `just ci` mirrors it: `fmt-check`, `lint-md`,
`lint-py`, `lint-config` (actionlint, taplo, yamllint over git-tracked files), `test-race`, `lint`
and `check-tidy`. The workflows add:

- `ci.yml`: `go test -race` on Ubuntu (amd64) and macOS (arm64).
- `security.yml`: gitleaks over the diff and the tree, govulncheck, CodeQL, and greps for
  `encoding/gob`, `os/exec` and filesystem `replace` directives.
- `release.yml`: tests, the tag checked against `const Version`, and a build for every supported
  platform.

**Local gates** skip under `-short` and without their variable. They are how the real checkpoints,
ONNX Runtime and the reference outputs are reached:

| Variable                                                  | Unlocks                                                                          |
| --------------------------------------------------------- | -------------------------------------------------------------------------------- |
| `LAYA_MODELS`                                             | the checkpoints (`<dir>/laya/…`): tokenizer corpora, sequence, native, e2e tests |
| `LAYA_ONNX_DIR`                                           | the local exports `laya-<name>.onnx`; `LAYA_ONNX_SUFFIX` picks `-dynamo` or `""` |
| `LAYA_ORT_LIB`                                            | a specific ONNX Runtime library; `LAYA_ORT_OLD_LIB` a pre-1.23 one to be refused |
| `LAYA_ORT_NET=1`                                          | `TestPinnedLive`, which downloads the pinned ORT archives                        |
| `LAYA_ORT_FINALIZER=1`                                    | the binding's finalizer data-race reproduction (D5)                              |
| `LAYA_CORPUS`                                             | the tokenizer differential corpus (`just corpus`, `just dump-stages`)            |
| `LAYA_ONNX_INT8_DIR`                                      | `TestInt8Exports` over the `export_onnx.py --int8` output                        |
| `LAYA_EVAL_DIR`, `LAYA_EVAL_CHECKPOINTS`, `LAYA_EVAL_OUT` | `TestInt8Calibration` over `cmd/laya-evalfetch`'s subsets                        |

Recipes for the gated runs: `just test-onnx`, `just test-onnx-race`, `just bench-onnx`,
`just bench-native`, `just diff-tokenizer`. A float32 checkpoint costs 1–2 GB, so the heavy runs take
one checkpoint per process under a memory cap (`systemd-run --user --scope -p MemoryMax=8G
-p MemorySwapMax=0`), as `just bench-native` does.

## 2. Oracles and provenance

- Every fixture is generated from `original/` in the pinned reference environment
  (`scripts/requirements-ref.txt`, D7) and is a reviewed diff, never a drive-by (R6).
- `TestGoldenProvenance` checks every `testdata/*.jsonl` header: library versions, the `compute`
  block, and `checkpoint_sha` against `pinnedRevision` itself. `TestOpsProvenance` and
  `TestHeadProvenance` check the native block fixtures' pins.
- Fixture tests pin their case names and each case's outcome, so a regenerated file that loses a
  case or flips an outcome fails rather than shrinking the test.
- Generators refuse an environment that would record different numbers: a host without AVX2+FMA
  for numpy and torch kernels, a torch not on AVX2 for `act_softmax.jsonl`, a version mismatch for
  the block oracles.
- `internal/backend/fake` replays `logits.jsonl` keyed on all five tensors **and** the checkpoint,
  because english and typed-decisions record identical batches with different logits.

## 3. Tolerances

- **Per checkpoint, from measurement.** Never a round global figure: on random inputs multilingual
  is ~15× looser against PyTorch than the others. `e2eTol` is 2× the measured worst case, `matrixTol`
  about 5×, each under the 1e-4 ceiling of the 1.0 gate.
- **Scaled diffs** `|got − want| / max(1, |want|)` for logits, which carry the −1e4 sentinel.
- **A NaN fails every comparison.** The end-to-end comparisons reject NaN explicitly
  (`TestCompareRejectsNaN`), since `NaN > tol` is false.
- **Exact where Python is exact.** Prompt bytes, answer JSON bytes, `Round4` results, the act
  softmax and numpy's exp/log are compared bit for bit; argmax flips always fail.
- Tokenizer tests assert ids **and** token strings.

## 4. Discriminating tests

A test counts as evidence only if a plausible wrong implementation fails it. The patterns in use:

- **Mutation runs.** Each behaviour is checked by breaking it on purpose and watching a named test
  fail; PRs record the mutation and the failing test.
- **Wrong readings asserted in the test.** Block tests also assert that the likely misreadings
  (tanh GELU, swapped GeGLU halves, q/k swapped, the `(heads, 3, head_dim)` split, a window one off)
  miss torch by a large multiple of the tolerance, so a tolerance can never hide them.
- **Precision cases.** `answers.jsonl` carries seeded cases that a float64 port or a left-to-right
  sum rounds differently; `TestAnswersDiscriminatePrecision` requires each wrong port to miss them.
- **Blocking hazards as failures.** FIFO tests run the open under a 5 s guard, so a regression fails
  with a message instead of hanging the suite.
- **Backends that fail when called** prove a refusal happens before the forward pass.

## 5. Known blind spots of the data

- The act heads recorded in `logits.jsonl` are saturated (every `act_probability` is 1.0), so the
  torch act softmax (D26) is covered by `act_softmax.jsonl` alone.
- Three synthetic `noul` cases in `answers.jsonl` have k ≠ 2, which `SystemOne` cannot produce;
  only `TestAnswerGolden` covers them, and `TestAnswerParity` asserts exactly three skips.
- `sequence.jsonl` never reaches `room == 0`; stub-tokenizer tests pin the budget edges instead.
- The ported `test_local_e2e.py` thresholds are weak: the english checkpoint also lands 8/8 on
  `billing`, and moderation's ≥ 2/3 passes with an empty state.

## 6. Platforms and compilers

- **FMA fusion.** Go fuses `a*b + c` into one `FMADD` on arm64 (and on amd64 under `GOAMD64=v3`),
  rounding once where numpy rounds twice. An explicit `float64(…)` conversion around the product
  forbids the fusion (`calib.ECE`). To check: `GOARCH=arm64 go test -exec qemu-aarch64-static` and
  `GOARCH=arm64 GOOS=darwin go build -a -gcflags='all=-S' ./...` searched for `FMADD`.
- The kernels' pure-Go fallbacks run under `GOARCH=386` and the `purego` tag.
- `-race` needs cgo, so `CGO_ENABLED=0` builds and race runs are separate commands.
- The ONNX backend builds only on D21's platform list; `GOOS=windows go vet` checks the stub.

## 7. README examples

`example_test.go` (package `laya_test`) holds one Example per README code block, so CI compiles
them. `TestREADMEExamples` runs the weight-bound ones locally: it re-executes the test binary once
per Example, since they end on `log.Fatal`, and checks the exit status, a sentinel printed after the
Example returns, and the lines the README promises. It stages the `-dynamo` exports as hard links in
a temporary directory beside them, so `$TMPDIR` may be on another filesystem.
