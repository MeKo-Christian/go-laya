# Repository Guidelines

## Project Overview

**go-laya** is a Go port of the Python `laya` decision engine: a non-autoregressive
System 1 model that answers typed questions (`choice`, `score`, `noul`) about arbitrary
state in a single forward pass and returns calibrated probabilities instead of generated
text.

- **Module:** `github.com/MeKo-Christian/go-laya`
- **Go version:** 1.26+
- **Upstream:** [`NandhaKishorM/laya`](https://github.com/NandhaKishorM/laya) 0.3.4, Apache-2.0. See `NOTICE`.
- **The plan is the spec.** `PLAN.md` drives the work; `docs/INVARIANTS.md` lists the 74
  behaviours a test must assert, each citing the Python line that defines it; `docs/API.md`
  gives the target Go types and the exact JSON shapes Python emits.

This is a **parity port**. The measure of correctness is not "is this good Go" but "does
this agree with `original/`". Where upstream is inconsistent, reproducing the inconsistency
is usually right and deviating from it always needs a line in `PLAN.md` saying so.

## Build & Development Commands

```bash
just build       # go build -v ./...
just test        # go test -v -count=1 ./...
just test-race   # with the race detector
just lint        # golangci-lint run
just lint-fix    # golangci-lint run --fix
just lint-md     # markdownlint (checker only)
just fmt         # treefmt: gofumpt + gci, prettier, shfmt + shellcheck
just fmt-check   # fail if anything is unformatted
just cover       # coverage profile + HTML report
just check-tidy  # go mod tidy, then fail on a dirty go.mod/go.sum
just vuln        # govulncheck ./...
just check       # developer loop: test + lint + cover
just ci          # what CI runs: fmt-check + lint-md + test-race + lint + check-tidy
```

`just ci` is the single command that mirrors the pipeline. Run it before you claim
anything is done.

## Coding Style & Naming

- Format with `gofumpt`; import grouping is `gci`'s job (standard, default, localmodule).
  Both run under `just fmt`. Do not hand-order imports.
- **GoDoc on every exported symbol.** Enforced by revive's `exported` rule on the root
  package; `internal/` and `cmd/` are exempt.
- Files are capped at **1500 lines** (revive `file-length-limit`). Split rather than grow.
- Markdown is formatted by **prettier only**. `markdownlint` runs as a checker and never
  with `--fix` — chaining both as formatters can fail to converge under `--fail-on-change`.
- Tests live beside their source as `*_test.go`, in the same package. `testpackage` is
  disabled on purpose: parity tests need the unexported internals.

## The frozen upstream: `original/`

`original/` holds the upstream Python, moved there unmodified.

- **Never edit anything under `original/`.** It is the parity reference and the source of
  the golden vectors. A change there silently invalidates every test that cites it.
- It is excluded from `treefmt`, from `golangci-lint`, and from the Markdown lint.
- To regenerate parity data, use `scripts/dump_python_parity.py` against the pinned
  reference environment (`scripts/requirements-ref.txt`) — never by hand.

## Testing Requirements

- **TDD.** Write the failing test, watch it fail, implement the minimum, watch it pass.
  `PLAN.md` states this per task and it is not decorative: most of these behaviours are
  ones where a wrong implementation returns a plausible answer rather than an error.
- Golden vectors in `testdata/` are checked in. **CI must never need Python or model
  weights.** Anything that does is gated behind `testing.Short()` and an env var
  (`LAYA_MODELS`).
- Regenerating `testdata/` is a reviewed diff, never a drive-by. `TestGoldenProvenance`
  asserts the recorded tokenizer/transformers versions still match what the Go code
  targets; a silent upstream tokenizer change is the realistic way parity regresses.
- Assert **ids and token strings**, not ids alone. An id diff tells you nothing;
  `["▁▁", "x"]` versus `["▁x"]` tells you which stage broke.

## Security

`laya.Open("someone/their-model")` downloads and parses a remote artefact, so the
deserialization path is an attack surface. CI enforces the cheap half:

- no `encoding/gob` and no `os/exec` outside tests,
- no filesystem `replace` directives in `go.mod`,
- `govulncheck` on reachable code paths, gitleaks over both the diff and the working tree.

The half CI cannot check is yours: verify safetensors/ONNX headers before use, check each
download's ETag/sha against Hub metadata, pin the runtime version, and make every download
cancellable via `context.Context`.

## Before Committing

```bash
just fmt
just ci
```

**Never revert uncommitted changes you did not create** — they cannot be recovered and
discarding them is data loss.

## Commit & Pull Request Guidelines

- **Conventional Commits:** `feat:`, `fix:`, `chore:`, `refactor:`, `test:`, `docs:`,
  `ci:`, `build:`, `style:`, `perf:`.
- Subject ~50 characters, blank line, body wrapped at 72. Say **why**, not what — the diff
  already says what.
- Technical content is written in **English**: commit messages, PR titles and bodies, issue
  descriptions, code comments, documentation.
- One logical change per commit. A formatting sweep and a behaviour change do not belong
  in the same commit.
- PRs state the change and its motivation, link the `PLAN.md` task, and include the output
  of the verification command that backs the claim.

## Keeping `PLAN.md` honest

- Tick a task only against a command you ran and can quote. A passing suite is not evidence
  for a task whose test it does not contain.
- Partly done stays unticked, with `(YYYY-MM-DD) — partial: <what remains>`.
- Work discovered along the way gets added as a new task under the right milestone. Do not
  widen a task's scope silently, and do not reinterpret an acceptance criterion to make it
  reachable.
