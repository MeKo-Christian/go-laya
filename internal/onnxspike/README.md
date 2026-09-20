# `internal/onnxspike` — Spikes S2 and S3

Throwaway. **M6 deletes this package** and replaces it with `internal/backend`.

It answers two questions. **S2:** can the S1 ONNX export be executed from Go **without CGO**, and
is `github.com/shota3506/onnxruntime-purego` stable enough to build on (risk R5)? **S3:** how long
does one forward pass actually take on a CPU (risk R3)? See §0 D5 and D6 for the answers, and §3 S2
and S3 for the evidence.

## Running it

Both tests skip unless an ONNX Runtime shared library is present, and `TestForwardPass`
additionally needs the S1 export. CI has neither, so CI skips them — `AGENTS.md`'s rule that
the pipeline never needs model weights.

```bash
just spike-onnx            # both tests, CGO-free
just spike-onnx-race 200   # the finalizer hunt, under the race detector
just bench-onnx            # the S3 latency sweep -- about 20 minutes
```

| Input              | Default                                       | Override                                |
| ------------------ | --------------------------------------------- | --------------------------------------- |
| ORT shared library | `/usr/local/lib`, `/usr/lib`, Homebrew        | `LAYA_ORT_LIB`, then `ORT_LIBRARY_PATH` |
| The `.onnx` export | `build/onnx/` (where `export_onnx.py` writes) | `LAYA_ONNX_DIR`                         |

A variable that is set but points nowhere is an error, not a fallback: quietly running
against a different runtime than the caller named would defeat the purpose of pinning one.

Regenerate `testdata/forward_pass.json` with the pinned reference environment
(`scripts/README.md`), never by hand:

```bash
.venv-ref/bin/python scripts/export_onnx.py --checkpoint english --dynamo --reuse \
    --suffix=-dynamo --no-check-attn --fixture internal/onnxspike/testdata/forward_pass.json
just fmt   # prettier owns the file's layout; the generator's is not the committed one
```

## Reading the benchmark

`bench-onnx` is two one-dimensional sweeps sharing a corner rather than a cross product: threads vary
at one fixed shape, shapes vary at one fixed thread count. A full product runs for hours and the
extra cells say nothing.

Three details that are not decoration:

- **`-benchtime=5x`, not a duration.** An iteration costs 0.15–20 s here, so letting `go test`
  calibrate `b.N` against a wall clock either overruns the timeout or returns a single sample.
- **p50 and p90, not the mean.** Upstream quotes p50, and a 15 W part throttles: the gap between the
  two is itself a result. Run the sweep more than once — run-to-run variation reached **2×** on the
  worst cell, which is why `BENCHMARKS.md` reports the minimum across three sweeps rather than one.
- **`BenchmarkSessionLoad` gets one process per checkpoint.** RSS growth is only meaningful the first
  time a session is built; a shared process reports the second checkpoint as roughly free.

## Two things worth knowing before M6 reuses any of this

**Every `*Value` must be closed explicitly.** The binding registers a `runtime.AddCleanup`
safety net on each one, and that net is not safe: `Runtime.Close` writes `r.apiFuncs = nil`
(`runtime.go:178`) while the GC's cleanup goroutine reads it in `releaseValuePtr`
(`value.go:152`), with no synchronisation anywhere in the package. `TestValueCleanup/abandoned`
reproduces the data race on the first iteration; `TestValueCleanup/closed` is green over 200
runs, because `Close` calls `cleanup.Stop()` and takes the finalizer out of play. The
abandoned case is opt-in via `LAYA_ONNXSPIKE_FINALIZER=1` so it does not fail `just ci`.

**`-race` implies CGO.** `CGO_ENABLED=0 go test -race` is refused by the toolchain, so the two
halves of the claim are two commands: `CGO_ENABLED=0` proves the _build_ needs no CGO, and a
separate `-race` run does the concurrency work. Only the race runtime pulls in cgo — the
binding itself never does.

## Sessions are created from a path, never from a reader

The exports exceed the 2 GB protobuf limit and are split into a graph file plus an
`.onnx.data` blob that ONNX Runtime resolves _relative to the model path_.
`NewSessionFromReader` has no path to resolve it against.
