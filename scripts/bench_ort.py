#!/usr/bin/env python3
"""Time one forward pass through Python's onnxruntime, the way BenchmarkForward does from Go.

PLAN.md Task 6.8.3. Spike S3 recorded the same graph running 2.7x slower from Python
(ORT 1.30.0) than from Go (ORT 1.23.0) on ``english``, but no script for the Python side
was ever kept, so the number could not be re-measured. This is that script, written to
match ``benchShapeRun`` in ``internal/backend/onnx/bench_test.go`` input for input:

* ids ``1 + (i*7919) % 997`` (``benchIDs``), an all-ones attention mask, markers at
  ``1 + 3i``, every marker real, qtype 0;
* one untimed warm-up run, then ``--reps`` timed runs, outputs fetched each time;
* p50/p90 as nearest rank on the sorted latencies (``quantile``), in milliseconds;
* ``intra_op_num_threads`` set, CPU provider, the default graph optimisation level --
  which is also what the C API, and so the Go binding, uses when nothing sets it.

It needs only numpy and onnxruntime, so any venv with the onnxruntime under test runs it:

    .venv-ref/bin/python scripts/bench_ort.py --model build/onnx/laya-english-dynamo.onnx
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


def bench_inputs(batch: int, seq: int, k: int) -> dict[str, np.ndarray]:
    """The five graph inputs newBenchInputs builds for one shape."""
    ids = 1 + (np.arange(batch * seq, dtype=np.int64) * 7919) % 997
    marker_pos = np.tile(1 + 3 * np.arange(k, dtype=np.int64), (batch, 1))
    return {
        "input_ids": ids.reshape(batch, seq),
        "attention_mask": np.ones((batch, seq), dtype=np.int64),
        "marker_pos": marker_pos,
        "marker_mask": np.ones((batch, k), dtype=bool),
        "qtype": np.zeros(batch, dtype=np.int64),
    }


def quantile(latencies: list[float], q: float) -> float:
    """Nearest rank on an already-sorted list, exactly as bench_test.go's quantile."""
    return latencies[min(int(q * len(latencies)), len(latencies) - 1)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--model", type=Path, required=True, help="an S1 export (.onnx)")
    ap.add_argument("--threads", type=int, default=8, help="intra_op_num_threads (default 8)")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--reps", type=int, default=10, help="timed runs after one warm-up")
    ap.add_argument(
        "--optimized-model",
        type=Path,
        default=None,
        help="also write the graph as ORT optimised it here (PLAN.md Task 6.8.3)",
    )
    args = ap.parse_args()

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = args.threads
    if args.optimized_model is not None:
        opts.optimized_model_filepath = args.optimized_model.as_posix()

    start = time.perf_counter()
    sess = ort.InferenceSession(
        args.model.as_posix(), sess_options=opts, providers=["CPUExecutionProvider"]
    )
    load_ms = (time.perf_counter() - start) * 1e3

    feed = bench_inputs(args.batch, args.seq, args.k)
    outputs = [o.name for o in sess.get_outputs()]

    # ORT pays arena allocation and graph optimisation on the first run at a shape.
    sess.run(outputs, feed)

    latencies = []
    for _ in range(args.reps):
        t0 = time.perf_counter()
        sess.run(outputs, feed)
        latencies.append((time.perf_counter() - t0) * 1e3)
    latencies.sort()

    print(
        json.dumps(
            {
                "onnxruntime": ort.__version__,
                "model": args.model.name,
                "threads": args.threads,
                "graph_optimization_level": str(opts.graph_optimization_level),
                "shape": f"b{args.batch}_s{args.seq}_k{args.k}",
                "reps": args.reps,
                "load_ms": round(load_ms, 1),
                "p50_ms": round(quantile(latencies, 0.5), 1),
                "p90_ms": round(quantile(latencies, 0.9), 1),
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
