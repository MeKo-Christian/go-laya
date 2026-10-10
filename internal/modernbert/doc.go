// Package modernbert holds the ModernBERT and mmBERT building blocks of the
// pure-Go native backend (PLAN.md M8, D8): the bias-free LayerNorm (Task 8.2)
// and, as Tasks 8.3-8.6 land, GeGLU, the fused QKV, the attention masks and
// RoPE, and the layer loop. Each block is tested against the real transformers
// modules through testdata/ops.json, which scripts/dump_modernbert_ops.py
// writes.
//
// The backend.Backend implementation is not here; it is Task 8.10's
// internal/backend/native, built on these blocks. Nor does this package load
// weights: Task 8.7 decides how they get in.
package modernbert
