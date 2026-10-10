// Package modernbert holds the ModernBERT and mmBERT building blocks of the
// pure-Go native backend (PLAN.md M8, D8): the bias-free LayerNorm (Task 8.2),
// the GeGLU MLP (8.3), the attention with its fused QKV, sliding-window mask
// and per-layer-type RoPE (8.4, 8.5) and, as the rest of M8 lands, the layer
// loop. Each block is tested against the real transformers
// modules through testdata/ops.json, which scripts/dump_modernbert_ops.py
// writes.
//
// The backend.Backend implementation is not here; it is Task 8.10's
// internal/backend/native, built on these blocks. Nor does this package load
// weights: Task 8.7 decides how they get in.
package modernbert
