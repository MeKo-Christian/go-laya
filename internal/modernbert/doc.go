// Package modernbert holds the ModernBERT and mmBERT encoder of the pure-Go
// native backend (PLAN.md M8, D8): the bias-free LayerNorm (Task 8.2), the
// GeGLU MLP (8.3), the attention with its fused QKV, sliding-window mask and
// per-layer-type RoPE (8.4, 8.5), and their assembly into the encoder layer
// and the Encoder, with the layer plan Config reads from config.json (8.5.1).
// Each is tested against the real transformers modules through
// testdata/ops.json, which scripts/dump_modernbert_ops.py writes.
//
// The backend.Backend implementation is not here; it is Task 8.10's
// internal/backend/native, built on these blocks. Nor does this package load
// weights: internal/backend/native decodes them from model.safetensors with
// internal/safetensors (Task 8.7) and hands them to the constructors here.
package modernbert
