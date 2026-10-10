// Package tensor is the dense float32 tensor and its CPU kernels -- dot,
// axpy and GEMM, with AVX2/FMA assembly for dot and axpy on amd64, NEON
// assembly for all three on arm64, and a pure-Go fallback elsewhere -- for
// the pure-Go native backend (PLAN.md M8, D8). It is lifted from
// go-pocket-tts; NOTICE records the provenance and what was changed.
//
// The package reads no weight files. internal/safetensors.ReadHeader stays
// the only safetensors header parser and the gate in front of every weight
// read: the native backend may build a tensor from a file's data only after
// ReadHeader has validated that file. internal/safetensors.File decodes F16
// and F32 data into float32 on the offsets ReadHeader returns (PLAN.md Task
// 8.7); BF16 is not decoded, as no shipped checkpoint stores it.
package tensor
