package tokenizer

import (
	"path/filepath"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// BenchmarkOpen is task 4.3.8's load-time measurement. The multilingual
// tokenizer.json is 34 MB; PLAN.md budgets under a second cold and says to
// measure rather than assume.
func BenchmarkOpen(b *testing.B) {
	root := golden.SkipWithoutModels(b)

	for _, checkpoint := range []string{golden.English, golden.Multilingual} {
		b.Run(checkpoint, func(b *testing.B) {
			dir := filepath.Join(golden.CheckpointDir(root, checkpoint), "tokenizer")
			b.ReportAllocs()
			for b.Loop() {
				if _, err := Open(dir); err != nil {
					b.Fatalf("Open: %v", err)
				}
			}
		})
	}
}

// BenchmarkEncode measures the steady-state cost, which is what build_sequence
// pays three times per question (common.py:63,68,83).
func BenchmarkEncode(b *testing.B) {
	root := golden.SkipWithoutModels(b)

	// A realistic serialised state rather than a short phrase: this is what
	// serialize_state feeds the tokenizer.
	const state = `{"subject": "invoice 4411", "body": "I was charged twice for ` +
		`invoice 4411, please refund it today.", "customer": {"id": 8812, "tier": ` +
		`"gold", "since": "2019-03-02"}, "attachments": ["receipt.pdf"], "amount": 42.5}`

	for _, checkpoint := range []string{golden.English, golden.Multilingual} {
		b.Run(checkpoint, func(b *testing.B) {
			tok := openCheckpoint(b, root, checkpoint)
			b.ReportAllocs()
			b.SetBytes(int64(len(state)))
			for b.Loop() {
				if ids := tok.Encode(state); len(ids) == 0 {
					b.Fatal("Encode returned nothing")
				}
			}
		})
	}
}
