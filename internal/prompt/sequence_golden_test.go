package prompt

import (
	"encoding/json"
	"fmt"
	"path/filepath"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// sequenceCase is one record of testdata/sequence.jsonl.
type sequenceCase struct {
	Checkpoint   string          `json:"checkpoint"`
	State        json.RawMessage `json:"state"`
	Q            json.RawMessage `json:"q"`
	MaxLen       int             `json:"max_len"`
	HeadMaxLen   int             `json:"head_max_len"`
	OptionOrder  []int           `json:"option_order"`
	TruncateLeft bool            `json:"truncate_left"`
	IDs          []int64         `json:"ids"`
	Markers      []int64         `json:"markers"`
	NOptions     int             `json:"n_options"`
	MarkersLost  int             `json:"markers_lost"`
}

// TestBuildSequenceGolden is Task 5.2.1: all 45 recorded build_sequence calls,
// three checkpoints by the matrix (qtype, criteria, instructions, state,
// max_len, head_max_len, option_order, truncate_left), ids and markers exact.
//
// It needs the real tokenizers, so it is gated like tokenizer's own golden
// test. The arithmetic it rests on is pinned without them by the stub-tokenizer
// tests in sequence_test.go, which CI does run.
func TestBuildSequenceGolden(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	toks := map[string]*tokenizer.HF{}
	for _, rec := range golden.Load(t, "sequence") {
		var c sequenceCase
		rec.Unmarshal(t, &c)

		tok, ok := toks[c.Checkpoint]
		if !ok {
			var err error
			tok, err = tokenizer.Open(filepath.Join(golden.CheckpointDir(root, c.Checkpoint), "tokenizer"))
			if err != nil {
				t.Fatalf("open %s: %v", c.Checkpoint, err)
			}
			toks[c.Checkpoint] = tok
		}

		t.Run(rec.Name, func(t *testing.T) {
			state, err := jsonx.Decode(c.State)
			if err != nil {
				t.Fatalf("decode the recorded state: %v", err)
			}
			q := internalFromGolden(t, c.Q)

			ids, markers, err := BuildSequence(tok, state, q, c.MaxLen, c.HeadMaxLen, c.OptionOrder, c.TruncateLeft)
			if err != nil {
				t.Fatalf("BuildSequence: %v", err)
			}

			if !slices.Equal(ids, c.IDs) {
				t.Errorf("ids differ (len %d, want %d)\n%s", len(ids), len(c.IDs), divergence(tok, ids, c.IDs))
			}
			if !slices.Equal(markers, c.Markers) {
				t.Errorf("markers = %v\n    want %v", markers, c.Markers)
			}
			if lost := c.NOptions - len(markers); lost != c.MarkersLost {
				t.Errorf("markers lost = %d, want %d", lost, c.MarkersLost)
			}
		})
	}
}

// divergence shows both sequences as ids and token strings around the first
// difference. An id diff alone says nothing; the token strings say whether the
// head, an option or the state went wrong (AGENTS.md).
func divergence(tok *tokenizer.HF, got, want []int64) string {
	i := 0
	for i < len(got) && i < len(want) && got[i] == want[i] {
		i++
	}
	lo := max(0, i-3)
	window := func(s []int64) string {
		var b strings.Builder
		for _, id := range s[min(lo, len(s)):min(i+5, len(s))] {
			fmt.Fprintf(&b, " %d:%q", id, tok.IDToToken(id))
		}
		return b.String()
	}
	return fmt.Sprintf("first divergence at %d\n got:%s\nwant:%s", i, window(got), window(want))
}
