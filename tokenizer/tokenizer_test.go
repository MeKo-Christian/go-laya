package tokenizer

import "testing"

// stub is the proof that the interface is implementable by something other than
// the real loader. Task 4.1.1's point is that build_sequence ports 1:1 against
// this interface and never against a concrete type, which is what keeps D2
// reversible: if the pure-Go tokenizer cannot reach parity, R1's fallback swaps
// daulet/tokenizers in behind these six methods and nothing upstream of it moves.
type stub struct {
	ids  []int64
	mask string
}

func (s stub) Encode(string) []int64 { return s.ids }
func (s stub) MaskToken() string     { return s.mask }
func (s stub) MaskID() int64         { return 50284 }
func (s stub) CLSID() int64          { return 50281 }
func (s stub) SEPID() int64          { return 50282 }
func (s stub) PADID() int64          { return 50283 }

var _ Tokenizer = stub{}

// TestInterfaceShape pins the five ids build_sequence splices in by hand
// (common.py:67,76,81 and agent.py:266) plus the mask string it scrubs with
// (common.py:62,68,83). Nothing else is in the interface because nothing else is
// called: laya never decodes, never asks for offsets and never passes
// add_special_tokens=true.
func TestInterfaceShape(t *testing.T) {
	var tok Tokenizer = stub{ids: []int64{1, 2, 3}, mask: "[MASK]"}

	if got := tok.Encode("anything"); len(got) != 3 {
		t.Errorf("Encode() = %v, want 3 ids", got)
	}
	if got := tok.MaskToken(); got != "[MASK]" {
		t.Errorf("MaskToken() = %q, want %q", got, "[MASK]")
	}
	for _, c := range []struct {
		name string
		got  int64
		want int64
	}{
		{"MaskID", tok.MaskID(), 50284},
		{"CLSID", tok.CLSID(), 50281},
		{"SEPID", tok.SEPID(), 50282},
		{"PADID", tok.PADID(), 50283},
	} {
		if c.got != c.want {
			t.Errorf("%s() = %d, want %d", c.name, c.got, c.want)
		}
	}
}
