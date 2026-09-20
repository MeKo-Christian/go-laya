package tokenizer

import "strings"

// Encode tokenises text with add_special_tokens=false, which is the only form
// laya uses (common.py:63,68,83).
//
// The pipeline, in HF's order:
//
//  1. sanitise to valid UTF-8;
//  2. extract added tokens in two phases, normalising what they do not claim;
//  3. pre-tokenise each remaining text segment;
//  4. BPE each piece.
//
// Step 1 is the API edge task 4.1.2 describes, and it is not defensive
// programming. Go strings may hold arbitrary bytes while Python's str cannot,
// so invalid UTF-8 can only ever arrive from the Go side. On English the
// consequence would be silent: its byte-level alphabet has no runes for
// 0xC0, 0xC1 and 0xF5-0xFF, and its BPE model declares unk_token null, so
// those bytes would be dropped rather than replaced -- and a dropped byte
// shifts every marker position after it.
func (t *HF) Encode(text string) []int64 {
	text = strings.ToValidUTF8(text, "�")
	if text == "" {
		return nil
	}

	var ids []int64
	for _, seg := range t.splitAdded(text) {
		if seg.token != nil {
			ids = append(ids, seg.token.id)
			continue
		}
		for _, piece := range t.pretok.preTokenize(seg.text) {
			ids = append(ids, t.bpe(piece)...)
		}
	}
	return ids
}

// Encode must satisfy the interface; asserted here rather than in tokenizer.go
// so the assertion sits with the method that completes it.
var _ Tokenizer = (*HF)(nil)
