package tokenizer

import "unicode/utf8"

// bpe turns one pre-token into ids: HF's BPE::merge_word followed by
// merge_all.
//
// Symbols are seeded per character, not per byte. A character absent from the
// vocabulary takes the fallback path below, and the resulting symbols are
// pushed *before* merging, so byte-fallback tokens can still participate in
// merges.
func (t *HF) bpe(w string) []int64 {
	if w == "" {
		return nil
	}

	var out word
	// pendingUnk tracks a run of unknown characters that fuse_unk collapses
	// into a single token.
	pendingUnk := false
	flushUnk := func() {
		if pendingUnk {
			out.add(int32(t.unkID))
			pendingUnk = false
		}
	}

	for _, r := range w {
		ch := string(r)
		if id, ok := t.vocab[ch]; ok {
			flushUnk()
			out.add(id)
			continue
		}

		// HF gates the whole fallback on unk_token being set: byte_fallback
		// with a null unk_token falls back to nothing and the character is
		// dropped. English is exactly that configuration, which is what makes
		// PLAN.md 1.4 item 3's "silently dropped" true and why Encode
		// sanitises to valid UTF-8 at the API edge.
		if !t.hasUnk {
			continue
		}

		if t.byteFallback {
			if ids, ok := t.byteTokens(ch); ok {
				flushUnk()
				for _, id := range ids {
					out.add(id)
				}
				continue
			}
			// All-or-nothing: if any byte lacks a <0xXX> token the character
			// takes the unk path rather than emitting a partial sequence.
		}

		if t.fuseUnk {
			pendingUnk = true // consecutive unknowns collapse into one
			continue
		}
		out.add(int32(t.unkID))
	}
	flushUnk()

	out.mergeAll(t.merges)
	return out.ids()
}

// byteTokens resolves a character to one <0xXX> id per UTF-8 byte, reporting
// false if any byte has no token.
func (t *HF) byteTokens(ch string) ([]int32, bool) {
	ids := make([]int32, 0, utf8.UTFMax)
	for i := range len(ch) {
		id, ok := t.vocab[byteFallbackToken(ch[i])]
		if !ok {
			return nil, false
		}
		ids = append(ids, id)
	}
	return ids, true
}
