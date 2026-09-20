package tokenizer

import (
	"unicode"
	"unicode/utf8"
)

// segment is one piece of the input after added-token extraction. A segment
// with a token is already resolved to an id; one without is ordinary text,
// normalized and waiting for the pre-tokenizer.
type segment struct {
	text  string
	token *addedToken
}

// addedMatcher finds added tokens in a string, leftmost-longest.
//
// A map keyed on content, probed from the longest possible length down, rather
// than an Aho-Corasick automaton: there are at most 249 patterns of at most 31
// bytes, so this is a bounded number of hash lookups per position and it is far
// easier to see that it implements leftmost-longest correctly. If profiling
// ever says otherwise, the interface here is one method wide.
type addedMatcher struct {
	byPattern map[string]*addedToken
	maxLen    int
}

func newAddedMatcher(tokens []*addedToken) addedMatcher {
	m := addedMatcher{byPattern: make(map[string]*addedToken, len(tokens))}
	for _, t := range tokens {
		if len(t.pattern) > m.maxLen {
			m.maxLen = len(t.pattern)
		}
		m.byPattern[t.pattern] = t
	}
	return m
}

// longestAt returns the longest added token matching at pos, or nil.
func (m addedMatcher) longestAt(s string, pos int) (*addedToken, int) {
	if len(m.byPattern) == 0 {
		return nil, 0
	}
	high := min(pos+m.maxLen, len(s))
	for end := high; end > pos; end-- {
		if t, ok := m.byPattern[s[pos:end]]; ok {
			return t, end - pos
		}
	}
	return nil, 0
}

// splitAdded implements HF's AddedVocabulary::extract_and_normalize.
//
// Phase 1 matches the normalized:false tokens against the raw string. Phase 2
// normalizes each segment phase 1 did not claim and matches the
// normalized:true tokens against the result. The partition is on `normalized`
// and not on `special`: the two coincide on both laya checkpoints, so keying on
// `special` would pass every test here and silently break on a checkpoint where
// they differ.
//
// The order is load-bearing in opposite directions on the two checkpoints. On
// English, [MASK] is phase 1 with lstrip, so it must swallow the spaces before
// the phase-2 space-run tokens can claim them. On multilingual, every added
// token is phase 1, which is what lets the U+2581 runs match the raw text
// before the Replace normalizer turns spaces into U+2581.
//
// Because phase 2 runs per segment, its lstrip and rstrip can never reach
// across a phase-1 boundary. That falls out of the structure rather than
// needing a check, which is the reason for splitting and recursing rather than
// making one pass with two matchers.
func (t *HF) splitAdded(text string) []segment {
	var out []segment
	for _, seg := range t.phase1.split(text, nil) {
		if seg.token != nil {
			out = append(out, seg)
			continue
		}
		normalized := t.norm.normalize(seg.text)
		out = append(out, t.phase2.split(normalized, nil)...)
	}
	return out
}

// split cuts s at every added token, appending to out.
func (m addedMatcher) split(s string, out []segment) []segment {
	// pending is the start of the text not yet emitted. It doubles as the
	// floor for lstrip: HF clamps the extension with max(trimmed, start_offset)
	// where start_offset is the end of the previous match, and that is exactly
	// where pending sits.
	pending := 0
	for i := 0; i < len(s); {
		tok, n := m.longestAt(s, i)
		if tok == nil {
			_, size := utf8.DecodeRuneInString(s[i:])
			i += size
			continue
		}

		start, end := i, i+n
		if tok.lstrip {
			start = trimSpaceLeftBounded(s, pending, start)
		}
		if tok.rstrip {
			end = trimSpaceRight(s, end)
		}
		if tok.singleWord && !isWordBounded(s, start, end) {
			_, size := utf8.DecodeRuneInString(s[i:])
			i += size
			continue
		}

		if start > pending {
			out = append(out, segment{text: s[pending:start]})
		}
		out = append(out, segment{text: s[start:end], token: tok})
		pending, i = end, end
	}
	if pending < len(s) {
		out = append(out, segment{text: s[pending:]})
	}
	return out
}

// trimSpaceLeftBounded walks back from at over White_Space, never past floor.
//
// unicode.IsSpace is exactly Unicode White_Space, which is also what Rust's
// char::is_whitespace implements, so this needs no table of its own. It is why
// NBSP before a mask is swallowed while ZWSP -- category Cf, not White_Space --
// is not.
func trimSpaceLeftBounded(s string, floor, at int) int {
	for at > floor {
		r, size := utf8.DecodeLastRuneInString(s[floor:at])
		if !unicode.IsSpace(r) {
			break
		}
		at -= size
	}
	return at
}

// trimSpaceRight extends over White_Space to the right.
//
// HF's rstrip has no clamp against the following match, unlike lstrip. That
// asymmetry looks like an upstream oversight, but it is unset on both laya
// checkpoints, so this reproduces the behaviour rather than improving on it.
func trimSpaceRight(s string, at int) int {
	for at < len(s) {
		r, size := utf8.DecodeRuneInString(s[at:])
		if !unicode.IsSpace(r) {
			break
		}
		at += size
	}
	return at
}

// isWordBounded reports whether [start,end) is not glued to a word character on
// either side. single_word is false on every token of both checkpoints;
// implemented because a loader that silently ignored it would be the same class
// of bug the loader's refusals exist to prevent.
func isWordBounded(s string, start, end int) bool {
	if start > 0 {
		if r, _ := utf8.DecodeLastRuneInString(s[:start]); isWordRune(r) {
			return false
		}
	}
	if end < len(s) {
		if r, _ := utf8.DecodeRuneInString(s[end:]); isWordRune(r) {
			return false
		}
	}
	return true
}

func isWordRune(r rune) bool {
	return isLetter(r) || isNumber(r) || r == '_'
}

// buildMatchers partitions the added tokens into the two phases and normalizes
// the phase-2 patterns, which is what HF does at load. It is a no-op under NFC
// for both checkpoints, but its absence would be a silent failure on the first
// checkpoint whose added tokens are not already normalized.
func (t *HF) buildMatchers() {
	var raw, norm []*addedToken
	for i := range t.added {
		a := &t.added[i]
		a.pattern = a.content
		if a.normalized {
			a.pattern = t.norm.normalize(a.content)
			norm = append(norm, a)
			continue
		}
		raw = append(raw, a)
	}
	t.phase1 = newAddedMatcher(raw)
	t.phase2 = newAddedMatcher(norm)
}
