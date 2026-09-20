package tokenizer

import "strings"

// metaspace is HF's Metaspace pre-tokenizer, the multilingual half of the port:
// {"replacement":"▁","prepend_scheme":"always","split":true}.
//
// It is far smaller than ByteLevel because there is no pattern to scan. The
// subtlety is entirely in the prepend guard and in what "split" means.
type metaspace struct {
	replacement   string
	prependAlways bool
	split         bool
}

// preTokenize turns one normalized segment into pieces.
//
// An empty segment yields no pieces at all. HF drops empty splits before the
// pre-tokenizer sees them, so prepend_scheme "always" never fires on one, and
// Encode("") is [] on multilingual as well as English. Prepending here instead
// would add a lone U+2581 and shift every marker position by one.
func (m metaspace) preTokenize(seg string) []string {
	if seg == "" {
		return nil
	}

	// HF's Metaspace replaces spaces itself as well as the Replace normalizer
	// having done so. Idempotent here -- kept because the fidelity is free and
	// its absence would be a silent dependency on normalizer ordering.
	seg = strings.ReplaceAll(seg, " ", m.replacement)

	// prepend_scheme "always" prepends the replacement unless the segment
	// already starts with it. The guard tests the replacement, not a raw space:
	// by this point the Replace normalizer has turned a leading space into
	// U+2581, which is why tok(" x") == tok("x") on this checkpoint and not on
	// the other (PLAN.md 1.4 item 1). Testing seg[0] != ' ' instead -- the bug
	// in gomlx's implementation -- would prepend a second U+2581 to every input
	// that starts with a space.
	if m.prependAlways && !strings.HasPrefix(seg, m.replacement) {
		seg = m.replacement + seg
	}

	if !m.split {
		return []string{seg}
	}
	return splitMergedWithNext(seg, m.replacement)
}

// splitMergedWithNext cuts before every occurrence of sep, so the delimiter
// attaches to what follows it and never to what precedes it. Text ahead of the
// first delimiter is a piece of its own.
func splitMergedWithNext(s, sep string) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); {
		idx := strings.Index(s[i:], sep)
		if idx < 0 {
			break
		}
		cut := i + idx
		if cut > start {
			out = append(out, s[start:cut])
			start = cut
		}
		i = cut + len(sep)
	}
	if start < len(s) {
		out = append(out, s[start:])
	}
	return out
}
