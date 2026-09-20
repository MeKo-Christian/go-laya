package tokenizer

import (
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

func mlMetaspace() metaspace {
	return metaspace{replacement: "▁", prependAlways: true, split: true}
}

// TestMetaspaceGolden runs all 103 multilingual stage vectors. No checkpoint
// needed: like the scanner, Metaspace is a pure string-to-pieces function.
func TestMetaspaceGolden(t *testing.T) {
	m := mlMetaspace()
	for _, rec := range golden.Load(t, "pretok_ml") {
		var want pretokCase
		rec.Unmarshal(t, &want)

		if got := m.preTokenize(want.Normalized); !equalStrings(got, want.Pretokens) {
			t.Errorf("%s: %q\n got %q\nwant %q", rec.Name, want.Text, got, want.Pretokens)
		}
	}
}

// TestMetaspacePrependGuard is PLAN.md 1.4 item 1 at stage level. prepend_scheme
// "always" prepends the replacement only when the segment does not already
// start with it -- the guard, not a test on the raw character, which is the bug
// gomlx's implementation has (it checks text[0] != ' '). This guard is the
// entire reason tok(" x") == tok("x") on multilingual: the Replace normalizer
// has already turned the leading space into U+2581, so the prepend suppresses
// itself.
func TestMetaspacePrependGuard(t *testing.T) {
	m := mlMetaspace()
	for _, c := range []struct {
		name string
		in   string
		want []string
	}{
		{"prepends when absent", "ab", []string{"▁ab"}},
		{"suppressed when present", "▁ab", []string{"▁ab"}},
		{"leading space already replaced", "▁x", []string{"▁x"}},
		{"bare replacement", "▁", []string{"▁"}},
		{"doubled replacement", "a▁▁b", []string{"▁a", "▁", "▁b"}},
		{"empty stays empty", "", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := m.preTokenize(c.in); !equalStrings(got, c.want) {
				t.Errorf("preTokenize(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestMetaspaceEmptyIsNoPieces is the trap the review caught: HF drops empty
// splits before Metaspace runs, so prepend_scheme "always" never fires on an
// empty string and Encode("") is [] on multilingual too. An implementation that
// normalises and then prepends over the whole string returns a lone U+2581 and
// is wrong -- silently, since a single extra leading token shifts every marker.
func TestMetaspaceEmptyIsNoPieces(t *testing.T) {
	if got := mlMetaspace().preTokenize(""); len(got) != 0 {
		t.Errorf("preTokenize(%q) = %q, want no pieces", "", got)
	}
}

// TestMetaspaceSplitsMergedWithNext: the delimiter attaches to what follows it,
// never to what precedes it, and text before the first delimiter is its own
// piece.
func TestMetaspaceSplitsMergedWithNext(t *testing.T) {
	m := mlMetaspace()
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"▁a▁b", []string{"▁a", "▁b"}},
		{"▁a▁", []string{"▁a", "▁"}},
		{"▁▁a", []string{"▁", "▁a"}},
	} {
		if got := m.preTokenize(c.in); !equalStrings(got, c.want) {
			t.Errorf("preTokenize(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}
