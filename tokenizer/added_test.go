package tokenizer

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

func mustOpenMini(tb testing.TB, name string) *HF {
	tb.Helper()

	tok, err := Open(filepath.Join("testdata", name))
	if err != nil {
		tb.Fatalf("Open %s: %v", name, err)
	}
	return tok
}

// renderSegments prints a split as "text|<token>|text" so a failure shows the
// boundaries rather than a slice of structs.
func renderSegments(segs []segment) string {
	var b strings.Builder
	for i, s := range segs {
		if i > 0 {
			b.WriteByte('|')
		}
		if s.token != nil {
			b.WriteString("<" + s.token.content + ">")
			continue
		}
		b.WriteString(s.text)
	}
	return b.String()
}

// TestAddedSplitPhases covers HF's two-phase contract on the synthetic
// fixtures, so it runs in CI. mini_en mirrors English: [MASK] is
// normalized:false with lstrip, while the space runs and [unused0] are
// normalized:true.
func TestAddedSplitPhases(t *testing.T) {
	tok := mustOpenMini(t, "mini_en")

	for _, c := range []struct {
		name string
		in   string
		want string
	}{
		{"lstrip beats the space-run token", "foo  [MASK]", "foo|<[MASK]>"},
		{"lstrip over a single space", "a [MASK]", "a|<[MASK]>"},
		{"phase 2 matches a space run", "a  b", "a|<  >|b"},
		{"leftmost longest picks three over two", "a   b", "a|<   >|b"},
		{"phase 2 token mid text", "x[unused0]y", "x|<[unused0]>|y"},
		{"no added token at all", "plain", "plain"},
		{"adjacent added tokens", "[unused0][unused0]", "<[unused0]>|<[unused0]>"},
		{"empty", "", ""},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := renderSegments(tok.splitAdded(c.in)); got != c.want {
				t.Errorf("splitAdded(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestLStripTakesWhiteSpaceNotFormat is the NBSP/ZWSP pair from PLAN.md 4.3.2.
// lstrip extends over Unicode White_Space, so NBSP is swallowed; ZWSP is
// category Cf and is not, so it survives as text.
func TestLStripTakesWhiteSpaceNotFormat(t *testing.T) {
	tok := mustOpenMini(t, "mini_en")

	if got := renderSegments(tok.splitAdded("a [MASK]")); got != "a|<[MASK]>" {
		t.Errorf("NBSP before [MASK]: got %q, want %q", got, "a|<[MASK]>")
	}
	if got := renderSegments(tok.splitAdded("a\u200b[MASK]")); got != "a\u200b|<[MASK]>" {
		t.Errorf("ZWSP before [MASK]: got %q, want %q", got, "a\u200b|<[MASK]>")
	}
}

// TestLStripStopsAtThePreviousMatch is the clamp HF applies as
// max(trimmed, start_offset). It is unreachable on both real checkpoints --
// only the mask tokens lstrip, and no added token ends in whitespace -- so the
// synthetic fixture is the only place it can be tested. It is two lines of
// implementation whose absence produces overlapping spans.
func TestLStripStopsAtThePreviousMatch(t *testing.T) {
	tok := mustOpenMini(t, "mini_en")

	// "  " is an added token in mini_en and [MASK] lstrips. The clamp keeps
	// [MASK] from reaching back through a match that was already made.
	got := renderSegments(tok.splitAdded("x  [MASK]"))
	if strings.Count(got, "<") != 1 {
		t.Errorf("splitAdded(%q) = %q; lstrip must not consume a preceding match", "x  [MASK]", got)
	}
}

// TestPhaseOrderOnMultilingual: every multilingual added token is
// normalized:false, so all of them match on the raw string -- before Replace
// turns spaces into U+2581. The run tokens depend on it.
func TestPhaseOrderOnMultilingual(t *testing.T) {
	tok := mustOpenMini(t, "mini_ml")

	// The text segments come back normalized -- phase 2 normalizes whatever
	// phase 1 did not claim -- so the space in " b" is already U+2581 here.
	for _, c := range []struct{ in, want string }{
		{"a\n b", "a|<\n>|\u2581b"},
		{"a▁▁b", "a|<▁▁>|b"},
		{"<start_of_turn>x", "<<start_of_turn>>|x"},
	} {
		if got := renderSegments(tok.splitAdded(c.in)); got != c.want {
			t.Errorf("splitAdded(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestAddedSplitOnTheRealCheckpoints runs PLAN.md 4.3.2's acceptance cases
// with the real added-token tables, where the id values are meaningful.
func TestAddedSplitOnTheRealCheckpoints(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	t.Run("english", func(t *testing.T) {
		tok := openCheckpoint(t, root, golden.English)
		for _, c := range []struct{ in, want string }{
			{"foo  [MASK]", "foo|<[MASK]>"},
			{"a [MASK]", "a|<[MASK]>"},
			{"a\u200b[MASK]", "a\u200b|<[MASK]>"},
			// 49 spaces: the table tops out at 24, so leftmost-longest takes
			// two 24-runs and leaves one space for the pre-tokenizer.
			{"a" + strings.Repeat(" ", 49) + "b", "a|<" + strings.Repeat(" ", 24) + ">|<" +
				strings.Repeat(" ", 24) + ">| b"},
		} {
			if got := renderSegments(tok.splitAdded(c.in)); got != c.want {
				t.Errorf("splitAdded(%q)\n got %q\nwant %q", c.in, got, c.want)
			}
		}
	})

	t.Run("multilingual", func(t *testing.T) {
		tok := openCheckpoint(t, root, golden.Multilingual)
		if got := renderSegments(tok.splitAdded("a\n<mask>")); got != "a|<\n>|<<mask>>" {
			t.Errorf("splitAdded(%q) = %q", "a\n<mask>", got)
		}
	})
}
