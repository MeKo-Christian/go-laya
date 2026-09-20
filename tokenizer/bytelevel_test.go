package tokenizer

import (
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// TestByteLevelGolden is the scanner against real Oniguruma output, all 103
// cases, with no checkpoint present. pretok_en records the pre-tokenizer's
// pieces after the NFC normalizer, which is the order the pipeline runs in.
func TestByteLevelGolden(t *testing.T) {
	for _, rec := range golden.Load(t, "pretok_en") {
		var want pretokCase
		rec.Unmarshal(t, &want)

		got := preTokenizeByteLevel(want.Normalized)
		if len(got) != len(want.Pretokens) {
			t.Errorf("%s: %q\n got %d pieces %q\nwant %d pieces %q",
				rec.Name, want.Text, len(got), got, len(want.Pretokens), want.Pretokens)
			continue
		}
		for i := range got {
			if got[i] != want.Pretokens[i] {
				t.Errorf("%s: %q piece %d = %q, want %q\n got %q\nwant %q",
					rec.Name, want.Text, i, got[i], want.Pretokens[i], got, want.Pretokens)
				break
			}
		}
	}
}

// TestByteLevelGiveBack pins the one rule in the scanner that cannot be read
// off the pattern without care. `\s+(?!\S)` is greedy and then backtracks, so a
// whitespace run followed by a non-space gives its last character back -- but
// only when the run is at least two long, and the character it gives back is
// absorbed by the next piece only if that piece starts with a literal U+0020.
// The optional prefix in ` ?\p{L}+` is a space, not "any whitespace", which is
// why a tab run leaves a stray single tab behind and a space run does not.
func TestByteLevelGiveBack(t *testing.T) {
	for _, c := range []struct {
		name string
		in   string
		want []string
	}{
		{"spaces absorbed by the next word", "a   b", []string{"a", "ĠĠ", "Ġb"}},
		{"single space is not given back", "a b", []string{"a", "Ġb"}},
		{"run reaching the end is taken whole", "a   ", []string{"a", "ĠĠĠ"}},
		{"tab given back is not absorbed", "a\t\tb", []string{"a", "ĉ", "ĉ", "b"}},
		{"whole string is whitespace", "   ", []string{"ĠĠĠ"}},
		{"empty", "", nil},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := preTokenizeByteLevel(c.in); !equalStrings(got, c.want) {
				t.Errorf("preTokenizeByteLevel(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestByteLevelContractions: the seven literals are lowercase-only, so "DON'T"
// splits on the apostrophe as punctuation while "don't" keeps "'t" together.
func TestByteLevelContractions(t *testing.T) {
	for _, c := range []struct {
		in   string
		want []string
	}{
		{"don't", []string{"don", "'t"}},
		{"DON'T", []string{"DON", "'", "T"}},
		{" 's", []string{"Ġ'", "s"}},
		{"we'll", []string{"we", "'ll"}},
	} {
		if got := preTokenizeByteLevel(c.in); !equalStrings(got, c.want) {
			t.Errorf("preTokenizeByteLevel(%q) = %q, want %q", c.in, got, c.want)
		}
	}
}

// TestByteLevelWhitespaceIsUnicode is the probe that settled PLAN.md 4.3.4's
// claim. Oniguruma's \s here is Unicode White_Space, not Ruby's ASCII default:
// NEL and IDEOGRAPHIC SPACE each split per the give-back rule instead of being
// swallowed by the greedy ` ?[^\s\p{L}\p{N}]+` alternative, and NBSP does not
// join an adjacent punctuation run.
func TestByteLevelWhitespaceIsUnicode(t *testing.T) {
	for _, c := range []struct {
		name string
		in   string
		want int // piece count between the two letters
	}{
		{"NBSP", "a  b", 4},
		{"EM SPACE", "a  b", 4},
		{"NEL", "a\u0085\u0085b", 4},
		{"IDEOGRAPHIC SPACE", "a　　b", 4},
		{"NBSP then punctuation", "a .b", 4},
	} {
		t.Run(c.name, func(t *testing.T) {
			if got := preTokenizeByteLevel(c.in); len(got) != c.want {
				t.Errorf("preTokenizeByteLevel(%q) = %q (%d pieces), want %d",
					c.in, got, len(got), c.want)
			}
		})
	}
}

// TestByteEncodeRoundTrips: the table is a bijection over all 256 bytes, which
// is what lets 4.5.1's fuzz invariant compare a re-joined encoding back to its
// input.
func TestByteEncodeRoundTrips(t *testing.T) {
	var all strings.Builder
	for b := range 256 {
		all.WriteByte(byte(b))
	}
	enc := byteEncode(all.String())
	dec, err := byteDecode(enc)
	if err != nil {
		t.Fatalf("byteDecode: %v", err)
	}
	if dec != all.String() {
		t.Errorf("round trip lost bytes: got %q", dec)
	}
	if n := len([]rune(enc)); n != 256 {
		t.Errorf("byteEncode produced %d runes over 256 bytes, want 256", n)
	}
}

func equalStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
