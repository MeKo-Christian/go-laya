package tokenizer

import (
	"crypto/sha256"
	"fmt"
	"slices"
	"strings"
	"testing"
	"unicode"

	"golang.org/x/text/unicode/norm"
)

// TestStaleCCCTable pins the allowlist TestDifferentialCorpus accepts NFC
// mismatches on. A codepoint dropped from it turns a decided deviation back
// into a failure; one added widens what the diff stays silent about, so either
// change has to come with a fresh probe run and a new provenance line.
func TestStaleCCCTable(t *testing.T) {
	if got := len(staleCCC); got != 108 {
		t.Fatalf("len(staleCCC) = %d, want 108 (scripts/probe_ccc.py at tokenizers 0.23.2)", got)
	}
	if !slices.IsSorted(staleCCC[:]) {
		t.Error("staleCCC is not sorted; hasStaleCCC binary-searches it")
	}
	if len(slices.Compact(slices.Clone(staleCCC[:]))) != len(staleCCC) {
		t.Error("staleCCC has a duplicate")
	}

	// The whole list, as `jq -r '.codepoints[]' build/ccc_disagreement.json |
	// sha256sum` prints it. Swapping one member for another keeps the size and
	// can keep every spot check below; this cannot.
	var list strings.Builder
	for _, r := range staleCCC {
		fmt.Fprintf(&list, "U+%04X\n", r)
	}
	const want = "2162d1d3c2569b5a0229f1324e9bc4f5b4869681cefefd68904d225d77fcc7cf"
	if got := fmt.Sprintf("%x", sha256.Sum256([]byte(list.String()))); got != want {
		t.Errorf("staleCCC hashes to %s, want %s (the probe output's)", got, want)
	}

	// First, last and a few from each range the probe's docstring names.
	for _, r := range []rune{0x07FD, 0x0898, 0x08D3, 0x1AC0, 0x1ACE, 0x1DF6, 0x10D24, 0x10EFF, 0x1E4EF} {
		if !hasStaleCCC(string(r)) {
			t.Errorf("U+%04X missing from staleCCC", r)
		}
	}

	// The probe's five method artefacts, an ordinary combining mark that both
	// tables agree on, and a starter.
	for _, r := range []rune{0x0334, 0x0340, 0x0341, 0x0343, 0x0344, 0x0301, 'a'} {
		if hasStaleCCC(string(r)) {
			t.Errorf("U+%04X is in staleCCC, but is not a disagreement the probe confirmed", r)
		}
	}

	// staleStarterNFC splits at the members and puts them back unchanged, which
	// is exact only if none of them decomposes or takes part in a composition.
	// Every canonical composite is the inverse of a canonical decomposition, so
	// it is enough that no codepoint decomposes to anything holding a member.
	for r := rune(0); r <= unicode.MaxRune; r++ {
		if d := norm.NFD.PropertiesString(string(r)).Decomposition(); d != nil && hasStaleCCC(string(d)) {
			t.Errorf("U+%04X decomposes to %q, which holds a staleCCC codepoint", r, d)
		}
	}

	// Each member is a mark that Go's NFC reorders after U+0334 (ccc 1), and the
	// Rust tables do not. A member with ccc ≤ 1 here could not cause that.
	for _, r := range staleCCC {
		if ccc := norm.NFC.PropertiesString(string(r)).CCC(); ccc <= 1 {
			t.Errorf("U+%04X has ccc %d in golang.org/x/text, want > 1", r, ccc)
		}
	}
}
