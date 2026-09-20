package tokenizer

import (
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// corpusTexts is every input string the two golden corpora hold, with and
// without the leading space common.py:68 adds. Fuzz seeds and cross-checks
// both start from it: the corpus is the set of inputs already known to be
// interesting, so seeding from anywhere else wastes the fuzzer's time.
func corpusTexts(tb testing.TB) []string {
	tb.Helper()

	seen := map[string]bool{}
	var out []string
	for _, fixture := range []string{"pretok_en", "pretok_ml"} {
		for _, rec := range golden.Load(tb, fixture) {
			var c pretokCase
			rec.Unmarshal(tb, &c)
			for _, s := range []string{c.Text, " " + c.Text, c.Normalized} {
				if !seen[s] {
					seen[s] = true
					out = append(out, s)
				}
			}
		}
	}
	return out
}
