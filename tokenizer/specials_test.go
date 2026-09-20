package tokenizer

import (
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
)

// TestSpecialTokenIDs is task 4.2.4 against PLAN.md 1.4's table. These six
// values are spliced into every sequence by hand (common.py:67,76,81), so a
// wrong one corrupts every prompt rather than one case.
func TestSpecialTokenIDs(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	for _, c := range []struct {
		checkpoint                         string
		mask                               string
		maskID, clsID, sepID, padID, unkID int64
		vocabSize                          int64
	}{
		{golden.English, "[MASK]", 50284, 50281, 50282, 50283, 50280, 50368},
		{golden.TypedDecisions, "[MASK]", 50284, 50281, 50282, 50283, 50280, 50368},
		{golden.Multilingual, "<mask>", 4, 2, 1, 0, 3, 256000},
	} {
		t.Run(c.checkpoint, func(t *testing.T) {
			tok := openCheckpoint(t, root, c.checkpoint)

			for _, f := range []struct {
				name      string
				got, want int64
			}{
				{"MaskID", tok.MaskID(), c.maskID},
				{"CLSID", tok.CLSID(), c.clsID},
				{"SEPID", tok.SEPID(), c.sepID},
				{"PADID", tok.PADID(), c.padID},
				{"UNKID", tok.UNKID(), c.unkID},
				{"VocabSize", tok.VocabSize(), c.vocabSize},
			} {
				if f.got != f.want {
					t.Errorf("%s() = %d, want %d", f.name, f.got, f.want)
				}
			}
			if got := tok.MaskToken(); got != c.mask {
				t.Errorf("MaskToken() = %q, want %q", got, c.mask)
			}
		})
	}
}

// TestSpecialsComeFromTheTokenizerNotTheEncoder is task 4.2.2, and it is worth
// its own test because the two configs genuinely disagree:
// multilingual/encoder/config.json declares cls_token_id 1 while its tokenizer
// declares <bos> = 2. The reference reads tok.cls_token_id, so 2 is what the
// weights were trained with -- taking the encoder's word for it would open
// every multilingual sequence with <eos>.
func TestSpecialsComeFromTheTokenizerNotTheEncoder(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	tok := openCheckpoint(t, root, golden.Multilingual)
	if got := tok.CLSID(); got != 2 {
		t.Errorf("CLSID() = %d, want 2 (the tokenizer's <bos>, not the encoder config's 1)", got)
	}
	if got := tok.IDToToken(tok.CLSID()); got != "<bos>" {
		t.Errorf("IDToToken(CLSID()) = %q, want %q", got, "<bos>")
	}
}

// TestVocabSizeIsNotVocabLength: on English 88 added tokens sit above the
// contiguous vocabulary, so len(vocab) is 50280 while the highest emittable id
// is 50367. Returning len(vocab) would make the fuzz invariant fail on [MASK].
func TestVocabSizeIsNotVocabLength(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	tok := openCheckpoint(t, root, golden.English)
	if int64(len(tok.vocab)) == tok.VocabSize() {
		t.Fatal("VocabSize() is len(vocab); it must cover added tokens above it")
	}
	if tok.MaskID() >= tok.VocabSize() {
		t.Errorf("MaskID() %d is not below VocabSize() %d", tok.MaskID(), tok.VocabSize())
	}
}
