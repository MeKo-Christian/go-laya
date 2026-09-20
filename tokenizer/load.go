package tokenizer

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
)

// ErrUnsupported reports a tokenizer.json this package cannot reproduce
// faithfully. It is deliberately a load-time failure: a tokenizer that ignores
// a field it does not understand keeps working and returns different ids, and
// because marker positions are token indices, different ids mean a confident
// wrong answer rather than an error (R1).
var ErrUnsupported = errors.New("tokenizer: unsupported tokenizer.json")

// fileTokenizerJSON and fileTokenizerConfig are the two names HF writes and
// laya ships. agent.py:174 looks for exactly this pair in a tokenizer/ subdir.
const (
	// #nosec G101 -- these are the two filenames HF writes, not credentials.
	// gosec matches on the substring "token".
	fileTokenizerJSON = "tokenizer.json"
	// #nosec G101 -- likewise a filename.
	fileTokenizerConfig = "tokenizer_config.json"
)

// maxTokenID is the largest id a tokenizer.json may declare. indexVocab
// allocates a dense table of highest+1 strings, so without a ceiling the
// allocation is a function of one number in the file rather than of the file's
// size: a two-kilobyte checkpoint declaring 2147483647 asks for 32 GB, and one
// declaring more than that wraps the int32 conversion. The largest vocabulary
// in production is multilingual's 256000; this leaves sixteen times that and
// caps the table at about 67 MB.
const maxTokenID = 1 << 22

// merge is one BPE merge rule. newID is resolved at load, so applying a merge
// can never fail to find its result -- HF rejects a merge whose concatenation
// is absent from the vocabulary (MergeTokenOutOfVocabulary) and so does Open.
type merge struct {
	rank  int32
	newID int32
}

// HF is a tokenizer loaded from a checkpoint's tokenizer.json and
// tokenizer_config.json.
type HF struct {
	vocab   map[string]int32
	idToTok []string // dense, indexed by id; "" where an id is unused
	merges  map[[2]int32]merge

	norm   normalizer
	pretok preTokenizer
	added  []addedToken
	// The two AddedVocabulary phases: normalized:false patterns matched
	// against the raw string, normalized:true against the normalized segments.
	phase1 addedMatcher
	phase2 addedMatcher

	byteFallback bool
	fuseUnk      bool
	unkID        int64
	hasUnk       bool

	specials specialIDs
	// vocabSize is max(len(vocab), highest added id + 1). Added tokens can sit
	// above the contiguous vocabulary -- on English 88 of them do, which is why
	// len(vocab) is 50280 while the encoder config says 50368.
	vocabSize int64
}

// preTokenizer is the stage between the normalizer and BPE.
type preTokenizer interface {
	preTokenize(segment string) []string
}

// byteLevel adapts the scanner to the preTokenizer interface.
type byteLevel struct{}

func (byteLevel) preTokenize(segment string) []string { return preTokenizeByteLevel(segment) }

// Open loads the tokenizer in dir, which must hold tokenizer.json and
// tokenizer_config.json -- the layout every laya checkpoint ships under
// tokenizer/.
func Open(dir string) (*HF, error) {
	return OpenFS(os.DirFS(dir))
}

// OpenFS loads from any filesystem, which is what makes the synthetic fixtures
// testable and what internal/hub will hand a cache directory in M6.
func OpenFS(fsys fs.FS) (*HF, error) {
	raw, err := fs.ReadFile(fsys, fileTokenizerJSON)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", fileTokenizerJSON, err)
	}
	var doc tokenizerJSON
	if err := json.Unmarshal(raw, &doc); err != nil {
		return nil, fmt.Errorf("tokenizer: decode %s: %w", fileTokenizerJSON, err)
	}

	cfgRaw, err := fs.ReadFile(fsys, fileTokenizerConfig)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", fileTokenizerConfig, err)
	}
	var cfg tokenizerConfigJSON
	if err := json.Unmarshal(cfgRaw, &cfg); err != nil {
		return nil, fmt.Errorf("tokenizer: decode %s: %w", fileTokenizerConfig, err)
	}

	return build(&doc, &cfg)
}

func build(doc *tokenizerJSON, cfg *tokenizerConfigJSON) (*HF, error) {
	if err := validate(doc); err != nil {
		return nil, err
	}
	if err := validateIDs(doc); err != nil {
		return nil, err
	}

	t := &HF{
		vocab:        doc.Model.Vocab,
		byteFallback: doc.Model.ByteFallback,
		fuseUnk:      doc.Model.FuseUnk,
	}
	var err error
	if t.norm, err = buildNormalizer(doc.Normalizer); err != nil {
		return nil, err
	}
	if t.pretok, err = buildPreTokenizer(doc.PreTokenizer); err != nil {
		return nil, err
	}
	if t.merges, err = buildMerges(doc.Model.Merges, t.vocab); err != nil {
		return nil, err
	}
	if err := t.resolveSpecials(doc, cfg); err != nil {
		return nil, err
	}
	t.buildAdded(doc.AddedTokens)
	t.buildMatchers()
	t.indexVocab()
	return t, nil
}

// validate refuses every declared behaviour this package does not implement.
func validate(doc *tokenizerJSON) error {
	m := &doc.Model
	switch {
	case m.Type != "BPE":
		return fmt.Errorf("%w: model type %q; only BPE is implemented", ErrUnsupported, m.Type)
	case m.IgnoreMerges:
		// Both real checkpoints set this false. True would short-circuit the
		// merge loop for whole tokens already in the vocabulary, which changes
		// ids -- plausibly, which is the dangerous kind.
		return fmt.Errorf("%w: ignore_merges is true", ErrUnsupported)
	case m.Dropout != nil && *m.Dropout != 0:
		return fmt.Errorf("%w: dropout %v is a training-time behaviour", ErrUnsupported, *m.Dropout)
	case m.ContinuingSubwordPrefix != nil && *m.ContinuingSubwordPrefix != "":
		return fmt.Errorf("%w: continuing_subword_prefix %q", ErrUnsupported, *m.ContinuingSubwordPrefix)
	case m.EndOfWordSuffix != nil && *m.EndOfWordSuffix != "":
		return fmt.Errorf("%w: end_of_word_suffix %q", ErrUnsupported, *m.EndOfWordSuffix)
	case !isJSONNull(doc.Truncation):
		// laya truncates by slicing ids itself (common.py:68,73,75,84,86), so a
		// tokenizer that also truncates would cut twice.
		return fmt.Errorf("%w: truncation is configured; laya truncates by slicing", ErrUnsupported)
	case !isJSONNull(doc.Padding):
		return fmt.Errorf("%w: padding is configured; collate_items pads", ErrUnsupported)
	}
	return nil
}

func isJSONNull(raw json.RawMessage) bool {
	return len(raw) == 0 || string(raw) == "null"
}

// validateIDs bounds every id before anything indexes or sizes a table with
// one. It runs before indexVocab because that is the amplifier: it turns the
// single largest id in the file into an allocation, and a negative one into a
// slice write at a negative index.
func validateIDs(doc *tokenizerJSON) error {
	for tok, id := range doc.Model.Vocab {
		if id < 0 || int64(id) > maxTokenID {
			return fmt.Errorf("%w: vocabulary entry %q has id %d, outside [0, %d]",
				ErrUnsupported, tok, id, maxTokenID)
		}
	}
	for _, a := range doc.AddedTokens {
		if a.ID < 0 || a.ID > maxTokenID {
			return fmt.Errorf("%w: added token %q has id %d, outside [0, %d]",
				ErrUnsupported, a.Content, a.ID, maxTokenID)
		}
	}
	return nil
}

func buildNormalizer(s *stageJSON) (normalizer, error) {
	if s == nil {
		return nil, fmt.Errorf("%w: no normalizer declared", ErrUnsupported)
	}
	switch s.Type {
	case "NFC":
		return nfcNormalizer{}, nil
	case "Replace":
		if s.Pattern == nil || s.Pattern.String == "" {
			return nil, fmt.Errorf("%w: Replace needs a String pattern; Regex is not implemented", ErrUnsupported)
		}
		return replaceNormalizer{pattern: s.Pattern.String, content: s.Content}, nil
	default:
		return nil, fmt.Errorf("%w: normalizer %q", ErrUnsupported, s.Type)
	}
}

func buildPreTokenizer(s *stageJSON) (preTokenizer, error) {
	if s == nil {
		return nil, fmt.Errorf("%w: no pre_tokenizer declared", ErrUnsupported)
	}
	switch s.Type {
	case "ByteLevel":
		if !s.UseRegex {
			// Without the pattern, ByteLevel maps bytes and splits nothing,
			// which is a different tokenizer.
			return nil, fmt.Errorf("%w: ByteLevel with use_regex false", ErrUnsupported)
		}
		if s.AddPrefixSpace {
			return nil, fmt.Errorf("%w: ByteLevel with add_prefix_space true", ErrUnsupported)
		}
		return byteLevel{}, nil
	case "Metaspace":
		if s.PrependScheme != "always" {
			return nil, fmt.Errorf("%w: Metaspace prepend_scheme %q", ErrUnsupported, s.PrependScheme)
		}
		if s.Replacement == "" {
			// splitMergedWithNext searches for the replacement and advances by
			// its length, so an empty one is an infinite loop on the first
			// Encode rather than a load error.
			return nil, fmt.Errorf("%w: Metaspace with an empty replacement", ErrUnsupported)
		}
		return metaspace{replacement: s.Replacement, prependAlways: true, split: s.Split}, nil
	default:
		return nil, fmt.Errorf("%w: pre_tokenizer %q", ErrUnsupported, s.Type)
	}
}

// buildMerges resolves both sides and the result of every merge to ids up
// front. Rank is the position in the file, which is what orders the merge loop.
func buildMerges(pairs [][2]string, vocab map[string]int32) (map[[2]int32]merge, error) {
	out := make(map[[2]int32]merge, len(pairs))
	for rank, p := range pairs {
		left, ok := vocab[p[0]]
		if !ok {
			return nil, fmt.Errorf("%w: merge %d has %q on the left, which is not in the vocabulary",
				ErrUnsupported, rank, p[0])
		}
		right, ok := vocab[p[1]]
		if !ok {
			return nil, fmt.Errorf("%w: merge %d has %q on the right, which is not in the vocabulary",
				ErrUnsupported, rank, p[1])
		}
		joined, ok := vocab[p[0]+p[1]]
		if !ok {
			// HF's MergeTokenOutOfVocabulary. Resolving it here is what lets
			// the merge loop apply a rule without a fallible lookup.
			return nil, fmt.Errorf("%w: merge %d produces %q, which is not in the vocabulary",
				ErrUnsupported, rank, p[0]+p[1])
		}
		key := [2]int32{left, right}
		if _, dup := out[key]; dup {
			continue // the earlier rank wins, as it does in HF
		}
		out[key] = merge{rank: int32(rank), newID: joined}
	}
	return out, nil
}

// indexVocab builds the dense id -> token table and settles vocabSize.
//
// Added tokens are folded in, not just model.vocab. On English the five
// specials and the 83 [unusedN] entries sit *above* the contiguous vocabulary,
// at 50280-50367, so a table built from model.vocab alone stops at 50279:
// IDToToken([MASK]) would return "" and VocabSize() would be 50280 rather than
// the 50368 the encoder config declares. Multilingual hides this -- all 249 of
// its added tokens are already inside model.vocab -- which is exactly why it
// needs asserting on English.
//
// validateIDs has already bounded every id to [0, maxTokenID], so neither the
// indexing below nor highest+1 can go out of range.
func (t *HF) indexVocab() {
	highest := int32(-1)
	for _, id := range t.vocab {
		if id > highest {
			highest = id
		}
	}
	for _, a := range t.added {
		if int32(a.id) > highest {
			highest = int32(a.id)
		}
	}

	t.idToTok = make([]string, highest+1)
	for tok, id := range t.vocab {
		t.idToTok[id] = tok
	}
	for _, a := range t.added {
		t.idToTok[a.id] = a.content
	}
	t.vocabSize = int64(highest) + 1
}

// byteFallbackToken is the "<0xXX>" spelling HF uses for a raw byte.
func byteFallbackToken(b byte) string {
	const hex = "0123456789ABCDEF"
	return string([]byte{'<', '0', 'x', hex[b>>4], hex[b&0xF], '>'})
}
