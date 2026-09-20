package tokenizer

import (
	"encoding/json"
	"fmt"
)

// The tokenizer.json shape, restricted to the fields the two laya pipelines
// declare. Unknown fields decode into nothing and are rejected by validate
// rather than ignored, so a checkpoint this port cannot reproduce fails at load
// instead of returning different ids at inference.
type tokenizerJSON struct {
	Version      string           `json:"version"`
	Truncation   json.RawMessage  `json:"truncation"`
	Padding      json.RawMessage  `json:"padding"`
	AddedTokens  []addedTokenJSON `json:"added_tokens"`
	Normalizer   *stageJSON       `json:"normalizer"`
	PreTokenizer *stageJSON       `json:"pre_tokenizer"`
	Model        modelJSON        `json:"model"`
	// post_processor and decoder are deliberately absent: laya always encodes
	// with add_special_tokens=false and never decodes, so both are dead for
	// this port. Both real checkpoints carry them, so they must not be refused.
}

// addedTokenJSON is one entry of "added_tokens".
type addedTokenJSON struct {
	ID         int64  `json:"id"`
	Content    string `json:"content"`
	SingleWord bool   `json:"single_word"`
	LStrip     bool   `json:"lstrip"`
	RStrip     bool   `json:"rstrip"`
	Normalized bool   `json:"normalized"`
	Special    bool   `json:"special"`
}

// stageJSON is a normalizer or pre-tokenizer. The union is small enough that
// one struct with optional fields beats a discriminated decode.
type stageJSON struct {
	Type string `json:"type"`

	// Replace
	Pattern *patternJSON `json:"pattern"`
	Content string       `json:"content"`

	// ByteLevel
	AddPrefixSpace bool `json:"add_prefix_space"`
	UseRegex       bool `json:"use_regex"`

	// Metaspace
	Replacement   string `json:"replacement"`
	PrependScheme string `json:"prepend_scheme"`
	Split         bool   `json:"split"`
}

// patternJSON is HF's Replace pattern: either a literal String or a Regex.
// Multilingual declares String; a Regex would need a different implementation,
// so it is refused rather than approximated.
// file itself -- they are Rust enum variant names, not our naming choice, and
// renaming them would simply fail to decode.
//
//nolint:tagliatelle // HF spells these fields with a leading capital in the
type patternJSON struct {
	String string `json:"String"`
	Regex  string `json:"Regex"`
}

type modelJSON struct {
	Type                    string           `json:"type"`
	Dropout                 *float64         `json:"dropout"`
	UnkToken                *string          `json:"unk_token"`
	ContinuingSubwordPrefix *string          `json:"continuing_subword_prefix"`
	EndOfWordSuffix         *string          `json:"end_of_word_suffix"`
	FuseUnk                 bool             `json:"fuse_unk"`
	ByteFallback            bool             `json:"byte_fallback"`
	IgnoreMerges            bool             `json:"ignore_merges"`
	Vocab                   map[string]int32 `json:"vocab"`
	Merges                  [][2]string      `json:"merges"`
}

// tokenizerConfigJSON is the half of tokenizer_config.json this port reads.
//
// Each token is either a bare string or a serialised AddedToken object, so they
// decode through specialToken. agent.py:21-46 rewrites this file in place to
// patch two upstream quirks; Go reads both shapes instead and never writes.
type tokenizerConfigJSON struct {
	CLSToken  specialToken `json:"cls_token"`
	SEPToken  specialToken `json:"sep_token"`
	PADToken  specialToken `json:"pad_token"`
	MaskToken specialToken `json:"mask_token"`
	UNKToken  specialToken `json:"unk_token"`
}

// specialToken decodes either "foo" or {"content":"foo",...} to the content.
type specialToken struct {
	Content string
}

func (s *specialToken) UnmarshalJSON(b []byte) error {
	var str string
	if err := json.Unmarshal(b, &str); err == nil {
		s.Content = str
		return nil
	}
	var obj struct {
		Content string `json:"content"`
	}
	if err := json.Unmarshal(b, &obj); err != nil {
		return fmt.Errorf("tokenizer: decode a special token: %w", err)
	}
	s.Content = obj.Content
	return nil
}
