package tokenizer

import "fmt"

// specialIDs are the five ids and one string build_sequence needs.
type specialIDs struct {
	maskToken string
	mask      int64
	cls       int64
	sep       int64
	pad       int64
	unk       int64
}

// addedToken is one entry of HF's AddedVocabulary, kept in the form the matcher
// wants rather than the form the file stores.
type addedToken struct {
	content    string
	id         int64
	lstrip     bool
	rstrip     bool
	singleWord bool
	// normalized selects the phase. HF partitions on this field, not on
	// `special` -- the two coincide on both laya checkpoints, so keying on
	// `special` would pass every test here and break on the next checkpoint.
	normalized bool
}

// resolveSpecials implements task 4.2: the token strings come from
// tokenizer_config.json, the ids from tokenizer.json.
//
// The tokenizer config wins over the encoder config, and that is not a style
// preference. multilingual/encoder/config.json says cls_token_id 1 while its
// tokenizer says <bos> = 2, and 2 is what the weights were trained with,
// because the reference code reads tok.cls_token_id (agent.py:174, 261).
// Trusting the encoder config would open every multilingual sequence with the
// wrong token.
func (t *HF) resolveSpecials(doc *tokenizerJSON, cfg *tokenizerConfigJSON) error {
	lookup := func(role, content string) (int64, error) {
		if content == "" {
			return 0, fmt.Errorf("%w: %s is missing from %s", ErrUnsupported, role, fileTokenizerConfig)
		}
		// added_tokens first: on English the five specials live only there,
		// above the contiguous vocabulary.
		for _, a := range doc.AddedTokens {
			if a.Content == content {
				return a.ID, nil
			}
		}
		if id, ok := doc.Model.Vocab[content]; ok {
			return int64(id), nil
		}
		return 0, fmt.Errorf("%w: %s %q is in %s but in neither added_tokens nor the vocabulary",
			ErrUnsupported, role, content, fileTokenizerConfig)
	}

	var err error
	s := specialIDs{maskToken: cfg.MaskToken.Content}
	for _, f := range []struct {
		role    string
		content string
		into    *int64
	}{
		{"mask_token", cfg.MaskToken.Content, &s.mask},
		{"cls_token", cfg.CLSToken.Content, &s.cls},
		{"sep_token", cfg.SEPToken.Content, &s.sep},
		{"pad_token", cfg.PADToken.Content, &s.pad},
	} {
		if *f.into, err = lookup(f.role, f.content); err != nil {
			return err
		}
	}

	// unk is optional: English declares [UNK] in its config but its BPE model
	// has unk_token null, so nothing is ever emitted as unknown -- an
	// out-of-alphabet byte is silently dropped instead (PLAN.md 1.4 item 3),
	// which is why Encode sanitises to valid UTF-8 at the edge.
	if doc.Model.UnkToken != nil && *doc.Model.UnkToken != "" {
		if s.unk, err = lookup("unk_token", *doc.Model.UnkToken); err != nil {
			return err
		}
		t.unkID, t.hasUnk = s.unk, true
	} else if cfg.UNKToken.Content != "" {
		if s.unk, err = lookup("unk_token", cfg.UNKToken.Content); err != nil {
			return err
		}
	}

	t.specials = s
	return nil
}

// buildAdded converts the file's added_tokens into matcher form. The order the
// file lists them in is not meaningful; the matcher resolves overlaps by
// leftmost-longest, not by declaration order.
func (t *HF) buildAdded(tokens []addedTokenJSON) {
	t.added = make([]addedToken, 0, len(tokens))
	for _, a := range tokens {
		t.added = append(t.added, addedToken{
			content:    a.Content,
			id:         a.ID,
			lstrip:     a.LStrip,
			rstrip:     a.RStrip,
			singleWord: a.SingleWord,
			normalized: a.Normalized,
		})
	}
}

// MaskToken returns the mask literal build_sequence scrubs user text with.
func (t *HF) MaskToken() string { return t.specials.maskToken }

// MaskID is the id that marks each option's start.
func (t *HF) MaskID() int64 { return t.specials.mask }

// CLSID opens the sequence.
func (t *HF) CLSID() int64 { return t.specials.cls }

// SEPID separates the sections of the sequence.
func (t *HF) SEPID() int64 { return t.specials.sep }

// PADID fills short rows in a collated batch.
func (t *HF) PADID() int64 { return t.specials.pad }

// UNKID is the unknown-token id. It is meaningful only on multilingual:
// English's BPE model declares unk_token null, so nothing is ever emitted as
// unknown.
func (t *HF) UNKID() int64 { return t.specials.unk }

// VocabSize is one past the highest id the tokenizer can emit.
//
// It is not len(vocab). Added tokens may sit above the contiguous vocabulary --
// on English 88 of them do, so the vocabulary holds 50280 entries while the
// highest id is 50367 and the encoder config declares 50368. Using len(vocab)
// would make task 4.5.1's "ids < VocabSize" invariant fail on the very first
// [MASK].
func (t *HF) VocabSize() int64 { return t.vocabSize }

// IDToToken returns the token string for an id, or "" if the id is unused.
// There is no decode path in inference; this exists so a golden failure can
// print ["a","b"] against ["ab"] and name the stage that broke.
func (t *HF) IDToToken(id int64) string {
	if id < 0 || id >= int64(len(t.idToTok)) {
		return ""
	}
	return t.idToTok[id]
}
