// Package tokenizer encodes text exactly as the HuggingFace `tokenizers`
// library does for the two pipelines the laya checkpoints ship, in pure Go and
// without cgo (D2).
//
// There are exactly two pipelines, because typed-decisions' tokenizer.json is
// byte-identical to English's:
//
//   - English and typed-decisions: NFC, then a ByteLevel pre-tokenizer
//     (add_prefix_space=false, use_regex=true), then BPE over a 50280-entry
//     vocabulary with no unk token and no byte fallback.
//   - Multilingual: Replace(" " -> U+2581), then a Metaspace pre-tokenizer
//     (prepend_scheme=always, split=true), then BPE over 256000 entries with
//     byte_fallback and fuse_unk both on.
//
// Both are driven entirely by the checkpoint's tokenizer.json, which is the only
// source of truth -- no tokenizer.model, vocab.txt or merges.txt exists in any
// of the three checkpoints.
//
// # Why this is the riskiest package in the port
//
// build_sequence returns marker positions as token *indices* (common.py:79), so
// a single extra or missing token shifts every marker after it and the model
// reads the wrong positions. That produces a confident, plausible, wrong answer
// rather than an error. Every semantic here is therefore pinned by a golden
// vector dumped from the Python (testdata/tokenizer_{en,ml}.jsonl) before the Go
// was written, and the vectors assert token *strings* as well as ids: an id diff
// tells you nothing, while ["a", "b"] against ["ab"] names the stage that broke.
package tokenizer

// Tokenizer is the surface build_sequence needs, and nothing more.
//
// laya calls the tokenizer in exactly one form -- tok(text,
// add_special_tokens=False)["input_ids"] (common.py:63,68,83) -- and splices
// CLS, SEP and the mask marker in by hand as raw ids (common.py:67,76,81), so
// the post_processor both checkpoints carry is dead code for this port. There is
// no decode path: nothing in inference turns ids back into text.
//
// The interface exists so that build_sequence ports against it rather than
// against a concrete type. If the pure-Go implementation cannot reach parity,
// R1's fallback is to put a cgo binding behind these six methods, and nothing
// upstream of the interface changes.
type Tokenizer interface {
	// Encode tokenises text with add_special_tokens=false.
	Encode(text string) []int64

	// MaskToken is the mask literal as a string. build_sequence scrubs it out
	// of the instructions, every option and the serialised state before
	// tokenising (common.py:62,68,83), so callers need the string and not only
	// the id.
	MaskToken() string

	// MaskID marks each option's start (common.py:67). The marker positions
	// build_sequence returns point at these.
	MaskID() int64

	// CLSID opens the sequence (common.py:76).
	CLSID() int64

	// SEPID separates head from options and options from state
	// (common.py:76,81,85).
	SEPID() int64

	// PADID fills short rows when collate_items pads a batch
	// (common.py:223, agent.py:266).
	PADID() int64
}
