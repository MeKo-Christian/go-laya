package tokenizer

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"math"
	"unicode/utf8"
)

// The vocabulary and merge table are most of a tokenizer.json -- on
// multilingual, 256 000 entries and 580 604 pairs in a 34 MB file -- and
// decoding them through encoding/json into map[string]int32 and [][2]string
// allocated a string for every merge side only to look it up and drop it. The
// scanners here read the two values in one pass instead and resolve each merge
// straight from the bytes.
//
// They are held to encoding/json's result rather than to the JSON spec, and the
// differential and fuzz tests in decode_test.go keep the pre-B.1 decode as the
// oracle. Two choices make that tractable:
//
//   - encoding/json still decodes the document, into rawOccurrences, so the
//     whole file is validated before a scanner sees a byte of it;
//   - a string that is not plain -- an escape, a control byte, invalid UTF-8 --
//     is handed to json.Unmarshal on its own, so escapes, surrogate pairs, lone
//     surrogates and U+FFFD replacement come out exactly as they did.

// rawOccurrences captures every occurrence of one object key, in document
// order. encoding/json matches keys case-insensitively and decodes a repeated
// key into the same field again -- a map accumulates, a null resets it, a slice
// is overwritten in place -- so keeping only the last occurrence, as
// json.RawMessage would, is not the same decode.
type rawOccurrences []json.RawMessage

// UnmarshalJSON records one occurrence. encoding/json passes null through
// rather than skipping it, which is what lets a later null reset the value.
func (r *rawOccurrences) UnmarshalJSON(b []byte) error {
	*r = append(*r, bytes.Clone(b))
	return nil
}

// errSyntax is malformed JSON. Production never reaches it -- encoding/json has
// validated the document by then -- but the scanners must still fail cleanly.
var errSyntax = errors.New("malformed JSON")

// decodeFailed prefixes a decode error the way OpenFS prefixes encoding/json's.
// A merge that does not resolve already wraps ErrUnsupported and passes through.
func decodeFailed(err error) error {
	if err == nil || errors.Is(err, ErrUnsupported) {
		return err
	}
	return fmt.Errorf("tokenizer: decode %s: %w", fileTokenizerJSON, err)
}

// decodeVocab is model.vocab as encoding/json decoded it into a
// map[string]int32: nil when absent or null, last key wins.
func decodeVocab(occ rawOccurrences) (map[string]int32, error) {
	if len(occ) == 1 {
		vocab, err := scanVocab(occ[0])
		return vocab, decodeFailed(err)
	}
	// Absent, or repeated. A repeated key accumulates into one map, which
	// encoding/json reproduces by construction; no real checkpoint has one.
	var vocab map[string]int32
	for _, raw := range occ {
		if err := json.Unmarshal(raw, &vocab); err != nil {
			return nil, decodeFailed(fmt.Errorf("model.vocab: %w", err))
		}
	}
	return vocab, nil
}

// decodeMerges is model.merges as encoding/json decoded it into [][2]string,
// resolved to ids. Rank is the position in the array, and on a repeated pair
// the earlier rank wins, as it does in HF.
func decodeMerges(occ rawOccurrences, vocab map[string]int32) (map[[2]int32]merge, error) {
	if len(occ) == 1 {
		merges, err := scanMerges(occ[0], vocab)
		return merges, decodeFailed(err)
	}
	// Absent, or repeated. A repeated key decodes into the same slice, where a
	// null pair keeps whatever an earlier occurrence left in that slot --
	// slack capacity included -- so only encoding/json reproduces it.
	var pairs [][2]string
	for _, raw := range occ {
		if err := json.Unmarshal(raw, &pairs); err != nil {
			return nil, decodeFailed(fmt.Errorf("model.merges: %w", err))
		}
	}
	mt := newMergeTable(vocab, len(pairs))
	for rank, p := range pairs {
		if err := mt.add(rank, []byte(p[0]), []byte(p[1])); err != nil {
			return nil, err
		}
	}
	return mt.out, nil
}

// mergeTable resolves both sides and the result of every merge to ids up
// front, so applying a merge can never fail to find its result.
type mergeTable struct {
	vocab  map[string]int32
	out    map[[2]int32]merge
	joined []byte // reused for every concatenation lookup
}

func newMergeTable(vocab map[string]int32, hint int) *mergeTable {
	return &mergeTable{vocab: vocab, out: make(map[[2]int32]merge, presize(hint))}
}

// maxPresize bounds a map's capacity hint. The real checkpoints stay below it
// (256 000 vocab entries, 580 604 merges), so they are still presized exactly.
const maxPresize = 1 << 20

// presize caps a capacity hint at maxPresize. The scanners' hints count
// separators in the raw value, and tokenizer.json comes from an untrusted Hub
// repo: one token of a million commas would otherwise presize a map ~30 times
// the size of the file before a single id is checked.
func presize(hint int) int {
	return min(hint, maxPresize)
}

// add records the merge at rank. The vocab[string(b)] lookups do not allocate.
func (m *mergeTable) add(rank int, left, right []byte) error {
	l, ok := m.vocab[string(left)]
	if !ok {
		return fmt.Errorf("%w: merge %d has %q on the left, which is not in the vocabulary",
			ErrUnsupported, rank, left)
	}
	r, ok := m.vocab[string(right)]
	if !ok {
		return fmt.Errorf("%w: merge %d has %q on the right, which is not in the vocabulary",
			ErrUnsupported, rank, right)
	}
	m.joined = append(append(m.joined[:0], left...), right...)
	joined, ok := m.vocab[string(m.joined)]
	if !ok {
		// HF's MergeTokenOutOfVocabulary. Resolving it here is what lets
		// the merge loop apply a rule without a fallible lookup.
		return fmt.Errorf("%w: merge %d produces %q, which is not in the vocabulary",
			ErrUnsupported, rank, m.joined)
	}
	key := [2]int32{l, r}
	if _, dup := m.out[key]; dup {
		return nil // the earlier rank wins, as it does in HF
	}
	m.out[key] = merge{rank: int32(rank), newID: joined}
	return nil
}

// scanVocab decodes one vocab value: an object of integer ids, or null.
func scanVocab(raw []byte) (map[string]int32, error) {
	s := scanner{data: raw}
	s.skipSpace()
	if s.null() {
		return nil, s.end("model.vocab")
	}
	if !s.consume('{') {
		return nil, s.typeError("model.vocab", "an object")
	}
	// A hint, not a count: a comma inside a token overcounts by one.
	vocab := make(map[string]int32, presize(bytes.Count(raw, []byte{','})+1))
	s.skipSpace()
	if s.consume('}') {
		return vocab, s.end("model.vocab")
	}
	for {
		s.skipSpace()
		key, err := s.str()
		if err != nil {
			return nil, fmt.Errorf("model.vocab: %w", err)
		}
		s.skipSpace()
		if !s.consume(':') {
			return nil, s.syntaxError("model.vocab")
		}
		s.skipSpace()
		id, err := s.id()
		if err != nil {
			return nil, fmt.Errorf("model.vocab[%q]: %w", key, err)
		}
		vocab[string(key)] = id

		s.skipSpace()
		switch {
		case s.consume(','):
		case s.consume('}'):
			return vocab, s.end("model.vocab")
		default:
			return nil, s.syntaxError("model.vocab")
		}
	}
}

// scanMerges decodes one merges value -- an array of [left, right] string
// pairs, or null -- and resolves it against vocab. A pair decodes the way
// encoding/json fills a [2]string: a missing or null element is "", elements
// past the second are skipped unread, and a non-string element is an error.
func scanMerges(raw []byte, vocab map[string]int32) (map[[2]int32]merge, error) {
	s := scanner{data: raw}
	s.skipSpace()
	// A hint, not a count: every pair closes with one ']', plus the outer one.
	mt := newMergeTable(vocab, bytes.Count(raw, []byte{']'}))
	if s.null() {
		return mt.out, s.end("model.merges")
	}
	if !s.consume('[') {
		return nil, s.typeError("model.merges", "an array")
	}
	s.skipSpace()
	if s.consume(']') {
		return mt.out, s.end("model.merges")
	}
	for rank := 0; ; rank++ {
		s.skipSpace()
		left, right, err := s.pair()
		if err != nil {
			return nil, fmt.Errorf("model.merges[%d]: %w", rank, err)
		}
		if err := mt.add(rank, left, right); err != nil {
			return nil, err
		}

		s.skipSpace()
		switch {
		case s.consume(','):
		case s.consume(']'):
			return mt.out, s.end("model.merges")
		default:
			return nil, s.syntaxError("model.merges")
		}
	}
}

// scanner is a cursor over one JSON value. Every method bounds-checks: in
// production its input has passed encoding/json already, but the fuzz test
// feeds it anything.
type scanner struct {
	data []byte
	pos  int
}

func (s *scanner) skipSpace() {
	for s.pos < len(s.data) {
		switch s.data[s.pos] {
		case ' ', '\t', '\n', '\r':
			s.pos++
		default:
			return
		}
	}
}

func (s *scanner) consume(c byte) bool {
	if s.pos < len(s.data) && s.data[s.pos] == c {
		s.pos++
		return true
	}
	return false
}

func (s *scanner) peek() byte {
	if s.pos < len(s.data) {
		return s.data[s.pos]
	}
	return 0
}

// null consumes a null literal if the input continues with one.
func (s *scanner) null() bool {
	if bytes.HasPrefix(s.data[s.pos:], []byte("null")) {
		s.pos += len("null")
		return true
	}
	return false
}

// end requires that nothing but whitespace follows the value.
func (s *scanner) end(what string) error {
	s.skipSpace()
	if s.pos != len(s.data) {
		return s.syntaxError(what)
	}
	return nil
}

func (s *scanner) syntaxError(what string) error {
	return fmt.Errorf("%s: %w at offset %d", what, errSyntax, s.pos)
}

func (s *scanner) typeError(what, want string) error {
	return fmt.Errorf("%s: want %s at offset %d", what, want, s.pos)
}

// str decodes the string token at the cursor. A plain token -- no escape, no
// control byte, valid UTF-8 -- is returned as a subslice of the input, which
// is what encoding/json's unquote returns for it too. Anything else goes
// through json.Unmarshal so the result is encoding/json's by construction.
func (s *scanner) str() ([]byte, error) {
	if !s.consume('"') {
		return nil, fmt.Errorf("want a string at offset %d", s.pos)
	}
	start := s.pos
	plain := true
	for s.pos < len(s.data) {
		switch c := s.data[s.pos]; {
		case c == '"':
			body := s.data[start:s.pos]
			s.pos++
			if plain && utf8.Valid(body) {
				return body, nil
			}
			var out string
			if err := json.Unmarshal(s.data[start-1:s.pos], &out); err != nil {
				return nil, fmt.Errorf("string at offset %d: %w", start-1, err)
			}
			return []byte(out), nil
		case c == '\\':
			// Skipping the escaped byte is enough to find the closing quote:
			// the hex digits of \uXXXX are never a quote or a backslash.
			plain = false
			s.pos += 2
		case c < ' ':
			plain = false
			s.pos++
		default:
			s.pos++
		}
	}
	return nil, fmt.Errorf("unterminated string at offset %d", start-1)
}

// id decodes a vocabulary id the way encoding/json fills an int32: an integer
// literal in range, or null for zero. A fraction or exponent is an error even
// when the value is integral, because encoding/json parses with ParseInt.
func (s *scanner) id() (int32, error) {
	if s.null() {
		return 0, nil
	}
	start := s.pos
	for s.pos < len(s.data) {
		c := s.data[s.pos]
		if (c < '0' || c > '9') && c != '-' && c != '+' && c != '.' && c != 'e' && c != 'E' {
			break
		}
		s.pos++
	}
	num := s.data[start:s.pos]
	neg := len(num) > 0 && num[0] == '-'
	if neg {
		num = num[1:]
	}
	if len(num) == 0 {
		return 0, fmt.Errorf("want an integer id at offset %d", start)
	}
	var n int64
	for _, c := range num {
		if c < '0' || c > '9' {
			return 0, fmt.Errorf("id %q at offset %d is not an integer", s.data[start:s.pos], start)
		}
		n = n*10 + int64(c-'0')
		if n > math.MaxInt32+1 {
			return 0, fmt.Errorf("id %q at offset %d overflows int32", s.data[start:s.pos], start)
		}
	}
	if neg {
		n = -n
	}
	if n > math.MaxInt32 {
		return 0, fmt.Errorf("id %q at offset %d overflows int32", s.data[start:s.pos], start)
	}
	return int32(n), nil
}

// pair decodes one merge into its two sides, as encoding/json fills a fresh
// [2]string element.
func (s *scanner) pair() (left, right []byte, err error) {
	if s.null() {
		return nil, nil, nil
	}
	if !s.consume('[') {
		return nil, nil, fmt.Errorf("want a [left, right] array at offset %d", s.pos)
	}
	s.skipSpace()
	if s.consume(']') {
		return nil, nil, nil
	}
	for i := 0; ; i++ {
		s.skipSpace()
		switch {
		case i >= 2:
			if err := s.skipValue(); err != nil {
				return nil, nil, err
			}
		case s.null():
		case s.peek() == '"':
			side, err := s.str()
			if err != nil {
				return nil, nil, err
			}
			if i == 0 {
				left = side
			} else {
				right = side
			}
		default:
			return nil, nil, fmt.Errorf("want a string at offset %d", s.pos)
		}

		s.skipSpace()
		switch {
		case s.consume(','):
		case s.consume(']'):
			return left, right, nil
		default:
			return nil, nil, fmt.Errorf("%w at offset %d", errSyntax, s.pos)
		}
	}
}

// skipValue steps over one value of any type without checking its grammar;
// it only has to find where the value ends.
func (s *scanner) skipValue() error {
	depth := 0
	for {
		s.skipSpace()
		if s.pos >= len(s.data) {
			return fmt.Errorf("%w: unexpected end", errSyntax)
		}
		switch c := s.data[s.pos]; c {
		case '"':
			if _, err := s.str(); err != nil {
				return err
			}
		case '[', '{':
			depth++
			s.pos++
		case ']', '}':
			if depth == 0 {
				return fmt.Errorf("%w at offset %d", errSyntax, s.pos)
			}
			depth--
			s.pos++
		case ',', ':':
			if depth == 0 {
				return fmt.Errorf("%w at offset %d", errSyntax, s.pos)
			}
			s.pos++
			continue
		default:
			start := s.pos
			for s.pos < len(s.data) && !isDelimiter(s.data[s.pos]) {
				s.pos++
			}
			if s.pos == start {
				return fmt.Errorf("%w at offset %d", errSyntax, s.pos)
			}
		}
		if depth == 0 {
			return nil
		}
	}
}

// isDelimiter reports a byte that ends a number or literal.
func isDelimiter(c byte) bool {
	switch c {
	case ',', ':', '[', ']', '{', '}', '"', ' ', '\t', '\n', '\r':
		return true
	}
	return false
}
