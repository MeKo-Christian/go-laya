package prompt

import (
	"fmt"
	"slices"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// stubTokenizer is a tokenizer.Tokenizer whose encoding is trivially
// predictable: one id per whitespace-separated word, assigned on first sight.
// It exists so that a failure in BuildSequence names the budget arithmetic
// rather than the tokenizer -- the real tokenizers are asserted in their own
// package, and the fixture-backed test is Task 5.2's.
type stubTokenizer struct {
	ids   map[string]int64
	words []string
}

const (
	stubPAD int64 = iota
	stubCLS
	stubSEP
	stubMASK
	stubFirstWord
)

const stubMaskToken = "[MASK]"

func newStub() *stubTokenizer {
	return &stubTokenizer{ids: map[string]int64{}}
}

func (s *stubTokenizer) Encode(text string) []int64 {
	fields := strings.Fields(text)
	out := make([]int64, 0, len(fields))
	for _, w := range fields {
		id, ok := s.ids[w]
		if !ok {
			id = stubFirstWord + int64(len(s.words))
			s.ids[w] = id
			s.words = append(s.words, w)
		}
		out = append(out, id)
	}
	return out
}

func (s *stubTokenizer) MaskToken() string { return stubMaskToken }
func (s *stubTokenizer) MaskID() int64     { return stubMASK }
func (s *stubTokenizer) CLSID() int64      { return stubCLS }
func (s *stubTokenizer) SEPID() int64      { return stubSEP }
func (s *stubTokenizer) PADID() int64      { return stubPAD }

// tokens maps ids back to words, so assertions read as the prompt rather than
// as numbers -- an id diff alone does not say which part of the layout moved.
func (s *stubTokenizer) tokens(ids []int64) []string {
	out := make([]string, len(ids))
	for i, id := range ids {
		switch id {
		case stubPAD:
			out[i] = "[PAD]"
		case stubCLS:
			out[i] = "[CLS]"
		case stubSEP:
			out[i] = "[SEP]"
		case stubMASK:
			out[i] = stubMaskToken
		default:
			out[i] = s.words[id-stubFirstWord]
		}
	}
	return out
}

// words returns "p0 p1 ... p(n-1)".
func words(prefix string, n int) string {
	w := make([]string, n)
	for i := range w {
		w[i] = fmt.Sprintf("%s%d", prefix, i)
	}
	return strings.Join(w, " ")
}

// toks splits space-separated expectations and concatenates them.
func toks(parts ...string) []string {
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		out = append(out, strings.Fields(p)...)
	}
	return out
}

func labels(keys ...string) jsonx.Obj {
	o := make(jsonx.Obj, len(keys))
	for i, k := range keys {
		o[i] = jsonx.Field{Key: k}
	}
	return o
}

type seqCase struct {
	name         string
	q            Internal
	state        any
	maxLen       int
	headMaxLen   int
	optionOrder  []int
	truncateLeft bool

	want        []string // the whole sequence, when the case pins it
	wantLen     int      // otherwise its length...
	wantPrefix  []string // ...and the part that matters
	wantSuffix  []string
	wantMarkers []int64
}

func runSequenceCases(t *testing.T, cases []seqCase) {
	t.Helper()
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			tok := newStub()
			maxLen, headMaxLen := c.maxLen, c.headMaxLen
			if maxLen == 0 {
				maxLen = 512
			}
			if headMaxLen == 0 {
				headMaxLen = 192
			}

			ids, markers, err := BuildSequence(tok, c.state, c.q, maxLen, headMaxLen, c.optionOrder, c.truncateLeft)
			if err != nil {
				t.Fatalf("BuildSequence: %v", err)
			}
			got := tok.tokens(ids)

			if c.want != nil && !slices.Equal(got, c.want) {
				t.Errorf("sequence\n got: %q\nwant: %q", got, c.want)
			}
			if c.wantLen != 0 && len(got) != c.wantLen {
				t.Errorf("len(ids) = %d, want %d\n got: %q", len(got), c.wantLen, got)
			}
			if c.wantPrefix != nil && !slices.Equal(got[:min(len(got), len(c.wantPrefix))], c.wantPrefix) {
				t.Errorf("prefix\n got: %q\nwant: %q", got[:min(len(got), len(c.wantPrefix))], c.wantPrefix)
			}
			if c.wantSuffix != nil && !slices.Equal(got[max(0, len(got)-len(c.wantSuffix)):], c.wantSuffix) {
				t.Errorf("suffix\n got: %q\nwant: %q", got[max(0, len(got)-len(c.wantSuffix)):], c.wantSuffix)
			}
			if !slices.Equal(markers, c.wantMarkers) {
				t.Errorf("markers = %v, want %v", markers, c.wantMarkers)
			}
			// Invariant #9 read the other way: every surviving marker points at
			// a mask token. A marker off by one points at option text instead,
			// and the head then scores the wrong position without complaint.
			for _, m := range markers {
				if ids[m] != stubMASK {
					t.Errorf("marker %d points at %q, not %s", m, got[m], stubMaskToken)
				}
			}
		})
	}
}

// TestBuildSequenceBudget is Task 5.1.1: the per-option [:48] cap, opt_budget,
// the single `< 16` fallback and the head's floor of 8 (invariants #3, #5-#7).
func TestBuildSequenceBudget(t *testing.T) {
	many := make(jsonx.Obj, 77)
	for i := range many {
		many[i] = jsonx.Field{Key: fmt.Sprintf("k%02d", i), Value: "d e"}
	}
	manyMarkers := make([]int64, 77)
	for i := range manyMarkers {
		manyMarkers[i] = 10 + 4*int64(i)
	}

	wide := jsonx.Obj{
		{Key: "a", Value: words("w", 19)},
		{Key: "b", Value: words("w", 19)},
		{Key: "c", Value: words("w", 19)},
	}

	runSequenceCases(t, []seqCase{
		{
			// " a: w0 .. w59" is 61 ids; [:48] keeps "a:" and w0..w46, and the
			// mask token comes on top of the 48 (invariant #3).
			name:  "option text capped at 48 ids plus the mask",
			q:     Internal{T: TypeChoice, Ins: "q", Crit: jsonx.Obj{{Key: "a", Value: words("w", 60)}}},
			state: "",
			want: toks("[CLS] choice question: q [SEP] [MASK] a:", words("w", 47),
				"[SEP] [SEP]"),
			wantMarkers: []int64{5},
		},
		{
			// 77 options of 4 ids: 308 against head_max_len 192. The fallback's
			// per = max(4, 176 // 77) = 4 truncates nothing, leaves the budget at
			// -116, and the head clamps to its floor of 8 (invariant #6's own
			// example, and the 320-id length sequence.jsonl records).
			name:        "77 options leave the budget negative and the head at 8",
			q:           Internal{T: TypeChoice, Ins: words("h", 20), Crit: many},
			state:       "",
			wantLen:     1 + 8 + 1 + 308 + 1 + 1,
			wantPrefix:  toks("[CLS] choice question:", words("h", 6), "[SEP] [MASK] k00: d e [MASK]"),
			wantSuffix:  toks("[MASK] k76: d e [SEP] [SEP]"),
			wantMarkers: manyMarkers,
		},
		{
			// Three options of 21 ids against head_max_len 40: budget -23, so
			// per = max(4, 24 // 3) = 8 cuts each to its mask plus 7 ids, the
			// budget becomes 40 - 24 = 16, and the head keeps 16.
			name:  "fallback truncates each option, mask included",
			q:     Internal{T: TypeChoice, Ins: words("h", 20), Crit: wide},
			state: "",
			want: toks("[CLS] choice question:", words("h", 14), "[SEP]",
				"[MASK] a:", words("w", 6),
				"[MASK] b:", words("w", 6),
				"[MASK] c:", words("w", 6),
				"[SEP] [SEP]"),
			headMaxLen:  40,
			wantMarkers: []int64{18, 26, 34},
		},
		{
			// head_max_len below 16 makes (head_max_len - 16) negative; Python's
			// floor division and Go's truncation differ there, but max(4, ...)
			// absorbs both, so the options survive whole and the head gets 8.
			name:        "head_max_len below 16",
			q:           Internal{T: TypeChoice, Ins: words("h", 20), Crit: labels("a", "b", "c")},
			state:       "",
			headMaxLen:  10,
			want:        toks("[CLS] choice question:", words("h", 6), "[SEP] [MASK] a [MASK] b [MASK] c [SEP] [SEP]"),
			wantMarkers: []int64{10, 12, 14},
		},
		{
			// No fallback: the budget is 20 - 2 = 18, and max(8, 18) gives the
			// head exactly 18 ids.
			name:        "the head takes the whole budget",
			q:           Internal{T: TypeChoice, Ins: words("h", 30), Crit: labels("a")},
			state:       "",
			headMaxLen:  20,
			want:        toks("[CLS] choice question:", words("h", 16), "[SEP] [MASK] a [SEP] [SEP]"),
			wantMarkers: []int64{20},
		},
	})
}

// TestBuildSequenceTruncation is Task 5.1.2: the state is cut from the right
// by default and from the left under truncate_left, into room = max(0,
// max_len - len(ids) - 1) (invariants #10, #11).
func TestBuildSequenceTruncation(t *testing.T) {
	// "[CLS] choice question: q [SEP] [MASK] a [SEP]" is 8 ids.
	q := Internal{T: TypeChoice, Ins: "q", Crit: labels("a")}
	prefix := "[CLS] choice question: q [SEP] [MASK] a [SEP]"
	state := words("s", 5)

	runSequenceCases(t, []seqCase{
		{
			name: "right truncation keeps the head of the state", q: q, state: state, maxLen: 11,
			want: toks(prefix, "s0 s1 [SEP]"), wantMarkers: []int64{5},
		},
		{
			name: "left truncation keeps its tail", q: q, state: state, maxLen: 11, truncateLeft: true,
			want: toks(prefix, "s3 s4 [SEP]"), wantMarkers: []int64{5},
		},
		{
			name: "room beyond the state keeps all of it, right", q: q, state: state, maxLen: 20,
			want: toks(prefix, state, "[SEP]"), wantMarkers: []int64{5},
		},
		{
			name: "room beyond the state keeps all of it, left", q: q, state: state, maxLen: 20, truncateLeft: true,
			want: toks(prefix, state, "[SEP]"), wantMarkers: []int64{5},
		},
		{
			// room == 0 and st[:0] is empty: the trailing [SEP] fits exactly.
			name: "room zero, right", q: q, state: state, maxLen: 9,
			want: toks(prefix, "[SEP]"), wantMarkers: []int64{5},
		},
		{
			// room == 0 and st[-0:] is Python's st[0:] -- the WHOLE state, not
			// none of it. Only the final ids[:max_len] clamp bounds it, and the
			// slot the right-truncating path gives the trailing [SEP] goes to s0
			// instead. sequence.jsonl never reaches room == 0 (PLAN.md 5.2.3),
			// so this case is the only thing pinning it.
			name: "room zero, left, keeps the whole state until the clamp", q: q, state: state,
			maxLen: 9, truncateLeft: true,
			want: toks(prefix, "s0"), wantMarkers: []int64{5},
		},
	})
}

// TestBuildSequenceClamp is Task 5.1.3: the result is ids[:max_len], and a
// marker survives iff it is < max_len (invariant #12).
func TestBuildSequenceClamp(t *testing.T) {
	// "[CLS] choice question: q [SEP]" then "[MASK] a" at 5 and "[MASK] b" at 7.
	q := Internal{T: TypeChoice, Ins: "q", Crit: labels("a", "b")}

	runSequenceCases(t, []seqCase{
		{
			name: "a marker at max_len is dropped", q: q, state: "s0 s1", maxLen: 7,
			want: toks("[CLS] choice question: q [SEP] [MASK] a"), wantMarkers: []int64{5},
		},
		{
			name: "a marker at max_len - 1 survives", q: q, state: "s0 s1", maxLen: 8,
			want: toks("[CLS] choice question: q [SEP] [MASK] a [MASK]"), wantMarkers: []int64{5, 7},
		},
		{
			name: "left truncation is clamped the same way", q: q, state: "s0 s1", maxLen: 7, truncateLeft: true,
			want: toks("[CLS] choice question: q [SEP] [MASK] a"), wantMarkers: []int64{5},
		},
		{
			// The options' own [SEP] is the last id kept; the trailing one and
			// the whole state are cut away.
			name: "the trailing [SEP] is clamped away", q: q, state: "s0 s1", maxLen: 10,
			want: toks("[CLS] choice question: q [SEP] [MASK] a [MASK] b [SEP]"), wantMarkers: []int64{5, 7},
		},
	})
}

// TestBuildSequenceLayout is Task 5.1.4: the layout and the marker positions,
// asserted as token indices (invariants #1, #2, #4, #8, #9).
func TestBuildSequenceLayout(t *testing.T) {
	q := Internal{T: TypeChoice, Ins: "pick one", Crit: jsonx.Obj{{Key: "a"}, {Key: "b", Value: "x y"}}}

	runSequenceCases(t, []seqCase{
		{
			// markers[0] == 1 + len(head_ids) + 1 == 1 + 4 + 1.
			name: "basic", q: q, state: "s1 s2",
			want:        toks("[CLS] choice question: pick one [SEP] [MASK] a [MASK] b: x y [SEP] s1 s2 [SEP]"),
			wantMarkers: []int64{6, 8},
		},
		{
			// option_order permutes the options; the markers follow them.
			name: "option_order", q: q, state: "s1 s2", optionOrder: []int{1, 0},
			want:        toks("[CLS] choice question: pick one [SEP] [MASK] b: x y [MASK] a [SEP] s1 s2 [SEP]"),
			wantMarkers: []int64{6, 10},
		},
		{
			// The mask literal becomes a space in all three text sources, so it
			// can never add a marker the model would score (invariant #1).
			name: "the mask literal is scrubbed everywhere",
			q: Internal{
				T: TypeChoice, Ins: "pick" + stubMaskToken + "one",
				Crit: labels("a" + stubMaskToken + "b"),
			},
			state:       "s1" + stubMaskToken + "s2",
			want:        toks("[CLS] choice question: pick one [SEP] [MASK] a b [SEP] s1 s2 [SEP]"),
			wantMarkers: []int64{6},
		},
		{
			// The head spells the qtype; noul renders false then true.
			name: "noul", q: Internal{T: TypeNoul, Ins: "q"}, state: "",
			want: toks("[CLS] noul question: q [SEP]",
				"[MASK] false: no, the statement does not hold",
				"[MASK] true: yes, the statement holds [SEP] [SEP]"),
			wantMarkers: []int64{5, 13},
		},
		{
			name: "score", q: Internal{T: TypeScore, Ins: "q", Crit: []any{"low", "high"}}, state: "",
			want:        toks("[CLS] score question: q [SEP] [MASK] level 0: low [MASK] level 1: high [SEP] [SEP]"),
			wantMarkers: []int64{5, 9},
		},
		{
			// A structured state is serialized with Python's separators before
			// it is tokenized (invariant #18).
			name: "structured state", q: q, state: []any{"a b", jsonx.Obj{{Key: "k", Value: "v"}}},
			wantSuffix:  toks(`[SEP] ["a b", {"k": "v"}] [SEP]`),
			wantMarkers: []int64{6, 8},
		},
		{
			name: "no options", q: Internal{T: TypeChoice, Ins: "q", Crit: jsonx.Obj{}}, state: "s",
			want:        toks("[CLS] choice question: q [SEP] [SEP] s [SEP]"),
			wantMarkers: []int64{},
		},
	})
}

// TestBuildSequenceStateError: serialize_state passes no default=, so a state
// Python cannot serialize raises, and here it is an error rather than a
// stringified stand-in the model would read as text.
func TestBuildSequenceStateError(t *testing.T) {
	_, _, err := BuildSequence(newStub(), make(chan int), Internal{T: TypeNoul, Ins: "q"}, 512, 192, nil, false)
	if err == nil {
		t.Fatal("BuildSequence accepted a state json.dumps would refuse")
	}
}

// TestBuildSequenceHugeMaxLen: max_len comes from the checkpoint's config, and
// a checkpoint is untrusted input (AGENTS.md, Security). Upstream's list grows
// with its contents, so an absurd max_len costs nothing until the clamp; a
// buffer sized from max_len would instead allocate it up front, or panic.
func TestBuildSequenceHugeMaxLen(t *testing.T) {
	runSequenceCases(t, []seqCase{{
		name: "max_len 1<<50", q: Internal{T: TypeChoice, Ins: "q", Crit: labels("a")}, state: "s",
		maxLen:      1 << 50,
		want:        toks("[CLS] choice question: q [SEP] [MASK] a [SEP] s [SEP]"),
		wantMarkers: []int64{5},
	}})
}
