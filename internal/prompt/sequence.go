package prompt

import (
	"strings"

	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// maxOptionIDs is the [:48] at common.py:68: the cap on one option's text ids,
// not counting the mask token put in front of them (invariant #3).
const maxOptionIDs = 48

// BuildSequence reproduces common.build_sequence (common.py:49-86, invariants
// #1-#12): the model's input ids,
//
//	[CLS] <type> question: <ins> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
//
// clamped to maxLen, and the index of each surviving option's mask token.
//
// The markers are token indices, so this function has no tolerance for being
// nearly right: one id too many in the head shifts every marker, and the model
// then scores the wrong positions and returns a confident, plausible answer.
// Upstream has no test for it at all, which is why every slice below is written
// with its Python spelling beside it.
//
// optionOrder and truncateLeft are internal parameters (Task 5.4, invariants #4
// and #11): the public API never sets them, and nil and false are what
// agent.py:261 passes.
//
// The error is serialize_state's: a state Python's json module would refuse
// raises upstream and fails here. The ValueError for lost markers
// (agent.py:262-263, invariant #13) belongs to the caller, as it does upstream.
func BuildSequence(
	tok tokenizer.Tokenizer,
	state any,
	q Internal,
	maxLen, headMaxLen int,
	optionOrder []int,
	truncateLeft bool,
) (ids, markers []int64, err error) {
	serialized, err := SerializeState(state)
	if err != nil {
		return nil, nil, err
	}

	mask := tok.MaskToken()
	opts := RenderOptions(q)
	order := optionOrder
	if order == nil {
		order = make([]int, len(opts))
		for i := range order {
			order[i] = i
		}
	}

	ins := strings.ReplaceAll(q.Ins, mask, " ")
	headIDs := tok.Encode(q.T + " question: " + ins)

	optIDs := make([][]int64, 0, len(order))
	for _, i := range order {
		text := tok.Encode(" " + strings.ReplaceAll(opts[i], mask, " "))
		optIDs = append(optIDs, append([]int64{tok.MaskID()}, pyHead(text, maxOptionIDs)...))
	}

	// Despite the name, opt_budget is what is left for the head, and it can go
	// negative (invariant #5). The fallback runs at most once and may leave it
	// below 16 (invariant #6). Python's // floors where Go's / truncates, which
	// differs only for a negative (headMaxLen - 16) -- and max(4, ...) absorbs
	// every negative quotient, so the two cannot be told apart here.
	optBudget := headMaxLen - totalLen(optIDs)
	if optBudget < 16 {
		per := max(4, (headMaxLen-16)/max(1, len(optIDs)))
		for i, o := range optIDs {
			optIDs[i] = pyHead(o, per)
		}
		optBudget = headMaxLen - totalLen(optIDs)
	}
	headIDs = pyHead(headIDs, max(8, optBudget))

	// Sized from the content, never from maxLen: max_len comes from the
	// checkpoint's config, which is untrusted, and upstream's list only grows
	// as far as its contents do.
	ids = make([]int64, 0, 3+len(headIDs)+totalLen(optIDs))
	ids = append(ids, tok.CLSID())
	ids = append(ids, headIDs...)
	ids = append(ids, tok.SEPID())
	allMarkers := make([]int64, 0, len(optIDs))
	for _, o := range optIDs {
		allMarkers = append(allMarkers, int64(len(ids)))
		ids = append(ids, o...)
	}
	ids = append(ids, tok.SEPID())

	room := max(0, maxLen-len(ids)-1)
	st := tok.Encode(strings.ReplaceAll(serialized, mask, " "))
	if truncateLeft {
		st = pyTail(st, room)
	} else {
		st = pyHead(st, room)
	}
	ids = append(ids, st...)
	ids = append(ids, tok.SEPID())

	markers = make([]int64, 0, len(allMarkers))
	for _, m := range allMarkers {
		if m < int64(maxLen) {
			markers = append(markers, m)
		}
	}
	return pyHead(ids, maxLen), markers, nil
}

// pyHead is Python's s[:n] for n >= 0: a stop past the end is the whole list.
func pyHead(s []int64, n int) []int64 {
	return s[:min(n, len(s))]
}

// pyTail is Python's s[-n:] for n >= 0, and n == 0 is the trap: -0 is 0, so
// s[-0:] is s[0:], the WHOLE list rather than none of it. Under truncate_left
// with no room left, build_sequence therefore appends the entire state and only
// the final ids[:max_len] clamp bounds it -- which puts the state's first id,
// not [SEP], in the last slot when the options end one short of max_len.
// sequence.jsonl never reaches room == 0 (PLAN.md 5.2.3), so
// TestBuildSequenceTruncation pins it with a stub tokenizer.
func pyTail(s []int64, n int) []int64 {
	if n == 0 {
		return s
	}
	return s[len(s)-min(n, len(s)):]
}

func totalLen(parts [][]int64) int {
	n := 0
	for _, p := range parts {
		n += len(p)
	}
	return n
}
