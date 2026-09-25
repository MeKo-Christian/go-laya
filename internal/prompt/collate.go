package prompt

import "github.com/MeKo-Christian/go-laya/backend"

// Item is one question's built sequence: the dict agent.py:265 appends before
// collating. The training-only keys collate_items also reads -- target, and
// label and meta on the way out -- are not ported (PLAN.md 5.3.1).
type Item struct {
	IDs, Markers []int64
	QType        int64
}

// Collate reproduces common.collate_items (common.py:218-251) for inference:
// every row right-padded with padID to the longest sequence L, an attention
// mask of 1 over real tokens and 0 over padding, marker positions zero-filled
// to the most markers any row has, and a marker mask that is true only over a
// row's real markers.
//
// Upstream takes a list of groups and flattens them first; agent.py:266 always
// passes one group, so this takes the flat list. Upstream returns None for no
// items at all, and this returns the zero Batch -- unreachable through the
// Agent, which rejects an empty question set before building a sequence.
//
// The rows are copies: the result goes to a runtime that may keep it.
func Collate(items []Item, padID int64) backend.Batch {
	if len(items) == 0 {
		return backend.Batch{}
	}

	l, kmax := 0, 0
	for _, it := range items {
		l = max(l, len(it.IDs))
		kmax = max(kmax, len(it.Markers))
	}

	b := backend.Batch{
		InputIDs:      make([][]int64, len(items)),
		AttentionMask: make([][]int64, len(items)),
		MarkerPos:     make([][]int64, len(items)),
		MarkerMask:    make([][]bool, len(items)),
		QType:         make([]int64, len(items)),
	}
	for i, it := range items {
		ids := make([]int64, l)
		att := make([]int64, l)
		copy(ids, it.IDs)
		for j := range ids {
			if j < len(it.IDs) {
				att[j] = 1
			} else {
				ids[j] = padID
			}
		}

		pos := make([]int64, kmax)
		mask := make([]bool, kmax)
		copy(pos, it.Markers)
		for k := range it.Markers {
			mask[k] = true
		}

		b.InputIDs[i], b.AttentionMask[i] = ids, att
		b.MarkerPos[i], b.MarkerMask[i] = pos, mask
		b.QType[i] = it.QType
	}
	return b
}
