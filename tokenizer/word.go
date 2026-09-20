package tokenizer

// symbol is one element of a word during merging. The word is a doubly linked
// list over a slice, so a merge is a pointer update rather than a reallocation
// and positions stay stable for the heap's stale-entry check.
type symbol struct {
	id         int32
	prev, next int32 // indices into word.syms; -1 for none
	// length counts the original symbols folded into this one. It is not used
	// by the merge loop itself, but a merged symbol's length being wrong is
	// the readable signature of a broken update.
	length int32
	// dead marks a symbol absorbed into its left neighbour.
	dead bool
}

// word is a sequence of symbols being merged.
type word struct {
	syms []symbol
}

// candidate is a pair of adjacent symbols that a merge rule matches.
type candidate struct {
	rank  int32
	pos   int32 // index of the left symbol
	newID int32
	// left and right record the ids the candidate was created for. If either
	// symbol has since changed, the candidate is stale and is skipped -- the
	// standard way to avoid deleting from the middle of a heap.
	left, right int32
}

// candidateHeap is a binary min-heap of candidates ordered by rank and then
// by position, which is HF's tie-break: the earliest occurrence of the
// lowest-ranked rule merges first.
//
// Hand-written rather than container/heap because that interface is built on
// `any`, and this is the encoder's hot loop -- a typed heap avoids boxing every
// candidate and keeps the comparison inlineable.
type candidateHeap []candidate

// less is the whole ordering policy, in one place.
func (h *candidateHeap) less(i, j int) bool {
	a, b := (*h)[i], (*h)[j]
	if a.rank != b.rank {
		return a.rank < b.rank
	}
	return a.pos < b.pos
}

func (h *candidateHeap) push(c candidate) {
	*h = append(*h, c)
	i := len(*h) - 1
	for i > 0 {
		parent := (i - 1) / 2
		if !h.less(i, parent) {
			break
		}
		(*h)[i], (*h)[parent] = (*h)[parent], (*h)[i]
		i = parent
	}
}

func (h *candidateHeap) pop() candidate {
	old := *h
	top := old[0]
	last := len(old) - 1
	old[0] = old[last]
	*h = old[:last]

	i, n := 0, last
	for {
		left, smallest := 2*i+1, i
		if left < n && h.less(left, smallest) {
			smallest = left
		}
		if right := left + 1; right < n && h.less(right, smallest) {
			smallest = right
		}
		if smallest == i {
			break
		}
		(*h)[i], (*h)[smallest] = (*h)[smallest], (*h)[i]
		i = smallest
	}
	return top
}

// mergeAll applies merges until no rule matches, lowest rank first.
func (w *word) mergeAll(merges map[[2]int32]merge) {
	pending := &candidateHeap{}
	for i := range w.syms {
		w.pushPair(pending, int32(i), merges)
	}

	for len(*pending) > 0 {
		c := pending.pop()

		left := &w.syms[c.pos]
		if left.dead || left.id != c.left || left.next < 0 {
			continue // stale: this symbol has already been merged into something
		}
		right := &w.syms[left.next]
		if right.dead || right.id != c.right {
			continue
		}

		// Fold the right symbol into the left one.
		left.id = c.newID
		left.length += right.length
		right.dead = true
		left.next = right.next
		if right.next >= 0 {
			w.syms[right.next].prev = c.pos
		}

		// The merge created at most two new adjacencies.
		w.pushPair(pending, c.pos, merges)
		if left.prev >= 0 {
			w.pushPair(pending, left.prev, merges)
		}
	}
}

// pushPair queues the merge of syms[i] with its right neighbour, if one exists
// and a rule matches.
func (w *word) pushPair(pending *candidateHeap, i int32, merges map[[2]int32]merge) {
	s := w.syms[i]
	if s.dead || s.next < 0 {
		return
	}
	right := w.syms[s.next]
	if right.dead {
		return
	}
	m, ok := merges[[2]int32{s.id, right.id}]
	if !ok {
		return
	}
	pending.push(candidate{
		rank:  m.rank,
		pos:   i,
		newID: m.newID,
		left:  s.id,
		right: right.id,
	})
}

// ids returns the surviving symbols in order.
func (w *word) ids() []int64 {
	out := make([]int64, 0, len(w.syms))
	for i := int32(0); i >= 0 && int(i) < len(w.syms); i = w.syms[i].next {
		if !w.syms[i].dead {
			out = append(out, int64(w.syms[i].id))
		}
		if w.syms[i].next <= i {
			break // defensive: a cycle would hang the encoder
		}
	}
	return out
}

// add appends a symbol.
func (w *word) add(id int32) {
	n := int32(len(w.syms))
	prev := n - 1
	if prev >= 0 {
		w.syms[prev].next = n
	}
	w.syms = append(w.syms, symbol{id: id, prev: prev, next: -1, length: 1})
}
