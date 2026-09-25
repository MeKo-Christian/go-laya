package prompt

import (
	"reflect"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
)

// collatePad is deliberately not 0: marker_pos is zero-filled, so a pad of 0
// would let a padding bug and a marker bug produce the same matrix.
const collatePad int64 = 9

// TestCollate is Task 5.3.1 on hand-built rows: the padding and masking rules
// of common.py:218-251, with every shape chosen so that each rule has a cell
// only it can get right. TestCollateGolden holds the same code to the recorded
// batches, which never go below three rows and never leave a row markerless.
func TestCollate(t *testing.T) {
	cases := []struct {
		name  string
		items []Item
		want  backend.Batch
	}{
		{
			name: "ragged ids and markers",
			items: []Item{
				{IDs: []int64{1, 20, 21, 2}, Markers: []int64{1, 2}, QType: 0},
				{IDs: []int64{1, 30}, Markers: []int64{1, 1, 1}, QType: 1},
				{IDs: []int64{1, 40, 41, 42, 43, 2}, Markers: nil, QType: 2},
			},
			want: backend.Batch{
				InputIDs: [][]int64{
					{1, 20, 21, 2, 9, 9},
					{1, 30, 9, 9, 9, 9},
					{1, 40, 41, 42, 43, 2},
				},
				AttentionMask: [][]int64{
					{1, 1, 1, 1, 0, 0},
					{1, 1, 0, 0, 0, 0},
					{1, 1, 1, 1, 1, 1},
				},
				MarkerPos: [][]int64{
					{1, 2, 0},
					{1, 1, 1},
					{0, 0, 0},
				},
				MarkerMask: [][]bool{
					{true, true, false},
					{true, true, true},
					{false, false, false},
				},
				QType: []int64{0, 1, 2},
			},
		},
		{
			name:  "single row needs no padding",
			items: []Item{{IDs: []int64{1, 5, 2}, Markers: []int64{1}, QType: 2}},
			want: backend.Batch{
				InputIDs:      [][]int64{{1, 5, 2}},
				AttentionMask: [][]int64{{1, 1, 1}},
				MarkerPos:     [][]int64{{1}},
				MarkerMask:    [][]bool{{true}},
				QType:         []int64{2},
			},
		},
		{
			// kmax 0: torch.zeros((n, 0)) is n empty rows, not no rows.
			name: "no row has markers",
			items: []Item{
				{IDs: []int64{1, 2}, QType: 0},
				{IDs: []int64{1}, QType: 0},
			},
			want: backend.Batch{
				InputIDs:      [][]int64{{1, 2}, {1, 9}},
				AttentionMask: [][]int64{{1, 1}, {1, 0}},
				MarkerPos:     [][]int64{{}, {}},
				MarkerMask:    [][]bool{{}, {}},
				QType:         []int64{0, 0},
			},
		},
		{
			// Python returns None; the Agent never gets here, since an empty
			// question set is rejected before any sequence is built.
			name:  "no items",
			items: nil,
			want:  backend.Batch{},
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			got := Collate(c.items, collatePad)
			if !reflect.DeepEqual(got, c.want) {
				t.Errorf("Collate =\n  %+v\nwant\n  %+v", got, c.want)
			}
		})
	}
}

// TestCollateDoesNotAlias guards the copy: the rows go to a runtime that may
// hold them past the call, and BuildSequence's output must not change under it.
func TestCollateDoesNotAlias(t *testing.T) {
	ids := []int64{1, 2}
	markers := []int64{1}
	b := Collate([]Item{{IDs: ids, Markers: markers}}, collatePad)
	ids[0], markers[0] = 99, 99
	if b.InputIDs[0][0] != 1 || b.MarkerPos[0][0] != 1 {
		t.Errorf("Collate aliases its input: %+v", b)
	}
}
