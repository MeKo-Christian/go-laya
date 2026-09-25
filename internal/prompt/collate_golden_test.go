package prompt

import (
	"encoding/json"
	"os"
	"path/filepath"
	"reflect"
	"testing"

	"github.com/MeKo-Christian/go-laya/internal/golden"
	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/tokenizer"
)

// logitsCase is the part of a testdata/logits.jsonl record that Collate is
// asserted against: the inputs of the batch and the tensors collate_items
// made from them (Task 1.7.1).
type logitsCase struct {
	Checkpoint  string                     `json:"checkpoint"`
	State       json.RawMessage            `json:"state"`
	Questions   json.RawMessage            `json:"questions"`
	QIDs        []string                   `json:"qids"`
	QTypes      []int64                    `json:"qtypes"`
	InputTokens int64                      `json:"input_tokens"`
	Collated    map[string]json.RawMessage `json:"collated"`
}

// checkpointBudget is what agent.py:257-258 reads from rl_agent_config.json.
type checkpointBudget struct {
	tok                *tokenizer.HF
	maxLen, headMaxLen int
}

// TestCollateGolden is Task 5.3.2: every one of the 30 recorded batches is
// rebuilt the way Agent.system_one builds it (agent.py:256-266) -- each
// question through BuildSequence under the checkpoint's own budgets, then
// Collate with the tokenizer's pad id -- and all five tensors must equal the
// recorded ones, shapes included. The summed attention mask must also equal
// the recorded input_tokens, which is invariant #33's usage figure.
//
// It needs the real tokenizers, so it is gated like TestBuildSequenceGolden;
// TestCollate pins the padding rules without them.
func TestCollateGolden(t *testing.T) {
	root := golden.SkipWithoutModels(t)

	ckpts := map[string]checkpointBudget{}
	for _, rec := range golden.Load(t, "logits") {
		var c logitsCase
		rec.Unmarshal(t, &c)

		ck, ok := ckpts[c.Checkpoint]
		if !ok {
			ck = openCheckpointBudget(t, golden.CheckpointDir(root, c.Checkpoint))
			ckpts[c.Checkpoint] = ck
		}

		t.Run(rec.Name, func(t *testing.T) {
			state, err := jsonx.Decode(c.State)
			if err != nil {
				t.Fatalf("decode the recorded state: %v", err)
			}
			qs := decodeObj(t, c.Questions)

			items := make([]Item, 0, len(c.QIDs))
			for _, qid := range c.QIDs {
				raw, found := qs.Get(qid)
				if !found {
					t.Fatalf("qid %q is not among the recorded questions", qid)
				}
				q := toInternal(t, raw)
				ids, markers, err := BuildSequence(ck.tok, state, q, ck.maxLen, ck.headMaxLen, nil, false)
				if err != nil {
					t.Fatalf("%s: BuildSequence: %v", qid, err)
				}
				items = append(items, Item{IDs: ids, Markers: markers, QType: qtypeOf(t, q.T)})
			}

			got := Collate(items, ck.tok.PADID())
			want := golden.CollatedBatch(t, c.Collated)
			assertEqual(t, "input_ids", got.InputIDs, want.InputIDs)
			assertEqual(t, "attention_mask", got.AttentionMask, want.AttentionMask)
			assertEqual(t, "marker_pos", got.MarkerPos, want.MarkerPos)
			assertEqual(t, "marker_mask", got.MarkerMask, want.MarkerMask)
			assertEqual(t, "qtype", got.QType, want.QType)
			assertEqual(t, "qtypes", got.QType, c.QTypes)

			var tokens int64
			for _, row := range got.AttentionMask {
				for _, v := range row {
					tokens += v
				}
			}
			if tokens != c.InputTokens {
				t.Errorf("sum(attention_mask) = %d, want input_tokens %d", tokens, c.InputTokens)
			}
		})
	}
}

func openCheckpointBudget(t *testing.T, dir string) checkpointBudget {
	t.Helper()

	tok, err := tokenizer.Open(filepath.Join(dir, "tokenizer"))
	if err != nil {
		t.Fatalf("open %s: %v", dir, err)
	}
	raw, err := os.ReadFile(filepath.Join(dir, "rl_agent_config.json"))
	if err != nil {
		t.Fatalf("read the checkpoint config: %v", err)
	}
	// agent.py:257-258's defaults, for a config that omits either key.
	cfg := struct {
		MaxLen     *int `json:"max_len"`
		HeadMaxLen *int `json:"head_max_len"`
	}{}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		t.Fatalf("decode the checkpoint config: %v", err)
	}
	ck := checkpointBudget{tok: tok, maxLen: 512, headMaxLen: 192}
	if cfg.MaxLen != nil {
		ck.maxLen = *cfg.MaxLen
	}
	if cfg.HeadMaxLen != nil {
		ck.headMaxLen = *cfg.HeadMaxLen
	}
	return ck
}

// toInternal is Agent._to_internal (agent.py:230-238) for the questions the
// fixture records. The public conversion is M7's; this one only has to agree
// with it on these inputs, and it refuses the one branch it does not port
// (non-string instructions, which upstream json.dumps with ensure_ascii).
func toInternal(t *testing.T, raw any) Internal {
	t.Helper()

	qdef, ok := raw.(jsonx.Obj)
	if !ok {
		t.Fatalf("recorded question is %T, not an object", raw)
	}
	var q Internal
	tv, _ := qdef.Get("type")
	q.T, _ = tv.(string)
	iv, _ := qdef.Get("instructions")
	if q.Ins, ok = iv.(string); !ok {
		t.Fatalf("non-string instructions %T are not ported here", iv)
	}
	q.Crit, _ = qdef.Get("criteria")

	// {c: None for c in crit}: a dict, so a repeated label collapses.
	if list, isList := q.Crit.([]any); isList && q.T == "choice" {
		var obj jsonx.Obj
		for _, c := range list {
			label, _ := c.(string)
			if _, dup := obj.Get(label); !dup {
				obj = append(obj, jsonx.Field{Key: label})
			}
		}
		q.Crit = obj
	}
	return q
}

// qtypeOf is common.py:11's QTYPES.
func qtypeOf(t *testing.T, qt string) int64 {
	t.Helper()

	switch qt {
	case "choice":
		return 0
	case "score":
		return 1
	case "noul":
		return 2
	}
	t.Fatalf("unknown question type %q", qt)
	return -1
}

func decodeObj(t *testing.T, raw json.RawMessage) jsonx.Obj {
	t.Helper()

	v, err := jsonx.Decode(raw)
	if err != nil {
		t.Fatalf("decode the recorded questions: %v", err)
	}
	obj, ok := v.(jsonx.Obj)
	if !ok {
		t.Fatalf("recorded questions are %T, not an object", v)
	}
	return obj
}

func assertEqual(t *testing.T, name string, got, want any) {
	t.Helper()

	if !reflect.DeepEqual(got, want) {
		t.Errorf("%s =\n  %v\nwant\n  %v", name, got, want)
	}
}
