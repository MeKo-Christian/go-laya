package fake

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
	"testing"

	"github.com/MeKo-Christian/go-laya/backend"
	"github.com/MeKo-Christian/go-laya/internal/golden"
)

var checkpoints = []string{"english", "multilingual", "typed-decisions"}

// recorded is a logits.jsonl case as the tests see it: the input the fake must
// recognise and the outputs it must hand back.
type recorded struct {
	name, checkpoint string
	batch            backend.Batch
	logits, act      [][]float32
}

func loadRecorded(t *testing.T) map[string]recorded {
	t.Helper()

	out := map[string]recorded{}
	for _, c := range golden.Load(t, "logits") {
		var r struct {
			Checkpoint string                     `json:"checkpoint"`
			Collated   map[string]json.RawMessage `json:"collated"`
			Logits     json.RawMessage            `json:"logits"`
			ActLogits  json.RawMessage            `json:"act_logits"`
		}
		c.Unmarshal(t, &r)
		out[c.Name] = recorded{
			name:       c.Name,
			checkpoint: r.Checkpoint,
			batch:      golden.CollatedBatch(t, r.Collated),
			logits:     golden.Matrix[float32](t, r.Logits, "logits"),
			act:        golden.Matrix[float32](t, r.ActLogits, "act_logits"),
		}
	}
	return out
}

// TestFakeReplaysEveryRecord is Task 6.1.2: every recorded batch, fed to its
// own checkpoint's fake, comes back as exactly the recorded logits and
// act_logits -- bit for bit, since both sides decode the same float32 text.
func TestFakeReplaysEveryRecord(t *testing.T) {
	recs := loadRecorded(t)
	fakes := map[string]*Backend{}
	for _, ck := range checkpoints {
		fakes[ck] = New(t, ck)
	}

	n := 0
	for _, r := range recs {
		t.Run(r.name, func(t *testing.T) {
			logits, act, err := fakes[r.checkpoint].Forward(context.Background(), r.batch)
			if err != nil {
				t.Fatalf("Forward: %v", err)
			}
			if !reflect.DeepEqual(logits, r.logits) {
				t.Errorf("logits =\n  %v\nwant\n  %v", logits, r.logits)
			}
			if !reflect.DeepEqual(act, r.act) {
				t.Errorf("act =\n  %v\nwant\n  %v", act, r.act)
			}
		})
		n++
	}
	if n != 30 {
		t.Errorf("replayed %d records, want the 30 logits.jsonl holds", n)
	}
}

// TestFakeIsPerCheckpoint guards the one ambiguity in the fixture: english and
// typed-decisions share a tokenizer and budgets, so their collated tensors are
// identical while their weights, and so their logits, are not. A fake keyed
// on tensors alone would answer one checkpoint with the other's numbers.
func TestFakeIsPerCheckpoint(t *testing.T) {
	recs := loadRecorded(t)
	en, td := recs["english/en/billing"], recs["typed-decisions/en/billing"]
	if !reflect.DeepEqual(en.batch, td.batch) {
		t.Fatal("the fixture no longer shares tensors across english and typed-decisions; revisit this test")
	}
	if reflect.DeepEqual(en.logits, td.logits) {
		t.Fatal("english and typed-decisions record the same logits; the test cannot tell them apart")
	}

	for _, want := range []recorded{en, td} {
		logits, act, err := New(t, want.checkpoint).Forward(context.Background(), want.batch)
		if err != nil {
			t.Fatalf("%s: Forward: %v", want.checkpoint, err)
		}
		if !reflect.DeepEqual(logits, want.logits) || !reflect.DeepEqual(act, want.act) {
			t.Errorf("%s answered with another checkpoint's outputs: logits %v", want.checkpoint, logits)
		}
	}
}

// TestFakeRejectsUnknownBatch is Task 6.1.3: a batch that is not a recording,
// however close, is an error wrapping ErrUnknownBatch and no outputs at all.
// Each row changes the smallest thing a prompt regression could change, and
// the message has to point at the recording it nearly was.
func TestFakeRejectsUnknownBatch(t *testing.T) {
	recs := loadRecorded(t)
	base := recs["english/en/billing"]

	cases := []struct {
		name   string
		mutate func(b *backend.Batch)
		near   string // the recording the error must name, "" when no shape matches
	}{
		{"one input id", func(b *backend.Batch) { b.InputIDs[1][5]++ }, "input_ids[1][5]"},
		{"one attention cell", func(b *backend.Batch) { b.AttentionMask[0][0] = 0 }, "attention_mask[0][0]"},
		{"one marker position", func(b *backend.Batch) { b.MarkerPos[2][0]++ }, "marker_pos[2][0]"},
		{"one marker mask cell", func(b *backend.Batch) { b.MarkerMask[1][3] = !b.MarkerMask[1][3] }, "marker_mask[1][3]"},
		{"one qtype", func(b *backend.Batch) { b.QType[2] = (b.QType[2] + 1) % 3 }, "qtype[2]"},
		{"a dropped row", func(b *backend.Batch) {
			b.InputIDs, b.AttentionMask = b.InputIDs[:2], b.AttentionMask[:2]
			b.MarkerPos, b.MarkerMask, b.QType = b.MarkerPos[:2], b.MarkerMask[:2], b.QType[:2]
		}, ""},
		{"a trimmed padding column", func(b *backend.Batch) {
			for i := range b.InputIDs {
				b.InputIDs[i] = b.InputIDs[i][:len(b.InputIDs[i])-1]
				b.AttentionMask[i] = b.AttentionMask[i][:len(b.AttentionMask[i])-1]
			}
		}, ""},
	}

	f := New(t, "english")
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			b := cloneBatch(base.batch)
			c.mutate(&b)
			logits, act, err := f.Forward(context.Background(), b)
			assertUnknown(t, logits, act, err)
			if c.near != "" && (!strings.Contains(err.Error(), base.name) || !strings.Contains(err.Error(), c.near)) {
				t.Errorf("error %q does not point at %s %s", err, base.name, c.near)
			}
		})
	}

	// Every multilingual recording is foreign to the english fake: a different
	// tokenizer, so a different batch even where the shapes coincide.
	for _, r := range recs {
		if r.checkpoint != "multilingual" {
			continue
		}
		t.Run("foreign/"+r.name, func(t *testing.T) {
			logits, act, err := f.Forward(context.Background(), r.batch)
			assertUnknown(t, logits, act, err)
		})
	}
}

func assertUnknown(t *testing.T, logits, act [][]float32, err error) {
	t.Helper()

	if !errors.Is(err, ErrUnknownBatch) {
		t.Fatalf("err = %v, want ErrUnknownBatch", err)
	}
	if logits != nil || act != nil {
		t.Errorf("an unknown batch returned outputs: logits %v, act %v", logits, act)
	}
}

// TestFakeDoesNotAlias: the caller owns what Forward returns, and softmax in
// place must not change the recording the next test replays.
func TestFakeDoesNotAlias(t *testing.T) {
	r := loadRecorded(t)["multilingual/de/billing"]
	f := New(t, "multilingual")

	logits, act, err := f.Forward(context.Background(), r.batch)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	logits[0][0], act[0][0] = 42, 42

	logits, act, err = f.Forward(context.Background(), r.batch)
	if err != nil {
		t.Fatalf("Forward: %v", err)
	}
	if !reflect.DeepEqual(logits, r.logits) || !reflect.DeepEqual(act, r.act) {
		t.Errorf("a caller's write reached the recording: logits %v", logits)
	}
}

func TestFakeLifecycle(t *testing.T) {
	r := loadRecorded(t)["english/en/spam"]

	t.Run("cancelled context", func(t *testing.T) {
		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		if _, _, err := New(t, "english").Forward(ctx, r.batch); !errors.Is(err, context.Canceled) {
			t.Errorf("err = %v, want context.Canceled", err)
		}
	})

	t.Run("closed", func(t *testing.T) {
		f := New(t, "english")
		if err := f.Close(); err != nil {
			t.Fatalf("Close: %v", err)
		}
		if _, _, err := f.Forward(context.Background(), r.batch); !errors.Is(err, ErrClosed) {
			t.Errorf("err = %v, want ErrClosed", err)
		}
	})
}

// fatalRecorder stops a test helper the way testing.T does, but lets the test
// see the message instead of failing.
type fatalRecorder struct {
	*testing.T

	msg string
}

func (r *fatalRecorder) Fatalf(format string, args ...any) {
	r.msg = fmt.Sprintf(format, args...)
	panic(r)
}

// TestNewRejectsUnknownCheckpoint: a misspelt checkpoint must not yield a fake
// that knows nothing and so rejects every batch -- the test using it would
// fail for a reason that has nothing to do with the code under test.
func TestNewRejectsUnknownCheckpoint(t *testing.T) {
	rec := &fatalRecorder{T: t}
	func() {
		defer func() {
			if p := recover(); p != nil && p != rec {
				panic(p)
			}
		}()
		New(rec, "englsh")
	}()
	if !strings.Contains(rec.msg, "englsh") {
		t.Errorf("New did not fail naming the checkpoint; message %q", rec.msg)
	}
}

func cloneBatch(b backend.Batch) backend.Batch {
	return backend.Batch{
		InputIDs:      clone2(b.InputIDs),
		AttentionMask: clone2(b.AttentionMask),
		MarkerPos:     clone2(b.MarkerPos),
		MarkerMask:    clone2(b.MarkerMask),
		QType:         append([]int64(nil), b.QType...),
	}
}
