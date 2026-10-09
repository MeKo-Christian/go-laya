package laya

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"reflect"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// The loader's agent is what Router.Predict runs (Task 7.7.1).
var _ Predictor = (*Agent)(nil)

// funcAgent is a Predictor whose SystemOne is a closure, for the cases stubAgent's
// fixed payload cannot express: errors, a nil result, what reached it.
type funcAgent func(ctx context.Context, state any, qs Questions) (*Result, error)

func (f funcAgent) SystemOne(ctx context.Context, state any, qs Questions) (*Result, error) {
	return f(ctx, state, qs)
}

func (funcAgent) Close() error { return nil }

// predictQs is a generic question set: it matches no typed-decisions workflow,
// so routing falls through to the language of the state.
var predictQs = Questions{{ID: "urgent", Q: NoulQuestion{Ins: "Is `message` urgent?"}}}

// Task 7.7.1: SystemOne is part of the Predictor interface.
func TestAgentInterface(t *testing.T) {
	var a Predictor = &stubAgent{name: "english"}
	res, err := a.SystemOne(context.Background(), "state", predictQs)
	if err != nil {
		t.Fatalf("SystemOne: %v", err)
	}
	if res.Model != "english" {
		t.Errorf("Model = %q, want the stub's name", res.Model)
	}
}

// The two routing fixtures of test_local_e2e.py:198 and :202.
var (
	english = Obj{{Key: "message", Value: "I was charged twice, please refund."}}
	hindi   = Obj{{Key: "message", Value: "मुझसे दो बार शुल्क लिया गया, कृपया पैसे वापस करें।"}}
)

// Task 7.7.2: Router.predict (router.py:293-309) routes, loads the chosen
// checkpoint and runs its system_one.
func TestRouterPredict(t *testing.T) {
	ctx := context.Background()

	t.Run("routes-loads-and-runs", func(t *testing.T) {
		r, l := stubbedRouter(t, 2)
		for _, c := range []struct {
			state any
			opts  []RouteOption
			want  string
		}{
			{english, nil, ModelEnglish},
			{hindi, nil, ModelMultilingual},
			{english, []RouteOption{ForModel("typed-decisions")}, ModelTypedDecisions},
			{english, []RouteOption{ForLang("de")}, ModelMultilingual},
		} {
			res, err := r.Predict(ctx, c.state, predictQs, c.opts...)
			if err != nil {
				t.Fatalf("Predict: %v", err)
			}
			want, err := r.Route(c.state, predictQs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if res.Routing == nil || !reflect.DeepEqual(*res.Routing, want) {
				t.Errorf("Routing = %+v, want Route's %+v", res.Routing, want)
			}
			if res.Model != c.want {
				t.Errorf("ran the %q agent, want %q", res.Model, c.want)
			}
		}
		if got, want := l.calls(), []string{ModelEnglish, ModelMultilingual, ModelTypedDecisions}; !slices.Equal(got, want) {
			t.Errorf("loader built %v, want %v", got, want)
		}
	})

	t.Run("passes-the-request-through", func(t *testing.T) {
		type key struct{}
		var gotValue, gotState any
		var gotQs Questions
		r, _ := stubbedRouter(t, 1)
		if err := r.Attach(ModelEnglish, funcAgent(func(ctx context.Context, state any, qs Questions) (*Result, error) {
			gotValue, gotState, gotQs = ctx.Value(key{}), state, qs
			return &Result{Model: modelName}, nil
		})); err != nil {
			t.Fatal(err)
		}
		if _, err := r.Predict(context.WithValue(ctx, key{}, "v"), english, predictQs); err != nil {
			t.Fatalf("Predict: %v", err)
		}
		if gotValue != "v" {
			t.Error("the agent did not get the caller's context")
		}
		if !reflect.DeepEqual(gotState, english) || !reflect.DeepEqual(gotQs, predictQs) {
			t.Errorf("the agent got state %v and questions %v", gotState, gotQs)
		}
	})
}

// Predict loads through the Router's cache, so the LRU rules hold for it.
func TestRouterPredictLoading(t *testing.T) {
	ctx := context.Background()

	t.Run("loads-lazily-once", func(t *testing.T) {
		r, l := stubbedRouter(t, 1)
		for range 2 {
			if _, err := r.Predict(ctx, english, predictQs); err != nil {
				t.Fatalf("Predict: %v", err)
			}
		}
		if got := l.calls(); !slices.Equal(got, []string{ModelEnglish}) {
			t.Errorf("loader built %v, want [english]", got)
		}
	})

	// test_local_e2e.py:204-206 with stubs: a language switch at max_loaded=1.
	t.Run("switch-evicts", func(t *testing.T) {
		r, l := stubbedRouter(t, 1)
		for _, state := range []any{english, hindi} {
			if _, err := r.Predict(ctx, state, predictQs); err != nil {
				t.Fatalf("Predict: %v", err)
			}
		}
		if got := r.Loaded(); !slices.Equal(got, []string{ModelMultilingual}) {
			t.Errorf("Loaded() = %v, want [multilingual]", got)
		}
		if n := l.made[ModelEnglish].closed.Load(); n != 1 {
			t.Errorf("the evicted english agent was closed %d times, want 1", n)
		}
	})

	t.Run("uses-an-attached-agent", func(t *testing.T) {
		r, l := stubbedRouter(t, 1)
		if err := r.Attach(ModelEnglish, &stubAgent{name: "attached"}); err != nil {
			t.Fatal(err)
		}
		res, err := r.Predict(ctx, english, predictQs)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		if res.Model != "attached" || len(l.calls()) != 0 {
			t.Errorf("ran %q and built %v, want the attached agent and nothing built", res.Model, l.calls())
		}
	})
}

// router.py:311: system_one = predict.
func TestRouterPredictAlias(t *testing.T) {
	ctx := context.Background()
	r, _ := stubbedRouter(t, 2)
	p, err := r.Predict(ctx, hindi, predictQs)
	if err != nil {
		t.Fatalf("Predict: %v", err)
	}
	s, err := r.SystemOne(ctx, hindi, predictQs)
	if err != nil {
		t.Fatalf("SystemOne: %v", err)
	}
	if !reflect.DeepEqual(p, s) {
		t.Errorf("SystemOne = %+v, Predict = %+v", s, p)
	}
}

// A failed route loads nothing, and a failed load or pass comes back wrapped.
func TestRouterPredictErrors(t *testing.T) {
	ctx := context.Background()
	r, l := stubbedRouter(t, 1)
	if _, err := r.Predict(ctx, english, predictQs, ForModel("nope")); !errors.Is(err, ErrUnknownModel) {
		t.Errorf("unknown model: err = %v, want ErrUnknownModel", err)
	}
	if got := l.calls(); len(got) != 0 {
		t.Errorf("a failed route still loaded %v", got)
	}

	errLoad := errors.New("load failed")
	failing, err := NewRouter(WithLoader(func(context.Context, string, ModelSpec) (Predictor, error) {
		return nil, errLoad
	}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := failing.Predict(ctx, english, predictQs); !errors.Is(err, errLoad) {
		t.Errorf("loader failure: err = %v, want it wrapped", err)
	}

	errRun := errors.New("forward failed")
	for _, c := range []struct {
		label string
		agent funcAgent
		want  error
	}{
		{"agent error", func(context.Context, any, Questions) (*Result, error) { return nil, errRun }, errRun},
		{"nil result", func(context.Context, any, Questions) (*Result, error) {
			return nil, nil //nolint:nilnil // the broken agent under test
		}, nil},
	} {
		r, _ := stubbedRouter(t, 1)
		if err := r.Attach(ModelEnglish, c.agent); err != nil {
			t.Fatal(err)
		}
		res, err := r.Predict(ctx, english, predictQs)
		if err == nil || res != nil {
			t.Errorf("%s: Predict = %v, %v, want an error", c.label, res, err)
		}
		if c.want != nil && !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want it wrapped", c.label, err)
		}
	}
}

// Tasks 7.7.3 and 7.7.4: the decision goes under "routing" as upstream's
// result["routing"] = dict(decision) (router.py:308).
func TestRouterPredictRouting(t *testing.T) {
	ctx := context.Background()

	// Task 7.7.3: the auto-workflow branch is where upstream leaks the raw
	// (repo, subfolder) tuple (invariant #46); the payload carries a string.
	t.Run("workflow-repo-is-a-string", func(t *testing.T) {
		r, _ := stubbedRouter(t, 1, WithAutoTaskDetection(true))
		res, err := r.Predict(ctx, english, qTD())
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		if res.Routing == nil {
			t.Fatal("no routing decision on the result")
		}
		want := DefaultModels()[ModelTypedDecisions].String()
		if res.Routing.Model != ModelTypedDecisions || res.Routing.Repo != want {
			t.Errorf("routing = %+v, want typed-decisions at %q", res.Routing, want)
		}
		b, err := jsonx.Marshal(res.Routing.Map())
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Contains(b, []byte(`"repo": "`+want+`"`)) {
			t.Errorf("routing JSON %s does not carry repo as a string", b)
		}
	})

	// Task 7.7.4: the routing block is serialized by jsonx, as json.dumps
	// would write result["routing"] = dict(decision): last, with Python's
	// separators.
	t.Run("marshals-through-jsonx", func(t *testing.T) {
		r, _ := stubbedRouter(t, 1)
		res, err := r.Predict(ctx, hindi, predictQs)
		if err != nil {
			t.Fatalf("Predict: %v", err)
		}
		if res.Routing == nil {
			t.Fatal("no routing decision on the result")
		}
		routing, err := jsonx.Marshal(res.Routing.Map())
		if err != nil {
			t.Fatal(err)
		}
		bare, err := jsonx.Marshal(Result{Model: res.Model, Answers: res.Answers, Usage: res.Usage}.Map())
		if err != nil {
			t.Fatal(err)
		}
		want := append(bytes.TrimSuffix(bare, []byte("}")), []byte(`, "routing": `+string(routing)+"}")...)
		got, err := jsonx.Marshal(res.Map())
		if err != nil {
			t.Fatal(err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("result JSON\n got %s\nwant %s", got, want)
		}
		if !bytes.Contains(routing, []byte(`"detection": {"script": "devanagari"`)) {
			t.Errorf("routing JSON %s lacks the detection block", routing)
		}

		// encoding/json compacts it (D12) but keeps it valid and in order.
		std, err := json.Marshal(res)
		if err != nil {
			t.Fatalf("json.Marshal: %v", err)
		}
		if !json.Valid(std) || bytes.Index(std, []byte(`"routing"`)) < bytes.Index(std, []byte(`"usage"`)) {
			t.Errorf("json.Marshal(result) = %s", std)
		}
	})
}
