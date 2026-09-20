//nolint:gosmopolitan // the Devanagari literals are routing fixtures, not UI text
package laya

import (
	"errors"
	"testing"

	"github.com/MeKo-Christian/go-laya/jsonx"
)

// The want strings are the output of
//
//	json.dumps(Router().route(**kwargs), ensure_ascii=False)
//
// run against the pinned reference environment. They pin the whole decision at
// once -- model, repo, the reason text including its Python repr quoting, the
// detection block and its key order, and the separators -- because every one of
// those is a way the port can be subtly wrong while still looking right.
//
// jsonx.Marshal is the encoder, not encoding/json: D12, and
// TestRouteDecisionMarshalledByEncodingJSONIsCompacted pins why.
func TestRouteDecisionBytesMatchPython(t *testing.T) {
	r, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	cases := []struct {
		name  string
		state any
		qs    Questions
		opts  []RouteOption
		want  string
	}{
		{
			name:  "detection-hindi",
			state: "नमस्ते दुनिया",
			qs:    idQuestions("a"),
			want: `{"model": "multilingual", "repo": "convaiinnovations/laya/multilingual", ` +
				`"reason": "non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it", ` +
				`"detection": {"script": "devanagari", "script_profile": {"devanagari": 1.0}, "language": null, ` +
				`"is_english": false, "non_latin_fraction": 1.0}, "workflow": null}`,
		},
		{
			name:  "detection-english",
			state: "Please refund my order",
			qs:    idQuestions("a"),
			want: `{"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text", ` +
				`"detection": {"script": "latin", "script_profile": {"latin": 1.0}, "language": "en", ` +
				`"is_english": true, "non_latin_fraction": 0.0}, "workflow": null}`,
		},
		{
			name:  "explicit-model",
			state: "hi",
			opts:  []RouteOption{ForModel("multilingual")},
			want: `{"model": "multilingual", "repo": "convaiinnovations/laya/multilingual", ` +
				`"reason": "explicit model='multilingual'", "detection": null, "workflow": null}`,
		},
		{
			name:  "explicit-lang",
			state: "Hello",
			opts:  []RouteOption{ForLang("de")},
			want: `{"model": "multilingual", "repo": "convaiinnovations/laya/multilingual", ` +
				`"reason": "explicit lang='de'", "detection": null, "workflow": null}`,
		},
		{
			name:  "unknown-script",
			state: "12345 6789",
			want: `{"model": "english", "repo": "convaiinnovations/laya", ` +
				`"reason": "no letters detected in state; using default (english)", ` +
				`"detection": {"script": "unknown", "script_profile": {}, "language": null, ` +
				`"is_english": true, "non_latin_fraction": 0.0}, "workflow": null}`,
		},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := r.Route(c.state, c.qs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			got, err := jsonx.Marshal(d.Map())
			if err != nil {
				t.Fatalf("Marshal: %v", err)
			}
			if string(got) != c.want {
				t.Errorf("got  %s\nwant %s", got, c.want)
			}
		})
	}
}

// tdIDs is the customer_service signature (router.py:76), the one the upstream
// routing tests use to exercise the workflow branch.
var tdIDs = []string{"action", "category", "churn_risk", "needs_human", "urgency"}

// Precedence is invariant #39, and the two rows that matter most are the ones
// where a *lower* branch would also have produced an answer: an explicit model
// on Devanagari text, and typed-decisions questions with auto detection off.
func TestRoutePrecedence(t *testing.T) {
	plain, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	auto, err := NewRouter(WithAutoTaskDetection(true))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	cases := []struct {
		name   string
		router *Router
		state  any
		qs     Questions
		opts   []RouteOption
		want   string
	}{
		{"model-beats-script", plain, "मुझसे दो बार", idQuestions("a"), []RouteOption{ForModel("english")}, ModelEnglish},
		{"model-beats-workflow", auto, "x", idQuestions(tdIDs...), []RouteOption{ForModel("multilingual")}, ModelMultilingual},
		{"task-beats-workflow", auto, "x", idQuestions(tdIDs...), []RouteOption{ForTask("en")}, ModelEnglish},
		{"workflow-beats-lang", auto, "Hello", idQuestions(tdIDs...), []RouteOption{ForLang("de")}, ModelTypedDecisions},
		{"workflow-off-falls-through", plain, "I was charged twice", idQuestions(tdIDs...), nil, ModelEnglish},
		{"lang-beats-detection", plain, "मुझसे दो बार", idQuestions("a"), []RouteOption{ForLang("en")}, ModelEnglish},
		{"lang-de-on-english-text", plain, "hello there", idQuestions("a"), []RouteOption{ForLang("de")}, ModelMultilingual},
		{"detection-decides", plain, "二重に請求されました", idQuestions("a"), nil, ModelMultilingual},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := c.router.Route(c.state, c.qs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if d.Model != c.want {
				t.Errorf("Model = %q, want %q (reason: %s)", d.Model, c.want, d.Reason)
			}
		})
	}
}

// The workflow leak (invariant #45, task 3.1.5). match_typed_decisions_workflow
// runs at router.py:264 whether or not auto detection is on, and the result is
// passed into the lang and detection branches -- so a decision can report a
// workflow that played no part in making it. Reproducing that is the point;
// suppressing it would be the tidier bug.
func TestRouteWorkflowLeaksIntoLaterBranches(t *testing.T) {
	plain, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	cases := []struct {
		name         string
		qs           Questions
		opts         []RouteOption
		wantWorkflow string // "" means nil
	}{
		{"lang-path-carries-it", idQuestions(tdIDs...), []RouteOption{ForLang("de")}, "customer_service"},
		{"detection-path-carries-it", idQuestions(tdIDs...), nil, "customer_service"},
		{"model-path-clears-it", idQuestions(tdIDs...), []RouteOption{ForModel("en")}, ""},
		{"task-path-clears-it", idQuestions(tdIDs...), []RouteOption{ForTask("en")}, ""},
		{"generic-questions", idQuestions("a", "b"), nil, ""},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := plain.Route("Hello", c.qs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			switch {
			case c.wantWorkflow == "" && d.Workflow != nil:
				t.Errorf("Workflow = %q, want nil", *d.Workflow)
			case c.wantWorkflow != "" && d.Workflow == nil:
				t.Errorf("Workflow = nil, want %q", c.wantWorkflow)
			case c.wantWorkflow != "" && *d.Workflow != c.wantWorkflow:
				t.Errorf("Workflow = %q, want %q", *d.Workflow, c.wantWorkflow)
			}
		})
	}
}

// Every reason string, verbatim from a reference run. These are the strings a
// caller reads in a log to understand a decision, and three of them embed a
// Python repr, which is why jsonx.ReprString exists.
func TestRouteReasons(t *testing.T) {
	plain, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	multiDefault, err := NewRouter(WithDefaultModel("multilingual"))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	auto, err := NewRouter(WithAutoTaskDetection(true))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	germanBody := "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung " +
		"fuer die Rechnung die nicht korrekt ist"

	cases := []struct {
		name   string
		router *Router
		state  any
		qs     Questions
		opts   []RouteOption
		want   string
	}{
		{"model", plain, "x", nil, []RouteOption{ForModel("multilingual")}, "explicit model='multilingual'"},
		{"model-alias-keeps-raw-text", plain, "x", nil, []RouteOption{ForModel("ML")}, "explicit model='ML'"},
		{"task-underscore", plain, "x", nil, []RouteOption{ForTask("typed_decisions")}, "explicit task='typed_decisions'"},
		{"task-hyphen", plain, "x", nil, []RouteOption{ForTask("typed-decisions")}, "explicit task='typed-decisions'"},
		{"task-alias", plain, "x", nil, []RouteOption{ForTask("en")}, "explicit task='en'"},
		{"lang", plain, "x", nil, []RouteOption{ForLang("de")}, "explicit lang='de'"},
		{"lang-region", plain, "मुझसे", nil, []RouteOption{ForLang("en-GB")}, "explicit lang='en-GB'"},
		{
			"workflow", auto, "Hello", idQuestions(tdIDs...), nil,
			"question ids match the 'customer_service' typed-decisions workflow",
		},
		{
			"no-letters", plain, "12345 6789", nil, nil,
			"no letters detected in state; using default (english)",
		},
		{
			"no-letters-custom-default", multiDefault, "12345 6789", nil, nil,
			"no letters detected in state; using default (multilingual)",
		},
		{
			"non-latin", plain, "मुझसे दो बार", nil, nil,
			"non-Latin script (devanagari, 100% of letters); the English checkpoint cannot read it",
		},
		{
			// 3 Latin and 5 Devanagari letters: non_latin_fraction is
			// exactly 0.625, so the percent is 62.5 and Python's "%.0f"
			// rounds it half-to-even, to 62. Half-away-from-zero would
			// say 63.
			"non-latin-half-way", plain, "abcनमसतप", nil, nil,
			"non-Latin script (devanagari, 62% of letters); the English checkpoint cannot read it",
		},
		{
			"latin-not-english", plain, germanBody, nil, nil,
			"Latin script but language looks like 'de', not English",
		},
		{"english", plain, "Please refund my order", nil, nil, "English Latin text"},
	}

	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			d, err := c.router.Route(c.state, c.qs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if d.Reason != c.want {
				t.Errorf("Reason =\n  %q\nwant\n  %q", d.Reason, c.want)
			}
		})
	}
}

// Python's "%.0f" rounds the decimal representation half-to-even. Go's
// strconv agrees and math.Round does not, so the reason text is one more place
// where reaching for the obvious rounding helper is wrong (invariant #29).
func TestPercentRoundsHalfToEven(t *testing.T) {
	cases := map[float64]string{
		0.0: "0", 0.005: "0", 0.015: "2", 0.025: "2",
		0.625: "62", 0.995: "100", 1.0: "100", 0.6667: "67", 0.4444: "44",
	}
	for in, want := range cases {
		if got := percent(in); got != want {
			t.Errorf("percent(%v) = %q, want %q", in, got, want)
		}
	}
}

// An unknown name must fail at the boundary, not route to the default.
func TestRouteRejectsUnknownNames(t *testing.T) {
	r, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	for _, opt := range []struct {
		name string
		opt  RouteOption
	}{{"model", ForModel("nope")}, {"task", ForTask("nope")}} {
		t.Run(opt.name, func(t *testing.T) {
			if _, err := r.Route("x", nil, opt.opt); !errors.Is(err, ErrUnknownModel) {
				t.Errorf("error = %v, want ErrUnknownModel", err)
			}
		})
	}
}
