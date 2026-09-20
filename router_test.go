package laya

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/lang"
)

// _repo_str, run against the pinned reference environment:
//
//	_repo_str(DEFAULT_MODELS["english"])       -> convaiinnovations/laya
//	_repo_str(DEFAULT_MODELS["multilingual"])  -> convaiinnovations/laya/multilingual
//	_repo_str(STANDALONE_MODELS["multilingual"]) -> convaiinnovations/laya-multilingual
//	_repo_str("/tmp/ml")                       -> /tmp/ml
//
// The last one is load-bearing: test_router.py:229-231 overrides the registry
// with local directories and expects the path back unchanged (invariant #47).
func TestModelSpecString(t *testing.T) {
	cases := []struct {
		name string
		spec ModelSpec
		want string
	}{
		{"bundle-root", ModelSpec{Repo: "convaiinnovations/laya"}, "convaiinnovations/laya"},
		{"bundle-sub", ModelSpec{Repo: "convaiinnovations/laya", Subfolder: "multilingual"}, "convaiinnovations/laya/multilingual"},
		{"standalone", ModelSpecFromString("convaiinnovations/laya-multilingual"), "convaiinnovations/laya-multilingual"},
		{"local-path", ModelSpecFromString("/tmp/ml"), "/tmp/ml"},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.spec.String(); got != c.want {
				t.Errorf("String() = %q, want %q", got, c.want)
			}
		})
	}
}

// The registries are handed out as copies. Python's Router does
// `dict(DEFAULT_MODELS)` on every construction (router.py:151) so a caller can
// never reach the package-level dict; a Go map returned by value would be the
// same map every time.
func TestRegistriesAreCopies(t *testing.T) {
	for _, reg := range []struct {
		name string
		fn   func() map[string]ModelSpec
	}{{"default", DefaultModels}, {"standalone", StandaloneModels}} {
		t.Run(reg.name, func(t *testing.T) {
			first := reg.fn()
			if len(first) != 3 {
				t.Fatalf("got %d models, want 3", len(first))
			}
			first["english"] = ModelSpec{Repo: "tampered"}
			delete(first, "multilingual")

			second := reg.fn()
			if second["english"].Repo == "tampered" {
				t.Error("mutating the returned map reached the package registry")
			}
			if _, ok := second["multilingual"]; !ok {
				t.Error("deleting from the returned map reached the package registry")
			}
		})
	}
}

// sorted(DEFAULT_MODELS) == sorted(STANDALONE_MODELS) is asserted upstream at
// test_router.py:228: the two registries must offer the same three names.
func TestRegistriesAgreeOnNames(t *testing.T) {
	def, std := DefaultModels(), StandaloneModels()
	for name := range def {
		if _, ok := std[name]; !ok {
			t.Errorf("%q is in DefaultModels but not StandaloneModels", name)
		}
	}
	for name := range std {
		if _, ok := def[name]; !ok {
			t.Errorf("%q is in StandaloneModels but not DefaultModels", name)
		}
	}
}

func TestNormalizeModelName(t *testing.T) {
	// Every alias upstream defines (router.py:65-70), plus the canonical names,
	// plus the two case variants test_router.py:104-108 pins and the
	// "convaiinnovations/laya".split("/")[-1] form it derives at :107.
	cases := map[string]string{
		"english": "english", "multilingual": "multilingual", "typed-decisions": "typed-decisions",
		"en": "english", "laya": "english", "default": "english",
		"multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
		"typed": "typed-decisions", "typed_decisions": "typed-decisions",
		"laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
		"ML": "multilingual", "English": "english",
		"  en  ": "english", "TYPED_DECISIONS": "typed-decisions",
	}
	for in, want := range cases {
		t.Run(in, func(t *testing.T) {
			got, err := NormalizeModelName(in)
			if err != nil {
				t.Fatalf("NormalizeModelName(%q): %v", in, err)
			}
			if got != want {
				t.Errorf("NormalizeModelName(%q) = %q, want %q", in, got, want)
			}
		})
	}
}

// normalise_name raises on anything outside DEFAULT_MODELS (router.py:103-106,
// test_router.py:109-113). Two shapes matter: a name nobody recognises, and a
// hub repo id, which is *not* a model name even though "laya" alone is.
func TestNormalizeModelNameRejectsUnknown(t *testing.T) {
	for _, in := range []string{"nope", "", "convaiinnovations/laya", "englsh", "typed decisions"} {
		t.Run(in, func(t *testing.T) {
			got, err := NormalizeModelName(in)
			if err == nil {
				t.Fatalf("NormalizeModelName(%q) = %q, want an error", in, got)
			}
			if !errors.Is(err, ErrUnknownModel) {
				t.Errorf("error %v does not wrap ErrUnknownModel", err)
			}
		})
	}
}

// Exact set equality, so a superset never matches (invariant #42,
// test_router.py:96-100). The four signatures are router.py:74-79.
func TestMatchTypedDecisionsWorkflow(t *testing.T) {
	sig := map[string][]string{
		"agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
		"customer_service":          {"action", "category", "churn_risk", "needs_human", "urgency"},
		"invoice_processing":        {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
		"security_incidents":        {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
	}
	for want, ids := range sig {
		t.Run(want, func(t *testing.T) {
			got, ok := MatchTypedDecisionsWorkflow(idQuestions(ids...))
			if !ok || got != want {
				t.Errorf("MatchTypedDecisionsWorkflow = (%q, %v), want (%q, true)", got, ok, want)
			}
		})
	}

	misses := map[string][]string{
		"partial-overlap": {"urgency", "category"},
		"superset":        {"action", "category", "churn_risk", "needs_human", "urgency", "extra"},
		"subset":          {"action", "category", "churn_risk", "needs_human"},
		"empty":           nil,
		"unrelated":       {"a", "b"},
	}
	for name, ids := range misses {
		t.Run(name, func(t *testing.T) {
			if got, ok := MatchTypedDecisionsWorkflow(idQuestions(ids...)); ok {
				t.Errorf("MatchTypedDecisionsWorkflow matched %q, want no match", got)
			}
		})
	}
}

// The five keys are always present and always in Python's insertion order
// (invariant #45). jsonx.Marshal is the outer encoder, per D12: routing it
// through encoding/json would compact the ", " and ": " separators back out,
// which TestRouteDecisionMarshalledByEncodingJSONIsCompacted pins.
func TestRouteDecisionMapOrder(t *testing.T) {
	d := RouteDecision{Model: "english", Repo: "convaiinnovations/laya", Reason: "English Latin text"}
	want := []string{"model", "repo", "reason", "detection", "workflow"}

	got := d.Map()
	if len(got) != len(want) {
		t.Fatalf("Map() has %d keys, want %d: %v", len(got), len(want), got)
	}
	for i, key := range want {
		if got[i].Key != key {
			t.Errorf("Map()[%d].Key = %q, want %q", i, got[i].Key, key)
		}
	}
}

// A decision with neither a detection nor a workflow must emit both as null,
// not omit them: Python sets them explicitly to None and json.dumps writes the
// keys (invariant #45).
func TestRouteDecisionMarshalsNullsNotOmissions(t *testing.T) {
	d := RouteDecision{Model: "english", Repo: "convaiinnovations/laya", Reason: "explicit model='en'"}

	got, err := jsonx.Marshal(d.Map())
	if err != nil {
		t.Fatalf("Marshal: %v", err)
	}
	want := `{"model": "english", "repo": "convaiinnovations/laya", ` +
		`"reason": "explicit model='en'", "detection": null, "workflow": null}`
	if string(got) != want {
		t.Errorf("got  %s\nwant %s", got, want)
	}
}

// D12, restated for this type. encoding/json runs compact() over whatever a
// MarshalJSON returns, so a RouteDecision handed to json.Marshal loses the
// separators. The test exists so that a later emitter reaching for json.Marshal
// finds a failing assertion rather than a silently wrong payload.
func TestRouteDecisionMarshalledByEncodingJSONIsCompacted(t *testing.T) {
	d := RouteDecision{Model: "english", Repo: "convaiinnovations/laya", Reason: "English Latin text"}

	viaJSON, err := json.Marshal(d)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}
	viaJSONX, err := jsonx.Marshal(d.Map())
	if err != nil {
		t.Fatalf("jsonx.Marshal: %v", err)
	}
	if string(viaJSON) == string(viaJSONX) {
		t.Fatal("encoding/json no longer compacts MarshalJSON output; D12 needs revisiting")
	}
}

// Detection is re-exported rather than redeclared: lang already owns it, and a
// second struct at the root would be D11's mistake again.
//
// The two assignments below are the assertion, and they are compile-time. Go
// function types are identical only when their parameter types are identical,
// so these lines do not build unless laya.Detection and lang.Detection are one
// type. A structurally identical but separately declared struct would fail
// here, which a conversion-based check would not catch.
var (
	_ func(Detection)      = func(lang.Detection) {}
	_ func(lang.Detection) = func(Detection) {}
)

func TestDetectionAliasCarriesLangResults(t *testing.T) {
	d := lang.Analyse("Hello")
	if d.Script != "latin" || !d.IsEnglish {
		t.Errorf("Analyse(%q) = %+v, want latin and English", "Hello", d)
	}

	// Map() comes from lang, and the root's RouteDecision embeds its output
	// verbatim, so the alias has to carry the methods too.
	if got := d.Map(); len(got) != 5 {
		t.Errorf("Detection.Map() has %d keys, want 5", len(got))
	}
}

// idQuestions builds a question set carrying just these ids. The workflow match
// looks at ids alone (router.py:113 takes set(questions)), so the question
// bodies are irrelevant and a noul with no criteria is the cheapest filler.
func idQuestions(ids ...string) Questions {
	qs := make(Questions, len(ids))
	for i, id := range ids {
		qs[i] = NamedQuestion{ID: id, Q: NoulQuestion{Ins: id}}
	}
	return qs
}
