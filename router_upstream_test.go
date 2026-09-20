//nolint:gosmopolitan // every non-Latin literal here is an upstream routing fixture
package laya

import (
	"context"
	"errors"
	"slices"
	"testing"

	"github.com/MeKo-Christian/go-laya/presets"
)

// A port of original/tests/test_router.py, which is M3's corpus: there is no
// generated router fixture in testdata/, and 3.3.1 asks for the whole file.
//
// Sections 29-86 of that file -- detect_script, is_english,
// guess_latin_language and state_text -- are *not* here. Upstream keeps its
// language tests in the router file, and M2 ported them to lang/lang_test.go
// with the same line citations (TestDetectScript, TestIsEnglish,
// TestGuessLatinLanguage, TestStateText, TestStateTextKeysIgnored). Copying
// them again would mean two tables to keep in step with one Python source.
//
// What follows is sections 88-279, plus section 1 of test_local_e2e.py:45-67.

// qGeneric is test_router.py:118-119: a question set that matches no workflow
// signature. The bodies are irrelevant to routing -- only the ids are read --
// but they are kept close to upstream's so the fixture reads the same.
func qGeneric() Questions {
	return Questions{{
		ID: "dept",
		Q:  ChoiceQuestion{Ins: "Which team?", Opts: Labels("billing", "tech")},
	}}
}

// qTD is test_router.py:120: the customer_service signature, as nouls.
func qTD() Questions { return idQuestions(tdIDs...) }

// test_router.py:89-100.
func TestUpstreamWorkflowSignatures(t *testing.T) {
	td := map[string][]string{
		"agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
		"customer_service":          {"action", "category", "churn_risk", "needs_human", "urgency"},
		"invoice_processing":        {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
		"security_incidents":        {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
	}
	for wf, ids := range td {
		if got, ok := MatchTypedDecisionsWorkflow(idQuestions(ids...)); !ok || got != wf {
			t.Errorf("workflow/%s = (%q, %v), want (%q, true)", wf, got, ok, wf)
		}
	}

	// :98-100 -- partial overlap, superset and empty all match nothing.
	if got, ok := MatchTypedDecisionsWorkflow(idQuestions("urgency", "category")); ok {
		t.Errorf("workflow/partial overlap matched %q", got)
	}
	superset := append(slices.Clone(td["customer_service"]), "extra")
	if got, ok := MatchTypedDecisionsWorkflow(idQuestions(superset...)); ok {
		t.Errorf("workflow/superset matched %q", got)
	}
	if got, ok := MatchTypedDecisionsWorkflow(nil); ok {
		t.Errorf("workflow/empty matched %q", got)
	}
}

// test_router.py:103-113. The last alias is upstream's
// "convaiinnovations/laya".split("/")[-1], written out.
func TestUpstreamNameNormalisation(t *testing.T) {
	for _, c := range []struct{ alias, want string }{
		{"en", "english"},
		{"laya", "english"},
		{"multi", "multilingual"},
		{"ML", "multilingual"},
		{"typed", "typed-decisions"},
		{"typed_decisions", "typed-decisions"},
		{"English", "english"},
		{"laya", "english"},
	} {
		got, err := NormalizeModelName(c.alias)
		if err != nil {
			t.Errorf("alias/%s: %v", c.alias, err)
			continue
		}
		if got != c.want {
			t.Errorf("alias/%s = %q, want %q", c.alias, got, c.want)
		}
	}

	if _, err := NormalizeModelName("nope"); err == nil {
		t.Error("alias/unknown: should have raised")
	}
}

// test_router.py:122-163, the routing-decision table. Upstream asserts on
// d["model"] alone here; the reasons and the payload bytes are pinned
// separately in route_test.go.
func TestUpstreamRoutingDecisions(t *testing.T) {
	r, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	cases := []struct {
		label string
		state any
		qs    Questions
		opts  []RouteOption
		want  string
	}{
		{"english text", map[string]any{"body": "I was charged twice, please refund."}, qGeneric(), nil, "english"},
		{"hindi text", map[string]any{"body": "मुझसे दो बार शुल्क लिया गया"}, qGeneric(), nil, "multilingual"},
		{"japanese text", map[string]any{"body": "二重に請求されました"}, qGeneric(), nil, "multilingual"},
		{"korean text", map[string]any{"body": "두 번 청구되었습니다"}, qGeneric(), nil, "multilingual"},
		{"arabic text", map[string]any{"body": "تم خصم المبلغ مرتين"}, qGeneric(), nil, "multilingual"},
		{
			"german text",
			map[string]any{"body": "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung " +
				"fuer die Rechnung die nicht korrekt ist"},
			qGeneric(), nil, "multilingual",
		},
		{"explicit model", map[string]any{"body": "anything"}, qGeneric(), []RouteOption{ForModel("multilingual")}, "multilingual"},
		{
			"explicit model overrides script",
			map[string]any{"body": "मुझसे दो बार"},
			qGeneric(),
			[]RouteOption{ForModel("english")},
			"english",
		},
		{"explicit task", map[string]any{"body": "x"}, qGeneric(), []RouteOption{ForTask("typed_decisions")}, "typed-decisions"},
		{"explicit lang en", map[string]any{"body": "मुझसे दो बार"}, qGeneric(), []RouteOption{ForLang("en")}, "english"},
		{"explicit lang de", map[string]any{"body": "hello there"}, qGeneric(), []RouteOption{ForLang("de")}, "multilingual"},
		{"td workflow, auto OFF", map[string]any{"body": "I was charged twice"}, qTD(), nil, "english"},
		{"empty state", map[string]any{}, qGeneric(), nil, "english"},
		{"none state", nil, qGeneric(), nil, "english"},
	}

	for _, c := range cases {
		t.Run(c.label, func(t *testing.T) {
			d, err := r.Route(c.state, c.qs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if d.Model != c.want {
				t.Errorf("model = %q, want %q (%s)", d.Model, c.want, d.Reason)
			}
		})
	}
}

// test_router.py:144-151: auto task detection is opt-in, and an explicit model
// still beats a detected workflow.
func TestUpstreamAutoTaskDetection(t *testing.T) {
	auto, err := NewRouter(WithAutoTaskDetection(true))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	state := map[string]any{"body": "I was charged twice"}

	cases := []struct {
		label string
		qs    Questions
		opts  []RouteOption
		want  string
	}{
		{"td workflow, auto ON", qTD(), nil, "typed-decisions"},
		{"auto ON but generic questions", qGeneric(), nil, "english"},
		{"explicit beats workflow", qTD(), []RouteOption{ForModel("multilingual")}, "multilingual"},
	}
	for _, c := range cases {
		t.Run(c.label, func(t *testing.T) {
			d, err := auto.Route(state, c.qs, c.opts...)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if d.Model != c.want {
				t.Errorf("model = %q, want %q", d.Model, c.want)
			}
		})
	}
}

// test_router.py:154-163: the decision payload, and the custom default.
// "decision/is dict" and "decision/.model property" have no Go counterpart --
// RouteDecision is a struct with exported fields, so the property is the field
// and the dict is Map(), pinned in router_test.go.
func TestUpstreamDecisionPayload(t *testing.T) {
	r, err := NewRouter()
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	d, err := r.Route(map[string]any{"body": "मुझसे दो बार शुल्क लिया गया"}, qGeneric())
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if d.Repo != "convaiinnovations/laya/multilingual" {
		t.Errorf("repo = %q", d.Repo)
	}
	if d.Reason == "" {
		t.Error("reason is empty")
	}
	if d.Detection == nil || d.Detection.Script != "devanagari" {
		t.Errorf("detection = %+v, want script devanagari", d.Detection)
	}
	if d.Model != "multilingual" {
		t.Errorf("model = %q, want multilingual", d.Model)
	}

	// :162-163 -- digits alone detect nothing, so the configured default wins.
	custom, err := NewRouter(WithDefaultModel("multilingual"))
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}
	got, err := custom.Route("12345", qGeneric())
	if err != nil {
		t.Fatalf("Route: %v", err)
	}
	if got.Model != "multilingual" {
		t.Errorf("custom default gave %q, want multilingual", got.Model)
	}
}

// test_router.py:192-209. The Go stub loader stands in for the _load_stub
// monkeypatch; "cap 1 agents match order" has no counterpart, because the two
// views upstream has to keep in step are one map and one slice here.
func TestUpstreamLRUBookkeeping(t *testing.T) {
	t.Run("cap 1 keeps newest", func(t *testing.T) {
		rr, _ := stubbedRouter(t, 1)
		mustLoad(t, rr, "english")
		mustLoad(t, rr, "multilingual")
		if got := rr.Loaded(); !slices.Equal(got, []string{"multilingual"}) {
			t.Errorf("loaded = %v", got)
		}
	})

	t.Run("cap 2 evicts oldest", func(t *testing.T) {
		rr, _ := stubbedRouter(t, 2)
		mustLoad(t, rr, "english")
		mustLoad(t, rr, "multilingual")
		mustLoad(t, rr, "typed-decisions")
		if got := rr.Loaded(); !slices.Equal(got, []string{"multilingual", "typed-decisions"}) {
			t.Errorf("loaded = %v", got)
		}
	})

	t.Run("touch protects, then unload", func(t *testing.T) {
		rr, _ := stubbedRouter(t, 2)
		mustLoad(t, rr, "english")
		mustLoad(t, rr, "multilingual")
		mustLoad(t, rr, "english")
		mustLoad(t, rr, "typed-decisions")

		got := slices.Sorted(slices.Values(rr.Loaded()))
		if !slices.Equal(got, []string{"english", "typed-decisions"}) {
			t.Errorf("loaded = %v", got)
		}

		if err := rr.Unload("english"); err != nil {
			t.Fatalf("Unload: %v", err)
		}
		if slices.Contains(rr.Loaded(), "english") {
			t.Error("unload one left english resident")
		}
		if err := rr.Unload(); err != nil {
			t.Fatalf("Unload all: %v", err)
		}
		if got := rr.Loaded(); len(got) != 0 {
			t.Errorf("unload all left %v", got)
		}
	})
}

// test_router.py:213-231. The three DEFAULT_MODELS tuples, _repo_str, the
// bundle/standalone split and the local-path override.
func TestUpstreamBundleVersusStandalone(t *testing.T) {
	def := DefaultModels()
	for name, want := range map[string]ModelSpec{
		"english":         {Repo: bundleRepo},
		"multilingual":    {Repo: bundleRepo, Subfolder: "multilingual"},
		"typed-decisions": {Repo: bundleRepo, Subfolder: "typed-decisions"},
	} {
		if def[name] != want {
			t.Errorf("bundle/%s = %+v, want %+v", name, def[name], want)
		}
	}
	for _, c := range []struct {
		label string
		spec  ModelSpec
		want  string
	}{
		{"root", ModelSpec{Repo: bundleRepo}, "convaiinnovations/laya"},
		{"sub", ModelSpec{Repo: bundleRepo, Subfolder: "multilingual"}, "convaiinnovations/laya/multilingual"},
		{"plain string", ModelSpecFromString("some/repo"), "some/repo"},
	} {
		if got := c.spec.String(); got != c.want {
			t.Errorf("repo_str/%s = %q, want %q", c.label, got, c.want)
		}
	}

	// :228 -- the two registries must offer the same three names.
	std := StandaloneModels()
	if len(std) != len(def) {
		t.Errorf("standalone map complete: %d names, want %d", len(std), len(def))
	}
	for name := range def {
		if _, ok := std[name]; !ok {
			t.Errorf("standalone map complete: %q missing", name)
		}
	}

	hindi := map[string]any{"m": "मुझसे दो बार"}
	english := map[string]any{"m": "I was charged twice"}

	cases := []struct {
		label string
		opts  []RouterOption
		state any
		want  string
	}{
		{"default router uses bundle", nil, hindi, "convaiinnovations/laya/multilingual"},
		{"opt-in uses own repo", []RouterOption{WithStandaloneRepos(true)}, hindi, "convaiinnovations/laya-multilingual"},
		{"english unchanged", []RouterOption{WithStandaloneRepos(true)}, english, "convaiinnovations/laya"},
		{
			"local path kept",
			[]RouterOption{WithModels(map[string]ModelSpec{
				"english":      ModelSpecFromString("/tmp/en"),
				"multilingual": ModelSpecFromString("/tmp/ml"),
			})},
			hindi, "/tmp/ml",
		},
	}
	for _, c := range cases {
		t.Run(c.label, func(t *testing.T) {
			r, err := NewRouter(c.opts...)
			if err != nil {
				t.Fatalf("NewRouter: %v", err)
			}
			d, err := r.Route(c.state, qGeneric())
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if d.Repo != c.want {
				t.Errorf("repo = %q, want %q", d.Repo, c.want)
			}
		})
	}
}

// test_router.py:240-255. Upstream raises max_loaded by hand because its
// preload is stubbed out; SetMaxLoaded is the Go equivalent (task 3.1.6), and
// Preload doing the raising itself is asserted in lru_test.go.
func TestUpstreamPreload(t *testing.T) {
	rp, _ := stubbedRouter(t, 1)
	rp.SetMaxLoaded(max(rp.MaxLoaded(), 3))
	for _, n := range []string{"english", "multilingual", "typed-decisions"} {
		mustLoad(t, rp, n)
	}
	got := slices.Sorted(slices.Values(rp.Loaded()))
	if !slices.Equal(got, []string{"english", "multilingual", "typed-decisions"}) {
		t.Errorf("all three resident: got %v", got)
	}
	if rp.MaxLoaded() < 3 {
		t.Errorf("max_loaded = %d, want >= 3", rp.MaxLoaded())
	}

	rp2, _ := stubbedRouter(t, 1)
	rp2.SetMaxLoaded(max(rp2.MaxLoaded(), 2))
	mustLoad(t, rp2, "english")
	mustLoad(t, rp2, "multilingual")
	mustLoad(t, rp2, "english") // a touch must not evict

	got = slices.Sorted(slices.Values(rp2.Loaded()))
	if !slices.Equal(got, []string{"english", "multilingual"}) {
		t.Errorf("subset resident: got %v", got)
	}
}

// test_router.py:259-270.
func TestUpstreamAttach(t *testing.T) {
	ra, _ := stubbedRouter(t, 1)
	sentinel := &stubAgent{name: "already-built"}

	if err := ra.Attach("english", sentinel); err != nil {
		t.Fatalf("Attach: %v", err)
	}
	if !slices.Contains(ra.Loaded(), "english") {
		t.Error("attach/counts as resident")
	}
	if ra.MaxLoaded() < 1 {
		t.Error("attach/raises max_loaded to hold it")
	}

	ra.SetMaxLoaded(max(ra.MaxLoaded(), 2))
	mustLoad(t, ra, "multilingual")

	got := slices.Sorted(slices.Values(ra.Loaded()))
	if !slices.Equal(got, []string{"english", "multilingual"}) {
		t.Errorf("attach/survives a later load: %v", got)
	}
	again, err := ra.Load(context.Background(), "english")
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if again != Agent(sentinel) {
		t.Error("attach/still the same object")
	}

	other, _ := stubbedRouter(t, 1)
	if err := other.Attach("en", &stubAgent{name: "x"}); err != nil {
		t.Errorf("attach/accepts aliases: %v", err)
	}
}

// Section 1 of original/tests/test_local_e2e.py:45-67, which the file itself
// labels "no weights loaded". It routes through a local-directory override,
// the same path test_router.py:229-231 exercises, and asserts on the model
// alone.
//
// German and French are deliberately long and accent-free -- "moechte",
// "ete facture" -- so they prove the Latin-script non-English branch fires on
// word shape rather than on diacritics.
func TestUpstreamLocalE2ERouting(t *testing.T) {
	r, err := NewRouter(
		WithMaxLoaded(1),
		WithModels(map[string]ModelSpec{
			"english":         ModelSpecFromString("/models/laya"),
			"multilingual":    ModelSpecFromString("/models/laya-multilingual"),
			"typed-decisions": ModelSpecFromString("/models/laya-typed-decisions"),
		}),
	)
	if err != nil {
		t.Fatalf("NewRouter: %v", err)
	}

	langs := []struct{ label, text, want string }{
		{"english", "I was charged twice for invoice 4411, please refund it today.", "english"},
		{
			"german", "Der Kunde wurde zweimal belastet und moechte eine Rueckerstattung fuer die " +
				"Rechnung die nicht korrekt ist und nicht bezahlt wurde", "multilingual",
		},
		{
			"french", "Le client a ete facture deux fois et il demande un remboursement pour la " +
				"facture qui a ete payee le mois dernier avec la carte", "multilingual",
		},
		{"hindi", "मुझसे इनवॉइस 4411 के लिए दो बार शुल्क लिया गया, कृपया आज ही धनवापसी करें।", "multilingual"},
		{"japanese", "請求書4411で二重に請求されました。本日中に返金してください。", "multilingual"},
		{"korean", "청구서 4411에 대해 두 번 청구되었습니다. 오늘 환불해 주세요.", "multilingual"},
		{"arabic", "تم خصم المبلغ مرتين للفاتورة 4411، يرجى رد المبلغ اليوم.", "multilingual"},
		{"tamil", "விலைப்பட்டியல் 4411க்கு இருமுறை கட்டணம் வசூலிக்கப்பட்டது, இன்றே திரும்பப் பெறவும்.", "multilingual"},
		{"russian", "С меня дважды списали деньги по счёту 4411, пожалуйста верните средства.", "multilingual"},
		{"chinese", "发票4411被重复扣款，请今天退款。", "multilingual"},
		{"thai", "ถูกเรียกเก็บเงินสองครั้งสำหรับใบแจ้งหนี้ 4411 กรุณาคืนเงินวันนี้", "multilingual"},
	}

	// laya.triage_questions() upstream; the ids are what routing reads.
	qs := presetTriage(t)

	for _, c := range langs {
		t.Run(c.label, func(t *testing.T) {
			d, err := r.Route(map[string]any{"message": c.text}, qs)
			if err != nil {
				t.Fatalf("Route: %v", err)
			}
			if d.Model != c.want {
				t.Errorf("model = %q, want %q (%s)", d.Model, c.want, d.Reason)
			}
		})
	}

	// Routing never loads: the whole section runs with a router that has no
	// loader at all, and would fail with ErrNoLoader the moment it tried.
	if _, err := r.Load(context.Background(), "english"); !errors.Is(err, ErrNoLoader) {
		t.Errorf("Load error = %v, want ErrNoLoader", err)
	}
}

// presetTriage is laya.triage_questions() (test_local_e2e.py:48). Routing only
// reads the ids, but using the real preset keeps the fixture honest: if the
// triage taxonomy ever grew into one of the four workflow signatures, this
// section would start routing to typed-decisions and say so.
func presetTriage(t *testing.T) Questions {
	t.Helper()
	return presets.TriageQuestions()
}
