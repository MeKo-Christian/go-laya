package laya

import (
	"context"
	"fmt"
	"maps"
	"slices"
	"strconv"
	"strings"
	"sync"

	"github.com/MeKo-Christian/go-laya/jsonx"
	"github.com/MeKo-Christian/go-laya/lang"
	"github.com/MeKo-Christian/go-laya/question"
)

// bundleRepo is the Hub repo that carries all three checkpoints; only the
// requested subfolder is downloaded (router.py:36).
const bundleRepo = "convaiinnovations/laya"

// The three checkpoint names, as DEFAULT_MODELS spells them (router.py:37-41).
// They are the canonical form every alias resolves to and the only values
// RouteDecision.Model ever takes.
const (
	ModelEnglish        = "english"
	ModelMultilingual   = "multilingual"
	ModelTypedDecisions = "typed-decisions"
)

// ModelSpec locates one checkpoint: a Hub repo id or a local directory, and
// optionally a subfolder within it.
//
// Upstream carries two shapes in one dict -- a (repo, subfolder) tuple in
// DEFAULT_MODELS and a bare string in STANDALONE_MODELS (router.py:37-48) --
// and _split normalises them at every use. One struct with an empty Subfolder
// covers both, so there is nothing left to normalise.
type ModelSpec struct {
	// Repo is a Hub repo id or a local filesystem path.
	Repo string
	// Subfolder is the directory within the repo, empty for the root.
	Subfolder string
}

// ModelSpecFromString builds a spec from a bare repo id or local path, which is
// upstream's string form. A local-directory override goes through here, and
// test_router.py:229-231 depends on the path surviving into the decision
// unchanged.
func ModelSpecFromString(repo string) ModelSpec {
	return ModelSpec{Repo: repo}
}

// String is the human-readable id of the spec: "repo/subfolder", or "repo"
// when there is no subfolder. It reproduces router._repo_str (invariant #47),
// and it is what RouteDecision.Repo carries.
func (s ModelSpec) String() string {
	if s.Subfolder == "" {
		return s.Repo
	}
	return s.Repo + "/" + s.Subfolder
}

// defaultModels is DEFAULT_MODELS (router.py:37-41): the bundle repo, one
// subfolder per checkpoint.
var defaultModels = map[string]ModelSpec{
	ModelEnglish:        {Repo: bundleRepo},
	ModelMultilingual:   {Repo: bundleRepo, Subfolder: ModelMultilingual},
	ModelTypedDecisions: {Repo: bundleRepo, Subfolder: ModelTypedDecisions},
}

// standaloneModels is STANDALONE_MODELS (router.py:44-48): the same three
// checkpoints in their own repos, for anyone who prefers them.
// English's own repo *is* the bundle repo -- upstream writes the string out a
// second time (router.py:46), and the two genuinely coincide.
var standaloneModels = map[string]ModelSpec{
	ModelEnglish:        {Repo: bundleRepo},
	ModelMultilingual:   {Repo: "convaiinnovations/laya-multilingual"},
	ModelTypedDecisions: {Repo: "convaiinnovations/laya-typed-decisions"},
}

// DefaultModels returns the bundle registry: one repo, three subfolders.
// The map is a fresh copy, so a caller can edit it and pass it to WithModels
// without reaching the package-level table.
func DefaultModels() map[string]ModelSpec { return copyModels(defaultModels) }

// StandaloneModels returns the per-checkpoint registry, the one
// WithStandaloneRepos selects. The map is a fresh copy.
func StandaloneModels() map[string]ModelSpec { return copyModels(standaloneModels) }

func copyModels(src map[string]ModelSpec) map[string]ModelSpec {
	out := make(map[string]ModelSpec, len(src))
	maps.Copy(out, src)
	return out
}

// modelAliases is _ALIASES (router.py:65-70): the names people are likely to
// type. The lookup is a single pass, not transitive -- every value here is
// already canonical.
var modelAliases = map[string]string{
	"en":                   ModelEnglish,
	"laya":                 ModelEnglish,
	"default":              ModelEnglish,
	"multi":                ModelMultilingual,
	"ml":                   ModelMultilingual,
	"laya-multilingual":    ModelMultilingual,
	"typed":                ModelTypedDecisions,
	"typed_decisions":      ModelTypedDecisions,
	"laya-typed-decisions": ModelTypedDecisions,
	"decisions":            ModelTypedDecisions,
}

// NormalizeModelName resolves a model name or alias to one of the three
// canonical names, reproducing router.normalise_name (invariant #40): trim,
// lower-case, one alias lookup, then membership.
//
// Membership is checked against the *default* registry even on a router built
// with WithStandaloneRepos or WithModels, because that is what upstream does
// (router.py:103 tests `key not in DEFAULT_MODELS` unconditionally). An
// override may replace where a checkpoint comes from; it may not invent a
// fourth name, and the checkpoint's type embedding is why.
//
// The error wraps ErrUnknownModel. Its text is Go's own: upstream interpolates
// Python list reprs into the message and no test asserts on it
// (test_router.py:109-113 only checks that it raises), so reproducing the
// formatting would be cosplay with nothing pinning it (PLAN.md task 3.1.6).
func NormalizeModelName(name string) (string, error) {
	key := strings.ToLower(strings.TrimSpace(name))
	if canonical, ok := modelAliases[key]; ok {
		key = canonical
	}
	if _, ok := defaultModels[key]; !ok {
		return "", fmt.Errorf("%w: %q; choose one of %s, or an alias: %s",
			ErrUnknownModel, name, sortedKeys(defaultModels), sortedKeys(modelAliases))
	}
	return key, nil
}

// typedDecisionWorkflow is one entry of _TYPED_DECISION_WORKFLOWS
// (router.py:74-79). A slice rather than a map so the scan order is upstream's;
// the four signatures are disjoint, so it cannot change an answer, but a
// reader should not have to prove that.
type typedDecisionWorkflow struct {
	name string
	ids  []string
}

// verbatim from router.py:74-79. Folding the repeated ones into constants
// would invite reuse across unrelated workflows and make the table harder to
// audit -- the same reasoning .golangci.yml records for presets/.
//
//nolint:goconst // the question ids are upstream signature data reproduced
var typedDecisionWorkflows = []typedDecisionWorkflow{
	{"agent_trace_observability", []string{"action", "needs_review", "outcome", "risk", "urgency"}},
	{"customer_service", []string{"action", "category", "churn_risk", "needs_human", "urgency"}},
	{"invoice_processing", []string{"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"}},
	{"security_incidents", []string{"credential_compromise", "disposition", "severity", "true_positive", "urgency"}},
}

// MatchTypedDecisionsWorkflow names the typed-decisions workflow whose question
// ids these are, if any.
//
// The match is exact set equality (invariant #42), so an unrelated schema that
// happens to contain "urgency" is never captured and neither is a superset that
// contains a whole signature. Duplicated ids cannot occur -- Questions.Validate
// rejects them -- but the comparison is set-based either way, as Python's is.
func MatchTypedDecisionsWorkflow(qs question.Questions) (string, bool) {
	ids := make(map[string]struct{}, len(qs))
	for _, nq := range qs {
		ids[nq.ID] = struct{}{}
	}

	for _, wf := range typedDecisionWorkflows {
		if len(wf.ids) != len(ids) {
			continue
		}
		if containsAll(ids, wf.ids) {
			return wf.name, true
		}
	}
	return "", false
}

func containsAll(have map[string]struct{}, want []string) bool {
	for _, id := range want {
		if _, ok := have[id]; !ok {
			return false
		}
	}
	return true
}

// RouteDecision is the routing outcome: which checkpoint, why, and what was
// detected. It is router.RouteDecision (router.py:82-97), which is a dict
// subclass upstream; here it is a struct with a Map for callers who want the
// dict.
//
// All five keys are always present. Detection is nil on every path but the
// detection one, and Workflow is nil on the model and task paths but carries
// the matched name on the lang and detection paths even though it did not
// drive the decision (invariant #45) -- upstream computes it before those
// branches run and passes it through.
type RouteDecision struct {
	// Model is the canonical checkpoint name.
	Model string `json:"model"`
	// Repo is the spec's String form. Always a string, including on the
	// auto-workflow path where upstream leaks a raw tuple (invariant #46).
	Repo string `json:"repo"`
	// Reason is the human-readable justification, reproduced verbatim from
	// upstream including its Python repr quoting.
	Reason string `json:"reason"`
	// Detection is the script and language analysis, nil off the detection
	// path.
	Detection *Detection `json:"detection"`
	// Workflow is the matched typed-decisions workflow, nil when none
	// matched.
	Workflow *string `json:"workflow"`
}

// Map returns the decision as the ordered object Python builds, in the keyword
// order of the RouteDecision(...) calls: model, repo, reason, detection,
// workflow. Hand this to jsonx.Marshal when the bytes have to match json.dumps.
func (d RouteDecision) Map() Obj {
	var detection any
	if d.Detection != nil {
		detection = d.Detection.Map()
	}
	var workflow any
	if d.Workflow != nil {
		workflow = *d.Workflow
	}

	return Obj{
		{Key: "model", Value: d.Model},
		{Key: "repo", Value: d.Repo},
		{Key: "reason", Value: d.Reason},
		{Key: "detection", Value: detection},
		{Key: "workflow", Value: workflow},
	}
}

// MarshalJSON emits the five keys in Python's order with Python's separators.
//
// Beware D12: encoding/json compacts whatever a MarshalJSON returns, so
// json.Marshal(decision) does *not* produce Python's bytes. Use
// jsonx.Marshal(d.Map()) wherever byte equality is the requirement; this method
// is for the callers who only need well-formed, correctly ordered JSON.
func (d RouteDecision) MarshalJSON() ([]byte, error) {
	b, err := jsonx.Marshal(d.Map())
	if err != nil {
		return nil, fmt.Errorf("laya: marshal route decision: %w", err)
	}
	return b, nil
}

// sortedKeys is only for error text: both registries and the alias table are
// maps, and an error that lists them in a different order every run is one
// nobody can diff.
func sortedKeys[V any](m map[string]V) []string {
	out := make([]string, 0, len(m))
	for k := range m {
		out = append(out, k)
	}
	slices.Sort(out)
	return out
}

// Router lazily loads laya checkpoints and sends each request to the right
// one. It is router.Router (router.py:122-314).
//
// Route is pure: it never loads or runs anything, never touches the network,
// and needs no context. That is upstream's own claim about it
// (test_router.py:1) and it is what makes the whole decision layer testable
// without weights.
//
// Unlike Python's, this Router is safe for concurrent use. Upstream's is not,
// and a Go server would reach for one from several goroutines on the first
// day.
//
// # Cost
//
// Routing is by language, as upstream, not by cost. Spike S3 measured
// laya-multilingual at 2.2-2.7x the speed of either ModernBERT-large
// checkpoint and the only one that answers in under a second on CPU; the
// per-checkpoint numbers are in BENCHMARKS.md. There is deliberately no
// cost-biased routing option (PLAN.md task 3.4.1): changing the default would
// break the end-to-end parity M7 asserts, and a caller who wants the cheap
// checkpoint can say so with ForModel("multilingual").
type Router struct {
	mu sync.Mutex

	// models, defaultModel and autoTaskDetection are fixed at construction
	// and never written again, so Route reads them without the lock and
	// stays as pure and as cheap as upstream promises it is. The mutex
	// guards the agent cache alone.
	models            map[string]ModelSpec
	defaultModel      string
	autoTaskDetection bool

	loader    func(context.Context, string, ModelSpec) (Agent, error)
	maxLoaded int
	agents    map[string]residentAgent
	order     []string // least recently used first
}

// NewRouter builds a Router. With no options it is upstream's default: the
// bundle registry, "english" as the fallback, and no automatic workflow
// detection.
func NewRouter(opts ...RouterOption) (*Router, error) {
	cfg := routerConfig{defaultModel: ModelEnglish}
	for _, opt := range opts {
		if err := opt(&cfg); err != nil {
			return nil, err
		}
	}

	models := DefaultModels()
	if cfg.standaloneRepos {
		models = StandaloneModels()
	}
	maps.Copy(models, cfg.overrides)

	return &Router{
		models:            models,
		defaultModel:      cfg.defaultModel,
		autoTaskDetection: cfg.autoTaskDetection,
		loader:            cfg.loader,
		maxLoaded:         max(1, cfg.maxLoaded),
		agents:            map[string]residentAgent{},
	}, nil
}

// Route decides which checkpoint to use, without loading or running anything.
//
// Precedence (invariant #39): explicit model, then explicit task, then a
// detected workflow if WithAutoTaskDetection is on, then explicit language,
// then the script and language of the state, and failing all of those the
// configured default.
//
// state is `any` rather than a narrower type because that is what lang.Analyse
// takes and what M2 already shipped: a string is the document itself, and a
// map, slice or struct is flattened by lang.StateText before detection.
func (r *Router) Route(state any, qs Questions, opts ...RouteOption) (RouteDecision, error) {
	var req routeRequest
	for _, opt := range opts {
		opt(&req)
	}

	if req.model != nil {
		return r.decideByName(*req.model, "explicit model="+jsonx.ReprString(*req.model))
	}
	if req.task != nil {
		return r.decideByName(taskToModelName(*req.task), "explicit task="+jsonx.ReprString(*req.task))
	}

	// Computed here, before the remaining branches, exactly as upstream does
	// at router.py:264 -- which is why it reaches the lang and detection
	// payloads even when it did not decide anything (invariant #45).
	name, matched := MatchTypedDecisionsWorkflow(qs)
	if matched && r.autoTaskDetection {
		return r.decideByWorkflow(name), nil
	}

	var workflow *string
	if matched {
		workflow = &name
	}

	if req.lang != nil {
		return r.decideByLang(*req.lang, workflow), nil
	}
	return r.decideByDetection(state, workflow), nil
}

// taskToModelName is router.py:260's inline conditional: the two spellings of
// the typed-decisions task map to the checkpoint, and every other task string
// is handed to NormalizeModelName as if it were a model name -- so ForTask("en")
// is legal and yields english (invariant #41).
func taskToModelName(task string) string {
	if strings.ReplaceAll(strings.ToLower(task), "-", "_") == "typed_decisions" {
		return ModelTypedDecisions
	}
	return task
}

func (r *Router) decideByName(name, reason string) (RouteDecision, error) {
	key, err := NormalizeModelName(name)
	if err != nil {
		return RouteDecision{}, err
	}
	return RouteDecision{Model: key, Repo: r.spec(key).String(), Reason: reason}, nil
}

// decideByWorkflow is router.py:265-268. Upstream hard-codes the model name
// here rather than normalising it, and -- alone among the five branches --
// passes the spec through unconverted, so `repo` comes out as a JSON array on
// the bundle path. This port always emits the string (invariant #46, task
// 3.1.4); the deviation is listed in the README.
func (r *Router) decideByWorkflow(name string) RouteDecision {
	return RouteDecision{
		Model:    ModelTypedDecisions,
		Repo:     r.spec(ModelTypedDecisions).String(),
		Reason:   "question ids match the " + jsonx.ReprString(name) + " typed-decisions workflow",
		Workflow: &name,
	}
}

// decideByLang is router.py:270-273. Two outcomes only: this branch never
// reaches typed-decisions, whatever the tag.
func (r *Router) decideByLang(code string, workflow *string) RouteDecision {
	key := ModelMultilingual
	primary, _, _ := strings.Cut(strings.ToLower(code), "-")
	switch primary {
	case "en", "eng", "english":
		key = ModelEnglish
	}

	return RouteDecision{
		Model:    key,
		Repo:     r.spec(key).String(),
		Reason:   "explicit lang=" + jsonx.ReprString(code),
		Workflow: workflow,
	}
}

// decideByDetection is router.py:275-290, the branch that actually looks at the
// state. The "unknown" test has to come first: analyse reports is_english true
// for a state with no letters, so testing English-ness first would send digits
// to the English checkpoint instead of the configured default.
func (r *Router) decideByDetection(state any, workflow *string) RouteDecision {
	det := lang.Analyse(state)

	var key, reason string
	switch {
	case det.Script == "unknown":
		key = r.defaultModel
		reason = "no letters detected in state; using default (" + key + ")"
	case det.Script != "latin":
		key = ModelMultilingual
		reason = "non-Latin script (" + det.Script + ", " + percent(det.NonLatinFraction) +
			"% of letters); the English checkpoint cannot read it"
	case !det.IsEnglish:
		key = ModelMultilingual
		reason = "Latin script but language looks like " + reprLanguage(det.Language) + ", not English"
	default:
		key = ModelEnglish
		reason = "English Latin text"
	}

	return RouteDecision{
		Model:     key,
		Repo:      r.spec(key).String(),
		Reason:    reason,
		Detection: &det,
		Workflow:  workflow,
	}
}

// percent renders Python's "%.0f" of 100 * fraction. Both languages round the
// decimal representation half-to-even, so 0.5 renders as "0" and 1.5 as "2";
// Go's math.Round would be half-away-from-zero and disagree on exactly those
// values (the same trap as jsonx.Round4, invariant #29).
func percent(fraction float64) string {
	return strconv.FormatFloat(100*fraction, 'f', 0, 64)
}

// reprLanguage is `%r` of det["language"]. The nil case is unreachable from
// this branch -- analyse sets is_english true whenever the Latin guess is
// undecided, so a nil language never gets here -- but Python would print None
// and silently emitting the empty string would be worse than saying so.
func reprLanguage(code *string) string {
	if code == nil {
		return "None"
	}
	return jsonx.ReprString(*code)
}

// spec is the registry entry for a canonical name. Every name NormalizeModelName
// returns is present: an override replaces an entry, it never removes one.
func (r *Router) spec(key string) ModelSpec {
	return r.models[key]
}
