package laya

import (
	"fmt"
	"maps"
	"slices"
	"strings"

	"github.com/MeKo-Christian/go-laya/jsonx"
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
// constant per key would add indirection to a table whose whole value is that
// it can be diffed against router.py:65-70 at a glance.
//
//nolint:goconst // alias keys are upstream data reproduced verbatim; a
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
