# Proposed Go API

> Referenced from `PLAN.md` §7. The type definitions the port targets, and the decisions that resolve
> Python's dynamic typing into explicit Go types. **This file is the type appendix; `PLAN.md` wins on
> any conflict** — in particular §2 of the plan owns the package layout, which used to be duplicated
> here and drifted within a day (review of 2026-09-20).

## Return-value shapes to reproduce

```jsonc
{
  "model": "laya-rl-agent",          // constant
  "answers": { "<qid>": <Answer>, ... },
  "usage": { "input_tokens": <int, sum of attention_mask over the whole batch>,
             "output_tokens": 0 },
  "routing": { ... }                 // only via Router.predict
}
```

`<Answer>`, three disjoint shapes — key sets differ per type:

```jsonc
// choice (agent.py:313-320)
{ "type": "choice",
  "choice": "<criteria key with the highest p>",
  "probabilities": { "<key>": <float, 4dp>, ... },   // criteria definition order
  "confidence": <float, 4dp>,                        // 1 - H(p)/ln k
  "action": { "act_probability": <float, 4dp> } }

// score (agent.py:321-330)
{ "type": "score",
  "score": <float, 4dp>,                             // sum(i * p[i]), NOT argmax
  "legend": { "0": <RAW criterion value>, "1": ..., },// raw, not rendered
  "probabilities": { "0": <float, 4dp>, ... },
  "confidence": <float, 4dp>,
  "action": { "act_probability": <float, 4dp> } }

// noul (agent.py:331-337)  — NO "probabilities", NO "legend"
{ "type": "noul",
  "noul": <float, 4dp>,                              // p[1] = P(true)
  "confidence": <float, 4dp>,                        // max(p1, 1-p1), NOT entropy
  "action": { "act_probability": <float, 4dp> } }
```

`RouteDecision` — always exactly five keys:

```jsonc
{ "model": "english" | "multilingual" | "typed-decisions",
  "repo":  "convaiinnovations/laya/multilingual",   // string... except one branch upstream, see INVARIANTS #46 / PLAN Task 3.1.4
  "reason": "<human-readable>",
  "detection": null | { "script": "devanagari",
                        "script_profile": {"devanagari": 1.0},
                        "language": null | "de",
                        "is_english": false,
                        "non_latin_fraction": 1.0 },
  "workflow": null | "customer_service" | ... }
```

### 3.3 Package layout

See `PLAN.md` §2 — the single owner of the layout. In short: module `github.com/MeKo-Christian/go-laya`;
public `lang/`, `mailtext/`, `presets/`, `jsonx/`, `question/`, `tokenizer/` and `backend/` (D9: the
`Backend` interface and `Batch` struct, zero dependencies); `internal/prompt`, `internal/calib`,
`internal/hub`, `internal/backend/onnx`, `internal/golden`. The `Obj` type below lives in `jsonx/`,
not at the root.

The code block below is headed `package laya`, and since 2026-09-20 that is true only through
aliases: the question types live in `question/` and the root re-exports them (`type Question =
question.Question`), so that `presets` can build questions without `go list -deps ./presets`
reaching the ONNX binding. See PLAN D11. `laya.ChoiceQuestion` still names the same type it always
did.

### 3.4 Concrete Go type definitions

```go
package laya

// ---------------------------------------------------------------- primitives

// QType is a decision primitive. The integer values are baked into the
// checkpoint's type embedding and must not be renumbered.
type QType uint8

const (
	Choice QType = 0
	Score  QType = 1
	Noul   QType = 2
)

func (t QType) String() string // "choice" | "score" | "noul"

// ---------------------------------------------------------------- ordered JSON

// Obj is an ordered JSON object. Python dicts preserve insertion order and laya
// builds prompts from serialized state, so a Go map would silently change the
// bytes the model sees (and the key order of emitted probabilities). Every place
// the Python code uses a dict whose order is observable, Go must use Obj.
type Obj []Field

type Field struct {
	Key   string
	Value any
}

// MarshalJSON reproduces json.dumps(x, ensure_ascii=False): separators ", " and
// ": ", no HTML escaping, no key sorting, non-ASCII emitted literally.
func (o Obj) MarshalJSON() ([]byte, error)

// ---------------------------------------------------------------- state

// State is the document a question is asked about: text, an ordered object, or a
// list (e.g. conversation turns). Python accepts str|dict|list at every call
// site; Go makes the three cases explicit.
type State interface {
	// Serialize reproduces common.serialize_state.
	Serialize() string
	// TextLeaves reproduces lang._iter_text: string leaves only, dict keys ignored.
	TextLeaves(depth int) []string
}

func TextState(s string) State   // str
func ObjState(o Obj) State       // dict
func ListState(v ...any) State   // list

// ---------------------------------------------------------------- questions

// Criterion is a criterion description. A string renders verbatim; nil and ""
// mean "no description"; anything else renders as Python-compatible compact
// JSON (use Obj for objects so key order survives).
type Criterion any

type Question interface {
	Type() QType
	Instructions() string
	// RenderOptions reproduces common.render_options, in label-index order.
	RenderOptions() []string
	validate() error
}

type ChoiceOption struct {
	Key  string
	Desc Criterion // nil or "" -> bare key
}

// ChoiceQuestion replaces Python's dict-or-list `criteria`: a []ChoiceOption
// covers both (a list of bare strings becomes options with Desc == nil, exactly
// what Agent._to_internal does at agent.py:233-234) and, unlike a Go map,
// preserves definition order, which drives both the prompt and the
// probabilities key order.
type ChoiceQuestion struct {
	Ins  string
	Opts []ChoiceOption
}

func Labels(keys ...string) []ChoiceOption // sugar for the list form

// ScoreQuestion's Levels deliberately cannot express one shape upstream allows:
// a score question whose `criteria` is a dict, which render_options enumerates
// by key ({"low": 1, "high": 2} -> ["level 0: low", "level 1: high"], recorded
// as testdata/render.jsonl's options/score/dict-criteria). That case is reachable
// only through internal/prompt's loose {t, ins, crit} shape, which is where it is
// tested. Widening Levels to admit it would put a Python-only ambiguity into the
// Go API for no caller's benefit; see PLAN Task 2.5.
type ScoreQuestion struct {
	Ins    string
	Levels []Criterion // level 0 .. n-1
}

type NoulQuestion struct {
	Ins   string
	False Criterion // nil/"" -> "no, the statement does not hold"
	True  Criterion // nil/"" -> "yes, the statement holds"
}

// Questions preserves definition order so results are byte-reproducible.
// A Python dict cannot hold two questions with the same id, so a duplicate is
// an error (ErrDuplicateQuestionID), never last-wins.
type Questions []NamedQuestion

type NamedQuestion struct {
	ID string
	Q  Question
}

func (qs Questions) IDs() []string
func (qs Questions) Get(id string) (Question, bool)
func (qs Questions) Validate() error // duplicate ids, per-question validate()

// ---------------------------------------------------------------- answers

type Action struct {
	ActProbability float64 `json:"act_probability"`
}

// Probs is an ordered probability map; key order mirrors Python's dict.
type Probs []ProbEntry

type ProbEntry struct {
	Key string
	P   float64
}

func (p Probs) MarshalJSON() ([]byte, error)
func (p Probs) Get(key string) float64
func (p Probs) Max() (key string, p float64) // first maximum on a tie, like numpy argmax (INVARIANTS #30a)

// Answer carries all three shapes. Which keys are emitted is decided by Type,
// through a hand-written MarshalJSON over jsonx.Obj — NOT by `omitempty`: a
// choice key of "" is legal in Python and omitempty would drop "choice", and an
// empty legend/probabilities would vanish the same way, breaking invariant #30.
// A noul answer has no "probabilities" and no "legend"; a score of 0.0 is still
// emitted. Key order per type is exactly the Python order shown above.
type Answer struct {
	Type          string
	Choice        string   // choice only
	Score         *float64 // score only
	Noul          *float64 // noul only
	Legend        Obj      // score only; raw criterion values
	Probabilities Probs    // choice and score
	Confidence    float64
	Action        Action
}

func (a Answer) MarshalJSON() ([]byte, error) // per-Type key set, no omitempty

type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"` // always 0
}

// AnswerSet is an ordered id -> Answer map mirroring the question order.
type AnswerSet []NamedAnswer

type NamedAnswer struct {
	ID string
	A  Answer
}

func (as AnswerSet) Get(id string) (Answer, bool)
func (as AnswerSet) MarshalJSON() ([]byte, error)

type Result struct {
	Model   string         `json:"model"` // "laya-rl-agent"
	Answers AnswerSet      `json:"answers"`
	Usage   Usage          `json:"usage"`
	Routing *RouteDecision `json:"routing,omitempty"`
}

// ---------------------------------------------------------------- agent

type Agent struct {
	// unexported: cfg, tokenizer, backend, temperature tables
	mu sync.RWMutex
}

func Open(ctx context.Context, ref string, opts ...Option) (*Agent, error)

func (a *Agent) SystemOne(ctx context.Context, state State, qs Questions) (*Result, error)
func (a *Agent) Predict(ctx context.Context, state State, qs Questions) (*Result, error) // alias
func (a *Agent) MaxLen() int
func (a *Agent) HeadMaxLen() int
func (a *Agent) SetLimits(maxLen, headMaxLen int) error // replaces agent.cfg[...] mutation
func (a *Agent) Close() error

// ---------------------------------------------------------------- options

type Option func(*agentConfig) error

func WithDevice(d string) Option        // "cpu" | "cuda" | "cuda:0" | "coreml" | "auto"
func WithHFToken(tok string) Option     // default: $HF_TOKEN
func WithSubfolder(sub string) Option
func WithCacheDir(dir string) Option    // default: $LAYA_CACHE or os.UserCacheDir()/laya
func WithLimits(maxLen, headMaxLen int) Option
func WithBackend(b backend.Backend) Option // public leaf package (PLAN D9): test doubles, alternative runtimes
func WithLogger(l *slog.Logger) Option

// ---------------------------------------------------------------- router

type ModelSpec struct {
	Repo      string // repo id or local path
	Subfolder string // "" for the repo root
}

func (s ModelSpec) String() string   // reproduces router._repo_str
func ModelSpecFromString(repo string) ModelSpec // the bare-repo / local-path form

func DefaultModels() map[string]ModelSpec    // copies: the bundle registry
func StandaloneModels() map[string]ModelSpec // copies: the per-checkpoint repos

func NormalizeModelName(name string) (string, error)           // router.normalise_name
func MatchTypedDecisionsWorkflow(qs Questions) (string, bool)   // exact id-set equality

// Detection is emitted inside the routing payload, so ScriptProfile's key order
// is observable there. It is ordered, not a map.
//
// Corrected 2026-09-20 (M2): the order is latin FIRST, then the other scripts in
// text-encounter order, with zero counts dropped. That is script_profile's order
// (lang.py:116 seeds counts with {"latin": 0}). "latin last" is detect_script's
// order (invariant #56) and belongs to a different function; the two genuinely
// differ and cannot be collapsed. PLAN Task 2.2.7 carried the same error.
//
// Declared in lang and re-exported at the root as `type Detection =
// lang.Detection` (added 2026-09-20, M3). An alias, not a second struct: lang
// owns it, and redeclaring would repeat the mistake D11 records.
//
// The struct tags below are illustrative: the implementation carries a
// MarshalJSON, which makes them dead. Byte-parity with Python goes through
// jsonx.Marshal(d.Map()), because encoding/json compacts a MarshalJSON's output
// and would strip Python's ", " and ": " separators (PLAN D12).
type Detection struct {
	Script           string         `json:"script"`
	ScriptProfile    Profile        `json:"script_profile"` // []ScriptShare, marshals as an ordered object
	Language         *string        `json:"language"`
	IsEnglish        bool           `json:"is_english"`
	NonLatinFraction float64        `json:"non_latin_fraction"`
}

type ScriptShare struct {
	Script   string
	Fraction float64
}

// RouteDecision always serialises all five keys; nil pointers become null,
// matching Python where detection/workflow are explicitly None.
type RouteDecision struct {
	Model     string     `json:"model"`
	Repo      string     `json:"repo"`
	Reason    string     `json:"reason"`
	Detection *Detection `json:"detection"`
	Workflow  *string    `json:"workflow"`
}

type Router struct {
	mu sync.Mutex // Python's Router is not concurrency-safe; Go's must be
	// models map[string]ModelSpec, agents, lru order, loader hook
}

func NewRouter(opts ...RouterOption) (*Router, error)

// Route is pure: it never loads or runs anything, never touches the network,
// and needs no context.
func (r *Router) Route(state State, qs Questions, ro ...RouteOption) (RouteDecision, error)

// Agent is an interface, not a struct, and today carries only Close(). The
// Router caches agents and must release what it evicts; that is the whole of
// what it asks of one. M7 widens it with SystemOne (PLAN D13). Corrected
// 2026-09-20 (M3): this block used to say *Agent, which M3 could not build
// against because the concrete agent arrives in M6/M7.
func (r *Router) Load(ctx context.Context, name string) (Agent, error)
func (r *Router) MaxLoaded() int
func (r *Router) SetMaxLoaded(n int) // clamped to >= 1; the upstream tests mutate it after construction
func (r *Router) Attach(name string, a Agent) error
func (r *Router) Preload(ctx context.Context, names ...string) error
func (r *Router) Loaded() []string // LRU order, least-recent first

// Unload and Close release what they drop -- eviction too. Python drops the
// agent from two dicts and lets refcounting free it, which in Go frees nothing
// and leaks an ORT session. Unload therefore returns an error where Python
// returns nothing (PLAN tasks 3.2.4-3.2.6). Only agents the Router's own loader
// built are closed; an attached one belongs to the caller.
func (r *Router) Unload(names ...string) error
func (r *Router) Close() error

// Deferred to M7 (PLAN Task 7.6): they call agent.system_one, so they need the
// widened Agent and the Result type.
func (r *Router) Predict(ctx context.Context, state State, qs Questions, ro ...RouteOption) (*Result, error)
func (r *Router) SystemOne(ctx context.Context, state State, qs Questions, ro ...RouteOption) (*Result, error) // alias, router.py:311

type RouterOption func(*routerConfig) error

func WithModels(m map[string]ModelSpec) RouterOption
func WithMaxLoaded(n int) RouterOption          // clamped to >= 1
func WithDefaultModel(name string) RouterOption // default "english"
func WithAutoTaskDetection(on bool) RouterOption
func WithStandaloneRepos(on bool) RouterOption

// WithLoader is required until M6 lands the default loader; without it Load
// returns ErrNoLoader. It is also the seam the upstream LRU tests need, which
// monkeypatch Router.load (test_router.py:177).
func WithLoader(fn func(context.Context, string, ModelSpec) (Agent, error)) RouterOption

// Deferred to M6 (PLAN Task 6.11): they configure the agent builder, and until
// there is one they would store values nothing reads.
func WithRouterDevice(d string) RouterOption
func WithRouterToken(tok string) RouterOption // falls back to $HF_TOKEN

type RouteOption func(*routeRequest)

func ForModel(name string) RouteOption
func ForTask(task string) RouteOption
func ForLang(code string) RouteOption

// ---------------------------------------------------------------- errors

var (
	ErrUnknownModel            = errors.New("laya: unknown model")
	ErrCheckpointNotFound      = errors.New("laya: checkpoint not found")
	ErrIncompatibleCheckpoint  = backend.ErrIncompatibleCheckpoint // defined in the leaf so a Backend can wrap it
	ErrOptionsExceedHeadBudget = errors.New("laya: question options exceed head_max_len")
	ErrEmptyQuestions          = errors.New("laya: no questions")          // Python: TypeError from collate_items returning None
	ErrDuplicateQuestionID     = errors.New("laya: duplicate question id") // Python: impossible in a dict
)

// OptionBudgetError names the offending question, replacing Python's
// ValueError("question %r options exceed head_max_len=%d") at agent.py:262-263.
type OptionBudgetError struct {
	QuestionID  string
	HeadMaxLen  int
	WantMarkers int
	GotMarkers  int
}

func (e *OptionBudgetError) Error() string
func (e *OptionBudgetError) Unwrap() error // ErrOptionsExceedHeadBudget
```

**Error handling.** Python raises `ValueError`/`FileNotFoundError` and _prints_ warnings to stdout (`agent.py:159,162,219-227,280`). Go: return wrapped sentinels everywhere; replace the three `print()` warnings with `slog` at `Warn` level through `WithLogger`. The silent CPU fallback at `agent.py:203-216` and the mid-inference GPU→CPU fallback at `agent.py:278-290` should become explicit policy: `WithDeviceFallback(bool)`, defaulting to **on** with a logged warning, matching Python.

**`context.Context`.** Python has none. In Go: `Open`, `Load`, `Preload` and `Predict`/`SystemOne` take a `ctx` (download and inference are both cancellable — ORT supports a run-level cancel via `RunOptions.Terminate`, and the HTTP client honours the ctx). `Route`, `lang.*`, `mailtext.*` and `presets.*` are pure and take none.

### 3.5 The dynamic-typing decisions, and what I propose

| Python                                                                                            | The ambiguity                                                                     | Go decision                                                                                                                                                                                                                              |
| ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `criteria` on `choice`: `dict[str, Any]` **or** `list[str]` (agent.py:233-234)                    | list means "bare labels"; dict values may be `None`/`""` (bare) or any JSON value | `[]ChoiceOption{Key, Desc any}` + a `Labels("a","b")` helper. One representation, order preserved, both forms expressible                                                                                                                |
| `criteria` on `score`: `list[Any]`                                                                | may hold strings, dicts, numbers, even `None` (test_criteria.py:92)               | `[]Criterion` where `Criterion = any`                                                                                                                                                                                                    |
| `criteria` on `noul`: `dict` with `"true"`/`"false"`, or absent                                   | ordering is fixed (false=0, true=1) regardless of dict order                      | explicit `True`/`False` fields — removes the possibility of getting the index order wrong                                                                                                                                                |
| `state`: `str \| dict \| list`                                                                    | serialized one way for the prompt, flattened another way for detection            | `State` interface with `TextState`/`ObjState`/`ListState`. Rejected `any` + reflection: it makes both behaviours implicit and makes `Obj` ordering easy to lose                                                                          |
| criterion value: anything (`dict`, `list`, `int`, `bool`, `float`, `None`, unserialisable object) | `render_criterion` JSON-encodes with `default=str`                                | `Criterion = any`; encoder mirrors `json.dumps`, and falls back to `fmt.Sprintf("%v", v)` for anything `encoding/json` refuses (matching `default=str`)                                                                                  |
| `instructions`: `str` or anything (agent.py:236-237)                                              | non-str is `json.dumps`'d with `ensure_ascii=True` (unlike everything else)       | `Ins string` only. Callers serialise themselves. Document the dropped edge case                                                                                                                                                          |
| dict iteration order                                                                              | drives prompt bytes, probabilities key order, and two tie-breaks                  | `Obj`/`Probs`/`Questions`/`AnswerSet`/`Detection.ScriptProfile` ordered slices. No Go `map` anywhere the order reaches JSON — `script_profile` sits inside the emitted `routing` payload, so it was never unobservable (PLAN Task 2.2.7) |
| `RouteDecision.repo`: `str` normally, tuple on one branch (router.py:266)                         | inconsistent                                                                      | always `string`. Document the deviation                                                                                                                                                                                                  |
| `RouteDecision` is a `dict` subclass                                                              | users index it _and_ use `.model`                                                 | a struct with JSON tags; add `func (d RouteDecision) Map() Obj` for anyone who wants the dict                                                                                                                                            |
| `Router.models` values: `str` or `(repo, sub)` tuple/list                                         | `_split` normalises                                                               | `ModelSpec{Repo, Subfolder}`; `WithModels(map[string]ModelSpec)`. Add `ModelSpecFromString(s)` for the local-path override that test_router.py:230 relies on                                                                             |
| `legend` values: raw criteria                                                                     | can be dicts                                                                      | `Obj` with `any` values                                                                                                                                                                                                                  |

---
