package laya

import "context"

// Functional options for the Router. They mirror the keyword arguments of
// Python's Router.__init__ (router.py:144-153) and Router.route
// (router.py:241-248); the names gain a Router/For prefix where the bare word
// would collide with an option of some other constructor later.

// RouterOption configures a Router at construction. An option may reject its
// argument -- WithModels normalises its keys and WithDefaultModel its value --
// so NewRouter reports the first failure rather than silently ignoring it.
type RouterOption func(*routerConfig) error

type routerConfig struct {
	standaloneRepos   bool
	maxLoaded         int
	loader            func(context.Context, string, ModelSpec) (Agent, error)
	overrides         map[string]ModelSpec
	defaultModel      string
	autoTaskDetection bool
}

// WithModels overrides where individual checkpoints come from, by canonical
// name or alias. Only the named entries change; the rest keep the registry's
// own specs, as Python's dict.update does (router.py:152).
//
// This is how a caller points the router at local directories, which is what
// the upstream demo and test_local_e2e.py both do.
func WithModels(m map[string]ModelSpec) RouterOption {
	return func(c *routerConfig) error {
		if c.overrides == nil {
			c.overrides = make(map[string]ModelSpec, len(m))
		}
		for name, spec := range m {
			key, err := NormalizeModelName(name)
			if err != nil {
				return err
			}
			c.overrides[key] = spec
		}
		return nil
	}
}

// WithStandaloneRepos selects the per-checkpoint repos instead of the bundle,
// for anyone who prefers them (router.py:151).
func WithStandaloneRepos(on bool) RouterOption {
	return func(c *routerConfig) error {
		c.standaloneRepos = on
		return nil
	}
}

// WithDefaultModel sets the checkpoint used when the state has no letters at
// all and detection therefore decides nothing. Defaults to "english"; the name
// is normalised, so an alias is accepted (router.py:155).
func WithDefaultModel(name string) RouterOption {
	return func(c *routerConfig) error {
		key, err := NormalizeModelName(name)
		if err != nil {
			return err
		}
		c.defaultModel = key
		return nil
	}
}

// WithAutoTaskDetection enables the workflow branch: when the question ids
// exactly match one of the four typed-decisions signatures, route there.
//
// Off by default, as upstream (router.py:151). With it off the match is still
// computed and still reported in the decision's Workflow -- it just does not
// decide anything (invariant #45).
func WithAutoTaskDetection(on bool) RouterOption {
	return func(c *routerConfig) error {
		c.autoTaskDetection = on
		return nil
	}
}

// RouteOption is a per-call override of the routing decision. Each corresponds
// to one keyword argument of Python's route(), and each is absent by default --
// "" is a value a caller can legitimately pass, so the options carry presence
// rather than relying on a zero value.
type RouteOption func(*routeRequest)

type routeRequest struct {
	model *string
	task  *string
	lang  *string
}

// ForModel pins the checkpoint by name or alias. Highest precedence: it beats
// an explicit task, a matched workflow, an explicit language and detection
// alike (invariant #39).
func ForModel(name string) RouteOption {
	return func(r *routeRequest) { r.model = &name }
}

// ForTask routes by task name. "typed_decisions" and "typed-decisions" both
// mean the typed-decisions checkpoint; anything else is treated as a model
// name, so ForTask("en") is legal and yields english (invariant #41).
func ForTask(task string) RouteOption {
	return func(r *routeRequest) { r.task = &task }
}

// ForLang routes by language tag, skipping detection. Only two outcomes: an
// English tag picks english, everything else picks multilingual. The tag is
// lower-cased and cut at the first "-", so "en-GB" is English (invariant #43).
func ForLang(code string) RouteOption {
	return func(r *routeRequest) { r.lang = &code }
}

// WithMaxLoaded caps how many checkpoints stay resident; the least recently
// used is evicted past the cap. Clamped to at least 1 (invariant #48), as
// max(1, int(max_loaded)) is upstream.
//
// The default is 1, which is right for a batch job and wrong for anything that
// alternates languages: a cold load costs seconds and detection costs
// microseconds, so at 1 an alternating workload reloads on every request. Use
// Preload for a server.
func WithMaxLoaded(n int) RouterOption {
	return func(c *routerConfig) error {
		c.maxLoaded = max(1, n)
		return nil
	}
}

// WithLoader supplies the function that builds an agent for a checkpoint.
//
// It is required until M6 lands the ONNX-backed default; without it Load
// returns ErrNoLoader. It is also the seam the upstream LRU tests need:
// they monkeypatch Router.load to avoid building a real checkpoint
// (test_router.py:177), and an injected loader is how that ports without
// reflection.
func WithLoader(fn func(context.Context, string, ModelSpec) (Agent, error)) RouterOption {
	return func(c *routerConfig) error {
		c.loader = fn
		return nil
	}
}
