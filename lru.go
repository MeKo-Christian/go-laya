package laya

import (
	"context"
	"errors"
	"fmt"
	"slices"
)

// The Router's agent cache: router.py:169-238, plus the releasing Python does
// not do.
//
// Upstream drops an evicted agent from two dicts and lets refcounting free it.
// In Go that frees nothing -- the ONNX Runtime session behind an agent is
// hundreds of megabytes and is released only by Close -- so eviction, Unload
// and Close all close what they drop. That is a deliberate deviation (PLAN.md
// tasks 3.2.4 and 3.2.6).
//
// It is bounded by ownership. The Router closes agents its own loader built;
// an agent handed to Attach belongs to whoever attached it and is only ever
// forgotten, never closed. Without that line a caller could not attach one
// checkpoint to two routers, and the second Close would be a double free.

// residentAgent is one cache entry. owned records whether the Router built it
// and may therefore close it.
type residentAgent struct {
	agent Agent
	owned bool
}

// preloadOrder is the order Preload builds checkpoints in when it is not given
// names. Python walks its models dict, whose order is the DEFAULT_MODELS
// literal (router.py:37-41); a Go map has none, and Loaded() would come back
// in a different order on every run.
var preloadOrder = []string{ModelEnglish, ModelMultilingual, ModelTypedDecisions}

// Load returns the agent for name, building it on first use and touching it on
// a hit. The name is normalised, so an alias works.
//
// The build happens under the Router's lock. Two goroutines asking for the same
// checkpoint therefore cannot both build it -- which is the point, given a
// build is seconds and hundreds of megabytes -- at the cost of serialising
// loads of *different* checkpoints too. Route is unaffected: it reads nothing
// the lock protects.
func (r *Router) Load(ctx context.Context, name string) (Agent, error) {
	key, err := NormalizeModelName(name)
	if err != nil {
		return nil, err
	}

	r.mu.Lock()
	defer r.mu.Unlock()
	return r.loadLocked(ctx, key)
}

func (r *Router) loadLocked(ctx context.Context, key string) (Agent, error) {
	if resident, ok := r.agents[key]; ok {
		r.touchLocked(key)
		return resident.agent, nil
	}
	if r.loader == nil {
		return nil, fmt.Errorf("%w: cannot build %q; pass WithLoader", ErrNoLoader, key)
	}

	agent, err := r.loader(ctx, key, r.models[key])
	if err != nil {
		return nil, fmt.Errorf("laya: load %q: %w", key, err)
	}

	r.agents[key] = residentAgent{agent: agent, owned: true}
	r.order = append(r.order, key)

	// Appended before evicting, as upstream does (router.py:178-180), so at a
	// cap of 1 the newcomer survives and the incumbent is dropped.
	return agent, r.evictLocked()
}

// Attach registers an already-built agent under name instead of loading a
// second copy, and raises the cap so the LRU cannot immediately evict it
// (invariant #50). The name is normalised, so an alias works.
//
// The agent stays the caller's: the Router will forget it when it is evicted or
// unloaded, but never close it. Attaching over an existing entry likewise
// replaces it without closing, because the entry being replaced may be one the
// caller attached earlier.
func (r *Router) Attach(name string, agent Agent) error {
	key, err := NormalizeModelName(name)
	if err != nil {
		return err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.agents[key] = residentAgent{agent: agent}
	r.touchLocked(key)
	r.maxLoaded = max(r.maxLoaded, len(r.agents))
	return nil
}

// Preload builds checkpoints up front so no request pays a model load. With no
// names it builds every checkpoint in the registry.
//
// The cap is raised to fit whatever is preloaded (invariant #51) -- otherwise
// the LRU would evict what this just built -- and names already resident are
// skipped, so an attached agent is not rebuilt.
func (r *Router) Preload(ctx context.Context, names ...string) error {
	keys := make([]string, 0, max(len(names), len(preloadOrder)))
	if len(names) == 0 {
		keys = append(keys, preloadOrder...)
	}
	for _, name := range names {
		key, err := NormalizeModelName(name)
		if err != nil {
			return err
		}
		keys = append(keys, key)
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.maxLoaded = max(r.maxLoaded, len(keys), len(r.agents))
	for _, key := range keys {
		if _, ok := r.agents[key]; ok {
			continue
		}
		if _, err := r.loadLocked(ctx, key); err != nil {
			return err
		}
	}
	return nil
}

// Unload frees the named checkpoints, or every one of them when called with no
// names. An agent the Router built is closed; an attached one is only
// forgotten. Unknown and absent names are ignored, as upstream's pop(key, None)
// is (invariant #52).
//
// It reports an error where Python returns nothing, for the same reason
// eviction closes at all: in Go "free" is a call that can fail, and a backend
// that would not shut down is worth hearing about.
func (r *Router) Unload(names ...string) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	if len(names) == 0 {
		names = slices.Clone(r.order)
	}

	errs := make([]error, 0, len(names))
	for _, name := range names {
		key, err := NormalizeModelName(name)
		if err != nil {
			continue
		}
		errs = append(errs, r.dropLocked(key))
	}
	return errors.Join(errs...)
}

// Close releases every agent the Router built and forgets the rest. It is
// idempotent: a Router that has been closed is empty, so closing it again
// closes nothing twice.
func (r *Router) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	resident := slices.Clone(r.order)
	errs := make([]error, 0, len(resident))
	for _, key := range resident {
		errs = append(errs, r.dropLocked(key))
	}
	return errors.Join(errs...)
}

// Loaded returns the resident checkpoint names, least recently used first.
func (r *Router) Loaded() []string {
	r.mu.Lock()
	defer r.mu.Unlock()
	return slices.Clone(r.order)
}

// MaxLoaded returns the residency cap.
func (r *Router) MaxLoaded() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.maxLoaded
}

// SetMaxLoaded changes the residency cap, clamped to at least 1. It exists
// because the upstream tests raise the cap after construction
// (test_router.py:241, 249, 266) and a constructor-only API cannot express
// that.
//
// Lowering it evicts nothing immediately; the surplus goes on the next Load, as
// upstream's _evict is only ever reached from load.
func (r *Router) SetMaxLoaded(n int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.maxLoaded = max(1, n)
}

// touchLocked makes key the most recently used. Remove then append, as
// upstream's _touch does (router.py:183-186).
func (r *Router) touchLocked(key string) {
	r.order = append(slices.DeleteFunc(r.order, func(k string) bool { return k == key }), key)
}

// evictLocked drops from the front while over the cap (invariant #49).
//
// Upstream follows this with a reconciliation pass over _agents
// (router.py:191-195), because two Python dicts can drift apart. One map and
// one slice maintained together cannot, so there is nothing here to reconcile.
func (r *Router) evictLocked() error {
	errs := make([]error, 0, max(0, len(r.order)-r.maxLoaded))
	for len(r.order) > r.maxLoaded {
		errs = append(errs, r.dropLocked(r.order[0]))
	}
	return errors.Join(errs...)
}

// dropLocked removes key from both views and closes the agent if the Router
// built it.
func (r *Router) dropLocked(key string) error {
	resident, ok := r.agents[key]
	if !ok {
		return nil
	}
	delete(r.agents, key)
	r.order = slices.DeleteFunc(r.order, func(k string) bool { return k == key })

	if !resident.owned {
		return nil
	}
	if err := resident.agent.Close(); err != nil {
		return fmt.Errorf("laya: close %q: %w", key, err)
	}
	return nil
}
