package laya

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
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
//
// And it is bounded by use (PLAN.md task 7.7.6, D27). In Python the agent a
// running predict holds stays alive after eviction, because that local
// reference keeps it; closing it on eviction would fail the pass instead. So
// Predict leases the agent it runs, and a dropped agent is closed only once its
// last lease is released. The drop itself is immediate -- Loaded() no longer
// lists it -- and the close runs after the Router's lock is released, in the
// call that dropped it, which therefore waits for those passes and still
// reports the close error.

// residentAgent is one cache entry. owned records whether the Router built it
// and may therefore close it; leases counts the Predict calls running on it.
type residentAgent struct {
	key    string
	agent  Predictor
	owned  bool
	leases sync.WaitGroup
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
//
// The agent is not leased. A concurrent load, Unload or Close may drop and
// close it while the caller still holds it; Predict is the way to run an agent
// under concurrency.
func (r *Router) Load(ctx context.Context, name string) (Predictor, error) {
	key, err := NormalizeModelName(name)
	if err != nil {
		return nil, err
	}

	var resident *residentAgent
	var dropped []*residentAgent
	r.withLock(func() { resident, dropped, err = r.loadLocked(ctx, key) })
	if err != nil {
		return nil, err
	}
	return resident.agent, closeDropped(dropped)
}

// lease is Load for Predict: the agent cannot be closed until release is
// called. The lease is taken under the lock that found the entry in the cache,
// so no drop can come between the two.
func (r *Router) lease(ctx context.Context, key string) (agent Predictor, release func(), err error) {
	var resident *residentAgent
	var dropped []*residentAgent
	r.withLock(func() {
		if resident, dropped, err = r.loadLocked(ctx, key); err == nil {
			resident.leases.Add(1)
		}
	})
	if err != nil {
		return nil, nil, err
	}
	release = sync.OnceFunc(resident.leases.Done)

	// The load may have evicted others; closing them waits for their own
	// leases, never for this one.
	if err := closeDropped(dropped); err != nil {
		release()
		return nil, nil, err
	}
	return resident.agent, release, nil
}

// withLock runs fn under the Router's lock and releases it however fn ends.
// The loads above cannot defer the unlock in their own scope, because they
// close what they evicted after unlocking; and a load runs the loader, which is
// caller code and may panic.
func (r *Router) withLock(fn func()) {
	r.mu.Lock()
	defer r.mu.Unlock()
	fn()
}

// loadLocked returns the entry for key, building it if need be, and the
// entries the build evicted, which the caller closes after unlocking.
func (r *Router) loadLocked(ctx context.Context, key string) (*residentAgent, []*residentAgent, error) {
	if resident, ok := r.agents[key]; ok {
		r.touchLocked(key)
		return resident, nil, nil
	}
	if r.loader == nil {
		return nil, nil, fmt.Errorf("%w: cannot build %q", ErrNoLoader, key)
	}

	agent, err := r.loader(ctx, key, r.models[key])
	if err != nil {
		return nil, nil, fmt.Errorf("laya: load %q: %w", key, err)
	}

	resident := &residentAgent{key: key, agent: agent, owned: true}
	r.agents[key] = resident
	r.order = append(r.order, key)

	// Appended before evicting, as upstream does (router.py:178-180), so at a
	// cap of 1 the newcomer survives and the incumbent is dropped.
	return resident, r.evictLocked(), nil
}

// Attach registers an already-built agent under name instead of loading a
// second copy, and raises the cap so the LRU cannot immediately evict it
// (invariant #50). The name is normalised, so an alias works.
//
// The agent stays the caller's: the Router will forget it when it is evicted or
// unloaded, but never close it. Attaching over an existing entry likewise
// replaces it without closing, because the entry being replaced may be one the
// caller attached earlier.
func (r *Router) Attach(name string, agent Predictor) error {
	key, err := NormalizeModelName(name)
	if err != nil {
		return err
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	r.agents[key] = &residentAgent{key: key, agent: agent}
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

	var dropped []*residentAgent
	var err error
	r.withLock(func() {
		r.maxLoaded = max(r.maxLoaded, len(keys), len(r.agents))
		for _, key := range keys {
			if _, ok := r.agents[key]; ok {
				continue
			}
			var evicted []*residentAgent
			if _, evicted, err = r.loadLocked(ctx, key); err != nil {
				return
			}
			dropped = append(dropped, evicted...)
		}
	})
	return errors.Join(err, closeDropped(dropped))
}

// Unload frees the named checkpoints, or every one of them when called with no
// names. An agent the Router built is closed; an attached one is only
// forgotten. Unknown and absent names are ignored, as upstream's pop(key, None)
// is (invariant #52).
//
// It reports an error where Python returns nothing, for the same reason
// eviction closes at all: in Go "free" is a call that can fail, and a backend
// that would not shut down is worth hearing about.
//
// An agent with a Predict still running on it leaves the cache at once and is
// closed when that pass ends; Unload waits for it.
func (r *Router) Unload(names ...string) error {
	r.mu.Lock()
	if len(names) == 0 {
		names = slices.Clone(r.order)
	}
	dropped := make([]*residentAgent, 0, len(names))
	for _, name := range names {
		key, err := NormalizeModelName(name)
		if err != nil {
			continue
		}
		if resident := r.dropLocked(key); resident != nil {
			dropped = append(dropped, resident)
		}
	}
	r.mu.Unlock()
	return closeDropped(dropped)
}

// Close releases every agent the Router built and forgets the rest. It is
// idempotent: a Router that has been closed is empty, so closing it again
// closes nothing twice. Like Unload, it waits for any Predict still running.
func (r *Router) Close() error {
	r.mu.Lock()
	dropped := make([]*residentAgent, 0, len(r.order))
	for _, key := range slices.Clone(r.order) {
		if resident := r.dropLocked(key); resident != nil {
			dropped = append(dropped, resident)
		}
	}
	r.mu.Unlock()
	return closeDropped(dropped)
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
func (r *Router) evictLocked() []*residentAgent {
	dropped := make([]*residentAgent, 0, max(0, len(r.order)-r.maxLoaded))
	for len(r.order) > r.maxLoaded {
		dropped = append(dropped, r.dropLocked(r.order[0]))
	}
	return dropped
}

// dropLocked removes key from both views and returns its entry, or nil if it
// was not resident. The caller closes it with closeDropped once unlocked.
func (r *Router) dropLocked(key string) *residentAgent {
	resident, ok := r.agents[key]
	if !ok {
		return nil
	}
	delete(r.agents, key)
	r.order = slices.DeleteFunc(r.order, func(k string) bool { return k == key })
	return resident
}

// closeDropped closes the entries the Router built, each after the last
// Predict leasing it has finished, and forgets the attached ones. It must run
// without the Router's lock: a lease is released without taking it, but the
// wait can be long, and nothing else should stall behind it.
//
// No new lease can start on a dropped entry, because leases are only taken on
// entries found in the cache, so the wait ends.
func closeDropped(dropped []*residentAgent) error {
	errs := make([]error, 0, len(dropped))
	for _, resident := range dropped {
		if !resident.owned {
			continue
		}
		resident.leases.Wait()
		if err := resident.agent.Close(); err != nil {
			errs = append(errs, fmt.Errorf("laya: close %q: %w", resident.key, err))
		}
	}
	return errors.Join(errs...)
}
